"""
04. Machine Learning: Multi-Timeframe Feature Synchronization (LTF + MTF + HTF)
Extracts multi-scale features and aligns them with zero lookahead bias.
Includes microstructure proxies, volatility dynamics, session/diurnal cycles,
momentum divergence, and cross-timeframe trend interactions.
"""
from typing import Tuple, List, Dict
import numpy as np
import pandas as pd
from config import ResearchConfig, default_config


class MultiTimeframeFeatureEngineer:
    def __init__(self, config: ResearchConfig = default_config):
        self.config = config

    def extract_features(
        self,
        ltf_df: pd.DataFrame,
        mtf_df: pd.DataFrame,
        htf_df: pd.DataFrame,
        regime_df: pd.DataFrame = None
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Extracts and aligns features across three distinct timeframes:
        - LTF (5m): Microstructure, candlestick momentum, oscillators, order flow proxies, session cycles
        - MTF (1h): Swing trend, moving average slopes, MACD
        - HTF (1D): Macro regime, daily ATR, 200 EMA distance
        """
        data = ltf_df.copy()
        pip = self.config.pip_size

        # --- 1. LTF Execution, Momentum & Oscillators ---
        c = data["close"]
        h = data["high"]
        l = data["low"]
        o = data["open"]
        v = data["volume"]

        # Log returns
        data["ltf_ret_1"] = np.log(c / c.shift(1))
        data["ltf_ret_3"] = np.log(c / c.shift(3))
        data["ltf_ret_6"] = np.log(c / c.shift(6))

        # RSI 14
        delta = c.diff()
        gain = delta.where(delta > 0, 0.0).ewm(alpha=1/14, adjust=False).mean()
        loss = (-delta.where(delta < 0, 0.0)).ewm(alpha=1/14, adjust=False).mean()
        rs = gain / (loss + 1e-10)
        data["ltf_rsi_14"] = 100.0 - (100.0 / (1.0 + rs))

        # ATR 14
        tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
        data["ltf_atr_pips"] = tr.rolling(14).mean() / pip

        # EMA Normalized Distances & Spread
        ema_9 = c.ewm(span=9, adjust=False).mean()
        ema_21 = c.ewm(span=21, adjust=False).mean()
        data["ltf_dist_ema_9"] = (c - ema_9) / (pip * 100)
        data["ltf_dist_ema_21"] = (c - ema_21) / (pip * 100)
        data["ltf_ema_spread"] = (ema_9 - ema_21) / (pip * 100)

        # Candlestick anatomy & Close Location Value (buying/selling pressure)
        data["ltf_body_ratio"] = (c - o) / (h - l + 1e-8)
        data["ltf_clv"] = ((c - l) - (h - c)) / (h - l + 1e-8)

        # --- Microstructure & Order Flow Proxies ---
        v_ma20 = v.rolling(20).mean() + 1e-5
        data["ltf_rel_vol"] = v / v_ma20
        tick_proxy = v / (tr + pip * 0.1)
        data["ltf_tick_intensity"] = tick_proxy / (tick_proxy.rolling(20).mean() + 1e-5)
        data["ltf_parkinson_vol"] = np.sqrt((np.log(h / (l + 1e-8)) ** 2) / (4.0 * np.log(2.0))) / pip
        data["ltf_signed_volume"] = np.sign(c - o) * np.log1p(v)

        # --- Volatility Regime & Dynamics ---
        tr_fast = tr.rolling(6).mean()
        tr_slow = tr.rolling(24).mean() + 1e-8
        data["ltf_vol_ratio"] = tr_fast / tr_slow
        data["ltf_vol_of_vol"] = tr.rolling(24).std() / tr_slow

        # --- Momentum Divergence & Return Autocorrelation ---
        data["ltf_rsi_slope_3"] = data["ltf_rsi_14"].diff(3)
        data["ltf_rsi_divergence"] = data["ltf_rsi_14"].diff(6) - (data["ltf_ret_6"] * 1000.0)
        data["ltf_autocorr_proxy"] = data["ltf_ret_1"] * data["ltf_ret_1"].shift(1) * 1e4

        # --- Session-Aware & Diurnal Cycle Features ---
        idx_dt = pd.to_datetime(data.index, utc=True)
        hours = idx_dt.hour.values
        dow = idx_dt.dayofweek.values
        data["session_hour_sin"] = np.sin(2.0 * np.pi * hours / 24.0)
        data["session_hour_cos"] = np.cos(2.0 * np.pi * hours / 24.0)
        data["session_dow_sin"] = np.sin(2.0 * np.pi * dow / 5.0)
        data["session_london_overlap"] = ((hours >= 12) & (hours < 16)).astype(float)
        data["session_asian"] = ((hours >= 0) & (hours < 7)).astype(float)

        # --- 2. MTF Features (1h Swing Setup - Shifted 1 bar for zero leakage) ---
        mtf_feat = pd.DataFrame(index=mtf_df.index)
        mtf_c = mtf_df["close"]
        mtf_ema_50 = mtf_c.ewm(span=50, adjust=False).mean()
        mtf_feat["mtf_trend_50"] = (mtf_c - mtf_ema_50) / (pip * 100)
        mtf_feat["mtf_ema_slope"] = mtf_ema_50.diff(3) / (pip * 100)

        # MTF MACD
        mtf_ema_12 = mtf_c.ewm(span=12, adjust=False).mean()
        mtf_ema_26 = mtf_c.ewm(span=26, adjust=False).mean()
        mtf_feat["mtf_macd_hist"] = ((mtf_ema_12 - mtf_ema_26) - (mtf_ema_12 - mtf_ema_26).ewm(span=9, adjust=False).mean()) / pip

        # Shift MTF by 1 bar and merge onto LTF index
        mtf_shifted = mtf_feat.shift(1)
        data = data.merge(mtf_shifted, how="left", left_index=True, right_index=True).ffill()

        # Cross-Timeframe Interaction
        data["cross_trend_align"] = np.sign(data["mtf_trend_50"]) * np.sign(data["ltf_ema_spread"])
        data["cross_pullback"] = data["ltf_dist_ema_21"] * np.sign(data["mtf_trend_50"])

        # --- 3. HTF Features (1D Macro Bias - Shifted 1 bar for zero leakage) ---
        htf_feat = pd.DataFrame(index=htf_df.index)
        htf_c = htf_df["close"]
        htf_ema_200 = htf_c.ewm(span=200, adjust=False).mean()
        htf_feat["htf_macro_dist_200"] = (htf_c - htf_ema_200) / (pip * 100)
        htf_shifted = htf_feat.shift(1)
        data = data.merge(htf_shifted, how="left", left_index=True, right_index=True).ffill()

        # --- 4. Market Regime Feature ---
        if regime_df is not None and "regime_id" in regime_df:
            reg_shifted = regime_df[["regime_id"]].shift(1)
            data = data.merge(reg_shifted, how="left", left_index=True, right_index=True).ffill()
            data["regime_id"] = data["regime_id"].fillna(0).astype(int)

        # --- 5. Target Labels (ATR-Adaptive or Fixed Threshold) ---
        future_c = data["close"].shift(-self.config.forward_bars)
        delta_pips = (future_c - data["close"]) / pip

        if getattr(self.config, "use_adaptive_target", True):
            atr_mult = getattr(self.config, "adaptive_target_atr_mult", 0.75)
            min_pips = getattr(self.config, "min_adaptive_target_pips", 3.0)
            target_th = np.maximum(min_pips, data["ltf_atr_pips"] * atr_mult)
        else:
            target_th = self.config.target_pip_threshold

        data["target_long"] = (delta_pips >= target_th).astype(int)
        data["target_short"] = (delta_pips <= -target_th).astype(int)
        data["target_delta_pips"] = delta_pips

        # Clean NaNs
        # Exclude targets, OHLCV, and regime_id (used separately for signal gating, not as ML input)
        exclude = {
            "open", "high", "low", "close", "volume",
            "target_long", "target_short", "target_delta_pips",
            "regime_id", "regime_label"
        }
        feature_cols = [col for col in data.columns if col not in exclude]

        cleaned = data.dropna().copy()
        return cleaned, feature_cols
