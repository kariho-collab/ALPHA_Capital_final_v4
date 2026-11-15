"""
03. Market Regime Detection: Semantic Regime Classifier & Transition Matrix
"""
import logging
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from .hmm_model import GaussianHMM

logger = logging.getLogger(__name__)


class MarketRegimeDetector:
    def __init__(self, n_states: int = 3, random_state: int = 42):
        self.n_states = n_states
        self.model = GaussianHMM(n_states=n_states, random_state=random_state)
        self.state_labels_: Dict[int, str] = {}
        self.feature_names_: list = []

    def extract_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extracts orthogonal features driving financial regimes:
        1. Normalized Log Return (Trend Direction)
        2. Realized Range / ATR (Volatility Level)
        3. Momentum Velocity (Distance from 50 EMA)
        """
        data = pd.DataFrame(index=df.index)
        close = df["close"]
        high = df["high"]
        low = df["low"]

        # 1. Return
        data["log_ret"] = np.log(close / close.shift(1))

        # 2. Normalized Volatility Range
        tr = high - low
        atr = tr.rolling(window=14).mean()
        data["norm_vol"] = tr / (atr + 1e-10)

        # 3. Trend Velocity (Normalized distance from 50 EMA)
        ema_50 = close.ewm(span=50, adjust=False).mean()
        data["trend_velocity"] = (close - ema_50) / (atr + 1e-10)

        data = data.dropna()
        self.feature_names_ = list(data.columns)
        return data

    def fit_and_predict(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        feat_df = self.extract_regime_features(df)
        X = feat_df.values

        # Standardize features
        mean = X.mean(axis=0)
        std = X.std(axis=0) + 1e-8
        X_norm = (X - mean) / std

        logger.info(f"Fitting Gaussian HMM across {len(X_norm):,} observations for {self.n_states} regimes...")
        self.model.fit(X_norm)
        state_seq = self.model.predict(X_norm)

        feat_df["regime_id"] = state_seq

        # Interpret states semantically based on cluster means
        # Return index 0 is log_ret, index 1 is norm_vol
        state_means = self.model.means_
        state_dict = {}

        # Use relative ranking of state means rather than absolute thresholds
        # This guarantees we always get distinct regime labels regardless of
        # the standardization scale.
        ret_means = state_means[:, 0]  # log_ret z-score means
        vol_means = state_means[:, 1]  # normalized vol z-score means

        # Sort states by return mean: lowest → Bear, highest → Bull, middle → Range
        ret_order = np.argsort(ret_means)  # ascending: [bear_idx, range_idx, bull_idx]
        vol_order = np.argsort(vol_means)  # ascending: lowest vol → highest vol

        # If highest-vol state is substantially more volatile, mark it as Shock
        # Otherwise use a pure return-based ranking
        high_vol_state = vol_order[-1]
        vol_spread = vol_means[high_vol_state] - np.median(vol_means)

        if self.n_states == 3:
            if vol_spread > 0.8:
                # High vol state is a genuine shock regime
                label_map = {}
                label_map[high_vol_state] = (
                    "High-Volatility Shock / Breakdown" if ret_means[high_vol_state] < 0
                    else "High-Volatility Expansion"
                )
                remaining = [s for s in ret_order if s != high_vol_state]
                if ret_means[remaining[0]] < ret_means[remaining[1]]:
                    label_map[remaining[0]] = "Bearish Momentum Trend"
                    label_map[remaining[1]] = "Bullish Momentum Trend"
                else:
                    label_map[remaining[0]] = "Bullish Momentum Trend"
                    label_map[remaining[1]] = "Bearish Momentum Trend"
                state_dict = label_map
            else:
                # No clear shock — assign Bear / Range / Bull by return ranking
                state_dict[ret_order[0]] = "Bearish Momentum Trend"
                state_dict[ret_order[1]] = "Low-Volatility Range (Mean-Reverting)"
                state_dict[ret_order[2]] = "Bullish Momentum Trend"
        else:
            # Generic fallback for n_states != 3
            for k in range(self.n_states):
                rm = ret_means[k]
                vm = vol_means[k]
                if vm > 0.5:
                    label = "High-Volatility Shock / Breakdown" if rm < 0 else "High-Volatility Expansion"
                elif rm > 0.15:
                    label = "Bullish Momentum Trend"
                elif rm < -0.15:
                    label = "Bearish Momentum Trend"
                else:
                    label = "Low-Volatility Range (Mean-Reverting)"
                state_dict[k] = label

        self.state_labels_ = state_dict
        feat_df["regime_label"] = [self.state_labels_[s] for s in state_seq]

        # Transition Matrix Analysis
        trans_mat = np.round(self.model.trans_mat_, 4)
        persistence = {
            self.state_labels_[k]: round(float(trans_mat[k, k] * 100), 2)
            for k in range(self.n_states)
        }

        # Expected duration in bars = 1 / (1 - P_ii)
        expected_durations = {
            self.state_labels_[k]: round(float(1.0 / (1.0 - min(0.999, trans_mat[k, k]))), 1)
            for k in range(self.n_states)
        }

        report = {
            "regime_labels": self.state_labels_,
            "transition_matrix": trans_mat.tolist(),
            "regime_persistence_pct": persistence,
            "expected_state_duration_bars": expected_durations,
            "regime_distribution": feat_df["regime_label"].value_counts(normalize=True).round(4).to_dict()
        }

        return feat_df, report
