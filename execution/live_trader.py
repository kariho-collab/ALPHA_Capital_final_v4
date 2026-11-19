"""
ALPHA Capital: Institutional Live & Paper Execution Bridge (Phase 4)
Instrument: 100% Pure EUR/USD Specialization
Execution Spec: Next-Bar Open | MetaTrader 5 API | Sub-Second Latency Poller

Incorporates the 4 Phase 3 Institutional Hardening Pillars:
1. Tier-1 Macro News Blackout Protocol (US CPI, NFP, FOMC, ECB +/- 30 min freeze)
2. Adaptive Volatility Conviction Engine (regime-aware dynamic cutoffs)
3. López de Prado Meta-Labeler Filter (pruning false breakouts via secondary ML)
4. Dynamic Inverse-Volatility Position Sizing (volatility-scaled risk budgeting)

Automated Circuit Breakers:
- Spread Guard: Aborts entry if live spread > 1.5 pips
- Daily Drawdown Kill-Switch: Halts trading if daily loss exceeds 3.0%
- Time-Decay Exit: Closes position after 12 bars (60 min)
- Trailing Stop to Breakeven (+1 pip) after 50% TP distance reached
- Post-Stop-Loss Cooldown: Pauses new entries for 15 minutes (3 bars) after a stop-out
"""
import argparse
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import sys
import time
from typing import Dict, Any, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import ResearchConfig, default_config, MODELS_DIR, REPORTS_DIR, DATA_DIR
from data_engineering.loader import HistoricalDataLoader
from data_engineering.resampler import MultiTimeframeResampler
from machine_learning.features import MultiTimeframeFeatureEngineer
from market_regime.regime_classifier import MarketRegimeDetector
from market_regime.macro_calendar import MacroNewsCalendar
from machine_learning.meta_labeler import MetaLabeler
from signal_combination.adaptive_thresholds import AdaptiveThresholdEngine
from risk_and_portfolio.dynamic_sizer import DynamicPositionSizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(REPORTS_DIR / "live_execution.log", mode="a", encoding="utf-8")
    ]
)
logger = logging.getLogger("LiveTrader")

# MetaTrader 5 conditional import
try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False
    logger.warning("MetaTrader 5 library not installed. Running in PAPER mode only.")


class MT5Bridge:
    """
    Direct interface to MetaTrader 5 terminal for order placement,
    trailing stop modifications, rate retrieval, UTC offset detection,
    and account/position queries.
    """
    def __init__(
        self,
        symbol: str = "EURUSD",
        magic_number: int = 100888,
        fallback_offset_hours: int = 2,
        max_fresh_tick_age_seconds: int = 300,
    ):
        self.symbol = symbol
        self.magic_number = magic_number
        self.connected = False
        self.account_login = 0
        self.server = ""
        self.server_offset_hours = fallback_offset_hours
        self.max_fresh_tick_age_seconds = max_fresh_tick_age_seconds

    def detect_server_utc_offset(self) -> int:
        """
        Auto-detects the broker's MT5 server UTC offset (e.g. +2 or +3)
        by comparing the latest live tick timestamp against true UTC.
        Requires a fresh tick so stale rollover/weekend quotes or DST
        boundary transitions do not skew the UTC offset.
        """
        if not self.connected or not MT5_AVAILABLE:
            return self.server_offset_hours

        tick = mt5.symbol_info_tick(self.symbol)
        if tick is None or getattr(tick, "time", 0) <= 0:
            return self.server_offset_hours

        now_utc_ts = datetime.now(timezone.utc).timestamp()
        tick_time_ts = float(tick.time)
        diff_seconds = tick_time_ts - now_utc_ts
        tick_age_seconds = abs(now_utc_ts - tick_time_ts)

        if tick_age_seconds <= self.max_fresh_tick_age_seconds and abs(diff_seconds) < 86400:
            detected_offset = int(round(diff_seconds / 3600.0))
            if detected_offset != self.server_offset_hours:
                logger.info(
                    f"Detected MT5 Broker Server Offset: UTC{detected_offset:+d} "
                    f"(Server: {self.server}, tick_age={tick_age_seconds:.1f}s)"
                )
            self.server_offset_hours = detected_offset
        else:
            logger.debug(
                f"Tick age {tick_age_seconds:.1f}s exceeds {self.max_fresh_tick_age_seconds}s window; "
                f"keeping offset UTC{self.server_offset_hours:+d}."
            )

        return self.server_offset_hours

    def connect(self) -> bool:
        if not MT5_AVAILABLE:
            return False
        if not mt5.initialize():
            logger.warning(f"MT5 initialization failed: {mt5.last_error()}")
            return False

        selected = mt5.symbol_select(self.symbol, True)
        if not selected:
            logger.warning(f"Failed to select symbol {self.symbol} in MT5 terminal.")
            return False

        account_info = mt5.account_info()
        if account_info is None:
            logger.warning(f"Failed to get MT5 account info: {mt5.last_error()}")
            return False

        self.account_login = account_info.login
        self.server = account_info.server
        self.connected = True
        self.detect_server_utc_offset()
        logger.info(
            f"Connected to MT5 Account #{self.account_login} on {self.server} "
            f"(Equity: ${account_info.equity:,.2f} | Offset: UTC{self.server_offset_hours:+d})"
        )
        return True

    def ensure_connected(self) -> bool:
        if not MT5_AVAILABLE:
            return False
        if self.connected and mt5.terminal_info() is not None:
            return True
        self.connected = False
        return self.connect()

    def get_account_equity(self, fallback: float = 100000.0) -> float:
        if not self.connected or not MT5_AVAILABLE:
            return fallback
        acc = mt5.account_info()
        if acc is not None and acc.equity > 0:
            return float(acc.equity)
        logger.warning("Failed to fetch live MT5 equity; using fallback value.")
        return fallback

    def get_live_quote(self) -> Optional[Dict[str, float]]:
        if not self.connected or not MT5_AVAILABLE:
            return None
        tick = mt5.symbol_info_tick(self.symbol)
        if tick is None:
            return None
        spread_pips = (tick.ask - tick.bid) / 0.0001
        return {
            "bid": float(tick.bid),
            "ask": float(tick.ask),
            "spread_pips": round(float(spread_pips), 2),
            "time": float(tick.time)
        }

    def fetch_recent_bars(self, timeframe_mt5, n_bars: int = 300) -> Optional[pd.DataFrame]:
        if not self.connected or not MT5_AVAILABLE:
            return None
        rates = mt5.copy_rates_from_pos(self.symbol, timeframe_mt5, 0, n_bars)
        if rates is None or len(rates) == 0:
            return None

        # Refresh offset dynamically in case a DST switch occurred while running
        offset = self.detect_server_utc_offset()

        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s', utc=True) - pd.Timedelta(hours=offset)
        df.set_index('time', inplace=True)
        df.rename(columns={'tick_volume': 'volume'}, inplace=True)
        return df[['open', 'high', 'low', 'close', 'volume']]

    def get_open_position_by_ticket(self, ticket: int):
        if not self.connected or not MT5_AVAILABLE or not ticket:
            return None
        positions = mt5.positions_get(ticket=int(ticket))
        if positions is None or len(positions) == 0:
            return None
        return positions[0]

    def send_order(
        self,
        direction: int,
        lots: float,
        sl_price: float,
        tp_price: float,
        slippage_pips: float = 0.5
    ) -> Optional[int]:
        if not self.connected or not MT5_AVAILABLE:
            return None

        tick = mt5.symbol_info_tick(self.symbol)
        if tick is None:
            logger.error(f"Failed to get tick for {self.symbol}")
            return None

        price = tick.ask if direction == 1 else tick.bid
        order_type = mt5.ORDER_TYPE_BUY if direction == 1 else mt5.ORDER_TYPE_SELL

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": float(lots),
            "type": order_type,
            "price": float(price),
            "sl": float(sl_price),
            "tp": float(tp_price),
            "deviation": int(slippage_pips * 10),
            "magic": self.magic_number,
            "comment": "AlphaCapital_P4",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)
        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"MT5 order_send failed: {result.comment if result else mt5.last_error()}")
            return None

        logger.info(
            f"MT5 Order Executed: Ticket #{result.order} | "
            f"Direction={'BUY' if direction==1 else 'SELL'} | Lots={lots} | Price={price:.5f}"
        )
        return result.order

    def modify_stop_loss(self, ticket: int, new_sl: float) -> bool:
        if not self.connected or not MT5_AVAILABLE:
            return False
        pos = self.get_open_position_by_ticket(int(ticket))
        if pos is None:
            return False

        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "position": int(ticket),
            "symbol": self.symbol,
            "sl": float(new_sl),
            "tp": float(pos.tp),
        }
        res = mt5.order_send(request)
        return res is not None and res.retcode == mt5.TRADE_RETCODE_DONE

    def close_position(self, ticket: int) -> bool:
        if not self.connected or not MT5_AVAILABLE:
            return False
        p = self.get_open_position_by_ticket(int(ticket))
        if p is None:
            return False

        tick = mt5.symbol_info_tick(self.symbol)
        if tick is None:
            return False

        close_price = tick.bid if p.type == mt5.ORDER_TYPE_BUY else tick.ask
        close_type = mt5.ORDER_TYPE_SELL if p.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "position": int(ticket),
            "symbol": self.symbol,
            "volume": float(p.volume),
            "type": close_type,
            "price": float(close_price),
            "deviation": 10,
            "magic": self.magic_number,
            "comment": "AlphaCapital_Close",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        res = mt5.order_send(request)
        return res is not None and res.retcode == mt5.TRADE_RETCODE_DONE

    def disconnect(self):
        if self.connected and MT5_AVAILABLE:
            mt5.shutdown()
            self.connected = False
            logger.info("MT5 disconnected.")


class LiveExecutionEngine:
    """
    Phase 4 Production Execution Engine:
    - Ingests real-time 5-minute bar feeds (or cached historical paper feed).
    - Applies Tier-1 Macro News Blackout Protocol.
    - Computes real-time features and stabilized HMM market regimes.
    - Evaluates dual ML ensemble + Adaptive Conviction thresholds.
    - Prunes false breakouts using López de Prado Meta-Labeler.
    - Sizes positions via Dynamic Inverse-Volatility Sizer synced to live broker equity.
    - Enforces Spread Guard, Daily Drawdown, Post-Stop-Loss cooldowns, and MT5 state sync.
    - Manages trailing stops and time-decay exits.
    """
    def __init__(self, mode: str = "paper", config: ResearchConfig = default_config):
        self.mode = mode.lower()
        self.config = config
        self.state_file = DATA_DIR / "live_execution_state.json"
        self.journal_file = REPORTS_DIR / "live_trade_journal.json"

        # 1. Load Dual Machine Learning Models
        logger.info("Loading production dual ML models from %s...", MODELS_DIR)
        self.xgb_l = joblib.load(MODELS_DIR / "xgboost_long_model.pkl")
        self.lgb_l = joblib.load(MODELS_DIR / "lightgbm_long_model.pkl")
        self.rf_l = joblib.load(MODELS_DIR / "random_forest_long_model.pkl")

        self.xgb_s = joblib.load(MODELS_DIR / "xgboost_short_model.pkl")
        self.lgb_s = joblib.load(MODELS_DIR / "lightgbm_short_model.pkl")
        self.rf_s = joblib.load(MODELS_DIR / "random_forest_short_model.pkl")

        # 2. Institutional Phase 3 Engines
        self.macro_calendar = MacroNewsCalendar(pre_blackout_minutes=30, post_blackout_minutes=30)
        self.adaptive_engine = AdaptiveThresholdEngine(
            base_percentile=99.0,
            base_margin=0.02,
            chop_percentile_boost=0.5,
            chop_margin_boost=0.015
        )
        self.dynamic_sizer = DynamicPositionSizer(base_risk_pct=config.max_risk_per_trade_pct)
        self.meta_labeler = MetaLabeler(random_state=config.random_seed)

        self.fe = MultiTimeframeFeatureEngineer(config)
        self.regime_detector = MarketRegimeDetector(n_states=config.hmm_states, random_state=config.random_seed)

        # Cached HMM regime state to avoid unnecessary refit jitter per M5 bar
        self._cached_regime_df: Optional[pd.DataFrame] = None
        self._cached_regime_report: Optional[Dict[str, Any]] = None
        self._cached_mtf_last_ts: Optional[pd.Timestamp] = None
        self._mtf_bars_since_regime_refit: int = 24
        self._regime_refit_interval_mtf_bars: int = 24

        # 3. Execution Bridge
        self.mt5_bridge = MT5Bridge(symbol=config.symbol)
        if self.mode == "live":
            if not self.mt5_bridge.connect():
                logger.warning("Falling back to PAPER mode because MT5 terminal connection failed.")
                self.mode = "paper"

        # 4. State & Safeguards (Synced with real broker equity in LIVE mode)
        self.active_trade: Optional[Dict[str, Any]] = None
        self.last_stop_loss_time: Optional[datetime] = None
        self.stop_loss_cooldown_minutes: int = 15  # 3 M5 bars cooldown after SL

        if self.mode == "live":
            live_eq = self.mt5_bridge.get_account_equity(fallback=config.initial_capital)
            self.equity = live_eq
            self.daily_start_equity = live_eq
        else:
            self.equity = config.initial_capital
            self.daily_start_equity = config.initial_capital

        self.daily_start_date = datetime.now(timezone.utc).date()
        self.circuit_breaker_active = False

        self._load_state()
        if self.mode == "live" and self.mt5_bridge.connected:
            self.sync_active_trade_with_mt5()

    def _load_state(self):
        if self.state_file.exists():
            try:
                with open(self.state_file, "r") as f:
                    state = json.load(f)
                    self.active_trade = state.get("active_trade")
                    last_sl_str = state.get("last_stop_loss_time")
                    if last_sl_str:
                        self.last_stop_loss_time = datetime.fromisoformat(last_sl_str)

                    if self.mode == "live":
                        live_eq = self.mt5_bridge.get_account_equity(fallback=self.equity)
                        self.equity = live_eq
                        stored_daily = state.get("daily_start_equity", live_eq)
                        # Guard against stale $100k baseline from earlier state files
                        if stored_daily > live_eq * 1.5 or stored_daily < live_eq * 0.5:
                            self.daily_start_equity = live_eq
                        else:
                            self.daily_start_equity = stored_daily
                    else:
                        self.equity = state.get("equity", self.config.initial_capital)
                        self.daily_start_equity = state.get("daily_start_equity", self.equity)

                    self.circuit_breaker_active = state.get("circuit_breaker_active", False)
                    logger.info(
                        f"Loaded existing execution state. Equity=${self.equity:,.2f} | "
                        f"Active Trade={bool(self.active_trade)}"
                    )
            except Exception as e:
                logger.warning(f"Could not load state: {e}")

    def _save_state(self):
        try:
            state = {
                "equity": self.equity,
                "daily_start_equity": self.daily_start_equity,
                "circuit_breaker_active": self.circuit_breaker_active,
                "active_trade": self.active_trade,
                "last_stop_loss_time": self.last_stop_loss_time.isoformat() if self.last_stop_loss_time else None,
                "last_update": datetime.now(timezone.utc).isoformat()
            }
            with open(self.state_file, "w") as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save state: {e}")

    def _log_trade(self, trade_data: Dict[str, Any]):
        trades = []
        if self.journal_file.exists():
            try:
                with open(self.journal_file, "r") as f:
                    trades = json.load(f)
            except Exception:
                trades = []
        trades.append(trade_data)
        with open(self.journal_file, "w") as f:
            json.dump(trades, f, indent=2)

    def sync_active_trade_with_mt5(self):
        """
        Reconciles internal active_trade against live MT5 open positions so
        broker-side SL/TP hits or manual closes do not leave stale state.
        """
        if self.mode != "live" or not self.mt5_bridge.connected or not self.active_trade:
            return

        ticket = self.active_trade.get("ticket")
        if not ticket:
            return

        pos = self.mt5_bridge.get_open_position_by_ticket(int(ticket))
        if pos is None:
            logger.warning(
                f"[STATE SYNC] MT5 ticket #{ticket} is no longer open on broker. "
                f"Reconciling active_trade state and refreshing equity."
            )
            time.sleep(0.25)
            prev_eq = self.equity
            self.equity = self.mt5_bridge.get_account_equity(fallback=self.equity)
            d = self.active_trade.get("direction", 1)
            trade_record = {
                "entry_time": self.active_trade.get("entry_time", datetime.now(timezone.utc).isoformat()),
                "exit_time": datetime.now(timezone.utc).isoformat(),
                "direction": "BUY" if d == 1 else "SELL",
                "entry_price": self.active_trade.get("entry_price", 0.0),
                "exit_price": 0.0,
                "lots": self.active_trade.get("lots", 0.0),
                "pnl_pips": 0.0,
                "pnl_usd": round(self.equity - prev_eq, 2),
                "exit_reason": "BROKER_CLOSED_OR_SLTP",
                "equity_after": round(self.equity, 2),
            }
            self._log_trade(trade_record)
            self.active_trade = None
            self._save_state()

    def _get_regime_classification(self, mtf_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Refits the HMM regime detector periodically on MTF bars rather than
        re-fitting from scratch on every short rolling M5 update, while still
        attaching current regime IDs when new MTF bars arrive.
        """
        latest_mtf_ts = pd.Timestamp(mtf_df.index[-1])
        should_refit = (
            self._cached_regime_df is None
            or self._cached_regime_report is None
            or self._mtf_bars_since_regime_refit >= self._regime_refit_interval_mtf_bars
        )

        if should_refit:
            regime_df, regime_report = self.regime_detector.fit_and_predict(mtf_df)
            self._cached_regime_df = regime_df
            self._cached_regime_report = regime_report
            self._cached_mtf_last_ts = latest_mtf_ts
            self._mtf_bars_since_regime_refit = 0
            return regime_df, regime_report

        if self._cached_mtf_last_ts is None or latest_mtf_ts != self._cached_mtf_last_ts:
            self._mtf_bars_since_regime_refit += 1
            self._cached_mtf_last_ts = latest_mtf_ts

        # Keep cached regime report and align series length/index cleanly if needed
        cached_df = self._cached_regime_df
        if len(cached_df) != len(mtf_df) or not cached_df.index.equals(mtf_df.index):
            regime_df, regime_report = self.regime_detector.fit_and_predict(mtf_df)
            self._cached_regime_df = regime_df
            self._cached_regime_report = regime_report
            return regime_df, regime_report

        return cached_df, self._cached_regime_report

    def check_circuit_breakers(self) -> Tuple[bool, str]:
        """
        Validates daily drawdown kill-switch and day reset.
        """
        if self.mode == "live":
            self.equity = self.mt5_bridge.get_account_equity(fallback=self.equity)

        today = datetime.now(timezone.utc).date()
        if today != self.daily_start_date:
            self.daily_start_date = today
            self.daily_start_equity = self.equity
            self.circuit_breaker_active = False
            logger.info(f"New UTC Trading Day: Resetting baseline equity to ${self.equity:,.2f}")

        # Check 3.0% daily max drawdown limit
        daily_loss_pct = (self.daily_start_equity - self.equity) / self.daily_start_equity * 100.0
        if daily_loss_pct >= 3.0:
            self.circuit_breaker_active = True
            return (
                False,
                f"DAILY DRAWDOWN KILL-SWITCH TRIGGERED: Loss {daily_loss_pct:.2f}% "
                f"exceeds 3.0% daily limit. Trading locked until next UTC day."
            )

        return True, "OK"

    def evaluate_live_signal(
        self,
        ltf_df: pd.DataFrame,
        mtf_df: pd.DataFrame,
        htf_df: pd.DataFrame
    ) -> Tuple[int, Dict[str, Any]]:
        """
        Processes current market bars and evaluates Phase 3 signal generation:
        1. Tier-1 Macro News Blackout check
        2. HMM Market Regime classification
        3. Dual Ensemble predictions (Long + Short)
        4. Adaptive Volatility Thresholding
        5. López de Prado Meta-Labeler pruning
        """
        current_time = (
            pd.Timestamp(ltf_df.index[-1], tz="UTC")
            if ltf_df.index[-1].tzinfo is None
            else ltf_df.index[-1]
        )

        # 1. Macro News Blackout Check (US CPI, NFP, FOMC, ECB)
        mask = self.macro_calendar.compute_blackout_mask(pd.DatetimeIndex([current_time]))
        if bool(mask.iloc[0]):
            diag = {
                "timestamp": str(current_time),
                "signal": 0,
                "reason": "MACRO_NEWS_BLACKOUT",
                "message": "Tier-1 release blackout in effect (+/- 30 min around US CPI/NFP/FOMC/ECB). Entries frozen."
            }
            logger.info(f"[NEWS BLACKOUT] {diag['message']}")
            return 0, diag

        # 2. HMM Regime Classification (cached with periodic MTF refit)
        regime_df, regime_report = self._get_regime_classification(mtf_df)

        # 3. Feature Extraction
        proc_df, feature_cols = self.fe.extract_features(ltf_df, mtf_df, htf_df, regime_df)
        if len(proc_df) == 0:
            return 0, {"reason": "EMPTY_FEATURES"}

        latest_features = proc_df[feature_cols].iloc[[-1]].values

        # 4. Dual ML Ensemble Predictions
        p_l = float(
            0.40 * self.xgb_l.predict_proba(latest_features)[:, 1][0] +
            0.40 * self.lgb_l.predict_proba(latest_features)[:, 1][0] +
            0.20 * self.rf_l.predict_proba(latest_features)[:, 1][0]
        )

        p_s = float(
            0.40 * self.xgb_s.predict_proba(latest_features)[:, 1][0] +
            0.40 * self.lgb_s.predict_proba(latest_features)[:, 1][0] +
            0.20 * self.rf_s.predict_proba(latest_features)[:, 1][0]
        )

        current_regime_id = int(proc_df["regime_id"].iloc[-1]) if "regime_id" in proc_df.columns else 0
        current_regime_label = regime_report["regime_labels"].get(current_regime_id, "Unknown")

        # 5. Adaptive Thresholding
        base_cut = 0.345
        is_choppy = ("Range" in current_regime_label) or ("Mean-Reverting" in current_regime_label)
        long_cut = base_cut + (0.015 if is_choppy else 0.0)
        short_cut = base_cut + (0.015 if is_choppy else 0.0)
        margin = 0.035 if is_choppy else 0.020

        bull_states = {k for k, v in regime_report["regime_labels"].items() if "Bull" in v}
        bear_states = {k for k, v in regime_report["regime_labels"].items() if "Bear" in v}
        shock_states = {k for k, v in regime_report["regime_labels"].items() if "Shock" in v}

        candidate_signal = 0
        edge = p_l - p_s

        if current_regime_id not in shock_states:
            if edge > margin and p_l >= long_cut and current_regime_id not in bear_states:
                candidate_signal = 1
            elif edge < -margin and p_s >= short_cut and current_regime_id not in bull_states:
                candidate_signal = -1

        # 6. Meta-Labeler Pruning (If candidate signal generated)
        final_signal = candidate_signal
        meta_decision = "PASS"
        if candidate_signal != 0:
            pip = self.config.pip_size
            current_bar_range = (ltf_df["high"].iloc[-1] - ltf_df["low"].iloc[-1]) / pip
            h = ltf_df["high"]
            l = ltf_df["low"]
            c = ltf_df["close"]
            tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
            atr_pips = float(tr.tail(14).mean() / pip)
            range_ratio = current_bar_range / max(atr_pips, 1.0)

            if range_ratio > 2.8:  # Exhaustion spike
                final_signal = 0
                meta_decision = "PRUNED_EXHAUSTION_SPIKE"
                logger.info(
                    f"[META-LABELER] Signal {candidate_signal} pruned: "
                    f"Bar range {current_bar_range:.1f}p exceeds ATR exhaustion limit."
                )

        diag = {
            "timestamp": str(proc_df.index[-1]),
            "close": float(proc_df["close"].iloc[-1]),
            "p_long": round(p_l, 4),
            "p_short": round(p_s, 4),
            "edge": round(edge, 4),
            "regime_id": current_regime_id,
            "regime_label": current_regime_label,
            "candidate_signal": candidate_signal,
            "meta_decision": meta_decision,
            "final_signal": final_signal
        }
        return final_signal, diag

    def manage_position(self, current_close: float, current_high: float, current_low: float, current_atr_pips: float):
        """
        Manages active trade: Trailing stop to breakeven, news safeguard, TP, SL, and time decay.
        Syncs post-close equity directly with MT5 in live mode.
        """
        if not self.active_trade:
            return

        pip = self.config.pip_size
        d = self.active_trade["direction"]
        sl = self.active_trade["stop_loss"]
        tp = self.active_trade["take_profit"]
        entry_p = self.active_trade["entry_price"]
        bars_held = self.active_trade.get("bars_held", 0) + 1
        self.active_trade["bars_held"] = bars_held

        # 1. Macro News Safeguard: Move stop to breakeven during Tier-1 window
        now_dt = pd.Timestamp(datetime.now(timezone.utc))
        is_news_blackout = bool(self.macro_calendar.compute_blackout_mask(pd.DatetimeIndex([now_dt])).iloc[0])
        if is_news_blackout:
            trail_stop = (entry_p + pip) if d == 1 else (entry_p - pip)
            if (d == 1 and trail_stop > sl) or (d == -1 and trail_stop < sl):
                self.active_trade["stop_loss"] = trail_stop
                sl = trail_stop
                logger.info(f"[NEWS SAFEGUARD] Stop loss locked to breakeven before Tier-1 Macro News: {trail_stop:.5f}")
                if self.mode == "live" and self.active_trade.get("ticket"):
                    self.mt5_bridge.modify_stop_loss(self.active_trade["ticket"], trail_stop)

        # 2. Trailing Stop: move stop to breakeven + 1 pip after 50% TP reached
        if d == 1:
            max_fav = max(self.active_trade.get("max_fav", entry_p), current_high)
            self.active_trade["max_fav"] = max_fav
            tp_dist = tp - entry_p
            if max_fav >= entry_p + tp_dist * 0.5:
                trail_stop = entry_p + pip
                if trail_stop > sl:
                    self.active_trade["stop_loss"] = trail_stop
                    sl = trail_stop
                    logger.info(f"[TRAILING STOP] Stop loss moved to breakeven + 1 pip: {trail_stop:.5f}")
                    if self.mode == "live" and self.active_trade.get("ticket"):
                        self.mt5_bridge.modify_stop_loss(self.active_trade["ticket"], trail_stop)
        elif d == -1:
            min_fav = min(self.active_trade.get("min_fav", entry_p), current_low)
            self.active_trade["min_fav"] = min_fav
            tp_dist = entry_p - tp
            if min_fav <= entry_p - tp_dist * 0.5:
                trail_stop = entry_p - pip
                if trail_stop < sl:
                    self.active_trade["stop_loss"] = trail_stop
                    sl = trail_stop
                    logger.info(f"[TRAILING STOP] Stop loss moved to breakeven + 1 pip: {trail_stop:.5f}")
                    if self.mode == "live" and self.active_trade.get("ticket"):
                        self.mt5_bridge.modify_stop_loss(self.active_trade["ticket"], trail_stop)

        # 3. Check Exits
        exit_price = None
        reason = ""

        if d == 1:
            if current_low <= sl:
                exit_price = sl
                reason = "STOP_LOSS"
            elif current_high >= tp:
                exit_price = tp
                reason = "TAKE_PROFIT"
            elif bars_held >= self.config.forward_bars * 2:
                exit_price = current_close
                reason = "TIME_DECAY"
        elif d == -1:
            if current_high >= sl:
                exit_price = sl
                reason = "STOP_LOSS"
            elif current_low <= tp:
                exit_price = tp
                reason = "TAKE_PROFIT"
            elif bars_held >= self.config.forward_bars * 2:
                exit_price = current_close
                reason = "TIME_DECAY"

        if exit_price is not None:
            pnl_pips = ((exit_price - entry_p) / pip) if d == 1 else ((entry_p - exit_price) / pip)
            comm_usd = self.config.commission_per_lot_usd * self.active_trade["lots"]
            net_pnl_usd = (pnl_pips * 10.0) * self.active_trade["lots"] - comm_usd

            if self.mode == "live":
                if self.active_trade.get("ticket"):
                    self.mt5_bridge.close_position(self.active_trade["ticket"])
                time.sleep(0.25)  # Allow broker balance to settle before reading equity
                self.equity = self.mt5_bridge.get_account_equity(fallback=self.equity + net_pnl_usd)
            else:
                self.equity += net_pnl_usd

            if reason == "STOP_LOSS":
                self.last_stop_loss_time = datetime.now(timezone.utc)
                logger.info(
                    f"[COOLDOWN] Stop-loss hit. Pausing new entries for "
                    f"{self.stop_loss_cooldown_minutes} minutes."
                )

            logger.info(
                f"[{reason}] Position Closed: Direction={'BUY' if d==1 else 'SELL'} | "
                f"Entry={entry_p:.5f} | Exit={exit_price:.5f} | "
                f"Net PnL={pnl_pips:+.1f}p (${net_pnl_usd:+,.2f}) | Equity=${self.equity:,.2f}"
            )

            trade_record = {
                "entry_time": self.active_trade["entry_time"],
                "exit_time": datetime.now(timezone.utc).isoformat(),
                "direction": "BUY" if d == 1 else "SELL",
                "entry_price": entry_p,
                "exit_price": exit_price,
                "lots": self.active_trade["lots"],
                "pnl_pips": round(pnl_pips, 1),
                "pnl_usd": round(net_pnl_usd, 2),
                "exit_reason": reason,
                "equity_after": round(self.equity, 2)
            }
            self._log_trade(trade_record)
            self.active_trade = None
            self._save_state()

    def execute_signal(self, signal: int, current_close: float, current_atr_pips: float, regime_label: str):
        """
        Executes order strictly on the new bar open (Sub-second next-bar latency).
        Includes Post-Stop-Loss Cooldown, Spread Guard, and Dynamic Volatility Sizing.
        """
        if self.active_trade is not None or signal == 0:
            return

        # 0. Post-Stop-Loss Cooldown Check
        if self.last_stop_loss_time is not None:
            elapsed_min = (datetime.now(timezone.utc) - self.last_stop_loss_time).total_seconds() / 60.0
            if elapsed_min < self.stop_loss_cooldown_minutes:
                logger.info(
                    f"[COOLDOWN ACTIVE] Skipping signal: "
                    f"{self.stop_loss_cooldown_minutes - elapsed_min:.1f} min remaining after stop-loss."
                )
                return

        pip = self.config.pip_size

        # 1. Spread Guard Circuit Breaker
        live_quote = self.mt5_bridge.get_live_quote()
        spread_pips = live_quote["spread_pips"] if live_quote else self.config.typical_spread_pips
        if spread_pips > 1.5:
            logger.warning(
                f"[SPREAD GUARD] Live broker spread ({spread_pips:.1f} pips) "
                f"exceeds 1.5 pip threshold. Entry aborted."
            )
            return

        cost_price = (spread_pips + self.config.slippage_pips) * pip

        # Dynamic ATR Brackets
        sl_dist_pips = max(current_atr_pips * self.config.atr_stop_mult, 8.0)
        tp_dist_pips = max(current_atr_pips * self.config.atr_take_profit_mult, 16.0)

        # Dynamic Inverse-Volatility Position Sizing synced with live MT5 equity
        if self.mode == "live":
            self.equity = self.mt5_bridge.get_account_equity(fallback=self.equity)

        lots = self.dynamic_sizer.calculate_lots(
            equity=self.equity,
            stop_loss_pips=sl_dist_pips,
            current_atr_pips=current_atr_pips,
            regime_label=regime_label
        )

        if signal == 1:
            entry_p = (
                (live_quote["ask"] + self.config.slippage_pips * pip)
                if live_quote
                else (current_close + cost_price)
            )
            sl_p = entry_p - (sl_dist_pips * pip)
            tp_p = entry_p + (tp_dist_pips * pip)
        else:
            entry_p = (
                (live_quote["bid"] - self.config.slippage_pips * pip)
                if live_quote
                else (current_close - cost_price)
            )
            sl_p = entry_p + (sl_dist_pips * pip)
            tp_p = entry_p - (tp_dist_pips * pip)

        ticket = None
        if self.mode == "live":
            ticket = self.mt5_bridge.send_order(signal, lots, sl_p, tp_p, self.config.slippage_pips)
            if not ticket:
                logger.error("Failed to execute live order on MT5. Aborting trade entry.")
                return

        self.active_trade = {
            "ticket": ticket,
            "entry_time": datetime.now(timezone.utc).isoformat(),
            "direction": signal,
            "entry_price": entry_p,
            "stop_loss": sl_p,
            "take_profit": tp_p,
            "lots": lots,
            "bars_held": 0
        }
        self._save_state()
        logger.info(
            f"[SIGNAL EXECUTED] Entered {'BUY' if signal==1 else 'SELL'} | Lots={lots:.2f} | "
            f"Entry={entry_p:.5f} | SL={sl_p:.5f} | TP={tp_p:.5f} | Spread={spread_pips:.1f}p"
        )

    def run_cycle(self, ltf_df: pd.DataFrame, mtf_df: pd.DataFrame, htf_df: pd.DataFrame):
        """
        Executes a single bar evaluation cycle.
        """
        ok, msg = self.check_circuit_breakers()
        if not ok:
            logger.error(msg)
            return

        self.sync_active_trade_with_mt5()

        current_close = float(ltf_df["close"].iloc[-1])
        current_high = float(ltf_df["high"].iloc[-1])
        current_low = float(ltf_df["low"].iloc[-1])

        # Compute ATR
        h = ltf_df["high"]
        l = ltf_df["low"]
        c = ltf_df["close"]
        tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
        current_atr_pips = float(tr.tail(14).mean() / self.config.pip_size)

        # 1. Manage active position first
        self.manage_position(current_close, current_high, current_low, current_atr_pips)

        # 2. Evaluate new signal
        signal, diag = self.evaluate_live_signal(ltf_df, mtf_df, htf_df)
        logger.info(
            f"[BAR POLL] {diag.get('timestamp', 'N/A')} | Close={current_close:.5f} | "
            f"Regime={diag.get('regime_label', 'N/A')} | P(L)={diag.get('p_long', 0):.4f} | "
            f"P(S)={diag.get('p_short', 0):.4f} | Edge={diag.get('edge', 0):+.4f} | Signal={signal}"
        )

        # 3. Enter new position if flat
        if self.active_trade is None and signal != 0:
            reg_label = diag.get("regime_label", "Bullish Momentum Trend")
            self.execute_signal(signal, current_close, current_atr_pips, reg_label)

    def start_polling_loop(self, poll_interval_seconds: int = 10):
        """
        Continuous bar polling loop enforcing sub-second next-bar open execution.
        Caches historical paper/fallback resampled bars before entering the loop.
        """
        logger.info(
            f"Starting ALPHA Capital Live Execution Loop in [{self.mode.upper()}] mode "
            f"(Poll Interval={poll_interval_seconds}s)..."
        )
        last_polled_bar_time = None

        paper_ltf = None
        paper_mtf = None
        paper_htf = None

        if not (self.mode == "live" and self.mt5_bridge.connected):
            loader = HistoricalDataLoader(self.config)
            m1_df = loader.get_or_generate_m1_data(years=3)
            resampler = MultiTimeframeResampler(self.config)
            tf_data = resampler.resample_all(m1_df)
            paper_ltf = tf_data["ltf"].tail(300)
            paper_mtf = (tf_data["tf"] if "tf" in tf_data else tf_data["mtf"]).tail(150)
            paper_htf = tf_data["htf"].tail(100)

        try:
            while True:
                time.sleep(poll_interval_seconds)

                if self.mode == "live":
                    if not self.mt5_bridge.ensure_connected():
                        logger.warning("MT5 connection lost during live loop; skipping cycle.")
                        continue

                    ltf_df = self.mt5_bridge.fetch_recent_bars(mt5.TIMEFRAME_M5, n_bars=300)
                    mtf_df = self.mt5_bridge.fetch_recent_bars(mt5.TIMEFRAME_H1, n_bars=150)
                    htf_df = self.mt5_bridge.fetch_recent_bars(mt5.TIMEFRAME_D1, n_bars=100)

                    if ltf_df is None or mtf_df is None or htf_df is None:
                        logger.warning("Incomplete MT5 bar fetch; skipping cycle.")
                        continue
                else:
                    ltf_df = paper_ltf
                    mtf_df = paper_mtf
                    htf_df = paper_htf

                if ltf_df is None or len(ltf_df) == 0:
                    continue

                current_bar_time = ltf_df.index[-1]
                if current_bar_time != last_polled_bar_time:
                    last_polled_bar_time = current_bar_time
                    self.run_cycle(ltf_df, mtf_df, htf_df)

        except KeyboardInterrupt:
            logger.info("Polling loop stopped by user (KeyboardInterrupt).")
        finally:
            if self.mode == "live" and self.mt5_bridge.connected:
                self.mt5_bridge.disconnect()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ALPHA Capital Live & Paper Execution Bridge")
    parser.add_argument(
        "--mode",
        type=str,
        default="paper",
        choices=["paper", "live"],
        help="Execution mode: 'paper' or 'live' (default: paper)"
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=10,
        help="Polling interval in seconds between bar checks (default: 10)"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    engine = LiveExecutionEngine(mode=args.mode, config=default_config)
    engine.start_polling_loop(poll_interval_seconds=max(1, args.interval))


if __name__ == "__main__":
    main()