"""
06. Risk and Portfolio Research: Multi-Timeframe Event Simulator
Simulates trades with institutional execution realism:
- Next-bar Open execution (1-bar decision latency)
- Dynamic spread, slippage, and round-turn broker commissions ($3.00/lot)
- Pessimistic intrabar fill resolution (SL before TP on ambiguous bars)
- Bar-by-bar Mark-to-Market equity tracking (Rule 1: no unrealized loss concealment)
"""
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
from config import ResearchConfig, default_config
from .position_sizing import PositionSizer
from .risk_engine import PortfolioRiskEngine


@dataclass
class SimulatedTrade:
    entry_time: pd.Timestamp
    direction: int         # +1 Buy, -1 Sell
    entry_price: float
    exit_price: float
    exit_time: pd.Timestamp
    lots: float
    pnl_usd: float
    pnl_pips: float
    commission_usd: float
    exit_reason: str


class InstitutionalBacktester:
    def __init__(self, config: ResearchConfig = default_config):
        self.config = config

    def run_simulation(
        self,
        df_ltf: pd.DataFrame,
        signals: pd.Series,
        regimes: pd.Series = None,
        regime_labels: Optional[Dict[int, str]] = None
    ) -> Dict[str, Any]:
        """
        Executes event-driven backtest with strict execution realism:
        - Next-bar Open fills (1-bar decision latency)
        - Spread + slippage + turnover commission
        - Dynamic ATR brackets with trailing stop to breakeven + 1 pip
        - Continuous bar-by-bar Mark-to-Market equity tracking
        """
        capital = float(self.config.initial_capital)
        equity_curve: List[float] = []
        equity_times: List[pd.Timestamp] = []
        trades: List[SimulatedTrade] = []

        active_trade: Optional[Dict[str, Any]] = None
        pending_signal: int = 0
        pending_signal_bar: int = -1

        pip = self.config.pip_size
        cost_pips = self.config.typical_spread_pips + self.config.slippage_pips
        cost_price = cost_pips * pip
        comm_per_lot = getattr(self.config, "commission_per_lot_usd", 3.0)
        delay_bars = getattr(self.config, "execution_delay_bars", 1)

        # Align signals to dataframe index
        common_idx = df_ltf.index.intersection(signals.index)
        data = df_ltf.loc[common_idx]
        sig = signals.loc[common_idx]
        reg = regimes.loc[common_idx] if regimes is not None else pd.Series(0, index=common_idx)

        close_arr = data["close"].to_numpy(dtype=np.float64)
        open_arr = data["open"].to_numpy(dtype=np.float64)
        high_arr = data["high"].to_numpy(dtype=np.float64)
        low_arr = data["low"].to_numpy(dtype=np.float64)
        atr_arr = data["ltf_atr_pips"].to_numpy(dtype=np.float64) if "ltf_atr_pips" in data.columns else np.full(len(data), 5.0)
        times_arr = data.index
        sig_arr = sig.to_numpy(dtype=np.int32)
        reg_arr = reg.to_numpy(dtype=np.int32)
        n_bars = len(data)

        for i in range(n_bars):
            close = close_arr[i]
            open_p = open_arr[i]
            high = high_arr[i]
            low = low_arr[i]
            atr_val = atr_arr[i] * pip
            signal_val = sig_arr[i]
            regime_val = reg_arr[i]
            t = times_arr[i]

            # -------------------------------------------------------------
            # STEP 1: Execute Pending Order on Bar Open (Configurable Execution Latency)
            # -------------------------------------------------------------
            if active_trade is None and pending_signal != 0 and i >= pending_signal_bar + delay_bars:
                sl_dist_pips = max(atr_val / pip * self.config.atr_stop_mult, 8.0)
                tp_dist_pips = max(atr_val / pip * self.config.atr_take_profit_mult, 16.0)

                # Position Sizing (Fixed Fractional or Dynamic Inverse-Volatility)
                if getattr(self.config, "use_dynamic_sizing", False):
                    from risk_and_portfolio.dynamic_sizer import DynamicPositionSizer
                    if not hasattr(self, "_dynamic_sizer") or self._dynamic_sizer is None:
                        self._dynamic_sizer = DynamicPositionSizer(base_risk_pct=self.config.max_risk_per_trade_pct)
                    reg_name = regime_labels.get(regime_val, "Bullish Momentum Trend") if regime_labels else "Bullish Momentum Trend"
                    lots = self._dynamic_sizer.calculate_lots(
                        equity=capital,
                        stop_loss_pips=sl_dist_pips,
                        current_atr_pips=atr_arr[i],
                        regime_label=reg_name
                    )
                else:
                    lots = PositionSizer.fixed_fractional(
                        equity=capital,
                        risk_pct=self.config.max_risk_per_trade_pct,
                        stop_loss_pips=sl_dist_pips
                    )

                # Fill at Bar Open + Spread + Slippage
                if pending_signal == 1:  # Long
                    entry_p = open_p + cost_price
                    sl_p = entry_p - (sl_dist_pips * pip)
                    tp_p = entry_p + (tp_dist_pips * pip)
                else:  # Short
                    entry_p = open_p - cost_price
                    sl_p = entry_p + (sl_dist_pips * pip)
                    tp_p = entry_p - (tp_dist_pips * pip)

                # Deduct entry half-turn commission
                comm_entry = (comm_per_lot / 2.0) * lots
                capital -= comm_entry

                active_trade = {
                    "entry_time": t,
                    "entry_bar_idx": i,
                    "direction": pending_signal,
                    "entry_price": entry_p,
                    "stop_loss": sl_p,
                    "take_profit": tp_p,
                    "lots": lots,
                    "comm_entry": comm_entry,
                    "max_favorable": entry_p,
                    "min_favorable": entry_p
                }
                pending_signal = 0
                pending_signal_bar = -1

            # -------------------------------------------------------------
            # STEP 2: Manage Active Trade (Trailing Stop & Exit Evaluation)
            # -------------------------------------------------------------
            if active_trade is not None:
                d = active_trade["direction"]
                sl = active_trade["stop_loss"]
                tp = active_trade["take_profit"]
                entry_p_active = active_trade["entry_price"]
                bars_held = i - active_trade["entry_bar_idx"]

                # Trailing stop: move stop to breakeven + 1 pip after 50% of TP reached
                if d == 1:
                    max_favorable = max(active_trade.get("max_favorable", entry_p_active), high)
                    active_trade["max_favorable"] = max_favorable
                    tp_dist = tp - entry_p_active
                    if max_favorable >= entry_p_active + tp_dist * 0.5:
                        trail_stop = entry_p_active + pip
                        if trail_stop > sl:
                            sl = trail_stop
                            active_trade["stop_loss"] = sl
                elif d == -1:
                    min_favorable = min(active_trade.get("min_favorable", entry_p_active), low)
                    active_trade["min_favorable"] = min_favorable
                    tp_dist = entry_p_active - tp
                    if min_favorable <= entry_p_active - tp_dist * 0.5:
                        trail_stop = entry_p_active - pip
                        if trail_stop < sl:
                            sl = trail_stop
                            active_trade["stop_loss"] = sl

                exit_price = None
                reason = ""

                # Pessimistic Intrabar Resolution: if both SL and TP in bar range, SL triggers first
                if d == 1:  # Long
                    hit_sl = (low <= sl)
                    hit_tp = (high >= tp)
                    if hit_sl and hit_tp:
                        exit_price = sl
                        reason = "STOP_LOSS_PESSIMISTIC"
                    elif hit_sl:
                        exit_price = sl
                        reason = "STOP_LOSS"
                    elif hit_tp:
                        exit_price = tp
                        reason = "TAKE_PROFIT"
                    elif bars_held >= self.config.forward_bars * 2:
                        exit_price = close
                        reason = "TIME_DECAY"
                    elif signal_val == -1:  # Reversal signal
                        exit_price = close - cost_price
                        reason = "SIGNAL_REVERSAL"
                elif d == -1:  # Short
                    hit_sl = (high >= sl)
                    hit_tp = (low <= tp)
                    if hit_sl and hit_tp:
                        exit_price = sl
                        reason = "STOP_LOSS_PESSIMISTIC"
                    elif hit_sl:
                        exit_price = sl
                        reason = "STOP_LOSS"
                    elif hit_tp:
                        exit_price = tp
                        reason = "TAKE_PROFIT"
                    elif bars_held >= self.config.forward_bars * 2:
                        exit_price = close
                        reason = "TIME_DECAY"
                    elif signal_val == 1:  # Reversal signal
                        exit_price = close + cost_price
                        reason = "SIGNAL_REVERSAL"

                if exit_price is not None:
                    # Deduct exit half-turn commission
                    comm_exit = (comm_per_lot / 2.0) * active_trade["lots"]
                    tot_commission = active_trade["comm_entry"] + comm_exit
                    capital -= comm_exit

                    # Settle trade PnL
                    pnl_pips = ((exit_price - active_trade["entry_price"]) / pip) if d == 1 else ((active_trade["entry_price"] - exit_price) / pip)
                    gross_pnl_usd = (pnl_pips * 10.0) * active_trade["lots"]
                    net_pnl_usd = gross_pnl_usd  # commissions already deducted from capital
                    capital += gross_pnl_usd

                    trades.append(SimulatedTrade(
                        entry_time=active_trade["entry_time"],
                        direction=d,
                        entry_price=active_trade["entry_price"],
                        exit_price=exit_price,
                        exit_time=t,
                        lots=active_trade["lots"],
                        pnl_usd=net_pnl_usd - tot_commission,
                        pnl_pips=pnl_pips,
                        commission_usd=tot_commission,
                        exit_reason=reason
                    ))
                    active_trade = None

            # -------------------------------------------------------------
            # STEP 3: Register Signal for Next-Bar Execution (If Flat)
            # -------------------------------------------------------------
            if active_trade is None and signal_val != 0 and i < n_bars - 1:
                pending_signal = signal_val
                pending_signal_bar = i

            # -------------------------------------------------------------
            # STEP 4: Mark-to-Market Bar Equity Calculation (Rule 1)
            # Marks open positions to market on every bar; prevents hidden unrealized loss
            # -------------------------------------------------------------
            if active_trade is not None:
                d = active_trade["direction"]
                unrealized_pips = ((close - active_trade["entry_price"]) / pip) if d == 1 else ((active_trade["entry_price"] - close) / pip)
                unrealized_usd = (unrealized_pips * 10.0) * active_trade["lots"]
                # Include expected exit commission
                comm_exit_est = (comm_per_lot / 2.0) * active_trade["lots"]
                bar_mtm_equity = capital + unrealized_usd - comm_exit_est
            else:
                bar_mtm_equity = capital

            equity_curve.append(bar_mtm_equity)
            equity_times.append(t)

        trade_pnls = [tr.pnl_usd for tr in trades]
        risk_metrics = PortfolioRiskEngine.compute_risk_analytics(
            equity_curve=equity_curve,
            trade_pnls=trade_pnls,
            equity_times=equity_times
        )
        risk_metrics["equity_curve"] = equity_curve
        risk_metrics["equity_times"] = equity_times
        risk_metrics["trade_objects"] = trades

        return risk_metrics
