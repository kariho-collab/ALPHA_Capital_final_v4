"""
06. Risk and Portfolio Research: Institutional Risk Analytics
Computes Mark-to-Market Daily VaR, CVaR, Sharpe, Sortino, Calmar, and Drawdown profiles.
Adheres strictly to Rule 1: Marks open positions to market every bar and computes risk on resampled Daily returns.
"""
from typing import Dict, Any, List
import numpy as np
import pandas as pd


class PortfolioRiskEngine:
    @staticmethod
    def compute_risk_analytics(
        equity_curve: List[float],
        trade_pnls: List[float],
        equity_times: List[pd.Timestamp] = None
    ) -> Dict[str, Any]:
        """
        Computes institutional risk analytics with mark-to-market daily returns:
        - Daily Resampled Returns R_d
        - Daily Annualized Sharpe (sqrt(252))
        - Daily Annualized Sortino (sqrt(252))
        - Daily 95% and 99% VaR and CVaR
        - Trade-level Profit Factor, Win Rate, Expectancy
        """
        if len(equity_curve) < 2:
            return {}

        eq_arr = np.array(equity_curve)
        initial_cap = float(eq_arr[0])
        final_cap = float(eq_arr[-1])
        total_ret_pct = ((final_cap - initial_cap) / initial_cap) * 100.0

        # 1. Peak-to-Trough Drawdown on continuous MTM equity
        peak = np.maximum.accumulate(eq_arr)
        drawdowns = (eq_arr - peak) / (peak + 1e-10)
        max_dd_pct = float(np.min(drawdowns) * 100.0)

        # 2. Resample to Daily (1D) Mark-to-Market Equity for Institutional Risk Ratios
        if equity_times is not None and len(equity_times) == len(equity_curve):
            eq_series = pd.Series(equity_curve, index=pd.to_datetime(equity_times, utc=True))
            daily_eq = eq_series.resample("1D").last().dropna()
            daily_returns = daily_eq.pct_change().dropna().values
        else:
            # Fallback if timestamps not provided: sample every 288 bars (1 day of 5m bars)
            daily_returns = np.diff(eq_arr[::288]) / eq_arr[::288][:-1] if len(eq_arr) > 288 else np.diff(eq_arr) / eq_arr[:-1]

        # Filter active days (remove zero-volume weekend gaps)
        daily_returns = daily_returns[~np.isnan(daily_returns)]

        if len(daily_returns) > 5:
            d_mean = float(np.mean(daily_returns))
            d_std = float(np.std(daily_returns)) + 1e-10
            daily_sharpe = float((d_mean / d_std) * np.sqrt(252))

            downside_d = daily_returns[daily_returns < 0]
            downside_std = float(np.std(downside_d)) + 1e-10 if len(downside_d) > 0 else 1e-10
            daily_sortino = float((d_mean / downside_std) * np.sqrt(252))

            # Daily VaR and CVaR (95% and 99%)
            var_95 = float(np.percentile(daily_returns, 5) * 100.0)
            cvar_95 = float(np.mean(daily_returns[daily_returns <= np.percentile(daily_returns, 5)]) * 100.0)
            var_99 = float(np.percentile(daily_returns, 1) * 100.0)
            cvar_99 = float(np.mean(daily_returns[daily_returns <= np.percentile(daily_returns, 1)]) * 100.0)
        else:
            daily_sharpe = 0.0
            daily_sortino = 0.0
            var_95 = 0.0
            cvar_95 = 0.0
            var_99 = 0.0
            cvar_99 = 0.0

        calmar = float((total_ret_pct / max(abs(max_dd_pct), 0.1))) if max_dd_pct < 0 else 0.0

        # 3. Trade-Level Analytics
        n_trades = len(trade_pnls)
        if n_trades > 0:
            pnls = np.array(trade_pnls)
            wins = pnls[pnls > 0]
            losses = pnls[pnls < 0]

            win_rate = float(len(wins) / n_trades * 100.0)
            gross_profit = float(np.sum(wins)) if len(wins) > 0 else 0.0
            gross_loss = float(abs(np.sum(losses))) if len(losses) > 0 else 1e-6
            profit_factor = float(gross_profit / gross_loss)
            expectancy_usd = float(np.mean(pnls))
        else:
            win_rate = 0.0
            profit_factor = 0.0
            expectancy_usd = 0.0

        return {
            "initial_capital": round(initial_cap, 2),
            "final_capital": round(final_cap, 2),
            "total_return_pct": round(total_ret_pct, 2),
            "max_drawdown_pct": round(max_dd_pct, 2),
            "annualized_sharpe": round(daily_sharpe, 2),
            "sortino_ratio": round(daily_sortino, 2),
            "calmar_ratio": round(calmar, 2),
            "var_95_pct": round(var_95, 3),
            "cvar_95_expected_shortfall_pct": round(cvar_95, 3),
            "var_99_pct": round(var_99, 3),
            "cvar_99_expected_shortfall_pct": round(cvar_99, 3),
            "total_trades": n_trades,
            "win_rate_pct": round(win_rate, 2),
            "profit_factor": round(profit_factor, 2),
            "trade_expectancy_usd": round(expectancy_usd, 2)
        }
