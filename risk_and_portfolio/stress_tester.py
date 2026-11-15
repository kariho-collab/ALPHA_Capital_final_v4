"""
06. Risk and Portfolio Research: Institutional Stress Testing & Monte Carlo Engine
Performs bootstrap trade path simulations, drawdown distributions, probability of ruin,
and transaction cost / friction sensitivity sweeps.
"""
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from config import ResearchConfig, default_config
from .backtester import InstitutionalBacktester, SimulatedTrade


class InstitutionalStressTester:
    def __init__(self, config: ResearchConfig = default_config):
        self.config = config

    def run_monte_carlo(
        self,
        trades: List[SimulatedTrade],
        initial_capital: float = 100000.0,
        n_simulations: int = 2500,
        random_seed: int = 42
    ) -> Dict[str, Any]:
        """
        Runs Monte Carlo bootstrap resampling on trade PnL outcomes.
        Simulates sequence risk and computes empirical drawdown distribution.
        """
        np.random.seed(random_seed)
        n_trades = len(trades)
        if n_trades == 0:
            return {}

        pnls = np.array([tr.pnl_usd for tr in trades])
        
        sim_equities = np.zeros((n_simulations, n_trades + 1))
        sim_max_dds = np.zeros(n_simulations)
        sim_returns = np.zeros(n_simulations)
        sim_equities[:, 0] = initial_capital

        ruin_count = 0
        dd_10_count = 0
        dd_5_count = 0

        for s in range(n_simulations):
            # Sample with replacement
            sampled_pnls = np.random.choice(pnls, size=n_trades, replace=True)
            equity_path = np.zeros(n_trades + 1)
            equity_path[0] = initial_capital

            current_eq = initial_capital
            peak_eq = initial_capital
            max_dd = 0.0

            for i, pnl in enumerate(sampled_pnls):
                current_eq += pnl
                if current_eq <= 0:
                    current_eq = 0.0
                equity_path[i + 1] = current_eq

                if current_eq > peak_eq:
                    peak_eq = current_eq
                dd = (current_eq - peak_eq) / peak_eq if peak_eq > 0 else 0.0
                if dd < max_dd:
                    max_dd = dd

            sim_equities[s, :] = equity_path
            sim_max_dds[s] = max_dd * 100.0 # in %
            sim_returns[s] = ((equity_path[-1] - initial_capital) / initial_capital) * 100.0

            if equity_path[-1] <= initial_capital * 0.5:
                ruin_count += 1
            if max_dd <= -0.10:
                dd_10_count += 1
            if max_dd <= -0.05:
                dd_5_count += 1

        percentiles = [5, 25, 50, 75, 95]
        equity_bands = {p: np.percentile(sim_equities, p, axis=0) for p in percentiles}

        return {
            "n_simulations": n_simulations,
            "n_trades": n_trades,
            "median_return_pct": float(np.median(sim_returns)),
            "return_5th_pct": float(np.percentile(sim_returns, 5)),
            "return_95th_pct": float(np.percentile(sim_returns, 95)),
            "median_max_dd_pct": float(np.median(sim_max_dds)),
            "max_dd_95th_pct": float(np.percentile(sim_max_dds, 5)),  # 5th percentile is severe drawdown
            "max_dd_99th_pct": float(np.percentile(sim_max_dds, 1)),
            "prob_dd_exceeds_5pct": float(dd_5_count / n_simulations * 100.0),
            "prob_dd_exceeds_10pct": float(dd_10_count / n_simulations * 100.0),
            "prob_ruin_50pct": float(ruin_count / n_simulations * 100.0),
            "sim_max_dds": sim_max_dds,
            "sim_returns": sim_returns,
            "equity_bands": equity_bands
        }

    def run_frictional_sensitivity_sweep(
        self,
        ltf_df: pd.DataFrame,
        signals: pd.Series,
        regimes: pd.Series,
        spread_values: List[float] = None,
        slippage_values: List[float] = None
    ) -> pd.DataFrame:
        """
        Tests strategy durability across varying institutional & retail transaction costs.
        Identifies the break-even friction threshold.
        """
        spreads = spread_values or [0.6, 0.8, 1.2, 1.5, 2.0, 2.5]
        slippage = slippage_values or [0.2]

        results = []

        for sp in spreads:
            for sl in slippage:
                cfg_test = ResearchConfig(
                    symbol=self.config.symbol,
                    pip_size=self.config.pip_size,
                    typical_spread_pips=sp,
                    slippage_pips=sl,
                    initial_capital=self.config.initial_capital,
                    max_risk_per_trade_pct=self.config.max_risk_per_trade_pct,
                    atr_stop_mult=self.config.atr_stop_mult,
                    atr_take_profit_mult=self.config.atr_take_profit_mult
                )
                bt = InstitutionalBacktester(cfg_test)
                res = bt.run_simulation(ltf_df, signals, regimes)

                results.append({
                    "spread_pips": sp,
                    "slippage_pips": sl,
                    "total_friction_pips": round(sp + sl, 2),
                    "total_return_pct": res["total_return_pct"],
                    "sharpe_ratio": res["annualized_sharpe"],
                    "max_drawdown_pct": res["max_drawdown_pct"],
                    "win_rate_pct": res["win_rate_pct"],
                    "profit_factor": res["profit_factor"],
                    "expectancy_usd": res["trade_expectancy_usd"],
                    "total_trades": res["total_trades"]
                })

        return pd.DataFrame(results)

    def run_risk_sizing_sweep(
        self,
        ltf_df: pd.DataFrame,
        signals: pd.Series,
        regimes: pd.Series,
        risk_pct_levels: List[float] = None
    ) -> pd.DataFrame:
        """
        Evaluates performance and drawdowns across different risk-per-trade fractions.
        Verifies optimality of half-Kelly position sizing.
        """
        risks = risk_pct_levels or [0.0025, 0.005, 0.0075, 0.010, 0.015]
        results = []

        for r_pct in risks:
            cfg_test = ResearchConfig(
                symbol=self.config.symbol,
                pip_size=self.config.pip_size,
                typical_spread_pips=self.config.typical_spread_pips,
                slippage_pips=self.config.slippage_pips,
                initial_capital=self.config.initial_capital,
                max_risk_per_trade_pct=r_pct,
                atr_stop_mult=self.config.atr_stop_mult,
                atr_take_profit_mult=self.config.atr_take_profit_mult
            )
            bt = InstitutionalBacktester(cfg_test)
            res = bt.run_simulation(ltf_df, signals, regimes)

            results.append({
                "risk_pct_per_trade": round(r_pct * 100, 2),
                "total_return_pct": res["total_return_pct"],
                "sharpe_ratio": res["annualized_sharpe"],
                "max_drawdown_pct": res["max_drawdown_pct"],
                "profit_factor": res["profit_factor"],
                "calmar_ratio": res["calmar_ratio"],
                "final_capital": res["final_capital"]
            })

        return pd.DataFrame(results)

    @staticmethod
    def plot_monte_carlo(mc_results: Dict[str, Any], save_path: Path):
        """
        Plots Monte Carlo equity confidence bands and drawdown distribution.
        """
        bands = mc_results["equity_bands"]
        dds = mc_results["sim_max_dds"]
        n_trades = mc_results["n_trades"]
        trade_x = np.arange(n_trades + 1)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={"height_ratios": [1.5, 1]})

        # Top: Confidence Bands
        ax1.plot(trade_x, bands[50], color="#2980b9", linewidth=2.0, label="Median Path (50th %ile)")
        ax1.fill_between(trade_x, bands[25], bands[75], color="#3498db", alpha=0.3, label="Interquartile Range (25th-75th %ile)")
        ax1.fill_between(trade_x, bands[5], bands[95], color="#2980b9", alpha=0.15, label="90% Confidence Interval (5th-95th %ile)")
        ax1.axhline(bands[50][0], color="gray", linestyle="--", alpha=0.7)
        ax1.set_title(f"Monte Carlo Equity Trajectory Distribution ({mc_results['n_simulations']:,} Bootstrap Simulations)", fontsize=12, fontweight="bold")
        ax1.set_ylabel("Portfolio Capital ($)")
        ax1.legend(loc="upper left")
        ax1.grid(True, alpha=0.3)

        # Bottom: Maximum Drawdown Histogram
        ax2.hist(dds, bins=40, color="#e74c3c", alpha=0.75, edgecolor="black", linewidth=0.5)
        ax2.axvline(mc_results["median_max_dd_pct"], color="#2c3e50", linestyle="-", linewidth=2, label=f"Median MaxDD: {mc_results['median_max_dd_pct']:.1f}%")
        ax2.axvline(mc_results["max_dd_95th_pct"], color="#c0392b", linestyle="--", linewidth=2, label=f"95% VaR MaxDD: {mc_results['max_dd_95th_pct']:.1f}%")
        ax2.axvline(mc_results["max_dd_99th_pct"], color="#7f1d1d", linestyle=":", linewidth=2, label=f"99% VaR MaxDD: {mc_results['max_dd_99th_pct']:.1f}%")
        ax2.set_title("Maximum Peak-to-Trough Drawdown Distribution", fontsize=11, fontweight="bold")
        ax2.set_xlabel("Maximum Drawdown (%)")
        ax2.set_ylabel("Simulation Frequency")
        ax2.legend(loc="upper left")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
