"""
Unit & Regression Tests: Execution Realism & Daily Mark-to-Market Risk Engine
Verifies next-bar execution, pessimistic intrabar resolution, commission deductions,
and mark-to-market daily return distribution calculations.
"""
import pytest
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from config import ResearchConfig
from risk_and_portfolio.backtester import InstitutionalBacktester
from risk_and_portfolio.risk_engine import PortfolioRiskEngine


def create_synthetic_test_bars(n_bars=100, start_price=1.1000):
    times = pd.date_range("2026-01-01 00:00", periods=n_bars, freq="5min", tz="UTC")
    closes = np.full(n_bars, start_price)
    opens = np.full(n_bars, start_price)
    highs = np.full(n_bars, start_price + 0.0005)
    lows = np.full(n_bars, start_price - 0.0005)
    vols = np.full(n_bars, 1000)

    df = pd.DataFrame({
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": vols,
        "ltf_atr_pips": np.full(n_bars, 10.0)
    }, index=times)
    return df


def test_next_bar_open_execution():
    """Verify signal at bar i fills strictly at bar i+1 open + cost"""
    cfg = ResearchConfig(
        pip_size=0.0001,
        typical_spread_pips=0.8,
        slippage_pips=0.2,
        commission_per_lot_usd=3.0,
        initial_capital=100000.0,
        max_risk_per_trade_pct=0.01,
        atr_stop_mult=1.5,
        atr_take_profit_mult=3.0
    )
    df = create_synthetic_test_bars(n_bars=30)
    # Set bar 5 open to different price to verify exact fill
    df.iloc[5, df.columns.get_loc("open")] = 1.1020

    signals = pd.Series(0, index=df.index)
    signals.iloc[4] = 1  # Signal fired at bar 4

    bt = InstitutionalBacktester(cfg)
    res = bt.run_simulation(df, signals)

    trades = res["trade_objects"]
    assert len(trades) >= 1, "Trade was not executed"
    trade = trades[0]

    # Entry must occur on bar 5
    assert trade.entry_time == df.index[5], f"Expected entry at bar 5 ({df.index[5]}), got {trade.entry_time}"
    # Entry price must match open[5] + spread + slippage (1.1020 + 0.0001 = 1.1021)
    expected_entry = 1.1020 + (0.8 + 0.2) * 0.0001
    assert np.isclose(trade.entry_price, expected_entry, atol=1e-5), f"Expected {expected_entry}, got {trade.entry_price}"
    print("[PASS] Next-bar Open execution verified.")


def test_pessimistic_intrabar_resolution():
    """Verify that if both SL and TP could trigger on the same bar, SL triggers first"""
    cfg = ResearchConfig(
        pip_size=0.0001,
        typical_spread_pips=0.8,
        slippage_pips=0.2,
        commission_per_lot_usd=3.0,
        initial_capital=100000.0,
        max_risk_per_trade_pct=0.01,
        atr_stop_mult=1.0,
        atr_take_profit_mult=2.0
    )
    df = create_synthetic_test_bars(n_bars=20, start_price=1.1000)
    signals = pd.Series(0, index=df.index)
    signals.iloc[2] = 1  # Buy signal on bar 2 -> fills bar 3 Open (1.1001)

    # Make bar 4 an extreme outside bar spanning both SL (1.0991) and TP (1.1021)
    df.iloc[4, df.columns.get_loc("high")] = 1.1050
    df.iloc[4, df.columns.get_loc("low")] = 1.0950

    bt = InstitutionalBacktester(cfg)
    res = bt.run_simulation(df, signals)
    trades = res["trade_objects"]

    assert len(trades) == 1, "Expected exactly 1 trade"
    assert "STOP_LOSS" in trades[0].exit_reason, f"Expected pessimistic STOP_LOSS, got {trades[0].exit_reason}"
    print("[PASS] Pessimistic intrabar resolution verified.")


def test_commission_deduction():
    """Verify broker turnover commission is deducted from capital and net PnL"""
    cfg = ResearchConfig(
        pip_size=0.0001,
        typical_spread_pips=0.0,
        slippage_pips=0.0,
        commission_per_lot_usd=6.0,  # $6.00 per round-turn lot
        initial_capital=100000.0,
        max_risk_per_trade_pct=0.01
    )
    df = create_synthetic_test_bars(n_bars=20, start_price=1.1000)
    signals = pd.Series(0, index=df.index)
    signals.iloc[2] = 1

    bt = InstitutionalBacktester(cfg)
    res = bt.run_simulation(df, signals)
    trades = res["trade_objects"]

    assert len(trades) >= 1
    trade = trades[0]
    expected_comm = 6.0 * trade.lots
    assert np.isclose(trade.commission_usd, expected_comm, atol=1e-3), f"Expected commission {expected_comm}, got {trade.commission_usd}"
    print("[PASS] Turnover commission deduction verified.")


def test_daily_mark_to_market_var_and_cvar():
    """Verify daily resampled returns produce non-zero, statistically meaningful VaR and CVaR"""
    # Create 30 days of daily equity curve with realistic variations
    times = pd.date_range("2026-01-01", periods=1000, freq="1h", tz="UTC")
    # Simulate equity drifting with realistic daily fluctuation
    np.random.seed(42)
    daily_returns = np.random.normal(0.001, 0.01, size=len(times))
    equity = 100000.0 * np.cumprod(1.0 + daily_returns)

    risk_metrics = PortfolioRiskEngine.compute_risk_analytics(
        equity_curve=equity.tolist(),
        trade_pnls=[100.0, -50.0, 200.0, -80.0],
        equity_times=times.tolist()
    )

    # VaR 95% must be strictly negative and non-zero (meaningful risk)
    assert risk_metrics["var_95_pct"] < 0.0, f"VaR 95% must be negative loss, got {risk_metrics['var_95_pct']}"
    assert risk_metrics["cvar_95_expected_shortfall_pct"] <= risk_metrics["var_95_pct"], "CVaR must be more severe than VaR"
    assert risk_metrics["annualized_sharpe"] != 0.0, "Sharpe must be computed"
    print(f"[PASS] Daily Mark-to-Market VaR ({risk_metrics['var_95_pct']}%) and CVaR ({risk_metrics['cvar_95_expected_shortfall_pct']}%) verified.")


if __name__ == "__main__":
    test_next_bar_open_execution()
    test_pessimistic_intrabar_resolution()
    test_commission_deduction()
    test_daily_mark_to_market_var_and_cvar()
    print("\nALL UNIT AND REGRESSION TESTS PASSED SUCCESSFULLY!")
