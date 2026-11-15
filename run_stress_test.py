"""
Institutional Stress Testing & Robustness Audit Runner
Executes Monte Carlo Trade Resampling, Drawdown Distribution Profiling,
Transaction Cost (Spread/Slippage) Sensitivity, and Risk-Sizing Optimization.
"""
import json
import logging
import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import default_config, DATA_DIR, MODELS_DIR, REPORTS_DIR
from data_engineering.loader import HistoricalDataLoader
from data_engineering.resampler import MultiTimeframeResampler
from machine_learning.features import MultiTimeframeFeatureEngineer
from market_regime.regime_classifier import MarketRegimeDetector
from signal_combination.meta_learner import RegimeGatedEnsemble
from risk_and_portfolio.backtester import InstitutionalBacktester
from risk_and_portfolio.stress_tester import InstitutionalStressTester

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("InstitutionalStressTest")


def run_full_stress_test():
    cfg = default_config
    print("=" * 78)
    print("       INSTITUTIONAL STRESS TESTING & ROBUSTNESS AUDIT: EUR/USD")
    print("=" * 78)

    # 1. Load Data
    print("\n[STEP 1: INGESTING VERIFIED MARKET DATA]")
    loader = HistoricalDataLoader(cfg)
    m1_df = loader.get_or_generate_m1_data(years=cfg.research_years)
    resampler = MultiTimeframeResampler(cfg)
    tf_data = resampler.resample_all(m1_df)
    ltf_df = tf_data["ltf"]
    mtf_df = tf_data["mtf"]
    htf_df = tf_data["htf"]

    # 2. Market Regime & Features
    print("\n[STEP 2: COMPUTING REGIMES & FEATURE MATRIX]")
    detector = MarketRegimeDetector(n_states=cfg.hmm_states, random_state=cfg.random_seed)
    regime_df, regime_report = detector.fit_and_predict(mtf_df)

    fe = MultiTimeframeFeatureEngineer(cfg)
    proc_df, feature_cols = fe.extract_features(ltf_df, mtf_df, htf_df, regime_df)

    split_idx = int(len(proc_df) * cfg.train_ratio)
    test_df = proc_df.iloc[split_idx:]
    X_test = test_df[feature_cols].values
    test_idx = test_df.index
    test_regimes = test_df["regime_id"].values if "regime_id" in test_df.columns else None

    # 3. Load Trained Model Artifacts
    print("\n[STEP 3: LOADING DUAL INSTITUTIONAL MODELS]")
    xgb_l = joblib.load(MODELS_DIR / "xgboost_long_model.pkl")
    lgb_l = joblib.load(MODELS_DIR / "lightgbm_long_model.pkl")
    rf_l = joblib.load(MODELS_DIR / "random_forest_long_model.pkl")

    xgb_s = joblib.load(MODELS_DIR / "xgboost_short_model.pkl")
    lgb_s = joblib.load(MODELS_DIR / "lightgbm_short_model.pkl")
    rf_s = joblib.load(MODELS_DIR / "random_forest_short_model.pkl")

    long_probs = {
        "xgboost": xgb_l.predict_proba(X_test)[:, 1],
        "lightgbm": lgb_l.predict_proba(X_test)[:, 1],
        "random_forest": rf_l.predict_proba(X_test)[:, 1]
    }
    short_probs = {
        "xgboost": xgb_s.predict_proba(X_test)[:, 1],
        "lightgbm": lgb_s.predict_proba(X_test)[:, 1],
        "random_forest": rf_s.predict_proba(X_test)[:, 1]
    }

    # 4. Generate Baseline Signals
    print("\n[STEP 4: GENERATING REGIME-GATED SIGNALS]")
    ensemble = RegimeGatedEnsemble()
    ensemble_df = ensemble.combine_dual_predictions(
        long_probs,
        short_probs,
        regimes=test_regimes,
        regime_labels=regime_report["regime_labels"],
        percentile_cutoff=99.0,
        margin=0.02
    )

    sig_series = pd.Series(ensemble_df["trade_signal"].values, index=test_idx)
    reg_series = pd.Series(ensemble_df["regime_id"].values, index=test_idx)

    # 5. Baseline Simulation
    backtester = InstitutionalBacktester(cfg)
    base_res = backtester.run_simulation(ltf_df, sig_series, reg_series)
    trades = base_res["trade_objects"]
    print(f"  > Baseline Verified: {len(trades)} trades | Return: {base_res['total_return_pct']:+.2f}% | MaxDD: {base_res['max_drawdown_pct']:.2f}% | Win Rate: {base_res['win_rate_pct']:.1f}% | PF: {base_res['profit_factor']:.2f}")

    # 6. Monte Carlo Simulation (2,500 Bootstrap Runs)
    print("\n[STEP 5: MONTE CARLO BOOTSTRAP PATH RESAMPLING (2,500 RUNS)]")
    stress_tester = InstitutionalStressTester(cfg)
    mc_res = stress_tester.run_monte_carlo(trades, initial_capital=cfg.initial_capital, n_simulations=2500)

    print("  ================ MONTE CARLO STRESS TEST RESULTS ================")
    print(f"  Simulations Run:                 {mc_res['n_simulations']:,}")
    print(f"  Trades Resampled per Run:        {mc_res['n_trades']:,}")
    print(f"  Median Expected Net Return:      {mc_res['median_return_pct']:+.2f}%")
    print(f"  90% Return Confidence Interval:  [{mc_res['return_5th_pct']:+.2f}%, {mc_res['return_95th_pct']:+.2f}%]")
    print(f"  Median Peak-to-Trough Drawdown:  {mc_res['median_max_dd_pct']:.2f}%")
    print(f"  95% Drawdown-at-Risk (95% VaR):  {mc_res['max_dd_95th_pct']:.2f}%")
    print(f"  99% Worst-Case Drawdown (99%):   {mc_res['max_dd_99th_pct']:.2f}%")
    print(f"  Probability of Drawdown > 5%:    {mc_res['prob_dd_exceeds_5pct']:.2f}%")
    print(f"  Probability of Drawdown > 10%:   {mc_res['prob_dd_exceeds_10pct']:.2f}%")
    print(f"  Probability of Capital Ruin:     {mc_res['prob_ruin_50pct']:.2f}%")
    print("  =================================================================")

    # Plot Monte Carlo Chart
    mc_plot_path = REPORTS_DIR / "monte_carlo_drawdown.png"
    stress_tester.plot_monte_carlo(mc_res, mc_plot_path)
    print(f"\n  [Visual Chart Saved] {mc_plot_path}")

    # 7. Transaction Cost / Friction Sensitivity Sweep
    print("\n[STEP 6: TRANSACTION FRICTION (SPREAD & SLIPPAGE) STRESS SWEEP]")
    friction_df = stress_tester.run_frictional_sensitivity_sweep(
        ltf_df, sig_series, reg_series,
        spread_values=[0.6, 0.8, 1.2, 1.5, 2.0, 2.5],
        slippage_values=[0.2]
    )

    print("  " + friction_df[["total_friction_pips", "total_return_pct", "sharpe_ratio", "max_drawdown_pct", "win_rate_pct", "profit_factor"]].to_string(index=False))

    # Identify break-even spread
    profitable_rows = friction_df[friction_df["profit_factor"] > 1.0]
    breakeven_friction = profitable_rows["total_friction_pips"].max() if len(profitable_rows) > 0 else 0.0
    print(f"\n  > Maximum Tolerable Friction Threshold: {breakeven_friction:.1f} pips total round-trip cost.")

    # 8. Sizing & Leverage Stress Sweep
    print("\n[STEP 7: RISK SIZING & LEVERAGE SENSITIVITY SWEEP]")
    sizing_df = stress_tester.run_risk_sizing_sweep(
        ltf_df, sig_series, reg_series,
        risk_pct_levels=[0.0025, 0.0050, 0.0075, 0.0100, 0.0150]
    )
    print("  " + sizing_df[["risk_pct_per_trade", "total_return_pct", "sharpe_ratio", "max_drawdown_pct", "profit_factor", "calmar_ratio"]].to_string(index=False))

    # 9. Save Tear Sheet JSON
    tear_sheet_path = REPORTS_DIR / "robustness_tear_sheet.json"
    tear_sheet_data = {
        "monte_carlo": {
            "n_simulations": mc_res["n_simulations"],
            "median_return_pct": round(mc_res["median_return_pct"], 2),
            "return_5th_pct": round(mc_res["return_5th_pct"], 2),
            "return_95th_pct": round(mc_res["return_95th_pct"], 2),
            "median_max_dd_pct": round(mc_res["median_max_dd_pct"], 2),
            "max_dd_95th_pct": round(mc_res["max_dd_95th_pct"], 2),
            "max_dd_99th_pct": round(mc_res["max_dd_99th_pct"], 2),
            "prob_dd_exceeds_5pct": round(mc_res["prob_dd_exceeds_5pct"], 2),
            "prob_dd_exceeds_10pct": round(mc_res["prob_dd_exceeds_10pct"], 2),
            "prob_ruin_50pct": round(mc_res["prob_ruin_50pct"], 2),
        },
        "friction_sensitivity": friction_df.to_dict(orient="records"),
        "risk_sizing_sensitivity": sizing_df.to_dict(orient="records"),
        "max_tolerable_friction_pips": breakeven_friction
    }

    with open(tear_sheet_path, "w") as f:
        json.dump(tear_sheet_data, f, indent=4)
    print(f"\n  [Audit Summary Saved] {tear_sheet_path}")
    print("\n>>> Robustness & Monte Carlo Stress Testing Completed Successfully <<<")


if __name__ == "__main__":
    run_full_stress_test()
