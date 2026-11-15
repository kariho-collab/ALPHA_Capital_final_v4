"""
ALPHA Capital: Master Phase 2 Statistical Proof & Institutional Robustness Runner
Executes comprehensive quantitative verification under the ethos:
"Research before complexity"

Pillars of Verification:
1. Deflated Sharpe Ratio (DSR) & Probabilistic Sharpe Ratio (PSR) [Bailey & López de Prado (2014)]
2. Multi-Period Rolling Walk-Forward Matrix (4 Chronological Slices, zero lookahead)
3. Regime-Stratified Performance Attribution (HMM Bull / Bear / Range)
4. Adverse Execution Latency & Transaction Friction Matrix (1-bar vs 2-bar latency, fee shocks)
5. Monte Carlo Drawdown & Permutation Overfitting Analysis (2,500 resampled paths, PBO)
6. Publication of institutional reports and charts
"""
import sys
import json
import logging
import shutil
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import default_config, REPORTS_DIR, MODELS_DIR
from data_engineering.loader import HistoricalDataLoader
from data_engineering.resampler import MultiTimeframeResampler
from market_regime.regime_classifier import MarketRegimeDetector
from machine_learning.features import MultiTimeframeFeatureEngineer
from machine_learning.tree_models import TabularMLTrainer
from signal_combination.meta_learner import RegimeGatedEnsemble
from signal_combination.calibrator import ProbabilityCalibrator
from risk_and_portfolio.backtester import InstitutionalBacktester
from risk_and_portfolio.phase2_robustness import (
    DeflatedSharpeCalculator,
    RollingWalkForwardEvaluator,
    RegimeAttributionAuditor,
    FrictionStressTester,
    MonteCarloPermutationTester,
    Phase2Plotter
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Phase2Proof")

# Artifact Directory
ARTIFACT_DIR = Path(r"C:\Users\lenovo\.gemini\antigravity\brain\e79b9f78-d430-4d9a-9055-42188e511cfd")


def run_phase2_suite():
    cfg = default_config
    print("=" * 80)
    print("      ALPHA CAPITAL: PHASE 2 STATISTICAL PROOF & INSTITUTIONAL ROBUSTNESS")
    print("      ETHOS: 'Research before complexity'")
    print("=" * 80)

    # -------------------------------------------------------------
    # 1. LOAD DATA & ENGINEER FEATURES
    # -------------------------------------------------------------
    print("\n[STEP 1: INGESTING VERIFIED DATA & REGIME DETECTION]")
    loader = HistoricalDataLoader(cfg)
    base_m1 = loader.get_or_generate_m1_data(years=cfg.research_years)
    resampler = MultiTimeframeResampler(cfg)
    tf_data = resampler.resample_all(base_m1)
    ltf_df = tf_data["ltf"]
    mtf_df = tf_data["mtf"]
    htf_df = tf_data["htf"]

    detector = MarketRegimeDetector(n_states=cfg.hmm_states, random_state=cfg.random_seed)
    regime_df, regime_report = detector.fit_and_predict(mtf_df)
    regime_labels = regime_report["regime_labels"]

    fe = MultiTimeframeFeatureEngineer(cfg)
    proc_df, feature_cols = fe.extract_features(ltf_df, mtf_df, htf_df, regime_df)
    print(f"  > Clean Features Synchronized: {len(proc_df):,} bars x {len(feature_cols)} features.")

    # -------------------------------------------------------------
    # 2. RUN STRICT 3-WAY SPLIT TO REPRODUCE PHASE 1 UNTOUCHED TEST SET
    # -------------------------------------------------------------
    print("\n[STEP 2: COMPUTING PHASE 1 UNTOUCHED BASELINE (3-WAY SPLIT)]")
    trainer = TabularMLTrainer(random_state=cfg.random_seed)
    dual_results = trainer.evaluate_dual(
        proc_df,
        feature_cols,
        train_ratio=cfg.train_ratio,
        val_ratio=cfg.val_ratio,
        run_cv=False
    )
    long_res = dual_results["long"]
    short_res = dual_results["short"]
    test_idx = dual_results["test_indices"]

    test_long_probs = {k: long_res[k]["test_probabilities"] for k in ("xgboost", "lightgbm", "random_forest")}
    test_short_probs = {k: short_res[k]["test_probabilities"] for k in ("xgboost", "lightgbm", "random_forest")}
    test_regimes = proc_df.loc[test_idx, "regime_id"].values if "regime_id" in proc_df.columns else None

    ensemble = RegimeGatedEnsemble()
    ensemble_df = ensemble.combine_dual_predictions(
        test_long_probs,
        test_short_probs,
        regimes=test_regimes,
        regime_labels=regime_labels,
        percentile_cutoff=99.0,
        margin=0.02
    )

    sig_series = pd.Series(ensemble_df["trade_signal"].values, index=test_idx)
    reg_series = pd.Series(ensemble_df["regime_id"].values, index=test_idx)

    backtester = InstitutionalBacktester(cfg)
    base_sim = backtester.run_simulation(ltf_df, sig_series, reg_series)
    trades = base_sim["trade_objects"]
    print(f"  > Baseline Verified: {len(trades)} trades | Net Return: {base_sim['total_return_pct']:+.2f}% | MaxDD: {base_sim['max_drawdown_pct']:.2f}% | Daily Sharpe: {base_sim['annualized_sharpe']:.2f} | PF: {base_sim['profit_factor']:.2f}")

    # Extract Daily Mark-to-Market Returns
    eq_series = pd.Series(base_sim["equity_curve"], index=pd.to_datetime(base_sim["equity_times"], utc=True))
    daily_eq = eq_series.resample("1D").last().dropna()
    daily_returns = daily_eq.pct_change().dropna().to_numpy()

    # -------------------------------------------------------------
    # 3. PILLAR A: DEFLATED SHARPE RATIO & PROBABILISTIC SHARPE RATIO
    # -------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  PILLAR A: DEFLATED SHARPE RATIO (DSR) & SELECTION BIAS AUDIT")
    print("  [Bailey & López de Prado (2014) - Extreme Value Asymptotic Theory]")
    print("=" * 70)

    psr_results = DeflatedSharpeCalculator.compute_psr(daily_returns, benchmark_sr=0.0)
    print(f"  Sample Days:                     {psr_results['sample_days']} daily observations")
    print(f"  Annualized Daily Sharpe (sqrt252): {psr_results['annualized_sharpe']:.4f}")
    print(f"  Return Skewness:                 {psr_results['skewness']:+.4f}")
    print(f"  Return Kurtosis (Pearson):       {psr_results['kurtosis']:.4f}")
    print(f"  Asymptotic Standard Error:       {psr_results['sr_std_err_ann']:.4f}")
    print(f"  Z-Statistic (vs H0: SR <= 0):    {psr_results['z_stat']:.4f}")
    print(f"  Probabilistic Sharpe Ratio (PSR):{psr_results['psr']:.6f} ({psr_results['prob_true_edge_pct']:.2f}% confidence)")

    dsr_trials = [10, 25, 50, 100, 250]
    dsr_table = []
    for n in dsr_trials:
        dsr_info = DeflatedSharpeCalculator.compute_dsr(daily_returns, n_trials=n)
        dsr_table.append(dsr_info)
        print(f"  DSR [N = {n:<3} Trials]: E[max(SR)] = {dsr_info['expected_max_sr_under_null']:.4f} | DSR = {dsr_info['deflated_sharpe_ratio']:.6f} (P(True Edge) = {dsr_info['prob_true_edge_pct']}%)")

    # -------------------------------------------------------------
    # 4. PILLAR B: MULTI-PERIOD ROLLING WALK-FORWARD MATRIX
    # -------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  PILLAR B: MULTI-PERIOD ROLLING WALK-FORWARD MATRIX")
    print("  [4 Chronological Out-of-Sample Slices — Zero Lookahead]")
    print("=" * 70)

    rolling_eval = RollingWalkForwardEvaluator(cfg)
    walk_forward_slices = rolling_eval.run_rolling_matrix(
        ltf_df=ltf_df,
        proc_df=proc_df,
        feature_cols=feature_cols,
        regime_labels=regime_labels,
        n_slices=4
    )

    wf_df = pd.DataFrame(walk_forward_slices)
    disp_cols = ["period_label", "total_return_pct", "annualized_sharpe", "sortino_ratio", "max_drawdown_pct", "win_rate_pct", "profit_factor", "total_trades"]
    print("  " + wf_df[disp_cols].to_string(index=False))

    wf_chart_path = REPORTS_DIR / "phase2_walk_forward_matrix.png"
    Phase2Plotter.plot_walk_forward_matrix(walk_forward_slices, wf_chart_path)
    print(f"  > Walk-Forward Matrix Visual Saved: {wf_chart_path}")

    # -------------------------------------------------------------
    # 5. PILLAR C: REGIME-STRATIFIED PERFORMANCE ATTRIBUTION
    # -------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  PILLAR C: LATENT MARKET REGIME STRATIFIED ATTRIBUTION")
    print("  [Auditing Performance Across HMM Bull, Bear, and Range States]")
    print("=" * 70)

    regime_summary = RegimeAttributionAuditor.audit_by_regime(
        trades=trades,
        regime_labels=regime_labels,
        regimes_series=reg_series
    )
    reg_df = pd.DataFrame(regime_summary)
    print("  " + reg_df[["regime", "total_trades", "win_rate_pct", "profit_factor", "total_pnl_usd", "mean_pnl_usd", "long_trades", "short_trades"]].to_string(index=False))

    reg_chart_path = REPORTS_DIR / "phase2_regime_attribution.png"
    Phase2Plotter.plot_regime_attribution(regime_summary, reg_chart_path)
    print(f"  > Regime Attribution Visual Saved: {reg_chart_path}")

    # -------------------------------------------------------------
    # 6. PILLAR D: ADVERSE LATENCY & FRICTION STRESS MATRIX
    # -------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  PILLAR D: ADVERSE LATENCY & FRICTION STRESS MATRIX")
    print("  [Next-Bar vs 2-Bar Latency, Commission Shocks, Spread Widening]")
    print("=" * 70)

    friction_tester = FrictionStressTester(cfg)
    friction_results = friction_tester.run_friction_matrix(ltf_df, sig_series, reg_series)
    fric_df = pd.DataFrame(friction_results)
    disp_fric = ["scenario", "spread_pips", "slippage_pips", "commission_usd", "latency_bars", "total_return_pct", "annualized_sharpe", "profit_factor", "is_profitable"]
    print("  " + fric_df[disp_fric].to_string(index=False))

    profitable_scenarios = [f for f in friction_results if f["is_profitable"]]
    print(f"\n  Surviving Stress Scenarios: {len(profitable_scenarios)} / {len(friction_results)} tested.")

    fric_chart_path = REPORTS_DIR / "phase2_friction_stress.png"
    Phase2Plotter.plot_friction_stress(friction_results, fric_chart_path)
    print(f"  > Friction Stress Visual Saved: {fric_chart_path}")

    # -------------------------------------------------------------
    # 7. PILLAR E: MONTE CARLO DRAWDOWN & PERMUTATION OVERFITTING
    # -------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  PILLAR E: MONTE CARLO DRAWDOWN & PERMUTATION OVERFITTING")
    print("  [2,500 Bootstrap Permutations of Untouched Test Trades]")
    print("=" * 70)

    mc_results = MonteCarloPermutationTester.run_permutation_test(
        trade_objects=trades,
        initial_capital=cfg.initial_capital,
        n_simulations=2500,
        random_state=cfg.random_seed
    )
    print(f"  Permutations Evaluated:          {mc_results['n_simulations']:,}")
    print(f"  Median Net Return:               {mc_results['median_return_pct']:+.2f}%")
    print(f"  90% Confidence Interval:         [{mc_results['return_5th_pct']:+.2f}%, {mc_results['return_95th_pct']:+.2f}%]")
    print(f"  Median Max Drawdown:             {mc_results['median_max_dd_pct']:.2f}%")
    print(f"  95% Drawdown-at-Risk (95% VaR):  {mc_results['max_dd_95th_pct']:.2f}%")
    print(f"  99% Extreme Drawdown (99%):      {mc_results['max_dd_99th_pct']:.2f}%")
    print(f"  Probability of Drawdown > 5%:    {mc_results['prob_dd_exceeds_5pct']:.2f}%")
    print(f"  Probability of Drawdown > 10%:   {mc_results['prob_dd_exceeds_10pct']:.2f}%")
    print(f"  Probability of Drawdown > 15%:   {mc_results['prob_dd_exceeds_15pct']:.2f}%")
    print(f"  Probability of Ruin (DD > 20%):  {mc_results['prob_ruin_pct']:.2f}%")
    print(f"  Probability of Loss (PBO est):   {mc_results['pbo_probability_of_loss']:.4f}")

    mc_chart_path = REPORTS_DIR / "phase2_monte_carlo_drawdown.png"
    Phase2Plotter.plot_monte_carlo(mc_results, mc_chart_path)
    print(f"  > Monte Carlo Visual Saved: {mc_chart_path}")

    # Copy visual charts to Artifact Directory
    if ARTIFACT_DIR.exists():
        for p in [wf_chart_path, reg_chart_path, fric_chart_path, mc_chart_path]:
            target_path = ARTIFACT_DIR / p.name
            shutil.copyfile(p, target_path)
            print(f"  > Synced Artifact Chart: {target_path}")

    # -------------------------------------------------------------
    # 8. SAVE COMPREHENSIVE PHASE 2 REPORT JSON
    # -------------------------------------------------------------
    report_path = REPORTS_DIR / "phase2_robustness_report.json"
    full_report = {
        "title": "ALPHA Capital Phase 2: Statistical Proof & Institutional Robustness",
        "ethos": "Research before complexity",
        "pillars": {
            "pillar_a_deflated_sharpe": {
                "psr": psr_results,
                "dsr_trials": dsr_table
            },
            "pillar_b_rolling_walk_forward_matrix": walk_forward_slices,
            "pillar_c_regime_attribution": regime_summary,
            "pillar_d_friction_and_latency_stress": friction_results,
            "pillar_e_monte_carlo_permutation": {k: v for k, v in mc_results.items() if k != "sample_equity_paths"}
        }
    }
    with open(report_path, "w") as f:
        json.dump(full_report, f, indent=4)
    print(f"\n  [Comprehensive Report Saved] {report_path}")
    print("=" * 80)
    print(">>> PHASE 2 STATISTICAL PROOF & ROBUSTNESS COMPLETED SUCCESSFULLY <<<")
    print("=" * 80)


if __name__ == "__main__":
    run_phase2_suite()
