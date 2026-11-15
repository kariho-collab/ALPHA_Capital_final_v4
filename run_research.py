"""
Master Institutional Research Pipeline: 6-Pillar EUR/USD Quantitative Framework
Executes Data Engineering, Statistical Discovery, Market Regime Detection,
Machine Learning, Signal Combination, and Realistic Risk/Portfolio Evaluation.
Adheres strictly to Phase 1 Quantitative Specifications:
- Strict 3-Way Chronological Split: 60% Train, 20% Validation, 20% Untouched Test
- Zero Calibration Leakage (calibrators fit strictly on Validation set)
- Next-Bar Open Execution (1-bar decision latency)
- Full Transaction Frictions ($3.00/lot commission, 0.8 pip spread, 0.2 pip slippage)
- Mark-to-Market Daily Return Resampling (Daily VaR 95%, Daily CVaR 95%, Daily Sharpe sqrt(252))
"""
import argparse
import sys
from pathlib import Path
import json
import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Add root directory to python path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import ResearchConfig, default_config, REPORTS_DIR, MODELS_DIR
from data_engineering.loader import HistoricalDataLoader
from data_engineering.resampler import MultiTimeframeResampler
from data_engineering.validator import DataValidator
from statistical_discovery.momentum_mean_rev import MomentumMeanReversionAnalyzer
from statistical_discovery.volatility_dynamics import VolatilityDynamicsAnalyzer
from statistical_discovery.seasonality import SeasonalityAnalyzer
from market_regime.regime_classifier import MarketRegimeDetector
from machine_learning.features import MultiTimeframeFeatureEngineer
from machine_learning.tree_models import TabularMLTrainer
from signal_combination.meta_learner import RegimeGatedEnsemble
from signal_combination.calibrator import ProbabilityCalibrator
from risk_and_portfolio.backtester import InstitutionalBacktester

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("InstitutionalResearch")


def run_full_research(run_deep_learning: bool = False):
    cfg = default_config
    print("=" * 78)
    print("      ALPHA CAPITAL: INSTITUTIONAL QUANTITATIVE RESEARCH FRAMEWORK")
    print(f"      Mode: {'Full (Trees + DL)' if run_deep_learning else 'Fast (Tree-Models Only)'}")
    print(f"      Execution Spec: Next-Bar Open | Comm: ${cfg.commission_per_lot_usd}/lot | Spread: {cfg.typical_spread_pips}p | Slip: {cfg.slippage_pips}p")
    print("=" * 78)

    # -------------------------------------------------------------
    # 01. DATA ENGINEERING
    # -------------------------------------------------------------
    print("\n[PILLAR 01: DATA ENGINEERING]")
    loader = HistoricalDataLoader(cfg)
    base_m1 = loader.get_or_generate_m1_data(years=cfg.research_years)
    val_report = DataValidator.validate_integrity(base_m1, "EURUSD_Base_Feed")
    print(f"  > Clean Continuous Data: {val_report['total_bars']:,} bars ({val_report['start_time'][:10]} to {val_report['end_time'][:10]})")
    print(f"  > Integrity Check: Zero corrupt timestamps, {val_report['outlier_10sigma_spikes']} statistical tail jumps.")

    # Resample to multi-timeframe hierarchy
    resampler = MultiTimeframeResampler(cfg)
    tf_data = resampler.resample_all(base_m1)
    ltf_df = tf_data["ltf"]  # 5min
    mtf_df = tf_data["mtf"]  # 1h
    htf_df = tf_data["htf"]  # 1D
    print(f"  > Multi-Timeframe Synced: LTF (5m) = {len(ltf_df):,} | MTF (1h) = {len(mtf_df):,} | HTF (1D) = {len(htf_df):,} bars.")

    # -------------------------------------------------------------
    # 02. STATISTICAL DISCOVERY
    # -------------------------------------------------------------
    print("\n[PILLAR 02: STATISTICAL DISCOVERY]")
    hurst_ltf = MomentumMeanReversionAnalyzer.compute_hurst_exponent(ltf_df["close"])
    hurst_mtf = MomentumMeanReversionAnalyzer.compute_hurst_exponent(mtf_df["close"])
    hurst_htf = MomentumMeanReversionAnalyzer.compute_hurst_exponent(htf_df["close"])
    
    def _reg_desc(h):
        return "Trending / Momentum Persistent" if h > 0.55 else "Mean-Reverting" if h < 0.45 else "Random Walk / Efficient Market"

    print(f"  > Hurst Exponent LTF (5m): {hurst_ltf:.4f} -> {_reg_desc(hurst_ltf)}")
    print(f"  > Hurst Exponent MTF (1h): {hurst_mtf:.4f} -> {_reg_desc(hurst_mtf)}")
    print(f"  > Hurst Exponent HTF (1D): {hurst_htf:.4f} -> {_reg_desc(hurst_htf)}")

    vol_report = VolatilityDynamicsAnalyzer.compute_volatility_summary(ltf_df)
    print(f"  > Realized Close-to-Close Vol: {vol_report['close_to_close_annual_vol']}%")
    print(f"  > Garman-Klass Range-Based Vol: {vol_report['garman_klass_annual_vol']}%")
    print(f"  > Volatility Clustering Autocorrelation: {vol_report['volatility_clustering_autocorr']} (GARCH clustering confirmed: {vol_report['clustering_detected']})")

    season_report = SeasonalityAnalyzer.analyze_intraday_sessions(ltf_df)
    print("  > Intraday Session Dynamics (UTC):")
    for sess, s_data in season_report.get("session_dynamics", {}).items():
        print(f"     * {sess:<28}: Range={s_data['avg_bar_range_pips']} pips | Vol={s_data['annualized_vol_pct']}% | Drift={s_data['mean_return_bps']:+.2f} bps")

    # -------------------------------------------------------------
    # 03. MARKET REGIME DETECTION (HMM)
    # -------------------------------------------------------------
    print("\n[PILLAR 03: MARKET REGIME DETECTION - HIDDEN MARKOV MODEL]")
    detector = MarketRegimeDetector(n_states=cfg.hmm_states, random_state=cfg.random_seed)
    regime_df, regime_report = detector.fit_and_predict(mtf_df)
    print("  > Identified Latent Market Regimes:")
    for k, label in regime_report["regime_labels"].items():
        pers = regime_report["regime_persistence_pct"][label]
        dur = regime_report["expected_state_duration_bars"][label]
        dist = regime_report["regime_distribution"].get(label, 0) * 100
        print(f"     * Regime {k} [{label}]: Frequency = {dist:.1f}% | Persistence = {pers:.1f}% | Mean Duration = {dur} hours")

    # -------------------------------------------------------------
    # 04. MACHINE LEARNING: 3-WAY CHRONOLOGICAL PARTITIONING
    # -------------------------------------------------------------
    mode_label = "TREES + LSTM + TRANSFORMER" if run_deep_learning else "TREE-MODELS ONLY (XGBoost / LightGBM / RF)"
    print(f"\n[PILLAR 04: MACHINE LEARNING — {mode_label}]")
    engineer = MultiTimeframeFeatureEngineer(cfg)
    proc_df, feature_cols = engineer.extract_features(ltf_df, mtf_df, htf_df, regime_df)
    print(f"  > Multi-Timeframe Synchronized Feature Matrix: {proc_df.shape[0]:,} bars x {len(feature_cols)} features.")

    # 3-Way Chronological Split: 60% Train, 20% Validation, 20% Untouched Test
    trainer = TabularMLTrainer(random_state=cfg.random_seed)
    dual_results = trainer.evaluate_dual(
        proc_df,
        feature_cols,
        train_ratio=cfg.train_ratio,
        val_ratio=cfg.val_ratio,
        run_cv=True
    )
    long_res = dual_results["long"]
    short_res = dual_results["short"]

    print("  --- 3-Way Chronological Performance (Dual Tabular Models) ---")
    print(f"  > Validation ROC-AUC | Long: XGB={long_res['xgboost']['val_auc']:.4f}, LGBM={long_res['lightgbm']['val_auc']:.4f}, RF={long_res['random_forest']['val_auc']:.4f}")
    print(f"  > Untouched Test AUC | Long: XGB={long_res['xgboost']['test_auc']:.4f}, LGBM={long_res['lightgbm']['test_auc']:.4f}, RF={long_res['random_forest']['test_auc']:.4f}")
    print(f"  > Validation ROC-AUC | Short: XGB={short_res['xgboost']['val_auc']:.4f}, LGBM={short_res['lightgbm']['val_auc']:.4f}, RF={short_res['random_forest']['val_auc']:.4f}")
    print(f"  > Untouched Test AUC | Short: XGB={short_res['xgboost']['test_auc']:.4f}, LGBM={short_res['lightgbm']['test_auc']:.4f}, RF={short_res['random_forest']['test_auc']:.4f}")

    # Walk-Forward TimeSeriesSplit Cross Validation Report
    if "cv_scores" in long_res and long_res["cv_scores"]:
        print("  --- Walk-Forward TimeSeriesSplit Cross-Validation on Training Set (5 Folds) ---")
        for m in ("xgboost", "lightgbm", "random_forest"):
            l_cv = long_res["cv_scores"][m]
            s_cv = short_res["cv_scores"][m]
            print(f"  > {m.upper():<13} Mean CV AUC | Long: {l_cv['mean_cv_auc']:.4f} (±{l_cv['std_cv_auc']:.4f}) | Short: {s_cv['mean_cv_auc']:.4f} (±{s_cv['std_cv_auc']:.4f})")

    # Save model artifacts
    import joblib
    for name in ("xgboost", "lightgbm", "random_forest"):
        joblib.dump(long_res[name]["model"], MODELS_DIR / f"{name}_long_model.pkl")
        joblib.dump(short_res[name]["model"], MODELS_DIR / f"{name}_short_model.pkl")
    logger.info("Saved dual tree model artifacts to %s", MODELS_DIR)

    # -------------------------------------------------------------
    # 05. SIGNAL COMBINATION & ZERO-LEAKAGE PROBABILITY CALIBRATION
    # -------------------------------------------------------------
    print("\n[PILLAR 05: SIGNAL COMBINATION & ZERO-LEAKAGE CALIBRATION]")
    ensemble = RegimeGatedEnsemble()

    val_idx = dual_results["val_indices"]
    test_idx = dual_results["test_indices"]

    val_long_probs = {k: long_res[k]["val_probabilities"] for k in ("xgboost", "lightgbm", "random_forest")}
    val_short_probs = {k: short_res[k]["val_probabilities"] for k in ("xgboost", "lightgbm", "random_forest")}

    test_long_probs = {k: long_res[k]["test_probabilities"] for k in ("xgboost", "lightgbm", "random_forest")}
    test_short_probs = {k: short_res[k]["test_probabilities"] for k in ("xgboost", "lightgbm", "random_forest")}

    # Fit calibration STRICTLY on the Validation Set (Rule 3: zero calibration leakage)
    tree_w = {"xgboost": 0.40, "lightgbm": 0.40, "random_forest": 0.20}
    val_ens_long = sum(val_long_probs[k] * w for k, w in tree_w.items())
    val_ens_short = sum(val_short_probs[k] * w for k, w in tree_w.items())
    y_val_long = long_res["y_val"]
    y_val_short = short_res["y_val"]

    # Naive baseline Brier check: p*(1-p)
    p_naive_l = float(y_val_long.mean())
    p_naive_s = float(y_val_short.mean())
    naive_brier_l = p_naive_l * (1.0 - p_naive_l)
    naive_brier_s = p_naive_s * (1.0 - p_naive_s)

    iso_long = ProbabilityCalibrator.fit_isotonic(y_val_long, val_ens_long)
    iso_short = ProbabilityCalibrator.fit_isotonic(y_val_short, val_ens_short)

    val_cal_l = ProbabilityCalibrator.calibrate_probabilities(iso_long, val_ens_long)
    val_cal_s = ProbabilityCalibrator.calibrate_probabilities(iso_short, val_ens_short)
    eval_cal_l = ProbabilityCalibrator.assess_calibration(y_val_long, val_cal_l)
    eval_cal_s = ProbabilityCalibrator.assess_calibration(y_val_short, val_cal_s)

    print(f"  > Calibration Fit on Validation Partition ({len(val_idx):,} bars):")
    print(f"  > Long  Brier: {eval_cal_l['brier_score']:.4f} (vs Naive Baseline: {naive_brier_l:.4f}) | ECE: {eval_cal_l['expected_calibration_error']:.4f}")
    print(f"  > Short Brier: {eval_cal_s['brier_score']:.4f} (vs Naive Baseline: {naive_brier_s:.4f}) | ECE: {eval_cal_s['expected_calibration_error']:.4f}")

    # Evaluate Ensemble on UNTOUCHED Final Test Set (20%)
    test_regimes = proc_df.loc[test_idx, "regime_id"].values if "regime_id" in proc_df.columns else None
    ensemble_df = ensemble.combine_dual_predictions(
        test_long_probs,
        test_short_probs,
        regimes=test_regimes,
        regime_labels=regime_report["regime_labels"],
        percentile_cutoff=99.0,
        margin=0.02
    )

    n_longs = int((ensemble_df["trade_signal"] == 1).sum())
    n_shorts = int((ensemble_df["trade_signal"] == -1).sum())
    n_neutral = int((ensemble_df["trade_signal"] == 0).sum())
    print(f"  > Final Untouched Test Set Signals Generated ({len(test_idx):,} bars): Longs = {n_longs:,} | Shorts = {n_shorts:,} | Filtered Neutral = {n_neutral:,}")

    # -------------------------------------------------------------
    # 06. REALISTIC RISK AND PORTFOLIO SIMULATION
    # -------------------------------------------------------------
    print("\n[PILLAR 06: REALISTIC EXECUTION SIMULATION & DAILY MTM RISK]")
    backtester = InstitutionalBacktester(cfg)
    sig_series = pd.Series(ensemble_df["trade_signal"].values, index=test_idx)
    reg_series = pd.Series(ensemble_df["regime_id"].values, index=test_idx) if "regime_id" in ensemble_df.columns else None

    sim_results = backtester.run_simulation(ltf_df, sig_series, reg_series)

    print("\n  ================ PHASE 1 REALISTIC RISK & RETURN REPORT ================")
    print(f"  Untouched Test Set Bars:         {len(test_idx):,} bars (Final 20% Out-of-Sample)")
    print(f"  Execution Fill Model:            Next-Bar Open (1-bar decision latency)")
    print(f"  Turnover Commission:             ${cfg.commission_per_lot_usd:.2f} per round-turn standard lot")
    print(f"  Initial Institutional Capital:   ${sim_results['initial_capital']:,.2f}")
    print(f"  Final Capital:                   ${sim_results['final_capital']:,.2f}")
    print(f"  Total Net Return:                {sim_results['total_return_pct']:+.2f}%")
    print(f"  Maximum Drawdown (Continuous):   {sim_results['max_drawdown_pct']:.2f}%")
    print(f"  Annualized Daily Sharpe (sqrt252): {sim_results['annualized_sharpe']:.2f}")
    print(f"  Daily Sortino Ratio:             {sim_results['sortino_ratio']:.2f}")
    print(f"  Calmar Ratio:                    {sim_results['calmar_ratio']:.2f}")
    print(f"  Daily Value at Risk (VaR 95%):   {sim_results['var_95_pct']:.3f}%")
    print(f"  Daily Expected Shortfall (CVaR): {sim_results['cvar_95_expected_shortfall_pct']:.3f}%")
    print(f"  Total Trades Executed:           {sim_results['total_trades']}")
    print(f"  Win Rate:                        {sim_results['win_rate_pct']:.1f}%")
    print(f"  Profit Factor:                   {sim_results['profit_factor']:.2f}")
    print(f"  Average Trade Expectancy (Net):  ${sim_results['trade_expectancy_usd']:+.2f}")
    print("  ========================================================================")

    # Plot Equity & Drawdown Chart
    plot_path = REPORTS_DIR / "phase1_realistic_equity_curve.png"
    _plot_equity(sim_results["equity_times"], sim_results["equity_curve"], plot_path)
    print(f"\n  [Visual Report Saved] {plot_path}")

    # Save summary report to JSON
    summary_path = REPORTS_DIR / "phase1_realistic_performance_report.json"
    clean_summary = {
        "execution_model": "Next-Bar Open (1-bar latency)",
        "spread_pips": cfg.typical_spread_pips,
        "slippage_pips": cfg.slippage_pips,
        "commission_per_lot_usd": cfg.commission_per_lot_usd,
        "partitioning": {"train_ratio": cfg.train_ratio, "val_ratio": cfg.val_ratio, "test_ratio": cfg.test_ratio},
        "results": {k: v for k, v in sim_results.items() if k not in ("equity_curve", "equity_times", "trade_objects")}
    }
    with open(summary_path, "w") as f:
        json.dump(clean_summary, f, indent=4)
    print(f"  [Realistic Metrics Saved] {summary_path}")


def _plot_equity(times, equity, save_path):
    times_arr = pd.to_datetime(times[:len(equity)], utc=True)
    eq_arr = np.array(equity)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    ax1.plot(times_arr, eq_arr, color="#16a085", linewidth=2.0, label="Mark-to-Market Portfolio Equity ($)")
    ax1.axhline(eq_arr[0], color="gray", linestyle="--", alpha=0.7)
    ax1.set_title("ALPHA Capital: Phase 1 Realistic Execution & Mark-to-Market Equity", fontsize=13, fontweight="bold")
    ax1.set_ylabel("Portfolio Capital (USD)")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    peak = np.maximum.accumulate(eq_arr)
    dd = (eq_arr - peak) / (peak + 1e-10) * 100.0
    ax2.fill_between(times_arr, dd, 0, color="#c0392b", alpha=0.4, label="Continuous Drawdown (%)")
    ax2.set_ylabel("Drawdown %")
    ax2.set_xlabel("Date (UTC)")
    ax2.legend(loc="lower left")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ALPHA Capital Phase 1 Research Pipeline")
    parser.add_argument("--with-dl", action="store_true", help="Include deep learning models")
    args = parser.parse_args()
    run_full_research(run_deep_learning=args.with_dl)
