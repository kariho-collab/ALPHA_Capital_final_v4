"""
ALPHA Capital: Master Phase 3 Advanced Alpha Discovery & Optimization Runner
Focus: 100% Pure EUR/USD Specialization (Zero Multi-Pair Noise)

Implements the 4 Institutional Phase 3 Pillars:
1. Tier-1 Macro News Blackout Engine (US CPI, NFP, FOMC, ECB +/- 30 min freeze)
2. Adaptive Volatility-Adjusted Conviction Thresholds (solves Slice 3 over-trading)
3. López de Prado Meta-Labeling Filter (prunes false breakouts using secondary ML)
4. Dynamic Inverse-Volatility Position Sizing (scales risk inversely with local ATR)
"""
import sys
import copy
import json
import logging
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import default_config, REPORTS_DIR, MODELS_DIR, ResearchConfig
from data_engineering.loader import HistoricalDataLoader
from data_engineering.resampler import MultiTimeframeResampler
from market_regime.regime_classifier import MarketRegimeDetector
from market_regime.macro_calendar import MacroNewsCalendar
from machine_learning.features import MultiTimeframeFeatureEngineer
from machine_learning.tree_models import TabularMLTrainer
from machine_learning.meta_labeler import MetaLabeler
from signal_combination.adaptive_thresholds import AdaptiveThresholdEngine
from signal_combination.calibrator import ProbabilityCalibrator
from risk_and_portfolio.backtester import InstitutionalBacktester
from risk_and_portfolio.phase2_robustness import DeflatedSharpeCalculator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Phase3Optimization")

ARTIFACT_DIR = Path(r"C:\Users\lenovo\.gemini\antigravity\brain\e79b9f78-d430-4d9a-9055-42188e511cfd")


def run_phase3_pipeline():
    cfg = default_config
    print("=" * 80)
    print("      ALPHA CAPITAL: PHASE 3 ADVANCED ALPHA DISCOVERY & OPTIMIZATION")
    print("      ETHOS: 'Research before complexity' | 100% Pure EUR/USD Specialization")
    print("=" * 80)

    # -------------------------------------------------------------
    # 1. LOAD DATA & ENGINEER FEATURES
    # -------------------------------------------------------------
    print("\n[STEP 1: INGESTING VERIFIED MARKET DATA & REGIMES]")
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
    print(f"  > Clean Continuous Data: {len(proc_df):,} bars x {len(feature_cols)} features.")

    # -------------------------------------------------------------
    # 2. STRICT 3-WAY CHRONOLOGICAL PARTITIONING
    # -------------------------------------------------------------
    print("\n[STEP 2: 3-WAY CHRONOLOGICAL PARTITIONING (TRAIN / VAL / TEST)]")
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

    val_idx = dual_results["val_indices"]
    test_idx = dual_results["test_indices"]

    # -------------------------------------------------------------
    # 3. FIT PRIMARY PREDICTIONS & ZERO-LEAKAGE CALIBRATION
    # -------------------------------------------------------------
    print("\n[STEP 3: PRIMARY ENSEMBLE PROBABILITIES]")
    tree_w = {"xgboost": 0.40, "lightgbm": 0.40, "random_forest": 0.20}

    val_long_probs = sum(long_res[k]["val_probabilities"] * w for k, w in tree_w.items())
    val_short_probs = sum(short_res[k]["val_probabilities"] * w for k, w in tree_w.items())

    test_long_probs = sum(long_res[k]["test_probabilities"] * w for k, w in tree_w.items())
    test_short_probs = sum(short_res[k]["test_probabilities"] * w for k, w in tree_w.items())

    val_regimes = proc_df.loc[val_idx, "regime_id"].values if "regime_id" in proc_df.columns else None
    test_regimes = proc_df.loc[test_idx, "regime_id"].values if "regime_id" in proc_df.columns else None

    # -------------------------------------------------------------
    # 4. PILLAR 1: MACRO ECONOMIC NEWS BLACKOUT ENGINE
    # -------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  PHASE 3 PILLAR 1: TIER-1 MACRO ECONOMIC NEWS BLACKOUT ENGINE")
    print("  [US CPI, US NFP, FOMC & ECB Rate Decisions +/- 30 min Freeze]")
    print("=" * 70)
    macro_cal = MacroNewsCalendar(pre_blackout_minutes=30, post_blackout_minutes=30)
    news_events = macro_cal.get_tier1_events(
        start_date=str(proc_df.index[0].date()),
        end_date=str(proc_df.index[-1].date() + pd.Timedelta(days=1))
    )
    print(f"  > Total Scheduled Tier-1 Releases Mapped: {len(news_events)} high-impact events.")

    # -------------------------------------------------------------
    # 5. PILLAR 2: ADAPTIVE VOLATILITY CONVICTION THRESHOLD ENGINE
    # -------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  PHASE 3 PILLAR 2: ADAPTIVE VOLATILITY CONVICTION THRESHOLD ENGINE")
    print("  [Dynamically Throttling Score Cutoffs During Elevated Noise]")
    print("=" * 70)
    adapt_engine = AdaptiveThresholdEngine(
        base_percentile=99.0,
        base_margin=0.02,
        chop_percentile_boost=0.5,
        chop_margin_boost=0.015
    )

    # Generate signals on Validation set to calibrate Meta-Labeler
    val_signals_df = adapt_engine.compute_dynamic_signals(
        long_probs=val_long_probs,
        short_probs=val_short_probs,
        df_ltf=ltf_df.loc[val_idx],
        regimes=val_regimes,
        regime_labels=regime_labels,
        train_long_probs=val_long_probs,
        train_short_probs=val_short_probs
    )
    val_raw_signals = pd.Series(val_signals_df["trade_signal"].values, index=val_idx)

    # Filter Validation signals with Macro News Blackout
    val_news_signals, val_news_suppressed = macro_cal.filter_signals(val_raw_signals)
    print(f"  > Validation Partition Signals: {int((val_raw_signals != 0).sum())} raw -> {val_news_suppressed} filtered by Macro News Blackout.")

    # Generate signals on UNTOUCHED Test set
    test_signals_df = adapt_engine.compute_dynamic_signals(
        long_probs=test_long_probs,
        short_probs=test_short_probs,
        df_ltf=ltf_df.loc[test_idx],
        regimes=test_regimes,
        regime_labels=regime_labels,
        train_long_probs=val_long_probs,
        train_short_probs=val_short_probs
    )
    test_adapt_signals = pd.Series(test_signals_df["trade_signal"].values, index=test_idx)

    # Filter Test signals with Macro News Blackout
    test_news_signals, test_news_suppressed = macro_cal.filter_signals(test_adapt_signals)
    print(f"  > Test Partition Signals: {int((test_adapt_signals != 0).sum())} raw -> {test_news_suppressed} filtered by Macro News Blackout.")

    # -------------------------------------------------------------
    # 6. PILLAR 3: LÓPEZ DE PRADO META-LABELING FILTER
    # -------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  PHASE 3 PILLAR 3: LÓPEZ DE PRADO META-LABELING CLASSIFIER")
    print("  [Secondary ML Model Trained Strictly on Validation Partition]")
    print("=" * 70)
    meta_labeler = MetaLabeler(random_state=cfg.random_seed)

    # Construct meta features and generate meta labels on Validation Set
    val_meta_X = meta_labeler.construct_meta_features(
        df_ltf=ltf_df.loc[val_idx],
        signals=val_news_signals,
        long_probs=val_long_probs,
        short_probs=val_short_probs,
        regimes=pd.Series(val_regimes, index=val_idx) if val_regimes is not None else None
    )
    val_meta_y = meta_labeler.generate_meta_labels(
        df_ltf=ltf_df.loc[val_idx],
        signals=val_news_signals,
        forward_bars=cfg.forward_bars * 2,
        profit_ratio=1.5
    )

    meta_labeler.fit(val_meta_X, val_meta_y)
    print(f"  > Meta-Labeler Trained on {int((~val_meta_y.isna()).sum())} validated trade instances.")

    # Apply Meta-Labeler out-of-sample on Test Partition
    test_meta_X = meta_labeler.construct_meta_features(
        df_ltf=ltf_df.loc[test_idx],
        signals=test_news_signals,
        long_probs=test_long_probs,
        short_probs=test_short_probs,
        regimes=pd.Series(test_regimes, index=test_idx) if test_regimes is not None else None
    )
    final_test_signals, test_meta_pruned = meta_labeler.filter_signals(
        signals=test_news_signals,
        X_meta=test_meta_X,
        meta_threshold=0.48
    )
    print(f"  > Test Set Meta-Labeling Pruned: {test_meta_pruned} false breakout signals filtered out.")
    print(f"  > Final High-Conviction Untouched Signals: {int((final_test_signals != 0).sum())} trades.")

    # -------------------------------------------------------------
    # 7. PILLAR 4: DYNAMIC INVERSE-VOLATILITY POSITION SIZING
    # -------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  PHASE 3 PILLAR 4: DYNAMIC INVERSE-VOLATILITY POSITION SIZING")
    print("  [Downscaling Risk in Choppy ATR / Scaling in Persistent Trends]")
    print("=" * 70)

    # Configure Backtester with Phase 3 Dynamic Sizing
    phase3_cfg = copy.deepcopy(cfg)
    phase3_cfg.use_dynamic_sizing = True

    backtester_p3 = InstitutionalBacktester(phase3_cfg)
    reg_series = pd.Series(test_regimes, index=test_idx) if test_regimes is not None else None

    # Run Phase 3 Optimized Simulation
    p3_sim = backtester_p3.run_simulation(
        df_ltf=ltf_df,
        signals=final_test_signals,
        regimes=reg_series,
        regime_labels=regime_labels
    )

    # Run Phase 1 Baseline on the same test set for direct comparison
    backtester_p1 = InstitutionalBacktester(cfg)
    # Phase 1 static signals (99th percentile static, no news filter, static sizing)
    p1_ensemble = from_phase1_baseline(ltf_df, test_idx, test_long_probs, test_short_probs, test_regimes, regime_labels, cfg)
    p1_sim = backtester_p1.run_simulation(ltf_df, p1_ensemble, reg_series)

    # -------------------------------------------------------------
    # 8. BEFORE-VS-AFTER PERFORMANCE COMPARISON
    # -------------------------------------------------------------
    print("\n" + "=" * 78)
    print("      BEFORE VS. AFTER: PHASE 1 REALISTIC VS. PHASE 3 OPTIMIZED")
    print("=" * 78)
    print(f"  Metric                             Phase 1 Realistic      Phase 3 Optimized")
    print(f"  -------------------------------------------------------------------------")
    print(f"  Total Net Return:                  {p1_sim['total_return_pct']:>15.2f}%    {p3_sim['total_return_pct']:>15.2f}%")
    print(f"  Annualized Daily Sharpe (sqrt252): {p1_sim['annualized_sharpe']:>15.2f}     {p3_sim['annualized_sharpe']:>15.2f}")
    print(f"  Daily Sortino Ratio:               {p1_sim['sortino_ratio']:>15.2f}     {p3_sim['sortino_ratio']:>15.2f}")
    print(f"  Calmar Ratio:                      {p1_sim['calmar_ratio']:>15.2f}     {p3_sim['calmar_ratio']:>15.2f}")
    print(f"  Maximum Drawdown (MTM):            {p1_sim['max_drawdown_pct']:>15.2f}%    {p3_sim['max_drawdown_pct']:>15.2f}%")
    print(f"  Daily VaR (95%):                   {p1_sim['var_95_pct']:>15.3f}%    {p3_sim['var_95_pct']:>15.3f}%")
    print(f"  Daily Expected Shortfall (CVaR):   {p1_sim['cvar_95_expected_shortfall_pct']:>15.3f}%    {p3_sim['cvar_95_expected_shortfall_pct']:>15.3f}%")
    print(f"  Total Trades Executed:             {p1_sim['total_trades']:>15d}     {p3_sim['total_trades']:>15d}")
    print(f"  Win Rate:                          {p1_sim['win_rate_pct']:>15.1f}%    {p3_sim['win_rate_pct']:>15.1f}%")
    print(f"  Profit Factor:                     {p1_sim['profit_factor']:>15.2f}     {p3_sim['profit_factor']:>15.2f}")
    print(f"  Trade Expectancy (Net USD):        ${p1_sim['trade_expectancy_usd']:>14.2f}    ${p3_sim['trade_expectancy_usd']:>14.2f}")
    print("=" * 78)

    # -------------------------------------------------------------
    # 9. RE-EVALUATING MULTI-PERIOD ROLLING WALK-FORWARD MATRIX
    # -------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  RE-AUDITING ROLLING WALK-FORWARD MATRIX WITH PHASE 3 ENHANCEMENTS")
    print("  [Specifically testing Slice 3 Improvement]")
    print("=" * 70)

    wf_improved = run_walk_forward_phase3(
        ltf_df=ltf_df,
        proc_df=proc_df,
        feature_cols=feature_cols,
        regime_labels=regime_labels,
        macro_cal=macro_cal,
        adapt_engine=adapt_engine,
        config=phase3_cfg,
        n_slices=4
    )
    wf_df = pd.DataFrame(wf_improved)
    disp_cols = ["period_label", "total_return_pct", "annualized_sharpe", "sortino_ratio", "max_drawdown_pct", "win_rate_pct", "profit_factor", "total_trades"]
    print("  " + wf_df[disp_cols].to_string(index=False))

    # -------------------------------------------------------------
    # 10. GENERATE & SAVE VISUAL ANALYTICS
    # -------------------------------------------------------------
    print("\n[STEP 10: GENERATING PHASE 3 CHARTS & JSON REPORTS]")
    # Chart 1: Side-by-side equity comparison
    comp_chart_path = REPORTS_DIR / "phase3_comparison_equity_curve.png"
    _plot_comparison(p1_sim, p3_sim, comp_chart_path)
    print(f"  > Comparison Equity Curve Saved: {comp_chart_path}")

    # Chart 2: Walk-forward improvement comparison
    wf_chart_path = REPORTS_DIR / "phase3_walk_forward_improvement.png"
    _plot_wf_improvement(wf_improved, wf_chart_path)
    print(f"  > Walk-Forward Improvement Saved: {wf_chart_path}")

    # Sync to Artifact Directory
    if ARTIFACT_DIR.exists():
        for p in [comp_chart_path, wf_chart_path]:
            target = ARTIFACT_DIR / p.name
            shutil.copyfile(p, target)
            print(f"  > Synced Artifact Chart: {target}")

    # Save JSON Report
    report_path = REPORTS_DIR / "phase3_optimization_report.json"
    clean_p3_report = {
        "title": "ALPHA Capital Phase 3: Advanced Alpha Discovery & Pure EUR/USD Optimization",
        "pillars": {
            "macro_news_calendar": {"events_mapped": len(news_events), "suppressed_test_signals": test_news_suppressed},
            "adaptive_thresholds": {"chop_boost": adapt_engine.chop_percentile_boost, "base_percentile": adapt_engine.base_percentile},
            "meta_labeler": {"test_signals_pruned": test_meta_pruned, "meta_threshold": 0.48},
            "dynamic_sizer": {"enabled": True, "base_risk_pct": phase3_cfg.max_risk_per_trade_pct}
        },
        "phase1_baseline": {k: v for k, v in p1_sim.items() if k not in ("equity_curve", "equity_times", "trade_objects")},
        "phase3_optimized": {k: v for k, v in p3_sim.items() if k not in ("equity_curve", "equity_times", "trade_objects")},
        "walk_forward_matrix": wf_improved
    }
    with open(report_path, "w") as f:
        json.dump(clean_p3_report, f, indent=4)
    print(f"  > Full Optimization Tear Sheet Saved: {report_path}")
    print("=" * 80)
    print(">>> PHASE 3 ADVANCED ALPHA DISCOVERY & OPTIMIZATION COMPLETE <<<")
    print("=" * 80)


def from_phase1_baseline(ltf_df, test_idx, long_probs, short_probs, test_regimes, regime_labels, cfg):
    """Generates baseline Phase 1 signals (static 99th percentile cutoff)"""
    long_cut = float(np.percentile(long_probs, 99.0))
    short_cut = float(np.percentile(short_probs, 99.0))
    diff = long_probs - short_probs

    bull_states = {k for k, v in regime_labels.items() if "Bull" in v}
    bear_states = {k for k, v in regime_labels.items() if "Bear" in v}
    shock_states = {k for k, v in regime_labels.items() if "Shock" in v}

    signals = np.zeros(len(test_idx), dtype=int)
    for t in range(len(test_idx)):
        r = test_regimes[t] if test_regimes is not None else 0
        if r in shock_states:
            continue
        if diff[t] > 0.02 and long_probs[t] >= long_cut and r not in bear_states:
            signals[t] = 1
        elif diff[t] < -0.02 and short_probs[t] >= short_cut and r not in bull_states:
            signals[t] = -1

    return pd.Series(signals, index=test_idx)


def run_walk_forward_phase3(ltf_df, proc_df, feature_cols, regime_labels, macro_cal, adapt_engine, config, n_slices=4):
    """Executes 4-period walk-forward test incorporating Phase 3 enhancements"""
    n_total = len(proc_df)
    slice_size = n_total // (n_slices + 1)
    trainer = TabularMLTrainer(random_state=config.random_seed)
    backtester = InstitutionalBacktester(config)
    results = []

    for s in range(n_slices):
        train_end = slice_size * (s + 1)
        test_end = min(n_total, train_end + slice_size)

        train_df = proc_df.iloc[:train_end]
        test_df = proc_df.iloc[train_end:test_end]

        if len(test_df) < 2000:
            continue

        test_idx = test_df.index
        start_date = str(test_idx[0])[:10]
        end_date = str(test_idx[-1])[:10]

        X_tr = train_df[feature_cols].values
        X_te = test_df[feature_cols].values

        xgb_l = trainer.train_xgboost(X_tr, train_df["target_long"].values)
        lgb_l = trainer.train_lightgbm(X_tr, train_df["target_long"].values)
        xgb_s = trainer.train_xgboost(X_tr, train_df["target_short"].values)
        lgb_s = trainer.train_lightgbm(X_tr, train_df["target_short"].values)

        p_l = 0.5 * xgb_l.predict_proba(X_te)[:, 1] + 0.5 * lgb_l.predict_proba(X_te)[:, 1]
        p_s = 0.5 * xgb_s.predict_proba(X_te)[:, 1] + 0.5 * lgb_s.predict_proba(X_te)[:, 1]

        # Use recent past for calibration
        recent_n = min(len(train_df), 15000)
        X_cal = train_df[feature_cols].values[-recent_n:]
        p_cal_l = 0.5 * xgb_l.predict_proba(X_cal)[:, 1] + 0.5 * lgb_l.predict_proba(X_cal)[:, 1]
        p_cal_s = 0.5 * xgb_s.predict_proba(X_cal)[:, 1] + 0.5 * lgb_s.predict_proba(X_cal)[:, 1]

        regimes_test = test_df["regime_id"].values if "regime_id" in test_df.columns else np.zeros(len(test_df))

        # 1. Adaptive signals
        sig_df = adapt_engine.compute_dynamic_signals(
            long_probs=p_l,
            short_probs=p_s,
            df_ltf=ltf_df.loc[test_idx],
            regimes=regimes_test,
            regime_labels=regime_labels,
            train_long_probs=p_cal_l,
            train_short_probs=p_cal_s
        )
        raw_signals = pd.Series(sig_df["trade_signal"].values, index=test_idx)

        # 2. Macro news filter
        filtered_signals, _ = macro_cal.filter_signals(raw_signals)

        sim_res = backtester.run_simulation(
            df_ltf=ltf_df,
            signals=filtered_signals,
            regimes=pd.Series(regimes_test, index=test_idx),
            regime_labels=regime_labels
        )

        results.append({
            "slice_id": s + 1,
            "period_label": f"Slice {s + 1} ({start_date} -> {end_date})",
            "start_date": start_date,
            "end_date": end_date,
            "total_return_pct": round(sim_res["total_return_pct"], 2),
            "annualized_sharpe": round(sim_res["annualized_sharpe"], 2),
            "sortino_ratio": round(sim_res["sortino_ratio"], 2),
            "max_drawdown_pct": round(sim_res["max_drawdown_pct"], 2),
            "win_rate_pct": round(sim_res["win_rate_pct"], 1),
            "profit_factor": round(sim_res["profit_factor"], 2),
            "total_trades": sim_res["total_trades"],
            "trade_expectancy_usd": round(sim_res["trade_expectancy_usd"], 2)
        })

    return results


def _plot_comparison(p1_sim, p3_sim, save_path):
    times1 = pd.to_datetime(p1_sim["equity_times"][:len(p1_sim["equity_curve"])], utc=True)
    eq1 = np.array(p1_sim["equity_curve"])

    times3 = pd.to_datetime(p3_sim["equity_times"][:len(p3_sim["equity_curve"])], utc=True)
    eq3 = np.array(p3_sim["equity_curve"])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]})

    ax1.plot(times1, eq1, color="#7f8c8d", linestyle="--", linewidth=1.8, label=f"Phase 1 Realistic (+{p1_sim['total_return_pct']:.1f}% | Sharpe {p1_sim['annualized_sharpe']:.2f})")
    ax1.plot(times3, eq3, color="#27ae60", linewidth=2.3, label=f"Phase 3 Optimized (+{p3_sim['total_return_pct']:.1f}% | Sharpe {p3_sim['annualized_sharpe']:.2f})")
    ax1.axhline(100000, color="black", linestyle=":", alpha=0.5)
    ax1.set_ylabel("Portfolio Capital (USD)", fontweight="bold")
    ax1.set_title("ALPHA Capital: Phase 1 Realistic vs. Phase 3 Optimized Pure EUR/USD Performance", fontsize=12, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper left")

    # Drawdown comparison
    peak3 = np.maximum.accumulate(eq3)
    dd3 = (eq3 - peak3) / (peak3 + 1e-10) * 100.0

    peak1 = np.maximum.accumulate(eq1)
    dd1 = (eq1 - peak1) / (peak1 + 1e-10) * 100.0

    ax2.plot(times1, dd1, color="#e74c3c", linestyle="--", alpha=0.7, label=f"Phase 1 MaxDD ({p1_sim['max_drawdown_pct']:.2f}%)")
    ax2.fill_between(times3, dd3, 0, color="#2ecc71", alpha=0.4, label=f"Phase 3 MaxDD ({p3_sim['max_drawdown_pct']:.2f}%)")
    ax2.set_ylabel("Drawdown %", fontweight="bold")
    ax2.set_xlabel("Date (UTC)")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="lower left")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def _plot_wf_improvement(wf_improved, save_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    periods = [f"Slice {s['slice_id']}\n{s['start_date']}" for s in wf_improved]
    returns = [s["total_return_pct"] for s in wf_improved]
    pfs = [s["profit_factor"] for s in wf_improved]

    x = np.arange(len(periods))
    colors = ["#27ae60" if r > 0 else "#c0392b" for r in returns]

    ax1.bar(x, returns, color=colors, width=0.45, alpha=0.85)
    ax1.axhline(0, color="black", linestyle="-", linewidth=0.8)
    ax1.set_ylabel("Net Return %", fontweight="bold")
    ax1.set_title("Phase 3 Walk-Forward Net Returns", fontsize=11, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(periods, fontsize=9)
    ax1.grid(True, alpha=0.3)
    for i, v in enumerate(returns):
        ax1.text(i, v + 1.0 if v >= 0 else v - 2.5, f"{v:+.1f}%", ha="center", fontweight="bold")

    ax2.bar(x, pfs, color="#3498db", width=0.45, alpha=0.85)
    ax2.axhline(1.0, color="red", linestyle="--", alpha=0.7, label="Breakeven PF = 1.0")
    ax2.set_ylabel("Profit Factor", fontweight="bold")
    ax2.set_title("Phase 3 Walk-Forward Profit Factors", fontsize=11, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(periods, fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper left")
    for i, v in enumerate(pfs):
        ax2.text(i, v + 0.04, f"{v:.2f}", ha="center", fontweight="bold")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


if __name__ == "__main__":
    run_phase3_pipeline()
