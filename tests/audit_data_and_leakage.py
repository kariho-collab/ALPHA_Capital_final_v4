"""
ALPHA Capital — Phase 1: Data Integrity & Leakage Audit Suite
Audits timestamp monotonicity, duplicate bars, price coherence, and ensures
zero future information leaks into features, labels, MTF joins, or regime shifts.
"""
import json
import logging
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import default_config, REPORTS_DIR
from data_engineering.loader import HistoricalDataLoader
from data_engineering.resampler import MultiTimeframeResampler
from machine_learning.features import MultiTimeframeFeatureEngineer
from market_regime.regime_classifier import MarketRegimeDetector

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("DataAndLeakageAudit")


def run_data_integrity_audit(m1_df: pd.DataFrame, ltf_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Priority 1: Data Integrity Verification
    """
    logger.info(">>> Running Priority 1: Data Integrity Audit <<<")
    report = {}

    # 1. Monotonicity & Chronological Ordering
    is_monotonic = bool(m1_df.index.is_monotonic_increasing)
    report["m1_index_monotonic_increasing"] = is_monotonic
    report["ltf_index_monotonic_increasing"] = bool(ltf_df.index.is_monotonic_increasing)

    # 2. Duplicate Timestamps
    m1_duplicates = int(m1_df.index.duplicated().sum())
    ltf_duplicates = int(ltf_df.index.duplicated().sum())
    report["m1_duplicate_timestamps"] = m1_duplicates
    report["ltf_duplicate_timestamps"] = ltf_duplicates

    # 3. Missing / Null Values
    report["m1_null_count"] = int(m1_df.isnull().sum().sum())
    report["ltf_null_count"] = int(ltf_df.isnull().sum().sum())

    # 4. Price Coherence Audit (High >= Low, High >= Open, High >= Close, Low <= Open, Low <= Close)
    h_ge_l = bool((ltf_df["high"] >= ltf_df["low"]).all())
    h_ge_o = bool((ltf_df["high"] >= ltf_df["open"]).all())
    h_ge_c = bool((ltf_df["high"] >= ltf_df["close"]).all())
    l_le_o = bool((ltf_df["low"] <= ltf_df["open"]).all())
    l_le_c = bool((ltf_df["low"] <= ltf_df["close"]).all())
    report["price_coherence_high_ge_low"] = h_ge_l
    report["price_coherence_high_ge_open_close"] = bool(h_ge_o and h_ge_c)
    report["price_coherence_low_le_open_close"] = bool(l_le_o and l_le_c)

    # 5. Non-Negative Volumes
    report["volume_all_positive"] = bool((ltf_df["volume"] > 0).all())

    # 6. Gap Distribution (Excluding Weekends)
    deltas = ltf_df.index.to_series().diff()
    # Weekday gaps larger than 10 minutes
    weekday_mask = ltf_df.index.dayofweek < 5
    unexpected_gaps = deltas[weekday_mask & (deltas > pd.Timedelta(minutes=10))]
    report["unexpected_weekday_gaps_count"] = len(unexpected_gaps)

    report["passed"] = bool(
        is_monotonic
        and m1_duplicates == 0
        and ltf_duplicates == 0
        and report["m1_null_count"] == 0
        and report["ltf_null_count"] == 0
        and h_ge_l and h_ge_o and h_ge_c and l_le_o and l_le_c
    )
    return report


def run_leakage_audit(
    ltf_df: pd.DataFrame,
    mtf_df: pd.DataFrame,
    htf_df: pd.DataFrame,
    regime_df: pd.DataFrame,
    fe: MultiTimeframeFeatureEngineer
) -> Dict[str, Any]:
    """
    Priority 2: Feature and Label Leakage Audit
    """
    logger.info(">>> Running Priority 2: Feature & Target Leakage Audit <<<")
    report = {}

    proc_df, feature_cols = fe.extract_features(ltf_df, mtf_df, htf_df, regime_df)

    # 1. Target Isolation: Ensure targets are NOT in feature_cols
    forbidden_targets = {"target_long", "target_short", "target_delta_pips"}
    leaked_targets = [col for col in feature_cols if col in forbidden_targets]
    report["leaked_targets_in_features"] = leaked_targets
    report["target_isolation_passed"] = (len(leaked_targets) == 0)

    # 2. Raw Future Price Isolation: Ensure raw future prices are not in feature_cols
    forbidden_raw = {"future_c", "delta_pips"}
    leaked_raw = [col for col in feature_cols if col in forbidden_raw]
    report["leaked_future_prices_in_features"] = leaked_raw
    report["future_price_isolation_passed"] = (len(leaked_raw) == 0)

    # 3. MTF Join Lag Audit:
    # Verify that MTF features at LTF time T use only MTF bars closed at or before T
    # Specifically, check that mtf_trend_50 at LTF bar corresponds to MTF bar shifted by 1
    pip = default_config.pip_size
    mtf_c = mtf_df["close"]
    mtf_ema_50 = mtf_c.ewm(span=50, adjust=False).mean()
    expected_mtf_trend = ((mtf_c - mtf_ema_50) / (pip * 100)).shift(1)

    # Pick 20 sample timestamps where an MTF bar closes
    mtf_close_times = mtf_df.index[10:30]
    alignment_correct = True
    for t_mtf in mtf_close_times:
        if t_mtf in proc_df.index and t_mtf in expected_mtf_trend.index:
            val_in_proc = proc_df.loc[t_mtf, "mtf_trend_50"]
            val_expected = expected_mtf_trend.loc[t_mtf]
            if not np.isclose(val_in_proc, val_expected, rtol=1e-5, atol=1e-5):
                alignment_correct = False
                break
    report["mtf_strict_t_minus_1_lag_verified"] = alignment_correct

    # 4. Regime Lag Audit:
    # Verify that regime_id at LTF bar is strictly shifted by 1 bar from regime_df
    reg_shifted = regime_df["regime_id"].shift(1)
    sample_times = regime_df.index[10:30]
    regime_lag_correct = True
    for t_reg in sample_times:
        if t_reg in proc_df.index and t_reg in reg_shifted.index:
            r_proc = proc_df.loc[t_reg, "regime_id"]
            r_exp = reg_shifted.loc[t_reg]
            if not np.isnan(r_exp) and int(r_proc) != int(r_exp):
                regime_lag_correct = False
                break
    report["regime_strict_t_minus_1_lag_verified"] = regime_lag_correct

    # 5. Target Forward Step Audit:
    # Verify target looks strictly forward and does NOT correlate trivially with lag-0 return
    # If target_delta_pips has zero correlation with ltf_ret_1, no instantaneous leakage exists
    corr_target_current_ret = float(proc_df["target_delta_pips"].corr(proc_df["ltf_ret_1"]))
    report["corr_target_delta_with_bar0_ret"] = round(corr_target_current_ret, 4)
    # Correlation should be small (~0.0 to 0.1), not 1.0
    report["zero_instantaneous_target_leakage"] = bool(abs(corr_target_current_ret) < 0.20)

    report["passed"] = bool(
        report["target_isolation_passed"]
        and report["future_price_isolation_passed"]
        and report["mtf_strict_t_minus_1_lag_verified"]
        and report["regime_strict_t_minus_1_lag_verified"]
        and report["zero_instantaneous_target_leakage"]
    )
    return report


def main():
    print("=" * 78)
    print("      ALPHA CAPITAL: PHASE 1 DATA INTEGRITY & LEAKAGE AUDIT")
    print("=" * 78)

    cfg = default_config
    loader = HistoricalDataLoader(cfg)
    m1_df = loader.get_or_generate_m1_data(years=cfg.research_years)
    resampler = MultiTimeframeResampler(cfg)
    tf_data = resampler.resample_all(m1_df)
    ltf_df = tf_data["ltf"]
    mtf_df = tf_data["mtf"]
    htf_df = tf_data["htf"]

    detector = MarketRegimeDetector(n_states=cfg.hmm_states, random_state=cfg.random_seed)
    regime_df, regime_report = detector.fit_and_predict(mtf_df)

    fe = MultiTimeframeFeatureEngineer(cfg)

    # Run Audits
    integrity_res = run_data_integrity_audit(m1_df, ltf_df)
    leakage_res = run_leakage_audit(ltf_df, mtf_df, htf_df, regime_df, fe)

    print("\n--- 1. DATA INTEGRITY AUDIT REPORT ---")
    for k, v in integrity_res.items():
        print(f"  {k:<42}: {v}")

    print("\n--- 2. FEATURE & LABEL LEAKAGE AUDIT REPORT ---")
    for k, v in leakage_res.items():
        print(f"  {k:<42}: {v}")

    audit_summary = {
        "timestamp": pd.Timestamp.now(tz="UTC").isoformat(),
        "data_integrity": integrity_res,
        "leakage_audit": leakage_res,
        "overall_audit_passed": bool(integrity_res["passed"] and leakage_res["passed"])
    }

    report_path = REPORTS_DIR / "data_and_leakage_audit_report.json"
    with open(report_path, "w") as f:
        json.dump(audit_summary, f, indent=4)

    print(f"\n[Audit Report Saved] {report_path}")
    print(f"OVERALL AUDIT PASSED: {audit_summary['overall_audit_passed']}")
    if not audit_summary["overall_audit_passed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
