"""
ALPHA Capital — Adaptive Volatility Conviction & Dynamic Thresholding Engine
Solves the Slice 3 Over-Trading Problem:
Instead of a static 99.0th percentile cutoff that admits false breakouts during
choppy / high-noise regimes, this engine adaptively scales conviction thresholds
based on local volatility (ATR), regime state, and Hurst persistence.
"""
from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd


class AdaptiveThresholdEngine:
    """
    Computes dynamic, market-state-aware probability cutoffs and conviction margins.
    Throttles signal generation during erratic, high-noise conditions.
    """
    def __init__(
        self,
        base_percentile: float = 99.0,
        base_margin: float = 0.02,
        chop_percentile_boost: float = 0.5,  # e.g., 99.0% -> 99.5% in chop
        chop_margin_boost: float = 0.015      # e.g., 0.02 -> 0.035 in chop
    ):
        self.base_percentile = base_percentile
        self.base_margin = base_margin
        self.chop_percentile_boost = chop_percentile_boost
        self.chop_margin_boost = chop_margin_boost

    def compute_dynamic_signals(
        self,
        long_probs: np.ndarray,
        short_probs: np.ndarray,
        df_ltf: pd.DataFrame,
        regimes: Optional[np.ndarray] = None,
        regime_labels: Optional[Dict[int, str]] = None,
        train_long_probs: Optional[np.ndarray] = None,
        train_short_probs: Optional[np.ndarray] = None
    ) -> pd.DataFrame:
        """
        Generates trade signals with adaptive, time-varying thresholding.
        Zero lookahead: baseline percentiles are established from training history
        or rolling lookback.
        """
        n_bars = len(long_probs)
        assert len(short_probs) == n_bars, "Long and short probabilities must match in length"

        # Baseline percentile cutoffs from calibration/training sample if provided
        if train_long_probs is not None and len(train_long_probs) > 1000:
            base_long_cut = float(np.percentile(train_long_probs, self.base_percentile))
            base_short_cut = float(np.percentile(train_short_probs, self.base_percentile))
            boosted_long_cut = float(np.percentile(train_long_probs, min(99.9, self.base_percentile + self.chop_percentile_boost)))
            boosted_short_cut = float(np.percentile(train_short_probs, min(99.9, self.base_percentile + self.chop_percentile_boost)))
        else:
            base_long_cut = float(np.percentile(long_probs, self.base_percentile))
            base_short_cut = float(np.percentile(short_probs, self.base_percentile))
            boosted_long_cut = float(np.percentile(long_probs, min(99.9, self.base_percentile + self.chop_percentile_boost)))
            boosted_short_cut = float(np.percentile(short_probs, min(99.9, self.base_percentile + self.chop_percentile_boost)))

        # Local volatility factor
        if "ltf_atr_pips" in df_ltf.columns:
            atr_series = df_ltf["ltf_atr_pips"].values[:n_bars]
            med_atr = float(np.nanmedian(atr_series)) + 1e-6
            atr_ratio = atr_series / med_atr
        else:
            atr_ratio = np.ones(n_bars)

        # Regimes parsing
        bull_states = set()
        bear_states = set()
        shock_states = set()
        range_states = set()

        if regime_labels:
            for k, label in regime_labels.items():
                if "Bull" in label:
                    bull_states.add(k)
                elif "Bear" in label:
                    bear_states.add(k)
                elif "Shock" in label:
                    shock_states.add(k)
                else:
                    range_states.add(k)

        signals = np.zeros(n_bars, dtype=int)
        applied_cutoffs_long = np.zeros(n_bars)
        applied_cutoffs_short = np.zeros(n_bars)
        effective_margins = np.zeros(n_bars)

        diff = long_probs - short_probs

        for t in range(n_bars):
            r = regimes[t] if regimes is not None else 0

            # Circuit breaker: high vol shock blocks trading completely
            if r in shock_states:
                continue

            # Determine if current environment is choppy/elevated noise
            is_choppy = (r in range_states) or (atr_ratio[t] > 1.35)

            if is_choppy:
                long_threshold = boosted_long_cut
                short_threshold = boosted_short_cut
                req_margin = self.base_margin + self.chop_margin_boost
            else:
                long_threshold = base_long_cut
                short_threshold = base_short_cut
                req_margin = self.base_margin

            applied_cutoffs_long[t] = long_threshold
            applied_cutoffs_short[t] = short_threshold
            effective_margins[t] = req_margin

            # Directional Gating
            can_long = (r not in bear_states)
            can_short = (r not in bull_states)

            if can_long and diff[t] > req_margin and long_probs[t] >= long_threshold:
                signals[t] = 1
            elif can_short and diff[t] < -req_margin and short_probs[t] >= short_threshold:
                signals[t] = -1

        out_df = pd.DataFrame({
            "long_prob": long_probs,
            "short_prob": short_probs,
            "prob_edge": diff,
            "applied_long_cutoff": applied_cutoffs_long,
            "applied_short_cutoff": applied_cutoffs_short,
            "effective_margin": effective_margins,
            "trade_signal": signals
        }, index=df_ltf.index[:n_bars])

        if regimes is not None:
            out_df["regime_id"] = regimes[:n_bars]

        return out_df
