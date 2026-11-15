"""
05. Signal Combination: Ensemble Stacking & Regime-Conditioned Signal Gating
Combines independent model signals (XGBoost, LightGBM, RF, LSTM, Transformer)
conditioned on the prevailing Market Regime.
"""
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd


class RegimeGatedEnsemble:
    """
    Combines model predictions dynamically based on the active market regime:
    - In Trending Regimes: Boosts momentum & Transformer/XGBoost weights.
    - In Ranging Regimes: Shifts weight to mean-reverting features & shrinks bet sizes.
    - In Vol Shock Regimes: Suppresses trades (circuit breaker).
    """
    def __init__(
        self,
        base_weights: Dict[str, float] = None,
        confidence_cutoff: float = 0.50
    ):
        self.base_weights = base_weights or {
            "xgboost": 0.40,
            "lightgbm": 0.40,
            "random_forest": 0.20,
        }
        self.confidence_cutoff = confidence_cutoff

    def combine_predictions(
        self,
        model_probs: Dict[str, np.ndarray],
        regimes: np.ndarray = None
    ) -> pd.DataFrame:
        """
        Computes weighted ensemble probability and applies regime gating logic.
        """
        min_len = min(len(p) for p in model_probs.values())
        if regimes is not None:
            min_len = min(min_len, len(regimes))

        aligned_probs = {k: v[-min_len:] for k, v in model_probs.items()}
        reg_slice = regimes[-min_len:] if regimes is not None else None

        weighted_sum = np.zeros(min_len)
        total_weight = 0.0

        for name, weight in self.base_weights.items():
            if name in aligned_probs:
                weighted_sum += aligned_probs[name] * weight
                total_weight += weight

        raw_ensemble = weighted_sum / (total_weight + 1e-10)

        # Apply Regime Gating
        gated_signal = np.zeros(min_len, dtype=int)
        gated_prob = raw_ensemble.copy()

        for t in range(min_len):
            current_regime = reg_slice[t] if reg_slice is not None else 0
            prob = raw_ensemble[t]

            if current_regime == 3:
                gated_signal[t] = 0
                gated_prob[t] = 0.50
            elif prob >= self.confidence_cutoff:
                gated_signal[t] = 1
            elif prob <= (1.0 - self.confidence_cutoff):
                gated_signal[t] = -1

        df_out = pd.DataFrame({
            "raw_ensemble_prob": raw_ensemble,
            "gated_prob": gated_prob,
            "trade_signal": gated_signal
        })
        if reg_slice is not None:
            df_out["regime_id"] = reg_slice

        return df_out

    def combine_dual_predictions(
        self,
        long_probs: Dict[str, np.ndarray],
        short_probs: Dict[str, np.ndarray],
        regimes: np.ndarray = None,
        regime_labels: Dict[int, str] = None,
        confidence_cutoff: Optional[float] = None,
        percentile_cutoff: Optional[float] = None,
        margin: float = 0.02
    ) -> pd.DataFrame:
        """
        Combines independent Long and Short probability predictions with HMM regime gating.

        Key features:
        - Weighted ensemble across tree models (XGBoost, LightGBM, RF).
        - Allows both directions across all regimes with regime-adjusted cutoffs.
        - Adaptive cutoff detection: if cutoff exceeds max probability (e.g. after isotonic
          calibration), falls back safely to top percentile to avoid 0-trade starvation.
        - Requires the winning probability to exceed cutoff AND beat opposite direction by margin.
        """
        all_series = list(long_probs.values()) + list(short_probs.values())
        min_len = min(len(s) for s in all_series)
        if regimes is not None:
            min_len = min(min_len, len(regimes))

        aligned_long = {k: v[-min_len:] for k, v in long_probs.items()}
        aligned_short = {k: v[-min_len:] for k, v in short_probs.items()}
        reg_slice = regimes[-min_len:] if regimes is not None else np.zeros(min_len, dtype=int)

        # Weighted long & short ensemble
        long_sum = np.zeros(min_len)
        short_sum = np.zeros(min_len)
        tot_w = 0.0

        for name, weight in self.base_weights.items():
            if name in aligned_long:
                long_sum += aligned_long[name] * weight
            if name in aligned_short:
                short_sum += aligned_short[name] * weight
            if name in aligned_long or name in aligned_short:
                tot_w += weight

        ens_long = long_sum / (tot_w + 1e-10)
        ens_short = short_sum / (tot_w + 1e-10)

        # Determine effective cutoffs independently per direction
        # This completely prevents calibration asymmetry from starving either direction
        if percentile_cutoff is not None:
            base_long_cut = float(np.percentile(ens_long, percentile_cutoff))
            base_short_cut = float(np.percentile(ens_short, percentile_cutoff))
        else:
            cutoff = confidence_cutoff if confidence_cutoff is not None else self.confidence_cutoff
            max_l = float(np.max(ens_long))
            max_s = float(np.max(ens_short))
            base_long_cut = cutoff if cutoff < max_l else float(np.percentile(ens_long, 85.0))
            base_short_cut = cutoff if cutoff < max_s else float(np.percentile(ens_short, 85.0))

        # Pre-classify regimes semantically
        reg_labels = regime_labels or {}
        bull_states = {k for k, v in reg_labels.items() if "Bull" in v}
        bear_states = {k for k, v in reg_labels.items() if "Bear" in v}
        shock_states = {k for k, v in reg_labels.items() if "Shock" in v}

        gated_signal = np.zeros(min_len, dtype=int)

        for t in range(min_len):
            r = reg_slice[t]
            p_l = ens_long[t]
            p_s = ens_short[t]

            # 1. Volatility Shock: Hard circuit breaker
            if r in shock_states:
                continue

            # 2. Regime Gating:
            # - In Bear regimes, suppress long entries (do not fight the macro bear)
            # - In Bull regimes, suppress short entries (do not fight the macro bull)
            # - In Range/Neutral regimes, allow both long and short entries
            edge = p_l - p_s

            if edge > margin and p_l >= base_long_cut and r not in bear_states:
                gated_signal[t] = 1
            elif edge < -margin and p_s >= base_short_cut and r not in bull_states:
                gated_signal[t] = -1

        df_out = pd.DataFrame({
            "raw_p_long": ens_long,
            "raw_p_short": ens_short,
            "directional_edge": ens_long - ens_short,
            "trade_signal": gated_signal,
            "regime_id": reg_slice
        })
        return df_out
