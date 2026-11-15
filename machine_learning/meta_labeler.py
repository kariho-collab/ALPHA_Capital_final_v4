"""
ALPHA Capital — López de Prado Meta-Labeling Engine
Reference: Marcos López de Prado (2018), 'Advances in Financial Machine Learning', Chapter 3.

Decouples side (direction) from bet sizing / execution:
1. Primary models predict direction: Long (+1) vs Short (-1).
2. Meta-Labeler predicts probability of success: y_meta in {0, 1}.
3. Prunes low-probability false breakouts, boosting precision and win rate.
"""
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
import joblib
from pathlib import Path


class MetaLabeler:
    """
    Trains a secondary binary classifier on trade events to filter out false breakouts.
    y_meta = 1 if primary signal achieved target profit, 0 otherwise.
    """
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.model: Optional[Any] = None
        self.meta_feature_names: List[str] = []

    def construct_meta_features(
        self,
        df_ltf: pd.DataFrame,
        signals: pd.Series,
        long_probs: np.ndarray,
        short_probs: np.ndarray,
        regimes: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """
        Extracts execution and microstructure meta-features at signal generation bars.
        """
        n = len(signals)
        pip = 0.0001

        atr = df_ltf["ltf_atr_pips"].values[:n] if "ltf_atr_pips" in df_ltf.columns else np.full(n, 5.0)
        close = df_ltf["close"].values[:n]
        high = df_ltf["high"].values[:n]
        low = df_ltf["low"].values[:n]
        bar_range = (high - low) / pip

        prob_max = np.maximum(long_probs, short_probs)
        prob_edge = np.abs(long_probs - short_probs)

        # Microstructure features
        range_atr_ratio = bar_range / (atr + 1e-6)
        hour_of_day = pd.to_datetime(df_ltf.index[:n]).hour.values

        reg_vals = regimes.values[:n] if regimes is not None else np.zeros(n)

        meta_df = pd.DataFrame({
            "prob_max": prob_max,
            "prob_edge": prob_edge,
            "atr_pips": atr,
            "bar_range_pips": bar_range,
            "range_atr_ratio": range_atr_ratio,
            "hour_of_day": hour_of_day,
            "regime_id": reg_vals,
            "signal_dir": signals.values[:n]
        }, index=df_ltf.index[:n])

        return meta_df

    def generate_meta_labels(
        self,
        df_ltf: pd.DataFrame,
        signals: pd.Series,
        forward_bars: int = 12,
        profit_ratio: float = 1.2
    ) -> pd.Series:
        """
        Computes binary meta-labels (1 for win, 0 for loss) for all signaled bars.
        Simulates whether the trade achieved profit target before stop loss.
        """
        pip = 0.0001
        close = df_ltf["close"].values
        high = df_ltf["high"].values
        low = df_ltf["low"].values
        atr = df_ltf["ltf_atr_pips"].values if "ltf_atr_pips" in df_ltf.columns else np.full(len(close), 5.0)
        n = len(signals)

        meta_labels = np.full(n, np.nan)
        sig_indices = np.where(signals.values[:n] != 0)[0]

        for idx in sig_indices:
            # Need forward bars
            if idx + forward_bars >= n:
                continue

            direction = signals.iloc[idx]
            entry_p = close[idx]
            local_atr = atr[idx] * pip
            sl_dist = max(local_atr * 1.5, 8.0 * pip)
            tp_dist = max(local_atr * 2.0, 16.0 * pip)

            if direction == 1:
                sl_p = entry_p - sl_dist
                tp_p = entry_p + tp_dist
            else:
                sl_p = entry_p + sl_dist
                tp_p = entry_p - tp_dist

            # Forward scan
            outcome = 0
            for f in range(1, forward_bars + 1):
                f_idx = idx + f
                h_val = high[f_idx]
                l_val = low[f_idx]

                if direction == 1:
                    if l_val <= sl_p:
                        outcome = 0
                        break
                    elif h_val >= tp_p:
                        outcome = 1
                        break
                else:
                    if h_val >= sl_p:
                        outcome = 0
                        break
                    elif l_val <= tp_p:
                        outcome = 1
                        break

            meta_labels[idx] = outcome

        return pd.Series(meta_labels, index=signals.index[:n], name="meta_label")

    def fit(self, X_meta: pd.DataFrame, y_meta: pd.Series):
        """
        Fits LightGBM meta-classifier on signaled instances only.
        """
        valid_mask = ~y_meta.isna()
        X_train = X_meta.loc[valid_mask]
        y_train = y_meta.loc[valid_mask].values.astype(int)

        self.meta_feature_names = list(X_train.columns)

        self.model = lgb.LGBMClassifier(
            n_estimators=100,
            learning_rate=0.03,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=self.random_state,
            verbose=-1
        )
        self.model.fit(X_train, y_train)

    def predict_success_proba(self, X_meta: pd.DataFrame) -> np.ndarray:
        """
        Returns P(y_meta = 1) for each instance.
        """
        if self.model is None:
            return np.ones(len(X_meta))
        return self.model.predict_proba(X_meta)[:, 1]

    def filter_signals(
        self,
        signals: pd.Series,
        X_meta: pd.DataFrame,
        meta_threshold: float = 0.50
    ) -> Tuple[pd.Series, int]:
        """
        Applies meta-labeler to filter out low-conviction signals.
        Returns filtered signals and count of pruned false breakouts.
        """
        sig_indices = np.where(signals.values != 0)[0]
        if len(sig_indices) == 0 or self.model is None:
            return signals.copy(), 0

        X_active = X_meta.iloc[sig_indices]
        probs = self.predict_success_proba(X_active)

        filtered = signals.copy()
        pruned_count = 0

        for i, idx in enumerate(sig_indices):
            if probs[i] < meta_threshold:
                filtered.iloc[idx] = 0
                pruned_count += 1

        return filtered, pruned_count
