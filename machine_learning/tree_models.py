"""
04. Machine Learning: Gradient Boosted Trees & Random Forest Benchmark
Implements XGBoost, LightGBM, and Random Forest with 3-Way Chronological Partitioning:
- Training (60%): Model parameter learning
- Validation (20%): Calibration fitting & probability threshold tuning
- Testing (20%): Final untouched out-of-sample evaluation
Includes Walk-Forward TimeSeriesSplit Cross-Validation.
"""
import logging
from typing import Dict, Any, Tuple, List
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb

logger = logging.getLogger(__name__)


class TabularMLTrainer:
    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    def train_xgboost(self, X_train: np.ndarray, y_train: np.ndarray) -> xgb.XGBClassifier:
        model = xgb.XGBClassifier(
            n_estimators=250,
            learning_rate=0.03,
            max_depth=4,
            subsample=0.85,
            colsample_bytree=0.75,
            min_child_weight=20,
            reg_alpha=2.0,
            reg_lambda=3.0,
            random_state=self.random_state,
            eval_metric="logloss",
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        return model

    def train_lightgbm(self, X_train: np.ndarray, y_train: np.ndarray) -> lgb.LGBMClassifier:
        model = lgb.LGBMClassifier(
            n_estimators=250,
            learning_rate=0.03,
            max_depth=4,
            num_leaves=16,
            min_child_samples=50,
            subsample=0.85,
            colsample_bytree=0.75,
            reg_alpha=2.0,
            reg_lambda=3.0,
            random_state=self.random_state,
            n_jobs=-1,
            verbose=-1
        )
        model.fit(X_train, y_train)
        return model

    def train_random_forest(self, X_train: np.ndarray, y_train: np.ndarray) -> RandomForestClassifier:
        model = RandomForestClassifier(
            n_estimators=150,
            max_depth=6,
            min_samples_leaf=50,
            max_features="sqrt",
            random_state=self.random_state,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        return model

    def run_walk_forward_cv(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_splits: int = 5
    ) -> Dict[str, Any]:
        """
        Executes anchored Walk-Forward cross-validation using TimeSeriesSplit on training data.
        """
        tscv = TimeSeriesSplit(n_splits=n_splits)
        scores = {"xgboost": [], "lightgbm": [], "random_forest": []}

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_tr, y_tr = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]

            # Fast evaluation models on fold (100 estimators for rapid CV)
            xgb_cv = xgb.XGBClassifier(
                n_estimators=100, learning_rate=0.04, max_depth=4, subsample=0.85,
                colsample_bytree=0.75, random_state=self.random_state, eval_metric="logloss", n_jobs=-1
            )
            lgb_cv = lgb.LGBMClassifier(
                n_estimators=100, learning_rate=0.04, max_depth=4, num_leaves=16,
                min_child_samples=50, random_state=self.random_state, n_jobs=-1, verbose=-1
            )
            rf_cv = RandomForestClassifier(
                n_estimators=80, max_depth=6, min_samples_leaf=50, max_features="sqrt",
                random_state=self.random_state, n_jobs=-1
            )
            xgb_cv.fit(X_tr, y_tr)
            lgb_cv.fit(X_tr, y_tr)
            rf_cv.fit(X_tr, y_tr)

            p_xgb = xgb_cv.predict_proba(X_val)[:, 1]
            p_lgb = lgb_cv.predict_proba(X_val)[:, 1]
            p_rf = rf_cv.predict_proba(X_val)[:, 1]

            scores["xgboost"].append(float(roc_auc_score(y_val, p_xgb)))
            scores["lightgbm"].append(float(roc_auc_score(y_val, p_lgb)))
            scores["random_forest"].append(float(roc_auc_score(y_val, p_rf)))

        return {
            model: {
                "mean_cv_auc": round(float(np.mean(vals)), 4),
                "std_cv_auc": round(float(np.std(vals)), 4),
                "fold_aucs": [round(v, 4) for v in vals]
            }
            for model, vals in scores.items()
        }

    def evaluate_all(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str = "target_long",
        train_ratio: float = 0.60,
        val_ratio: float = 0.20,
        run_cv: bool = True
    ) -> Dict[str, Any]:
        """
        Trains and benchmarks XGBoost, LightGBM, and Random Forest on strict 3-way chronological split:
        - Train (60%): Model fit
        - Val (20%): Calibration and threshold tuning
        - Test (20%): Final untouched out-of-sample audit
        """
        n = len(df)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        X = df[feature_cols].values
        y = df[target_col].values

        X_train, y_train = X[:train_end], y[:train_end]
        X_val, y_val = X[train_end:val_end], y[train_end:val_end]
        X_test, y_test = X[val_end:], y[val_end:]

        logger.info(f"Chronological 3-Way Partitioning: Train={len(X_train):,} | Val={len(X_val):,} | Test={len(X_test):,} bars.")

        # Walk-Forward Cross Validation on training segment
        cv_report = {}
        if run_cv:
            logger.info("Computing 5-fold Walk-Forward TimeSeriesSplit cross-validation on training partition...")
            cv_step = max(1, len(X_train) // 40000)
            cv_report = self.run_walk_forward_cv(X_train[::cv_step], y_train[::cv_step], n_splits=5)

        # 1. XGBoost
        xgb_m = self.train_xgboost(X_train, y_train)
        p_xgb_val = xgb_m.predict_proba(X_val)[:, 1]
        p_xgb_test = xgb_m.predict_proba(X_test)[:, 1]

        # 2. LightGBM
        lgb_m = self.train_lightgbm(X_train, y_train)
        p_lgb_val = lgb_m.predict_proba(X_val)[:, 1]
        p_lgb_test = lgb_m.predict_proba(X_test)[:, 1]

        # 3. Random Forest
        rf_m = self.train_random_forest(X_train, y_train)
        p_rf_val = rf_m.predict_proba(X_val)[:, 1]
        p_rf_test = rf_m.predict_proba(X_test)[:, 1]

        results = {
            "xgboost": {
                "val_auc": float(roc_auc_score(y_val, p_xgb_val)),
                "test_auc": float(roc_auc_score(y_test, p_xgb_test)),
                "val_brier": float(brier_score_loss(y_val, p_xgb_val)),
                "test_brier": float(brier_score_loss(y_test, p_xgb_test)),
                "model": xgb_m,
                "val_probabilities": p_xgb_val,
                "test_probabilities": p_xgb_test,
                "probabilities": p_xgb_test
            },
            "lightgbm": {
                "val_auc": float(roc_auc_score(y_val, p_lgb_val)),
                "test_auc": float(roc_auc_score(y_test, p_lgb_test)),
                "val_brier": float(brier_score_loss(y_val, p_lgb_val)),
                "test_brier": float(brier_score_loss(y_test, p_lgb_test)),
                "model": lgb_m,
                "val_probabilities": p_lgb_val,
                "test_probabilities": p_lgb_test,
                "probabilities": p_lgb_test
            },
            "random_forest": {
                "val_auc": float(roc_auc_score(y_val, p_rf_val)),
                "test_auc": float(roc_auc_score(y_test, p_rf_test)),
                "val_brier": float(brier_score_loss(y_val, p_rf_val)),
                "test_brier": float(brier_score_loss(y_test, p_rf_test)),
                "model": rf_m,
                "val_probabilities": p_rf_val,
                "test_probabilities": p_rf_test,
                "probabilities": p_rf_test
            },
            "cv_scores": cv_report,
            "val_indices": df.index[train_end:val_end],
            "test_indices": df.index[val_end:],
            "y_val": y_val,
            "y_test": y_test
        }

        # Feature Importance comparison
        feat_imp = pd.DataFrame({
            "feature": feature_cols,
            "xgb_importance": xgb_m.feature_importances_,
            "lgb_importance": lgb_m.feature_importances_ / (lgb_m.feature_importances_.sum() + 1e-10),
            "rf_importance": rf_m.feature_importances_
        }).sort_values("xgb_importance", ascending=False)

        results["feature_importance"] = feat_imp
        return results

    def evaluate_dual(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        train_ratio: float = 0.60,
        val_ratio: float = 0.20,
        run_cv: bool = True
    ) -> Dict[str, Any]:
        """
        Trains and benchmarks Dual ML architectures on 3-way chronological split:
        - Long Models: predict probability of upward breakout (target_long)
        - Short Models: predict probability of downward breakdown (target_short)
        """
        logger.info(">>> Training Dual Institutional Models (Long + Short) with 3-Way Chronological Split <<<")
        long_res = self.evaluate_all(df, feature_cols, target_col="target_long", train_ratio=train_ratio, val_ratio=val_ratio, run_cv=run_cv)
        short_res = self.evaluate_all(df, feature_cols, target_col="target_short", train_ratio=train_ratio, val_ratio=val_ratio, run_cv=run_cv)

        return {
            "long": long_res,
            "short": short_res,
            "val_indices": long_res["val_indices"],
            "test_indices": long_res["test_indices"]
        }
