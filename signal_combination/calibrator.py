"""
05. Signal Combination: Probability Calibrator & Reliability Diagrams
Supports both assessment and active recalibration via isotonic regression.
"""
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss


class ProbabilityCalibrator:
    @staticmethod
    def assess_calibration(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> Dict[str, Any]:
        """
        Computes empirical reliability curve and Brier score.
        A lower Brier score denotes a more calibrated, trustworthy probability output.
        """
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="uniform")
        brier = brier_score_loss(y_true, y_prob)

        # Expected Calibration Error (ECE)
        ece = float(np.mean(np.abs(prob_true - prob_pred)))

        return {
            "brier_score": round(float(brier), 4),
            "expected_calibration_error": round(ece, 4),
            "prob_true_bins": prob_true.tolist(),
            "prob_pred_bins": prob_pred.tolist()
        }

    @staticmethod
    def fit_isotonic(y_true: np.ndarray, y_prob: np.ndarray) -> IsotonicRegression:
        """
        Fits an isotonic regression model to map raw model probabilities to
        calibrated probabilities. The isotonic model preserves ordering while
        adjusting the scale so that predicted probabilities match observed frequencies.
        
        Args:
            y_true: Binary labels from the calibration set (NOT the test set).
            y_prob: Raw model probabilities from the calibration set.
        
        Returns:
            Fitted IsotonicRegression model.
        """
        iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        iso.fit(y_prob, y_true)
        return iso

    @staticmethod
    def calibrate_probabilities(
        iso_model: IsotonicRegression,
        y_prob: np.ndarray
    ) -> np.ndarray:
        """
        Transforms raw model probabilities through a fitted isotonic model.
        """
        return iso_model.predict(y_prob)
