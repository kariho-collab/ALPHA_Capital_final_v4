"""
02. Statistical Discovery: Volatility Dynamics & Estimators
Implements Parkinson, Garman-Klass, and Yang-Zhang volatility estimators.
"""
from typing import Dict, Any
import numpy as np
import pandas as pd


class VolatilityDynamicsAnalyzer:
    @staticmethod
    def parkinson_volatility(df: pd.DataFrame, window: int = 20) -> pd.Series:
        """
        Parkinson estimator using High-Low spread (5x more efficient than close-to-close).
        sigma_p = sqrt( (1 / (4 * ln(2))) * sum(ln(H/L)^2) / N )
        """
        factor = 1.0 / (4.0 * np.log(2.0))
        log_hl = np.log(df["high"] / df["low"]) ** 2
        roll_hl = log_hl.rolling(window=window).mean()
        return np.sqrt(factor * roll_hl)

    @staticmethod
    def garman_klass_volatility(df: pd.DataFrame, window: int = 20) -> pd.Series:
        """
        Garman-Klass estimator: incorporates Open, High, Low, Close (8x more efficient).
        sigma_gk = sqrt( 0.5 * ln(H/L)^2 - (2*ln(2) - 1) * ln(C/O)^2 )
        """
        log_hl = 0.5 * (np.log(df["high"] / df["low"]) ** 2)
        log_co = (2.0 * np.log(2.0) - 1.0) * (np.log(df["close"] / df["open"]) ** 2)
        gk = log_hl - log_co
        return np.sqrt(np.maximum(0.0, gk.rolling(window=window).mean()))

    @staticmethod
    def compute_volatility_summary(df: pd.DataFrame, annualization_factor: float = np.sqrt(252 * 288)) -> Dict[str, Any]:
        """
        Computes comparative annualized volatility metrics and clustering coefficients.
        """
        log_ret = np.log(df["close"] / df["close"].shift(1)).dropna()
        c2c_vol = float(log_ret.std() * annualization_factor)

        gk_series = VolatilityDynamicsAnalyzer.garman_klass_volatility(df).dropna()
        gk_vol = float(gk_series.mean() * annualization_factor)

        park_series = VolatilityDynamicsAnalyzer.parkinson_volatility(df).dropna()
        park_vol = float(park_series.mean() * annualization_factor)

        # Volatility Clustering (Autocorrelation of absolute returns)
        abs_ret = np.abs(log_ret)
        vol_persistence = float(abs_ret.autocorr(lag=1))

        return {
            "close_to_close_annual_vol": round(c2c_vol * 100, 2),
            "garman_klass_annual_vol": round(gk_vol * 100, 2),
            "parkinson_annual_vol": round(park_vol * 100, 2),
            "volatility_clustering_autocorr": round(vol_persistence, 4),
            "clustering_detected": vol_persistence > 0.10
        }
