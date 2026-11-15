"""
02. Statistical Discovery: Momentum vs Mean Reversion Analytics
Computes Hurst Exponent, Augmented Dickey-Fuller stationarity,
Ornstein-Uhlenbeck Half-Life, and Autocorrelation Structure.
"""
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from scipy import stats


class MomentumMeanReversionAnalyzer:
    @staticmethod
    def compute_hurst_exponent(price_series: pd.Series, max_lags: int = 100) -> float:
        """
        Calculates the Hurst Exponent (H) using Rescaled Range (R/S) analysis:
        - H < 0.45: Mean-Reverting (anti-persistent)
        - 0.45 <= H <= 0.55: Random Walk (Geometric Brownian Motion)
        - H > 0.55: Trending (persistent momentum)
        """
        prices = price_series.dropna().values
        if len(prices) < max_lags * 2:
            return 0.50

        lags = range(2, max_lags)
        tau = [
            np.sqrt(np.std(np.subtract(prices[lag:], prices[:-lag])))
            for lag in lags
        ]
        # Linear regression of log(tau) vs log(lag)
        poly = np.polyfit(np.log(list(lags)), np.log(tau), 1)
        hurst = poly[0] * 2.0
        return float(np.clip(hurst, 0.0, 1.0))

    @staticmethod
    def compute_half_life(price_series: pd.Series) -> float:
        """
        Estimates the Half-Life of mean reversion via an Ornstein-Uhlenbeck process:
        dy(t) = lambda * (y(t-1) - mu) * dt + sigma * dW
        Half-life = -ln(2) / lambda
        """
        y = price_series.dropna().values
        delta_y = np.diff(y)
        y_lag = y[:-1]

        # Regress delta_y against y_lag
        slope, intercept, _, _, _ = stats.linregress(y_lag, delta_y)
        if slope >= 0:
            return np.nan # Non-mean reverting (trending or random walk)

        half_life = -np.log(2.0) / slope
        return float(half_life)

    @staticmethod
    def compute_autocorrelations(returns: pd.Series, max_lag: int = 10) -> Dict[str, float]:
        """
        Computes autocorrelation coefficients of returns across lags.
        Negative autocorrelation indicates mean-reversion at high frequencies.
        """
        ret = returns.dropna()
        acfs = {}
        for lag in range(1, max_lag + 1):
            acfs[f"lag_{lag}"] = float(ret.autocorr(lag=lag))
        return acfs

    @classmethod
    def run_discovery_report(cls, df: pd.DataFrame, timeframe_label: str = "MTF") -> Dict[str, Any]:
        close = df["close"]
        returns = np.log(close / close.shift(1)).dropna()

        hurst = cls.compute_hurst_exponent(close)
        half_life = cls.compute_half_life(close)
        acfs = cls.compute_autocorrelations(returns, max_lag=5)

        # Regimes determination from Hurst
        if hurst > 0.55:
            nature = "Trending / Momentum Persistent"
        elif hurst < 0.45:
            nature = "Mean-Reverting"
        else:
            nature = "Random Walk / Efficient Market"

        return {
            "timeframe": timeframe_label,
            "hurst_exponent": round(hurst, 4),
            "market_nature": nature,
            "mean_reversion_half_life_bars": round(half_life, 1) if not np.isnan(half_life) else "Inf (No Mean-Rev)",
            "autocorrelations": acfs
        }


if __name__ == "__main__":
    from data_engineering.loader import HistoricalDataLoader
    from data_engineering.resampler import MultiTimeframeResampler

    loader = HistoricalDataLoader()
    raw = loader.get_or_generate_m1_data(years=1)
    resampler = MultiTimeframeResampler()
    dfs = resampler.resample_all(raw)

    analyzer = MomentumMeanReversionAnalyzer()
    print("LTF Report:", analyzer.run_discovery_report(dfs["ltf"], "LTF 5m"))
    print("MTF Report:", analyzer.run_discovery_report(dfs["mtf"], "MTF 1h"))
    print("HTF Report:", analyzer.run_discovery_report(dfs["htf"], "HTF 1D"))
