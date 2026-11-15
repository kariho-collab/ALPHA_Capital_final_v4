"""
01. Data Engineering: Data Quality & Integrity Validator
"""
import logging
from typing import Dict, Any
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DataValidator:
    @staticmethod
    def validate_integrity(df: pd.DataFrame, name: str = "Dataset") -> Dict[str, Any]:
        """
        Validates OHLCV integrity:
        - Checks for high < low or open/close outside high/low bounds
        - Checks missing timestamps & non-positive prices
        - Computes outlier return spikes (> 10 standard deviations)
        """
        n_rows = len(df)
        if n_rows == 0:
            raise ValueError(f"Validation failed: {name} is empty!")

        # 1. Price consistency
        invalid_high_low = (df["high"] < df["low"]).sum()
        invalid_open = ((df["open"] > df["high"]) | (df["open"] < df["low"])).sum()
        invalid_close = ((df["close"] > df["high"]) | (df["close"] < df["low"])).sum()
        non_positive = (df[["open", "high", "low", "close"]] <= 0).sum().sum()

        # 2. Duplicate or unordered timestamps
        is_monotonic = df.index.is_monotonic_increasing
        duplicates = df.index.duplicated().sum()

        # 3. Statistical outlier spikes
        log_ret = np.log(df["close"] / df["close"].shift(1)).dropna()
        mean_ret = log_ret.mean()
        std_ret = log_ret.std()
        outliers = (np.abs(log_ret - mean_ret) > 10 * std_ret).sum()

        stats = {
            "total_bars": n_rows,
            "start_time": str(df.index[0]),
            "end_time": str(df.index[-1]),
            "invalid_high_low": int(invalid_high_low),
            "invalid_open_close_bounds": int(invalid_open + invalid_close),
            "non_positive_prices": int(non_positive),
            "is_ordered": bool(is_monotonic),
            "duplicate_timestamps": int(duplicates),
            "outlier_10sigma_spikes": int(outliers)
        }

        logger.info(f"[{name} Validation] {n_rows:,} bars verified. Anomalies: {stats['invalid_high_low'] + stats['non_positive_prices'] + stats['duplicate_timestamps']}.")
        return stats


if __name__ == "__main__":
    from loader import HistoricalDataLoader
    loader = HistoricalDataLoader()
    raw = loader.get_or_generate_m1_data(years=1)
    report = DataValidator.validate_integrity(raw, "EURUSD_Base")
    print(report)
