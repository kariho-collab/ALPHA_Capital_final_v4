"""
01. Data Engineering: Multi-Timeframe Resampler (LTF, MTF, HTF)
Resamples base candles into synchronized lower, medium, and higher timeframes.
"""
import logging
from typing import Dict
import pandas as pd
from config import ResearchConfig, default_config

logger = logging.getLogger(__name__)


class MultiTimeframeResampler:
    def __init__(self, config: ResearchConfig = default_config):
        self.config = config

    def resample_all(self, df_base: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Takes raw base candles and creates synchronized timeframe tiers:
        - LTF: Lower Timeframe (e.g. 5min) - for execution & micro entry
        - MTF: Medium Timeframe (e.g. 1h) - for swing momentum & structure
        - HTF: Higher Timeframe (e.g. 1D) - for macro regime & major trend
        """
        logger.info(
            f"Resampling into LTF ({self.config.ltf_timeframe}), "
            f"MTF ({self.config.mtf_timeframe}), HTF ({self.config.htf_timeframe})..."
        )

        ltf_df = self._resample_timeframe(df_base, self.config.ltf_timeframe)
        mtf_df = self._resample_timeframe(df_base, self.config.mtf_timeframe)
        htf_df = self._resample_timeframe(df_base, self.config.htf_timeframe)

        logger.info(
            f"Resampling complete: LTF={len(ltf_df):,} bars, "
            f"MTF={len(mtf_df):,} bars, HTF={len(htf_df):,} bars."
        )

        return {
            "ltf": ltf_df,
            "mtf": mtf_df,
            "htf": htf_df
        }

    def _resample_timeframe(self, df: pd.DataFrame, freq: str) -> pd.DataFrame:
        agg_rules = {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum"
        }
        resampled = df.resample(freq).agg(agg_rules)
        # Drop intervals with zero trading activity (e.g. weekends)
        resampled = resampled.dropna(subset=["close"])
        resampled = resampled[(resampled["high"] > 0) & (resampled["low"] > 0)]
        return resampled

    def align_higher_to_lower(
        self,
        ltf_df: pd.DataFrame,
        higher_df: pd.DataFrame,
        prefix: str = "htf_"
    ) -> pd.DataFrame:
        """
        Forward-fills completed higher timeframe bars into lower timeframe index.
        CRITICAL: Shifts by 1 higher-timeframe bar to guarantee ZERO lookahead bias!
        """
        shifted_higher = higher_df.shift(1).copy()
        shifted_higher.columns = [f"{prefix}{c}" for c in shifted_higher.columns]

        # Merge asof / reindex forward fill
        aligned = ltf_df.merge(
            shifted_higher,
            how="left",
            left_index=True,
            right_index=True
        ).ffill().bfill()

        return aligned


if __name__ == "__main__":
    from loader import HistoricalDataLoader
    loader = HistoricalDataLoader()
    raw = loader.get_or_generate_m1_data(years=2)
    resampler = MultiTimeframeResampler()
    dfs = resampler.resample_all(raw)
    print("LTF Sample:", dfs["ltf"].tail(2))
    print("MTF Sample:", dfs["mtf"].tail(2))
    print("HTF Sample:", dfs["htf"].tail(2))
