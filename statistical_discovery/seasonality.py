"""
02. Statistical Discovery: Seasonality & Intraday Diurnal Cycles
"""
from typing import Dict, Any
import numpy as np
import pandas as pd


class SeasonalityAnalyzer:
    @staticmethod
    def analyze_intraday_sessions(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyzes mean return and volatility across major Forex market sessions (UTC):
        - Asian Session: 00:00 - 07:00 UTC (Tokyo/Sydney)
        - London Open / Morning: 07:00 - 12:00 UTC (European Liquidity)
        - London / New York Overlap: 12:00 - 16:00 UTC (Global Peak Institutional Volume)
        - New York Afternoon: 16:00 - 21:00 UTC
        - Quiet / Transition: 21:00 - 24:00 UTC
        """
        data = df.copy()
        if not hasattr(data.index, "hour"):
            return {}

        hours = data.index.hour
        data["log_ret"] = np.log(data["close"] / data["close"].shift(1))
        data["range_pips"] = (data["high"] - data["low"]) / 0.0001

        sessions = {
            "Asian (00-07 UTC)": (hours >= 0) & (hours < 7),
            "London Morning (07-12 UTC)": (hours >= 7) & (hours < 12),
            "London-NY Overlap (12-16 UTC)": (hours >= 12) & (hours < 16),
            "NY Afternoon (16-21 UTC)": (hours >= 16) & (hours < 21),
            "Quiet (21-24 UTC)": (hours >= 21)
        }

        results = {}
        for name, mask in sessions.items():
            subset = data[mask]
            if len(subset) > 0:
                results[name] = {
                    "bar_count": int(len(subset)),
                    "avg_bar_range_pips": round(float(subset["range_pips"].mean()), 2),
                    "annualized_vol_pct": round(float(subset["log_ret"].std() * np.sqrt(252 * 288) * 100), 2),
                    "mean_return_bps": round(float(subset["log_ret"].mean() * 10000), 3)
                }

        # Day of week seasonality
        days = {0: "Monday", 1: "Tuesday", 2: "Wednesday", 3: "Thursday", 4: "Friday"}
        day_results = {}
        for day_num, day_name in days.items():
            sub_day = data[data.index.dayofweek == day_num]
            if len(sub_day) > 0:
                day_results[day_name] = {
                    "avg_daily_pips": round(float(sub_day["range_pips"].mean()), 2),
                    "volatility_pct": round(float(sub_day["log_ret"].std() * np.sqrt(252 * 288) * 100), 2)
                }

        return {
            "session_dynamics": results,
            "day_of_week_dynamics": day_results
        }
