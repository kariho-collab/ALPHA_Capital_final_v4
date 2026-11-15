"""
ALPHA Capital — Macro Economic Calendar & News Blackout Engine
Implements the Anticipatory Macro News Blackout Protocol for EUR/USD:
- US Consumer Price Index (CPI)
- US Non-Farm Payrolls (NFP) & Unemployment
- Federal Open Market Committee (FOMC) Rate Decisions & Press Conferences
- European Central Bank (ECB) Rate Decisions & Press Conferences
- US Core PCE Price Index

Protects the quantitative strategy against spread blowouts, extreme slippage,
and the initial 5-minute whipsaw that occurs during Tier-1 macro releases.
"""
from typing import List, Tuple, Set, Dict, Optional
import datetime
import numpy as np
import pandas as pd


class MacroNewsCalendar:
    """
    Identifies Tier-1 macroeconomic news events and computes blackout windows.
    Default blackout window: 30 minutes pre-release to 30 minutes post-release.
    """
    def __init__(
        self,
        pre_blackout_minutes: int = 30,
        post_blackout_minutes: int = 30
    ):
        self.pre_blackout = datetime.timedelta(minutes=pre_blackout_minutes)
        self.post_blackout = datetime.timedelta(minutes=post_blackout_minutes)
        self._events_cache: Optional[List[pd.Timestamp]] = None

    def get_tier1_events(self, start_date: str = "2024-01-01", end_date: str = "2026-12-31") -> List[pd.Timestamp]:
        """
        Generates scheduled high-impact Tier-1 event timestamps for EUR/USD.
        In live trading, this can be augmented/replaced by live MT5 calendar API.
        """
        if self._events_cache is not None:
            return self._events_cache

        events = []
        dt_start = pd.Timestamp(start_date, tz="UTC")
        dt_end = pd.Timestamp(end_date, tz="UTC")

        # Iterate over all months in range
        cur = pd.Timestamp(f"{dt_start.year}-{dt_start.month:02d}-01", tz="UTC")
        while cur <= dt_end:
            year = cur.year
            month = cur.month

            # -----------------------------------------------------------------
            # 1. US Non-Farm Payrolls (NFP): 1st Friday of the month
            # Time: 12:30 UTC (during Daylight Saving) / 13:30 UTC (Standard)
            # -----------------------------------------------------------------
            first_day = pd.Timestamp(f"{year}-{month:02d}-01", tz="UTC")
            # Day of week: Monday=0, Friday=4
            days_to_friday = (4 - first_day.weekday()) % 7
            nfp_friday = first_day + pd.Timedelta(days=days_to_friday)
            
            # Daylight saving check (US DST: ~March second Sunday to Nov first Sunday)
            is_dst = (month > 3 and month < 11) or (month == 3 and nfp_friday.day >= 8) or (month == 11 and nfp_friday.day < 7)
            nfp_hour = 12 if is_dst else 13
            events.append(nfp_friday.replace(hour=nfp_hour, minute=30))

            # -----------------------------------------------------------------
            # 2. US CPI: Typically 2nd Wednesday of the month at 12:30 / 13:30 UTC
            # -----------------------------------------------------------------
            days_to_wed = (2 - first_day.weekday()) % 7
            first_wed = first_day + pd.Timedelta(days=days_to_wed)
            cpi_wed = first_wed + pd.Timedelta(days=7)  # 2nd Wednesday
            events.append(cpi_wed.replace(hour=nfp_hour, minute=30))

            # -----------------------------------------------------------------
            # 3. US Core PCE: Typically last Friday of the month at 12:30 / 13:30 UTC
            # -----------------------------------------------------------------
            next_m = cur + pd.DateOffset(months=1)
            last_day = next_m - pd.Timedelta(days=1)
            days_back_to_friday = (last_day.weekday() - 4) % 7
            last_friday = last_day - pd.Timedelta(days=days_back_to_friday)
            events.append(last_friday.replace(hour=nfp_hour, minute=30))

            # -----------------------------------------------------------------
            # 4. FOMC Meetings (8 per year: ~Jan, Mar, May, Jun, Jul, Sep, Nov, Dec)
            # Released Wednesday: Announcement at 18:00/19:00 UTC + Press Conf 18:30/19:30 UTC
            # -----------------------------------------------------------------
            if month in (1, 3, 5, 6, 7, 9, 11, 12):
                # Third Wednesday
                third_wed = first_wed + pd.Timedelta(days=14)
                fomc_hour = 18 if is_dst else 19
                events.append(third_wed.replace(hour=fomc_hour, minute=0))
                events.append(third_wed.replace(hour=fomc_hour, minute=30))  # Press conf

            # -----------------------------------------------------------------
            # 5. ECB Meetings (8 per year: ~Jan, Mar, Apr, Jun, Jul, Sep, Oct, Dec)
            # Released Thursday: Rate at 12:15 UTC + Press Conf 12:45 UTC
            # -----------------------------------------------------------------
            if month in (1, 3, 4, 6, 7, 9, 10, 12):
                days_to_thu = (3 - first_day.weekday()) % 7
                first_thu = first_day + pd.Timedelta(days=days_to_thu)
                second_thu = first_thu + pd.Timedelta(days=7)
                events.append(second_thu.replace(hour=12, minute=15))
                events.append(second_thu.replace(hour=12, minute=45))  # Press conf

            cur = cur + pd.DateOffset(months=1)

        events = sorted([e for e in events if dt_start <= e <= dt_end])
        self._events_cache = events
        return events

    def compute_blackout_mask(self, index: pd.DatetimeIndex) -> pd.Series:
        """
        Given a DatetimeIndex of market bars (e.g., 5-minute bars),
        returns a boolean Series where True indicates the bar is inside
        a high-impact macro news blackout window (pre-release or post-release).
        """
        # Ensure UTC timezone
        idx = pd.to_datetime(index, utc=True)
        events = self.get_tier1_events(
            start_date=str(idx[0].date()),
            end_date=str(idx[-1].date() + datetime.timedelta(days=1))
        )

        blackout_mask = np.zeros(len(idx), dtype=bool)

        for event in events:
            window_start = event - self.pre_blackout
            window_end = event + self.post_blackout
            in_window = (idx >= window_start) & (idx <= window_end)
            blackout_mask = blackout_mask | in_window

        return pd.Series(blackout_mask, index=index, name="is_macro_news_blackout")

    def filter_signals(self, signals: pd.Series) -> Tuple[pd.Series, int]:
        """
        Applies macro news blackout to a trading signal series.
        Suppresses signals inside the blackout window to 0 (flat/hold).
        Returns the filtered signals series and the count of suppressed signals.
        """
        mask = self.compute_blackout_mask(signals.index)
        filtered = signals.copy()
        suppressed_count = int(((signals != 0) & mask).sum())
        filtered[mask] = 0
        return filtered, suppressed_count
