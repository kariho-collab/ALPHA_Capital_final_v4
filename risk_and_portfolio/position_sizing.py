"""
06. Risk and Portfolio Research: Institutional Position Sizing Algorithms
Implements Fixed Fractional, Volatility-Targeted, and Half-Kelly Sizing.
"""
import numpy as np


class PositionSizer:
    @staticmethod
    def fixed_fractional(
        equity: float,
        risk_pct: float,
        stop_loss_pips: float,
        pip_value_per_lot: float = 10.0
    ) -> float:
        """
        Calculates position size (in standard lots) risking a fixed fraction of equity.
        Lots = (Equity * Risk_Pct) / (Stop_Loss_Pips * Pip_Value)
        """
        risk_amount = equity * risk_pct
        lot_size = risk_amount / (max(stop_loss_pips, 1.0) * pip_value_per_lot)
        return float(np.clip(lot_size, 0.01, 100.0))

    @staticmethod
    def half_kelly_criterion(
        equity: float,
        win_rate: float,
        win_loss_ratio: float,
        max_cap_pct: float = 0.02
    ) -> float:
        """
        Half-Kelly Criterion for optimal long-term geometric capital growth with drawndown dampening.
        Kelly fraction f* = (p * b - q) / b * 0.5
        """
        p = win_rate
        q = 1.0 - p
        b = max(win_loss_ratio, 0.1)

        f_star = (p * b - q) / b
        half_kelly = max(0.0, f_star * 0.5)

        # Cap maximum risk per trade
        effective_fraction = min(half_kelly, max_cap_pct)
        return float(effective_fraction)

    @staticmethod
    def volatility_targeting(
        equity: float,
        target_annual_vol: float,
        current_atr_pips: float,
        annualization_factor: float = np.sqrt(252 * 288)
    ) -> float:
        """
        Scales lot sizes dynamically to target a constant annualized portfolio volatility.
        """
        current_vol = (current_atr_pips * 0.0001) * annualization_factor
        vol_scalar = target_annual_vol / max(current_vol, 0.02)
        base_lots = (equity / 100000.0) * vol_scalar
        return float(np.clip(base_lots, 0.01, 50.0))
