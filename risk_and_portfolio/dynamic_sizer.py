"""
ALPHA Capital — Dynamic Volatility & Regime-Aware Position Sizing Engine
Replaces static 1.0% fixed fractional risk with an institutional risk-budgeting model:
1. Downscales risk budget in choppy or highly turbulent volatility spikes.
2. Allocates full conviction budget during stable, persistent momentum regimes.
3. Enforces hard institutional bounds: 0.01 lots minimum, 10.0 lots maximum.
"""
from typing import Dict, Any, Optional
import numpy as np


class DynamicPositionSizer:
    """
    Computes adaptive position sizing based on portfolio equity, stop-loss distance,
    local volatility (ATR), and latent market regime.
    """
    def __init__(
        self,
        base_risk_pct: float = 0.010,       # 1.0% base risk per trade
        min_risk_pct: float = 0.004,        # 0.4% floor during extreme chop
        max_risk_pct: float = 0.012,        # 1.2% cap during clean momentum
        min_lots: float = 0.01,
        max_lots: float = 10.0,
        lot_step: float = 0.01
    ):
        self.base_risk_pct = base_risk_pct
        self.min_risk_pct = min_risk_pct
        self.max_risk_pct = max_risk_pct
        self.min_lots = min_lots
        self.max_lots = max_lots
        self.lot_step = lot_step

    def calculate_lots(
        self,
        equity: float,
        stop_loss_pips: float,
        current_atr_pips: float,
        baseline_atr_pips: float = 5.0,
        regime_label: str = "Bullish Momentum Trend",
        pip_value_per_std_lot: float = 10.0
    ) -> float:
        """
        Calculates optimal lot size with volatility-inverse dampening and regime scaling.
        """
        if equity <= 0 or stop_loss_pips <= 0:
            return self.min_lots

        # Volatility dampener: if ATR is twice normal, reduce risk
        vol_ratio = max(0.5, current_atr_pips / max(baseline_atr_pips, 1.0))
        vol_mult = 1.0 / np.sqrt(vol_ratio)

        # Regime multiplier
        if "Momentum" in regime_label or "Trend" in regime_label:
            reg_mult = 1.10
        elif "Range" in regime_label or "Mean-Reverting" in regime_label:
            reg_mult = 0.70
        elif "Shock" in regime_label:
            reg_mult = 0.30
        else:
            reg_mult = 1.0

        # Effective risk percentage
        effective_risk = self.base_risk_pct * vol_mult * reg_mult
        effective_risk = min(self.max_risk_pct, max(self.min_risk_pct, effective_risk))

        risk_amount_usd = equity * effective_risk
        raw_lots = risk_amount_usd / (stop_loss_pips * pip_value_per_std_lot)

        # Quantize to lot_step
        lots = round(raw_lots / self.lot_step) * self.lot_step
        lots = min(self.max_lots, max(self.min_lots, lots))

        return float(lots)
