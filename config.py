"""
Institutional EUR/USD Research Framework Configuration
Multi-Timeframe, Regime-Switching, ML & Deep Learning Pipeline
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"

for p in (DATA_DIR, MODELS_DIR, REPORTS_DIR):
    p.mkdir(parents=True, exist_ok=True)


@dataclass
class ResearchConfig:
    # 1. Asset & Horizon Specs
    symbol: str = "EURUSD"
    pip_size: float = 0.0001
    typical_spread_pips: float = 0.8       # Institutional EUR/USD ECN spread
    slippage_pips: float = 0.2             # Realistic execution slippage
    commission_per_lot_usd: float = 3.0    # ECN round-turn commission per standard lot
    execution_delay_bars: int = 1          # Next-bar Open execution (1-bar decision latency)

    # Timeframe Hierarchy:
    # LTF: Lower Timeframe (Execution / Microstructure)
    # MTF: Medium Timeframe (Swing Momentum / Setup)
    # HTF: Higher Timeframe (Macro Regime / Trend)
    ltf_timeframe: str = "5min"            # Execution bars (M5 or M1)
    mtf_timeframe: str = "1h"              # Setup / Momentum bars (H1)
    htf_timeframe: str = "1D"              # Macro trend bars (D1)

    # 2. Research Data Period & 3-Way Chronological Partitioning
    start_year: int = 2016
    end_year: int = 2026
    research_years: int = 3                # Default research window (3 years = ~220,000 bars)
    train_ratio: float = 0.60              # 60% Train (fit tree models)
    val_ratio: float = 0.20                # 20% Validation (fit calibration & threshold tuning)
    test_ratio: float = 0.20               # 20% Final Untouched Test (single audit evaluation)

    # 3. Regime Detection (HMM)
    hmm_states: int = 3                    # 0: Bull Trend, 1: Bear Trend, 2: Mean-Reverting Range
    regime_features: List[str] = field(default_factory=lambda: ["log_return", "volatility", "trend_strength"])

    # 4. Target Generation
    forward_bars: int = 6                  # Horizon for signal prediction (6 * 5m = 30m)
    target_pip_threshold: float = 4.0      # Min net move in pips to justify spread
    use_adaptive_target: bool = True       # Dynamic volatility-scaled target
    adaptive_target_atr_mult: float = 0.75 # 0.75 * ATR(14)
    min_adaptive_target_pips: float = 3.0  # Floor for adaptive target

    # 5. Machine Learning Models
    models_to_train: List[str] = field(default_factory=lambda: ["xgboost", "random_forest", "lightgbm", "lstm", "transformer"])
    cv_folds: int = 5
    random_seed: int = 42

    # 6. Deep Learning Hyperparameters
    sequence_length: int = 24              # 24 lookback bars for sequence models
    lstm_units: int = 64
    transformer_heads: int = 4
    transformer_dim: int = 64
    batch_size: int = 64
    epochs: int = 15

    # 7. Risk & Portfolio Parameters
    initial_capital: float = 100000.0      # $100,000 Institutional base
    max_risk_per_trade_pct: float = 0.005  # 0.5% equity risk per trade (institutional standard)
    kelly_fraction: float = 0.5            # Half-Kelly multiplier for capital safety
    max_drawdown_limit_pct: float = 10.0   # 10% hard circuit breaker
    atr_stop_mult: float = 1.5             # Stop Loss = 1.5 * ATR
    atr_take_profit_mult: float = 3.0      # Take Profit = 3.0 * ATR (1:2 R:R)
    use_dynamic_sizing: bool = False       # Phase 3 Dynamic Volatility & Regime Position Sizing


default_config = ResearchConfig()
