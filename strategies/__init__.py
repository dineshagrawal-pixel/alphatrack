"""
Strategies package
Exports all trading strategies and their configurations.
"""

from strategies.breadth import run_breadth_backtest
from strategies.volatility import run_volatility_strategy
from strategies.ninesig import run_9sig_strategy as run_9sig_strategy
from strategies.moving_average import run_moving_average_strategy
from strategies.dual_momentum import run_dual_momentum_strategy
from strategies.volatility_vix import run_volatility_vix_strategy
from strategies.combined import run_combined_strategy


# Strategy configurations with UI parameters
STRATEGIES = {
    "Market Breadth Strategy": {
        "function": run_breadth_backtest,
        "params": [
            {"name": "ticker", "label": "Stock Symbol", "type": "text_input", "default": "TQQQ", "help": "The asset to trade based on market breadth signals."},
            {"name": "start_date", "label": "Start Date", "type": "date_input", "default": "2010-02-11", "help": "Beginning date for the backtest."},
            {"name": "initial_capital", "label": "Initial Capital", "type": "number_input", "default": 100000.0, "help": "Starting account balance."},
            {"name": "end_date", "label": "End Date", "type": "date_input", "default": "today", "help": "Ending date for the backtest."},
            {"name": "cash_floor_pct", "label": "Cash Floor Percentage (%)", "type": "slider", "min": 0.0, "max": 100.0, "value": 20.0, "step": 1.0, "transform": lambda x: x / 100, "help": "Minimum cash cushion based on portfolio high-water mark.", "advanced": True},
            {"name": "rebalance_pct", "label": "Rebalance Percentage (%)", "type": "slider", "min": 0.0, "max": 100.0, "value": 100.0, "step": 1.0, "transform": lambda x: x / 100, "help": "How much of the available signal stretch to use for rebalancing.", "advanced": True},
            {"name": "starting_cash_pct", "label": "Starting Cash Percentage (%)", "type": "slider", "min": 0.0, "max": 100.0, "value": 100.0, "step": 1.0, "transform": lambda x: x / 100, "help": "Initial allocation to cash vs. stocks.", "advanced": True},
            {"name": "cash_yield_apr", "label": "Cash Yield APR (%)", "type": "number_input", "min": 0.0, "value": 0.0, "step": 0.1, "format": "%.2f", "transform": lambda x: x / 100, "help": "Annual interest on cash holdings (e.g. 4.0 for 4%).", "advanced": True},
            {"name": "benchmark_symbol", "label": "Benchmark Symbol", "type": "text_input", "default": "SPY", "help": "Index comparison symbol.", "advanced": True},
            {"name": "slippage_bps", "label": "Slippage (bps)", "type": "number_input", "min": 0.0, "value": 5.0, "step": 1.0, "help": "Trading slippage in basis points (1 bp = 0.01%).", "advanced": True},
            {"name": "commission", "label": "Commission ($)", "type": "number_input", "min": 0.0, "value": 0.0, "step": 0.01, "help": "Fixed dollar cost per trade execution.", "advanced": True},
        ]
    },
    "9 Sig Strategy": {
        "function": run_9sig_strategy,
        "params": [
            {"name": "symbol_val", "label": "Stock Symbol (TQQQ)", "type": "text_input", "default": "TQQQ", "help": "The primary growth asset (e.g., TQQQ)."},
            {"name": "side_fund_val", "label": "Side Fund (SHY)", "type": "text_input", "default": "SHY", "help": "The safe-haven asset for the side fund (e.g., SHY or SGOV)."},
            {"name": "start_date_val", "label": "Start Date", "type": "date_input", "default": "2010-02-11", "help": "Beginning date for the backtest history."},
            {"name": "end_date", "label": "End Date", "type": "date_input", "default": "today", "help": "Ending date for the backtest."},
            {"name": "initial_capital_val", "label": "Initial Capital", "type": "number_input", "default": 100000.0, "help": "Starting portfolio balance."},
            {"name": "monthly_dep_val", "label": "Monthly Deposit ($)", "type": "number_input", "min": 0, "value": 0, "step": 100, "help": "Amount of 'New Money' to deposit into the portfolio every month (Rule VI)."},
            {"name": "target_tqqq_pct", "label": "Home Base Allocation %", "type": "slider", "min": 0.0, "max": 100.0, "value": 60.0, "step": 1.0, "transform": lambda x: x / 100, "help": "The target percentage for the growth asset (Rule I). Defaults to 60%."},
            {"name": "new_money_to_signal", "label": "New Money to Signal %", "type": "slider", "min": 0.0, "max": 100.0, "value": 60.0, "step": 1.0, "transform": lambda x: x / 100, "help": "Percentage of new deposits that increase the Signal Line (Rule VI).", "advanced": True},
            {"name": "gap_rebalance_pct", "label": "Gap Rebalance %", "type": "slider", "min": 0.0, "max": 100.0, "value": 50.0, "step": 1.0, "transform": lambda x: x / 100, "help": "Percentage of the gap between actual TQQQ value and the Signal Line to rebalance (Rule III).", "advanced": True},
            {"name": "q_growth_target", "label": "Quarterly Growth Target (%)", "type": "number_input", "min": 0.0, "value": 9.0, "step": 0.1, "format": "%.1f", "transform": lambda x: 1 + (x / 100), "help": "The default quarterly growth rate for the Signal Line (e.g., 9% per quarter)."},
            {"name": "q_growth_cap", "label": "Quarterly Growth Cap (%)", "type": "number_input", "min": 0.0, "value": 30.0, "step": 0.1, "format": "%.1f", "transform": lambda x: 1 + (x / 100), "help": "The maximum quarterly growth allowed for the Signal Line (Rule II).", "advanced": True},
            {"name": "signal_floor_pct", "label": "Signal Floor %", "type": "slider", "min": 50.0, "max": 100.0, "value": 88.0, "step": 1.0, "transform": lambda x: x / 100, "help": "Minimum Signal Line value as a percentage of its all-time high water mark (Rule II).", "advanced": True},
            {"name": "cash_floor_hwm_pct", "label": "Cash Floor (HWM) %", "type": "slider", "min": 0.0, "max": 50.0, "value": 20.0, "step": 1.0, "transform": lambda x: x / 100, "help": "Minimum cash (Side Fund) balance as a percentage of the total portfolio high-water mark (Rule VII).", "advanced": True},
            {"name": "crash_threshold", "label": "30-Down Rule Threshold (%)", "type": "number_input", "min": 0.0, "max": 100.0, "value": 70.0, "step": 1.0, "format": "%.1f", "transform": lambda x: x / 100, "help": "Triggers protection if the price falls below this percentage (e.g. 70%) of its 8-quarter high. Protection pauses selling (Rule IV).", "advanced": True},
            {"name": "up100_threshold", "label": "100-Up Reset Threshold (%)", "type": "number_input", "min": 100.0, "max": 300.0, "value": 200.0, "step": 1.0, "format": "%.1f", "transform": lambda x: x / 100, "help": "If the price doubles (200.0%) in a quarter, the strategy resets to the target Home Base allocation (Rule V).", "advanced": True},
            {"name": "benchmark_symbol", "label": "Benchmark Symbol", "type": "text_input", "default": "SPY", "help": "Standard index used for comparison in the report.", "advanced": True},
            {"name": "rebalance_sensitivity", "label": "Rebalance Sensitivity (%)", "type": "slider", "min": 0.0, "max": 10.0, "value": 5.0, "step": 0.1, "transform": lambda x: x / 100, "help": "Only rebalance if the required trade size exceeds this percentage of total equity. Reduces churn.", "advanced": True},
            {"name": "slippage_bps", "label": "Slippage (bps)", "type": "number_input", "min": 0.0, "value": 5.0, "step": 1.0, "help": "Trading slippage in basis points (1 bp = 0.01%).", "advanced": True},
            {"name": "commission", "label": "Commission ($)", "type": "number_input", "min": 0.0, "value": 0.0, "step": 0.01, "help": "Fixed dollar cost per trade execution.", "advanced": True},
        ]
    },
    "Volatility Strategy": {
        "function": run_volatility_strategy,
        "params": [
            {"name": "symbol_val", "label": "Stock Symbol", "type": "text_input", "default": "TQQQ", "help": "The asset to trade based on volatility signals."},
            {"name": "start_date_val", "label": "Start Date", "type": "date_input", "default": "2010-02-11", "help": "Beginning date for the backtest."},
            {"name": "end_date", "label": "End Date", "type": "date_input", "default": "today", "help": "Ending date for the backtest."},
            {"name": "initial_capital_val", "label": "Initial Capital", "type": "number_input", "default": 100000.0, "help": "Starting account balance."},
            {"name": "cash_floor_pct_val", "label": "Cash Floor Percentage (%)", "type": "slider", "min": 0.0, "max": 100.0, "value": 20.0, "step": 1.0, "transform": lambda x: x / 100, "help": "Minimum percentage of the portfolio high-water mark to keep in cash.", "advanced": True},
            {"name": "cash_yield_daily_val", "label": "Cash Yield APR (%)", "type": "number_input", "min": 0.0, "value": 0.0, "step": 0.1, "format": "%.1f", "transform": lambda x: (1 + (x/100))**(1/252), "help": "Annual interest rate earned on cash holdings (e.g., 4.0 for 4%).", "advanced": True},
            {"name": "benchmark_symbol", "label": "Benchmark Symbol", "type": "text_input", "default": "SPY", "help": "Standard index (e.g., SPY) to compare performance against.", "advanced": True},
            {"name": "slippage_bps", "label": "Slippage (bps)", "type": "number_input", "min": 0.0, "value": 5.0, "step": 1.0, "help": "Trading slippage in basis points (1 bp = 0.01%).", "advanced": True},
            {"name": "commission", "label": "Commission ($)", "type": "number_input", "min": 0.0, "value": 0.0, "step": 0.01, "help": "Fixed dollar cost per trade execution.", "advanced": True},
        ]
    },
    "Strategic Alpha Bundle": {
        "function": run_combined_strategy,
        "params": [
            {"name": "symbol_val", "label": "Stock Symbol", "type": "text_input", "default": "TQQQ", "help": "The asset to trade for all sub-strategies (TQQQ recommended)."},
            {"name": "start_date_val", "label": "Start Date", "type": "date_input", "default": "2010-02-11", "help": "Beginning date for the backtest."},
            {"name": "end_date", "label": "End Date", "type": "date_input", "default": "today", "help": "Ending date for the backtest."},
            {"name": "initial_capital_val", "label": "Initial Capital", "type": "number_input", "default": 100000.0, "help": "Starting account balance (split 1/3 each)."},
            {"name": "side_fund_val", "label": "9-Sig Side Fund", "type": "text_input", "default": "SHY", "help": "The safe asset used by the 9-Sig component."},
            {"name": "cash_floor_pct", "label": "Shared Cash Floor %", "type": "slider", "min": 0.0, "max": 50.0, "value": 20.0, "step": 1.0, "transform": lambda x: x / 100, "help": "The cash floor maintained by all sub-strategies."},
            {"name": "cash_yield_apr", "label": "Cash Yield APR (%)", "type": "number_input", "min": 0.0, "value": 0.0, "step": 0.1, "format": "%.2f", "transform": lambda x: x / 100, "help": "Interest earned on cash holdings."},
            {"name": "benchmark_symbol", "label": "Benchmark Symbol", "type": "text_input", "default": "SPY", "help": "Index used for comparison.", "advanced": True},
            {"name": "slippage_bps", "label": "Slippage (bps)", "type": "number_input", "min": 0.0, "value": 5.0, "step": 1.0, "help": "Trading slippage.", "advanced": True},
            {"name": "commission", "label": "Commission ($)", "type": "number_input", "min": 0.0, "value": 0.0, "step": 0.01, "help": "Trade commission.", "advanced": True},
        ]
    },
    "Moving Average Strategy": {
        "function": run_moving_average_strategy,
        "params": [
            {"name": "symbol_val", "label": "Stock Symbol", "type": "text_input", "default": "TQQQ", "help": "The asset to trade when above the moving average."},
            {"name": "start_date_val", "label": "Start Date", "type": "date_input", "default": "2010-02-11", "help": "Beginning date for the backtest."},
            {"name": "end_date", "label": "End Date", "type": "date_input", "default": "today", "help": "Ending date for the backtest."},
            {"name": "initial_capital_val", "label": "Initial Capital", "type": "number_input", "default": 100000.0, "help": "Starting account balance."},
            {"name": "cash_floor_pct", "label": "Cash Floor Percentage (%)", "type": "slider", "min": 0.0, "max": 50.0, "value": 20.0, "step": 1.0, "transform": lambda x: x / 100, "help": "Minimum percentage of the portfolio high-water mark to keep in cash. Refilled on rebalance days.", "advanced": True},
            {"name": "lookback_period_months", "label": "SMA Lookback (Months)", "type": "number_input", "min": 1, "value": 11, "step": 1, "help": "Number of months for the Simple Moving Average calculation."},
            {"name": "eval_frequency", "label": "Evaluation Frequency", "type": "selectbox", "options": ["Daily", "Weekly", "Monthly"], "default": "Weekly", "help": "How often the strategy checks the signal and executes trades."},
            {"name": "cash_yield_apr", "label": "Cash Yield APR (%)", "type": "number_input", "min": 0.0, "value": 0.0, "step": 0.1, "format": "%.2f", "transform": lambda x: x / 100, "help": "Annual interest rate earned on cash holdings (e.g., 4.0 for 4%).", "advanced": True},
            {"name": "benchmark_symbol", "label": "Benchmark Symbol", "type": "text_input", "default": "SPY", "help": "Standard index (e.g., SPY) to compare performance against.", "advanced": True},
            {"name": "rebalance_sensitivity", "label": "Rebalance Sensitivity (%)", "type": "slider", "min": 0.0, "max": 10.0, "value": 0.0, "step": 0.1, "transform": lambda x: x / 100, "help": "Only rebalance if the required trade size exceeds this percentage of total equity.", "advanced": True},
            {"name": "slippage_bps", "label": "Slippage (bps)", "type": "number_input", "min": 0.0, "value": 5.0, "step": 1.0, "help": "Trading slippage in basis points (1 bp = 0.01%).", "advanced": True},
            {"name": "commission", "label": "Commission ($)", "type": "number_input", "min": 0.0, "value": 0.0, "step": 0.01, "help": "Fixed dollar cost per trade execution.", "advanced": True},
        ]
    },
    "Volatility (VIX) Strategy": {
        "function": run_volatility_vix_strategy,
        "params": [
            {"name": "symbol_val", "label": "Stock Symbol", "type": "text_input", "default": "TQQQ", "help": "The asset to trade based on VIX signals."},
            {"name": "start_date_val", "label": "Start Date", "type": "date_input", "default": "2010-02-11", "help": "Beginning date for the backtest."},
            {"name": "end_date", "label": "End Date", "type": "date_input", "default": "today", "help": "Ending date for the backtest."},
            {"name": "initial_capital_val", "label": "Initial Capital", "type": "number_input", "default": 100000.0, "help": "Starting account balance."},
            {"name": "cash_floor_pct", "label": "Cash Floor Percentage (%)", "type": "slider", "min": 0.0, "max": 50.0, "value": 20.0, "step": 1.0, "transform": lambda x: x / 100, "help": "Minimum percentage of the portfolio high-water mark to keep in cash.", "advanced": True},
            {"name": "vix_ema_len", "label": "VIX EMA Length", "type": "number_input", "min": 1, "value": 3, "step": 1, "help": "Period for the first EMA of VIX."},
            {"name": "vix_ma_len", "label": "VIX MA Length", "type": "number_input", "min": 1, "value": 20, "step": 1, "help": "Period for the EMA of the VIX EMA."},
            {"name": "vix_threshold", "label": "VIX Absolute Threshold", "type": "number_input", "min": 0.0, "value": 20.0, "step": 1.0, "help": "Absolute VIX level for entry/exit logic."},
            {"name": "short_multiplier", "label": "VIX Exit Multiplier", "type": "number_input", "min": 1.0, "value": 1.15, "step": 0.05, "help": "Exit if VIX EMA > VIX MA * this multiplier."},
            {"name": "cash_yield_apr", "label": "Cash Yield APR (%)", "type": "number_input", "min": 0.0, "value": 0.0, "step": 0.1, "format": "%.2f", "transform": lambda x: x / 100, "help": "Annual interest on cash.", "advanced": True},
            {"name": "benchmark_symbol", "label": "Benchmark Symbol", "type": "text_input", "default": "SPY", "help": "Standard index to compare performance.", "advanced": True},
            {"name": "slippage_bps", "label": "Slippage (bps)", "type": "number_input", "min": 0.0, "value": 5.0, "step": 1.0, "help": "Trading slippage in basis points.", "advanced": True},
            {"name": "commission", "label": "Commission ($)", "type": "number_input", "min": 0.0, "value": 0.0, "step": 0.01, "help": "Fixed dollar cost per trade execution.", "advanced": True},
        ]
    },
    "Dual Momentum Strategy": {
        "function": run_dual_momentum_strategy,
        "params": [
            {"name": "growth_symbol", "label": "Growth Asset", "type": "text_input", "default": "TQQQ", "help": "The risky asset (e.g., TQQQ or SPY) used for relative momentum."},
            {"name": "defensive_symbol", "label": "Defensive Asset", "type": "text_input", "default": "TLT", "help": "The safe-haven asset (e.g., TLT or AGG) used if Growth is underperforming."},
            {"name": "start_date", "label": "Start Date", "type": "date_input", "default": "2010-02-11", "help": "Beginning date for the backtest."},
            {"name": "end_date", "label": "End Date", "type": "date_input", "default": "today", "help": "Ending date for the backtest."},
            {"name": "initial_capital", "label": "Initial Capital", "type": "number_input", "default": 100000.0, "help": "Starting account balance."},
            {"name": "lookback_months", "label": "Lookback Period (Months)", "type": "number_input", "min": 1, "value": 12, "step": 1, "help": "The window (typically 12 months) used to calculate momentum."},
            {"name": "eval_frequency", "label": "Evaluation Frequency", "type": "selectbox", "options": ["Daily", "Weekly", "Monthly"], "default": "Monthly", "help": "How often the strategy evaluates signals. Dual Momentum typically uses 'Monthly'."},
            {"name": "cash_yield_apr", "label": "Cash Yield APR (%)", "type": "number_input", "min": 0.0, "value": 0.0, "step": 0.1, "format": "%.2f", "transform": lambda x: x / 100, "help": "Annual interest earned on cash when both assets are negative.", "advanced": True},
            {"name": "benchmark_symbol", "label": "Benchmark Symbol", "type": "text_input", "default": "SPY", "help": "Index used for comparison.", "advanced": True},
            {"name": "slippage_bps", "label": "Slippage (bps)", "type": "number_input", "min": 0.0, "value": 5.0, "step": 1.0, "help": "Trading slippage in basis points.", "advanced": True},
            {"name": "commission", "label": "Commission ($)", "type": "number_input", "min": 0.0, "value": 0.0, "step": 0.01, "help": "Fixed dollar cost per trade execution.", "advanced": True},
        ]
    }
}


__all__ = [
    'STRATEGIES',
    'run_breadth_backtest',
    'run_volatility_strategy',
    'run_9sig_strategy',
    'run_moving_average_strategy',
    'run_dual_momentum_strategy',
    'run_volatility_vix_strategy',
    'run_combined_strategy',
]
