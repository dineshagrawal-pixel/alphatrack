import pytest
import pandas as pd
import reporting
from strategies.breadth import run_breadth_backtest

# ==============================================================================
# ASSISTANT IMMUNITY PROTOCOL - PLEASE READ BEFORE EDITING
# ==============================================================================
# ATTENTION DEVELOPER: This file contains the "Golden Baseline" for the 
# Market Breadth strategy. These constants characterize the strategy's 
# historical performance as of February 21, 2026.
#
# REGRESSION RULE: Do NOT update these constants to fix a failing test 
# without explicit, separate user permission. If a code change alters 
# these results, it must be flagged as a regression for human review.
# ==============================================================================

EXPECTED_CAGR = "34.67%"
EXPECTED_MDD = "-37.37%"
EXPECTED_TRADES = 29
REFERENCE_END_DATE = "2026-02-21"

def test_market_breadth_performance_regression():
    """
    Verifies that the Market Breadth strategy preserves its core performance
    profile on a fixed historical dataset.
    """
    # Use standard default parameters
    params = {
        "ticker": "TQQQ",
        "initial_capital": 100000.0,
        "cash_floor_pct": 0.2,
        "rebalance_pct": 1.0,
        "starting_cash_pct": 1.0,
        "cash_yield_apr": 0.0,
        "benchmark_symbol": "SPY",
        "slippage_bps": 5.0,
        "commission": 0.0,
        "end_date": REFERENCE_END_DATE  # Pin the end date to prevent drift
    }
    
    result = run_breadth_backtest(**params)
    
    df = result.df_results
    trade_log = result.trade_log
    
    # Calculate metrics exactly as the UI does
    strat_rets = df['ret'].dropna()
    bench_rets = df['BH'].pct_change().dropna()
    if params["benchmark_symbol"] in df.columns:
        bench_rets = df[params["benchmark_symbol"]].pct_change().dropna()
        
    metrics = reporting.calculate_pv_metrics(strat_rets, bench_rets, df['Strategy'])
    
    # Internal keys: GM_A (CAGR), MDD (Max Drawdown)
    actual_cagr = f"{metrics.get('GM_A', 0)*100:.2f}%"
    actual_mdd = f"{metrics.get('MDD', 0)*100:.2f}%"
    actual_trades = len(trade_log)
    
    # Regression Assertions
    assert actual_trades == EXPECTED_TRADES, \
        f"Trade count regression! Expected {EXPECTED_TRADES}, got {actual_trades}"
    
    assert actual_cagr == EXPECTED_CAGR, \
        f"CAGR regression! Expected {EXPECTED_CAGR}, got {actual_cagr}"
        
    assert actual_mdd == EXPECTED_MDD, \
        f"Max Drawdown regression! Expected {EXPECTED_MDD}, got {actual_mdd}"

if __name__ == "__main__":
    # Allow running this file directly for quick check
    pytest.main([__file__])
