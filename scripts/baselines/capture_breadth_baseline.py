import pandas as pd
import numpy as np
from strategies.breadth import run_breadth_backtest
from core.common import load_breadth_data_v2, calculate_pnl_from_trades, create_results_dataframe, create_rolling_returns_df, create_cash_pct_df
from core import reporting

def capture_baseline():
    # Standard parameters (matching defaults in strategies/__init__.py)
    params = {
        "ticker": "TQQQ",
        "initial_capital": 100000.0,
        "cash_floor_pct": 0.2,
        "rebalance_pct": 1.0,
        "starting_cash_pct": 1.0,
        "cash_yield_apr": 0.0,
        "benchmark_symbol": "SPY",
        "slippage_bps": 5.0,
        "commission": 0.0
    }
    
    print("Running Market Breadth Backtest for baseline...")
    result = run_breadth_backtest(**params)
    
    df = result.df_results
    trade_log = result.trade_log
    
    # Calculate Metrics (using the same logic as the UI)
    strat_rets = df['ret'].dropna()
    bench_rets = df['BH'].pct_change().dropna()
    if params["benchmark_symbol"] in df.columns:
        bench_rets = df[params["benchmark_symbol"]].pct_change().dropna()
        
    metrics = reporting.calculate_pv_metrics(strat_rets, bench_rets, df['Strategy'])
    
    # Internal keys: GM_A (CAGR), MDD (Max Drawdown)
    cagr_raw = metrics.get('GM_A', 0)
    mdd_raw = metrics.get('MDD', 0)
    
    cagr = f"{cagr_raw*100:.2f}%"
    mdd = f"{mdd_raw*100:.2f}%"
    trade_count = len(trade_log)
    
    print("\n--- Market Breadth Baseline Results ---")
    print(f"CAGR: {cagr}")
    print(f"Max Drawdown: {mdd}")
    print(f"Total Trades: {trade_count}")
    print("---------------------------------------")
    
    # Output in a format easy to copy into test
    print("\nTest Baseline Snippet:")
    print(f"expected_cagr = '{cagr}'")
    print(f"expected_mdd = '{mdd}'")
    print(f"expected_trades = {trade_count}")

if __name__ == "__main__":
    capture_baseline()
