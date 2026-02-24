import sys
import os
import pandas as pd
import numpy as np

# Add parent directory to path to import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from strategies.moving_average import run_moving_average_strategy
from reporting import get_all_metrics

def test_moving_average_performance():
    """
    Regression test for the Moving Average Strategy.
    Ensures that the core performance metrics remain consistent.
    """
    print("Running Moving Average Regression Test...")
    
    # Run with standard default parameters
    # Symbol: TQQQ, Start: 2010-02-11, Cap: 10000, SMA: 11 mo, Monthly
    result = run_moving_average_strategy(
        symbol_val="TQQQ",
        start_date_val=pd.to_datetime("2010-02-11").date(),
        initial_capital_val=10000.0,
        lookback_period_months=11,
        cash_yield_apr=0.0,
        eval_frequency="Monthly"
    )
    
    df = result.df_results
    rets = df['ret']
    
    metrics = get_all_metrics(
        df['Strategy'], 
        result.pnl_list, 
        result.initial_capital, 
        rets,
        df['BH'].pct_change().fillna(0)
    )
    
    # Check key metrics against known historical baselines for these parameters
    # (Values are approximate based on TQQQ history)
    
    ending_val = float(metrics['Ending Value'].replace('$', '').replace(',', ''))
    cagr = float(metrics['CAGR'].replace('%', ''))
    mdd = float(metrics['Max Drawdown'].replace('%', ''))
    
    print(f"Ending Value: {ending_val:,.2f}")
    print(f"CAGR: {cagr:.2f}%")
    print(f"Max Drawdown: {mdd:.2f}%")
    
    # Regression assertions
    # If these fail, it means the backtest logic or metric calculation has changed significantly
    assert ending_val > 500000, f"Ending value too low: {ending_val}"
    assert cagr > 30, f"CAGR too low: {cagr}"
    assert mdd < -70, f"Max drawdown too high: {mdd}"
    
    print("✅ Moving Average Regression Test Passed!")

if __name__ == "__main__":
    try:
        test_moving_average_performance()
    except Exception as e:
        print(f"❌ Test Failed: {e}")
        sys.exit(1)
