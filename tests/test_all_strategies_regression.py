import os
import sys
import pandas as pd
# Add project root to path for imports
sys.path.append(os.getcwd())
import reporting
from strategies.breadth import run_breadth_backtest
from strategies.volatility import run_volatility_strategy
from strategies.ninesig import run_9sig_strategy
from strategies.moving_average import run_moving_average_strategy
from strategies.dual_momentum import run_dual_momentum_strategy

# ==============================================================================
# ASSISTANT IMMUNITY PROTOCOL - PLEASE READ BEFORE EDITING
# ==============================================================================
# ATTENTION DEVELOPER: This file contains the "Golden Baselines" for ALL 
# strategies. These characterize historical performance as of Feb 21, 2026.
# LAST UPDATED: 2026-02-22 (Strategic Trade Count Refinement)
# Note: Trades count now reflects ONLY strategic trades, excluding FLOOR REFILL.
#
# REGRESSION RULE: Do NOT update these constants to fix a failing test 
# without explicit, separate user permission. Any logic change that alters 
# these results must be flagged for manual review.
# ==============================================================================

REFERENCE_END_DATE = "2026-02-21"

BASELINES = {
    "Market Breadth": {
        "expected_cagr": "33.66%",
        "expected_mdd": "-37.49%",
        "expected_trades": 29
    },
    "Volatility": {
        "expected_cagr": "33.49%",
        "expected_mdd": "-55.67%",
        "expected_trades": 31
    },
    "Moving Average": {
        "expected_cagr": "23.65%",
        "expected_mdd": "-51.43%",
        "expected_trades": 16
    },
    "9 Sig": {
        "expected_cagr": "33.57%",
        "expected_mdd": "-61.50%",
        "expected_trades": 34
    },
    "Dual Momentum": {
        "expected_cagr": "14.55%",
        "expected_mdd": "-73.82%",
        "expected_trades": 22
    }
}

def verify_performance(result, baseline, strategy_name):
    df = result.df_results
    trade_log = result.trade_log
    
    strat_rets = df['ret'].dropna()
    bench_rets = df['BH'].pct_change().dropna()
    
    # NEW SCHEMA CHECK: Ensure HWM is present for the new heatmap
    assert 'HWM' in df.columns, f"{strategy_name} missing 'HWM' column in results!"
    
    metrics = reporting.calculate_pv_metrics(strat_rets, bench_rets, df['Strategy'])
    
    actual_cagr = f"{metrics.get('GM_A', 0)*100:.2f}%"
    actual_mdd = f"{metrics.get('MDD', 0)*100:.2f}%"
    
    # COUNTING LOGIC (Strategic Only) - Matches reporting logic
    strategic_trades_list = [t for t in trade_log.to_dict('records') if t.get('Status') != 'FLOOR REFILL']
    actual_trades = len(strategic_trades_list)
    
    assert actual_trades == baseline["expected_trades"], \
        f"{strategy_name} Trades error! Expected {baseline['expected_trades']}, got {actual_trades}"
    assert actual_cagr == baseline["expected_cagr"], \
        f"{strategy_name} CAGR error! Expected {baseline['expected_cagr']}, got {actual_cagr}"
    assert actual_mdd == baseline["expected_mdd"], \
        f"{strategy_name} MDD error! Expected {baseline['expected_mdd']}, got {actual_mdd}"

def test_market_breadth_regression():
    params = {
        "ticker": "TQQQ", "initial_capital": 100000.0, "cash_floor_pct": 0.2,
        "rebalance_pct": 1.0, "starting_cash_pct": 1.0, "cash_yield_apr": 0.0,
        "benchmark_symbol": "SPY", "slippage_bps": 5.0, "commission": 0.0,
        "end_date": REFERENCE_END_DATE
    }
    result = run_breadth_backtest(**params)
    verify_performance(result, BASELINES["Market Breadth"], "Market Breadth")

def test_volatility_regression():
    params = {
        "symbol_val": "TQQQ", "start_date_val": "2010-02-11", "initial_capital_val": 100000.0,
        "cash_floor_pct_val": 0.2, "cash_yield_daily_val": 1.0, # 0% APR Multiplier
        "benchmark_symbol": "SPY", "slippage_bps": 5.0, "commission": 0.0,
        "end_date": REFERENCE_END_DATE
    }
    result = run_volatility_strategy(**params)
    verify_performance(result, BASELINES["Volatility"], "Volatility")

def test_moving_average_regression():
    params = {
        "symbol_val": "TQQQ", "start_date_val": "2010-02-11", "initial_capital_val": 10000.0,
        "lookback_period_months": 11, "cash_yield_apr": 0.0, "eval_frequency": "Weekly",
        "cash_floor_pct": 0.20, "benchmark_symbol": "SPY", "rebalance_sensitivity": 0.0,
        "slippage_bps": 5.0, "commission": 0.0, "end_date": REFERENCE_END_DATE
    }
    result = run_moving_average_strategy(**params)
    verify_performance(result, BASELINES["Moving Average"], "Moving Average")

def test_9sig_regression():
    params = {
        "symbol_val": "TQQQ", "side_fund_val": "SHY", "start_date_val": "2010-02-11",
        "initial_capital_val": 10000.0, "target_tqqq_pct": 0.6, "new_money_to_signal": 0.6,
        "gap_rebalance_pct": 0.5, "q_growth_target": 1.09, "q_growth_cap": 1.30,
        "signal_floor_pct": 0.88, "cash_floor_hwm_pct": 0.2, "crash_threshold": 0.7,
        "up100_threshold": 2.0, "monthly_dep_val": 0, "benchmark_symbol": "SPY",
        "rebalance_sensitivity": 0.05, "slippage_bps": 5.0, "commission": 0.0,
        "end_date": REFERENCE_END_DATE
    }
    result = run_9sig_strategy(**params)
    verify_performance(result, BASELINES["9 Sig"], "9 Sig")

def test_dual_momentum_regression():
    params = {
        "growth_symbol": "TQQQ", "defensive_symbol": "TLT", "start_date": "2010-02-11",
        "initial_capital": 100000.0, "lookback_months": 12, "eval_frequency": "Monthly",
        "cash_yield_apr": 0.0, "benchmark_symbol": "SPY", "slippage_bps": 5.0,
        "commission": 0.0, "end_date": REFERENCE_END_DATE
    }
    result = run_dual_momentum_strategy(**params)
    verify_performance(result, BASELINES["Dual Momentum"], "Dual Momentum")

if __name__ == "__main__":
    import sys
    # Manual run if pytest not available
    print("Running Regression Suite...")
    success = True
    for test in [test_market_breadth_regression, test_volatility_regression, 
                 test_moving_average_regression, test_9sig_regression, 
                 test_dual_momentum_regression]:
        try:
            print(f"Executing {test.__name__}...", end=" ", flush=True)
            test()
            print("PASSED")
        except Exception as e:
            print(f"\nFAILED: {e}")
            success = False
    sys.exit(0 if success else 1)
