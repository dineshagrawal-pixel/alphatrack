import pandas as pd
import numpy as np
from core import reporting
from strategies.breadth import run_breadth_backtest
from strategies.volatility import run_volatility_strategy
from strategies.ninesig import run_9sig_strategy
from strategies.moving_average import run_moving_average_strategy
from strategies.dual_momentum import run_dual_momentum_strategy

REFERENCE_END_DATE = "2026-02-21"

def capture_all_baselines():
    strategies = {
        "Market Breadth": {
            "func": run_breadth_backtest,
            "params": {
                "ticker": "TQQQ",
                "initial_capital": 100000.0,
                "cash_floor_pct": 0.2,
                "rebalance_pct": 1.0,
                "starting_cash_pct": 1.0,
                "cash_yield_apr": 0.0,
                "benchmark_symbol": "SPY",
                "slippage_bps": 5.0,
                "commission": 0.0,
                "end_date": REFERENCE_END_DATE
            }
        },
        "Volatility": {
            "func": run_volatility_strategy,
            "params": {
                "symbol_val": "TQQQ",
                "start_date_val": "2010-02-11",
                "initial_capital_val": 100000.0,
                "cash_floor_pct_val": 0.2,
                "cash_floor_pct_val": 0.2,
                "cash_yield_daily_val": 1.0, # Multiplier for 0% APR: (1+0)**(1/252)
                "benchmark_symbol": "SPY",
                "slippage_bps": 5.0,
                "commission": 0.0,
                "end_date": REFERENCE_END_DATE
            }
        },
        "Moving Average": {
            "func": run_moving_average_strategy,
            "params": {
                "symbol_val": "TQQQ",
                "start_date_val": "2010-02-11",
                "initial_capital_val": 10000.0,
                "lookback_period_months": 11,
                "cash_yield_apr": 0.0,
                "eval_frequency": "Weekly",
                "cash_floor_pct": 0.20,
                "benchmark_symbol": "SPY",
                "rebalance_sensitivity": 0.0,
                "slippage_bps": 5.0,
                "commission": 0.0,
                "end_date": REFERENCE_END_DATE
            }
        },
        "9 Sig": {
            "func": run_9sig_strategy,
            "params": {
                "symbol_val": "TQQQ",
                "side_fund_val": "SHY",
                "start_date_val": "2010-02-11",
                "initial_capital_val": 10000.0,
                "target_tqqq_pct": 0.6,
                "new_money_to_signal": 0.6,
                "gap_rebalance_pct": 0.5,
                "q_growth_target": 1.09,
                "q_growth_cap": 1.30,
                "signal_floor_pct": 0.88,
                "cash_floor_hwm_pct": 0.2,
                "crash_threshold": 0.7,
                "up100_threshold": 2.0,
                "monthly_dep_val": 0,
                "benchmark_symbol": "SPY",
                "rebalance_sensitivity": 0.05,
                "slippage_bps": 5.0,
                "commission": 0.0,
                "end_date": REFERENCE_END_DATE
            }
        },
        "Dual Momentum": {
            "func": run_dual_momentum_strategy,
            "params": {
                "growth_symbol": "TQQQ",
                "defensive_symbol": "TLT",
                "start_date": "2010-02-11",
                "initial_capital": 100000.0,
                "lookback_months": 12,
                "eval_frequency": "Monthly",
                "cash_yield_apr": 0.0,
                "benchmark_symbol": "SPY",
                "slippage_bps": 5.0,
                "commission": 0.0,
                "end_date": REFERENCE_END_DATE
            }
        }
    }
    
    print(f"--- Capturing All strategy Baselines (End Date: {REFERENCE_END_DATE}) ---\n")
    
    for name, config in strategies.items():
        print(f"Processing {name}...")
        try:
            result = config["func"](**config["params"])
            df = result.df_results
            trade_log = result.trade_log
            
            if df.empty:
                print(f"  > WARNING: Result DF is empty for {name}")
                continue
                
            print(f"  > Data Rows: {len(df)}")
            
            strat_rets = df['ret'].dropna()
            bench_rets = df['BH'].pct_change().dropna()
            if config["params"].get("benchmark_symbol") in df.columns:
                bench_rets = df[config["params"]["benchmark_symbol"]].pct_change().dropna()
                
            metrics = reporting.calculate_pv_metrics(strat_rets, bench_rets, df['Strategy'])
            
            cagr = f"{metrics.get('GM_A', 0)*100:.2f}%"
            mdd = f"{metrics.get('MDD', 0)*100:.2f}%"
            trades = len(trade_log)
            
            print(f"  > CAGR: {cagr} | MDD: {mdd} | Trades: {trades}")
        except Exception as e:
            print(f"  > ERROR: {str(e)}")
    
    print("\n---------------------------------------------------------")

if __name__ == "__main__":
    capture_all_baselines()
