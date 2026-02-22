"""
Combined Strategy (Alpha Bundle)
Allocates 1/3 capital to each: Market Breadth, 9-Sig, and Volatility.
"""

import pandas as pd
import numpy as np
from common import build_strategy_result, make_empty_result
from strategies.breadth import run_breadth_backtest
from strategies.ninesig import run_9sig_strategy
from strategies.volatility import run_volatility_strategy

def run_combined_strategy(
    symbol_val,
    start_date_val,
    initial_capital_val,
    side_fund_val="SHY",
    benchmark_symbol="SPY",
    cash_floor_pct=0.20,
    cash_yield_apr=0.0,
    slippage_bps=0.0,
    commission=0.0,
    end_date=None
):
    """
    Combined Strategy: Divides initial capital into 3 parts and runs three sub-strategies.
    Merges the results into a single portfolio view.
    """
    part_cap = initial_capital_val / 3.0
    
    # 1. Run Sub-Strategies with 1/3 capital each
    # Use standard defaults for internal parameters
    
    res_breadth = run_breadth_backtest(
        initial_capital=part_cap,
        cash_floor_pct=cash_floor_pct,
        rebalance_pct=1.0,
        starting_cash_pct=1.0,
        cash_yield_apr=cash_yield_apr,
        ticker=symbol_val,
        benchmark_symbol=benchmark_symbol,
        slippage_bps=slippage_bps,
        commission=commission,
        start_date=start_date_val,
        end_date=end_date
    )
    
    res_ninesig = run_9sig_strategy(
        symbol_val=symbol_val,
        side_fund_val=side_fund_val,
        start_date_val=start_date_val,
        initial_capital_val=part_cap,
        target_tqqq_pct=0.6,
        new_money_to_signal=0.6,
        gap_rebalance_pct=0.5,
        q_growth_target=9.0,
        q_growth_cap=30.0,
        signal_floor_pct=0.88,
        cash_floor_hwm_pct=cash_floor_pct,
        crash_threshold=0.7,
        up100_threshold=2.0,
        monthly_dep_val=0,
        benchmark_symbol=benchmark_symbol,
        rebalance_sensitivity=0.05,
        slippage_bps=slippage_bps,
        commission=commission,
        end_date=end_date
    )
    
    res_vol = run_volatility_strategy(
        symbol_val=symbol_val,
        start_date_val=start_date_val,
        initial_capital_val=part_cap,
        cash_floor_pct_val=cash_floor_pct,
        cash_yield_daily_val=cash_yield_apr, # standard volatility uses apr-to-daily transform in __init__
        benchmark_symbol=benchmark_symbol,
        slippage_bps=slippage_bps,
        commission=commission,
        end_date=end_date
    )
    
    # Safety Check: If any failed completely, return empty
    if res_breadth.df_results.empty or res_ninesig.df_results.empty or res_vol.df_results.empty:
        return make_empty_result(start_date_val, initial_capital_val, symbol_val, benchmark_symbol)

    # 2. Merge Equity Curves
    h1 = res_breadth.df_results
    h2 = res_ninesig.df_results
    h3 = res_vol.df_results
    
    # Ensure they are aligned on the same dates
    # We take the intersection of indices or outer join and ffill
    combined_df = pd.DataFrame(index=h1.index.union(h2.index).union(h3.index))
    
    # Merge Strategy Value
    combined_df['Strategy'] = (
        h1['Strategy'].reindex(combined_df.index).ffill().fillna(part_cap) +
        h2['Strategy'].reindex(combined_df.index).ffill().fillna(part_cap) +
        h3['Strategy'].reindex(combined_df.index).ffill().fillna(part_cap)
    )
    
    # Merge Buy & Hold (for the group)
    combined_df['BH'] = (
        h1['BH'].reindex(combined_df.index).ffill().fillna(part_cap) +
        h2['BH'].reindex(combined_df.index).ffill().fillna(part_cap) +
        h3['BH'].reindex(combined_df.index).ffill().fillna(part_cap)
    )
    
    # Benchmark stays as is (it's the same symbol)
    combined_df[benchmark_symbol] = h1[benchmark_symbol].reindex(combined_df.index).ffill()
    
    # Cash Balance
    combined_df['Cash'] = (
        h1['Cash'].reindex(combined_df.index).ffill().fillna(part_cap) +
        h2['Cash'].reindex(combined_df.index).ffill().fillna(part_cap) +
        h3['Cash'].reindex(combined_df.index).ffill().fillna(part_cap)
    )
    
    # Close price (for reference)
    combined_df['close'] = h1['close'].reindex(combined_df.index).ffill()
    
    # 3. Merge Trade Logs
    # Add tags to identify which sub-strategy made the trade
    # trade_log is a DataFrame in BacktestResult, so convert to records list first
    t1_list = res_breadth.trade_log.to_dict('records') if not res_breadth.trade_log.empty else []
    t2_list = res_ninesig.trade_log.to_dict('records') if not res_ninesig.trade_log.empty else []
    t3_list = res_vol.trade_log.to_dict('records') if not res_vol.trade_log.empty else []
    
    t1 = [{**t, 'Strategy': 'Breadth'} for t in t1_list]
    t2 = [{**t, 'Strategy': '9-Sig'} for t in t2_list]
    t3 = [{**t, 'Strategy': 'Volatility'} for t in t3_list]
    
    combined_trades = sorted(t1 + t2 + t3, key=lambda x: x.get('Date', x.get('Entry Date')))
    
    # 4. Prepare History for build_strategy_result
    # build_strategy_result expects a list of dicts or a dataframe
    # We'll convert back to hist format
    combined_hist = combined_df.reset_index().rename(columns={'index': 'Date'}).to_dict('records')
    
    return build_strategy_result(
        combined_hist, 
        combined_trades, 
        initial_capital_val, 
        symbol_val, 
        benchmark_symbol,
        strategy_name="Combined Alpha (1/3 Base)",
        params=locals()
    )
