"""
Dual Momentum Strategy
Based on Gary Antonacci's Global Equities Momentum (GEM).
Combines Relative Momentum (which asset is better) and Absolute Momentum (is the asset itself positive).
"""

import pandas as pd
import numpy as np
from common import (
    download_price_data,
    download_multiple_tickers,
    get_data_start_date,
    calculate_daily_yield,
    create_trade_log_entry,
    close_trade_log_entry,
    create_historical_record,
    build_strategy_result,
    make_empty_result,
    apply_trading_costs
)


def run_dual_momentum_strategy(
    growth_symbol,
    defensive_symbol,
    start_date,
    initial_capital,
    lookback_months=12,
    eval_frequency="Monthly",
    cash_yield_apr=0.0,
    benchmark_symbol="SPY",
    slippage_bps=5.0,
    commission=0.0,
    end_date=None
):
    """
    Dual Momentum Backtest:
    1. Calculate N-month total return for Growth Asset and Defensive Asset.
    2. Absolute Momentum: If Growth Asset return > Cash/BIL return, stay in Growth.
    3. Relative Momentum: Not fully implemented in 2-asset version, but if Growth is negative, we switch.
    """
    # 1. DATA FETCHING (Include consistent buffer)
    start_dt = pd.to_datetime(start_date)
    download_start = get_data_start_date(start_date)
    
    tickers = [growth_symbol, defensive_symbol, benchmark_symbol]
    all_prices = download_multiple_tickers(tickers, download_start)
    
    if all_prices.empty or growth_symbol not in all_prices.columns:
        return make_empty_result(start_date, initial_capital, growth_symbol, benchmark_symbol)

    # 2. INDICATORS
    # We need to calculate lookback returns
    # We'll use a specific day count for months (approx 21 trading days)
    lookback_days = int(lookback_months * 21)
    
    prices_df = all_prices.copy()
    
    # Calculate ROC (Rate of Change)
    def calculate_roc(series, window):
        return (series / series.shift(window)) - 1

    prices_df[f'{growth_symbol}_ROC'] = calculate_roc(prices_df[growth_symbol], lookback_days)
    prices_df[f'{defensive_symbol}_ROC'] = calculate_roc(prices_df[defensive_symbol], lookback_days)
    
    # Filter backtest period
    df = prices_df[prices_df.index >= start_dt].copy()
    if end_date:
        df = df[df.index <= pd.Timestamp(end_date)]
    if df.empty:
        return make_empty_result(start_date, initial_capital, growth_symbol, benchmark_symbol)

    # 3. INITIALIZATION
    cash = initial_capital
    stock_val = 0.0
    bond_val = 0.0
    current_asset = "CASH" # CASH, GROWTH, DEFENSIVE
    trade_log = []
    hist = []
    daily_yield = calculate_daily_yield(cash_yield_apr)
    max_equity = 0.0
    
    first_prices = {t: df[t].iloc[0] for t in tickers if t in df.columns}
    
    # 4. BACKTEST LOOP
    for i in range(len(df)):
        date = df.index[i]
        p_growth = float(df[growth_symbol].iloc[i])
        p_defensive = float(df[defensive_symbol].iloc[i])
        p_bench = float(df[benchmark_symbol].iloc[i]) if benchmark_symbol in df.columns else p_growth
        
        # A. Position Updates
        if stock_val > 0:
            prev_p = float(df[growth_symbol].iloc[i-1])
            stock_val = (stock_val / prev_p) * p_growth
        if bond_val > 0:
            prev_p = float(df[defensive_symbol].iloc[i-1])
            bond_val = (bond_val / prev_p) * p_defensive
            
        cash *= (1 + daily_yield)
        total_equity = stock_val + bond_val + cash
        
        # Update monthly HWM peak
        is_first_day = (i == 0)
        is_month_change = (i > 0 and date.month != df.index[i-1].month)
        if is_first_day or is_month_change:
            if total_equity > max_equity:
                max_equity = total_equity

        # B. Trading Logic
        is_rebalance_day = False
        if i == 0 or i == len(df) - 1:
            is_rebalance_day = True
        elif eval_frequency == "Monthly" and date.month != df.index[i+1].month:
            is_rebalance_day = True
        elif eval_frequency == "Weekly" and date.weekday() == 4: # Friday
            is_rebalance_day = True
        elif eval_frequency == "Daily":
            is_rebalance_day = True

        if is_rebalance_day:
            # Signal based on PREVIOUS day to avoid bias
            idx_in_full = prices_df.index.get_loc(date)
            if idx_in_full > 0:
                roc_growth = float(prices_df[f'{growth_symbol}_ROC'].iloc[idx_in_full-1])
                roc_defensive = float(prices_df[f'{defensive_symbol}_ROC'].iloc[idx_in_full-1])
            else:
                roc_growth = 0
                roc_defensive = 0
            
            # Dual Momentum Decision
            target_asset = "CASH"
            if roc_growth > 0 and roc_growth >= roc_defensive:
                target_asset = "GROWTH"
            elif roc_defensive > 0:
                target_asset = "DEFENSIVE"
            else:
                target_asset = "CASH"
                
            if target_asset != current_asset:
                # Execution
                # 1. Exit Current
                if current_asset == "GROWTH":
                    proceeds, cost = apply_trading_costs(-stock_val, p_growth, slippage_bps, commission)
                    cash += abs(proceeds)
                    stock_val = 0.0
                    if trade_log and trade_log[-1]['Status'] == 'OPEN':
                        close_trade_log_entry(trade_log[-1], date, p_growth, total_equity)
                elif current_asset == "DEFENSIVE":
                    proceeds, cost = apply_trading_costs(-bond_val, p_defensive, slippage_bps, commission)
                    cash += abs(proceeds)
                    bond_val = 0.0
                    if trade_log and trade_log[-1]['Status'] == 'OPEN':
                        close_trade_log_entry(trade_log[-1], date, p_defensive, total_equity)
                
                # 2. Enter Target
                if target_asset == "GROWTH":
                    net_spend, cost = apply_trading_costs(cash, p_growth, slippage_bps, commission)
                    stock_val = cash - cost # Simplified
                    cash = 0.0
                    trade_log.append(create_trade_log_entry(date, p_growth, None, 'OPEN', total_equity, amount=stock_val))
                elif target_asset == "DEFENSIVE":
                    net_spend, cost = apply_trading_costs(cash, p_defensive, slippage_bps, commission)
                    bond_val = cash - cost
                    cash = 0.0
                    trade_log.append(create_trade_log_entry(date, p_defensive, None, 'OPEN', total_equity, amount=bond_val))
                
                current_asset = target_asset

        # C. History
        bh_val = initial_capital * (p_growth / first_prices[growth_symbol])
        bench_val = initial_capital * (p_bench / first_prices[benchmark_symbol]) if benchmark_symbol in first_prices else bh_val
        
        # Get ROCs for current date
        idx_in_full = prices_df.index.get_loc(date)
        roc_g = float(prices_df[f'{growth_symbol}_ROC'].iloc[idx_in_full])
        roc_d = float(prices_df[f'{defensive_symbol}_ROC'].iloc[idx_in_full])

        hist.append(create_historical_record(
            date, total_equity, bh_val, benchmark_symbol, bench_val, cash, p_growth, 
            HWM=max_equity, Growth_ROC=roc_g, Defensive_ROC=roc_d
        ))

    return build_strategy_result(
        hist, trade_log, initial_capital, growth_symbol, benchmark_symbol, 
        strategy_name="Dual Momentum",
        params=locals()
    )
