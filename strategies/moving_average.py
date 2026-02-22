"""
Moving Average Strategy
Uses a simple moving average crossover to generate trading signals.
Aligned with Portfolio Visualizer's tactical asset allocation model.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from common import (
    download_price_data,
    get_data_start_date,
    calculate_daily_yield,
    create_trade_log_entry,
    close_trade_log_entry,
    create_historical_record,
    build_strategy_result,
    make_empty_result,
    apply_trading_costs,
    BacktestResult
)


def run_moving_average_strategy(
    symbol_val,
    start_date_val,
    initial_capital_val,
    lookback_period_months,
    cash_yield_apr,
    eval_frequency="Weekly",
    cash_floor_pct=0.20,
    benchmark_symbol="SPY",
    rebalance_sensitivity=0.0,
    slippage_bps=0.0,
    commission=0.0,
    end_date=None
):
    """
    Moving Average Strategy: Uses SMA crossover to switch between stocks and cash.
    - Evaluation: Custom (Daily, Weekly, Monthly).
    - Signal: Price vs moving average of daily closing prices.
    - Cash Floor: Maintains a minimum percentage of the high-water mark in cash.
    """
    # 1. DATA FETCHING (Include consistent buffer)
    start_dt = pd.to_datetime(start_date_val)
    download_start = get_data_start_date(start_date_val)
    
    prices = download_price_data(symbol_val, download_start)
    if prices.empty:
        return make_empty_result(start_date_val, initial_capital_val, symbol_val, benchmark_symbol)

    benchmark_prices = download_price_data(benchmark_symbol, download_start)
    if benchmark_prices.empty:
        return make_empty_result(start_date_val, initial_capital_val, symbol_val, benchmark_symbol)

    # 2. INDICATORS
    sma_days = int(lookback_period_months * 21)
    prices['SMA'] = prices['Close'].rolling(window=sma_days).mean()
    
    # Merge Benchmark
    benchmark_col = f"{benchmark_symbol}_Close"
    benchmark_prices.rename(columns={"Close": benchmark_col}, inplace=True)
    full_df = pd.merge(prices, benchmark_prices, left_index=True, right_index=True, how="left")
    full_df[benchmark_col] = full_df[benchmark_col].ffill()
    
    # Filter backtest period
    df = full_df[full_df.index >= start_dt].copy()
    if end_date:
        df = df[df.index <= pd.Timestamp(end_date)]

    if df.empty:
        return make_empty_result(start_date_val, initial_capital_val, symbol_val, benchmark_symbol)

    # 3. INITIALIZATION
    cash = initial_capital_val
    stock_val = 0.0
    currently_long = False
    trade_log = []
    hist = []
    daily_yield = calculate_daily_yield(cash_yield_apr)
    portfolio_hwm = initial_capital_val

    first_price = df['Close'].iloc[0]
    first_bench_p = df[benchmark_col].iloc[0]
    
    # 4. BACKTEST LOOP
    for i in range(len(df)):
        date = df.index[i]
        p = float(df['Close'].iloc[i])
        sma = float(df['SMA'].iloc[i])
        bp = float(df[benchmark_col].iloc[i])

        # A. Position Updates
        if stock_val > 0:
            prev_p = float(df['Close'].iloc[i-1])
            stock_val = (stock_val / prev_p) * p
        cash *= (1 + daily_yield)

        total_equity = stock_val + cash
        # Update limit monthly (Monthly HWM Tracking)
        is_first_day = (i == 0)
        is_month_change = (i > 0 and date.month != df.index[i-1].month)
        
        if is_first_day or is_month_change:
            if total_equity > portfolio_hwm:
                portfolio_hwm = total_equity
            cash_limit = portfolio_hwm * cash_floor_pct
        
        # B. Trading Logic (Use PREVIOUS day's signal to avoid look-ahead bias)
        idx_in_full = full_df.index.get_loc(date)
        if idx_in_full > 0:
            sig_p = float(full_df['Close'].iloc[idx_in_full-1])
            sig_sma = float(full_df['SMA'].iloc[idx_in_full-1])
        else:
            sig_p = p
            sig_sma = sma

        is_rebalance_day = False
        if i == len(df) - 1:
            is_rebalance_day = True 
        elif eval_frequency == "Daily":
            is_rebalance_day = True
        elif eval_frequency == "Weekly":
            if date.isocalendar()[1] != df.index[i+1].isocalendar()[1]:
                is_rebalance_day = True
        elif eval_frequency == "Monthly":
            if date.month != df.index[i+1].month:
                is_rebalance_day = True

        if is_rebalance_day:
            # Force Cash Refill (if floor is breached)
            if cash < cash_limit:
                shortfall = cash_limit - cash
                stock_to_sell = min(stock_val, shortfall)
                if stock_to_sell > 0:
                    # Apply costs
                    net_proceeds, cost = apply_trading_costs(-stock_to_sell, p, slippage_bps, commission)
                    stock_val -= stock_to_sell
                    cash += abs(net_proceeds)
                    trade_log.append({
                        'Date': date.date() if hasattr(date, 'date') else date,
                        'Status': 'FLOOR REFILL',
                        'Amt': round(stock_to_sell, 2),
                        'Entry Price': round(p, 2),
                        'Exit Price': 'N/A',
                        'Profit %': 'N/A'
                    })

            if sig_p >= sig_sma and not currently_long:
                # ENTRY (Invest only amount above floor)
                available_to_invest = max(0, cash - cash_limit)
                if available_to_invest > (total_equity * rebalance_sensitivity):
                    # Apply costs
                    net_spend, cost = apply_trading_costs(available_to_invest, p, slippage_bps, commission)
                    stock_val += available_to_invest
                    cash -= net_spend
                    currently_long = True
                    trade_log.append(create_trade_log_entry(date, p, None, 'OPEN', total_equity, amount=available_to_invest))
            elif sig_p < sig_sma and currently_long:
                # EXIT (All stocks to cash)
                currently_long = False
                # Find the actual open trade to close
                trade = next((t for t in reversed(trade_log) if t.get('Status') == 'OPEN'), None)
                if trade:
                    close_trade_log_entry(trade, date, p, total_equity)
                
                # Apply costs
                proceeds, cost = apply_trading_costs(-stock_val, p, slippage_bps, commission)
                cash += abs(proceeds)
                stock_val = 0.0

        # C. History Tracking
        total_equity = stock_val + cash
        bh_val = initial_capital_val * (p / first_price)
        bench_val = initial_capital_val * (bp / first_bench_p)
        
        hist.append(create_historical_record(
            date, total_equity, bh_val, benchmark_symbol, bench_val, cash, p, 
            HWM=portfolio_hwm, Signal=sma
        ))

    return build_strategy_result(
        hist, trade_log, initial_capital_val, symbol_val, benchmark_symbol, 
        strategy_name="Moving Average",
        params=locals()
    )


