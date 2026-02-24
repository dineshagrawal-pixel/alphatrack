"""
Volatility Strategy
Uses volatility signals and trend following to switch between stocks and cash.
Matches the user's provided V5 script logic.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from core.common import (
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


def run_volatility_strategy(
    symbol_val,
    start_date_val,
    initial_capital_val,
    cash_floor_pct_val,
    cash_yield_daily_val, # Note: Parameter is actually and APR now due to __init__.py change
    benchmark_symbol="SPY",
    slippage_bps=0.0,
    commission=0.0,
    end_date=None
):
    """
    Volatility Strategy: Uses vol14 < vol200 or price > sma200 to go long.
    Matches user's reference script V5.
    """
    # 1. FETCH DATA (Include consistent buffer)
    start_dt = pd.to_datetime(start_date_val)
    download_start = get_data_start_date(start_date_val)
    
    prices = download_price_data(symbol_val, download_start)
    if prices.empty:
        return make_empty_result(start_date_val, initial_capital_val, symbol_val, benchmark_symbol)

    benchmark_prices = download_price_data(benchmark_symbol, download_start)
    if benchmark_prices.empty:
        return make_empty_result(start_date_val, initial_capital_val, symbol_val, benchmark_symbol)

    # 2. INDICATORS (Matching V5 script)
    full_df = prices.copy()
    full_df['ret'] = full_df['Close'].pct_change()
    full_df['vol200'] = full_df['ret'].rolling(200).std() * np.sqrt(252)
    full_df['vol14'] = full_df['ret'].rolling(14).std() * np.sqrt(252)
    full_df['sma200'] = full_df['Close'].rolling(200).mean()
    
    # Merge Benchmark (Left join to prevent data loss)
    benchmark_col = f"{benchmark_symbol}_Close"
    benchmark_prices.rename(columns={"Close": benchmark_col}, inplace=True)
    full_df = pd.merge(full_df, benchmark_prices, left_index=True, right_index=True, how="left")
    full_df[benchmark_col] = full_df[benchmark_col].ffill()
    
    # Filter backtest period
    df = full_df[full_df.index >= start_dt].copy()
    if end_date:
        df = df[df.index <= pd.Timestamp(end_date)]
    
    df = df.dropna(subset=['vol200', 'vol14', 'sma200']).copy()

    if df.empty:
        return make_empty_result(start_date_val, initial_capital_val, symbol_val, benchmark_symbol)

    # 3. INITIALIZATION
    cash = initial_capital_val
    stock_val = 0.0
    bh_val = initial_capital_val
    max_equity = initial_capital_val
    pos = 0 
    trade_log = []  
    hist = []
    
    # Check if cash_yield_daily_val is the daily multiplier (legacy) or APR (new)
    # The __init__.py transform now passes (1+(x/100))**(1/252)
    daily_yield_multiplier = cash_yield_daily_val 

    first_price = df['Close'].iloc[0]
    first_bench_p = df[benchmark_col].iloc[0]

    # 4. BACKTEST LOOP
    for i in range(len(df)):
        date = df.index[i]
        p = float(df['Close'].iloc[i])
        vol14 = float(df['vol14'].iloc[i])
        vol200 = float(df['vol200'].iloc[i])
        sma200 = float(df['sma200'].iloc[i])
        bp = float(df[benchmark_col].iloc[i])

        if i > 0:
            prev_p = float(df['Close'].iloc[i-1])
            stock_val *= (p / prev_p)
            cash *= daily_yield_multiplier
            
        total_equity = stock_val + cash
            
        # Update floor monthly (Monthly HWM Tracking)
        is_first_day = (i == 0)
        is_month_change = (i > 0 and date.month != df.index[i-1].month)
        
        if is_first_day or is_month_change:
            if total_equity > max_equity:
                max_equity = total_equity
            cash_floor = max_equity * cash_floor_pct_val
            
            # Refill cash floor if diluted by growth (Cash Drift Protection)
            if pos == 1 and cash < cash_floor:
                shortfall = cash_floor - cash
                sell_amt = min(stock_val, shortfall)
                if sell_amt > 0:
                    proceeds, cost = apply_trading_costs(-sell_amt, p, slippage_bps, commission)
                    stock_val -= sell_amt
                    cash += abs(proceeds)
                    trade_log.append({
                        'Entry Date': date.date() if hasattr(date, 'date') else date,
                        'Status': 'FLOOR REFILL',
                        'Amt': round(sell_amt, 2),
                        'Entry Price': round(p, 2),
                        'Exit Price': 'N/A',
                        'Profit %': 'N/A'
                    })

        # V5 Logic for Entry/Exit
        is_long = (vol14 < vol200) or (p > sma200)
        is_short = (vol14 > vol200) and (p < sma200)

        # Logic: Entry
        if pos == 0 and is_long:
            # Use PREVIOUS day's signal to avoid look-ahead bias
            idx_in_full = full_df.index.get_loc(date)
            if idx_in_full > 0:
                sig_vol14 = float(full_df['vol14'].iloc[idx_in_full-1])
                sig_vol200 = float(full_df['vol200'].iloc[idx_in_full-1])
                sig_p = float(full_df['Close'].iloc[idx_in_full-1])
                sig_sma200 = float(full_df['sma200'].iloc[idx_in_full-1])
                
                trigger_buy = (sig_vol14 < sig_vol200) or (sig_p > sig_sma200)
            else:
                trigger_buy = False

            if trigger_buy:
                pos = 1
                investment_amt = max(0, cash - cash_floor)
                net_invested, cost = apply_trading_costs(investment_amt, p, slippage_bps, commission)
                cash -= net_invested
                stock_val += investment_amt # We bought 'investment_amt' worth of stock
                trade_log.append(create_trade_log_entry(date, p, None, 'OPEN', total_equity, amount=investment_amt))
        
        # Logic: Exit
        elif pos == 1:
            idx_in_full = full_df.index.get_loc(date)
            if idx_in_full > 0:
                sig_vol14 = float(full_df['vol14'].iloc[idx_in_full-1])
                sig_vol200 = float(full_df['vol200'].iloc[idx_in_full-1])
                sig_p = float(full_df['Close'].iloc[idx_in_full-1])
                sig_sma200 = float(full_df['sma200'].iloc[idx_in_full-1])
                
                trigger_sell = (sig_vol14 > sig_vol200) and (sig_p < sig_sma200)
            else:
                trigger_sell = False

            if trigger_sell:
                pos = 0
                # Find the actual open trade to close
                trade = next((t for t in reversed(trade_log) if t.get('Status') == 'OPEN'), None)
                if trade:
                    close_trade_log_entry(trade, date, p, total_equity)
                
                # Apply trading costs on sell
                proceeds, cost = apply_trading_costs(-stock_val, p, slippage_bps, commission)
                cash += abs(proceeds) # Proceeds is negative
                stock_val = 0
            
        # History tracking
        total_equity = stock_val + cash
        bh_val = initial_capital_val * (p / first_price)
        bench_val = initial_capital_val * (bp / first_bench_p)
        hist.append(create_historical_record(
            date, total_equity, bh_val, benchmark_symbol, bench_val, cash, p, 
            HWM=max_equity, Vol14=vol14, Vol200=vol200, Signal=sma200
        ))

    return build_strategy_result(
        hist, trade_log, initial_capital_val, symbol_val, benchmark_symbol, 
        strategy_name="Volatility Strategy",
        params=locals()
    )


