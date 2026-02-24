"""
Market Breadth Strategy
Uses breadth data (high-low quantity) to generate trading signals.
"""

import pandas as pd
import yfinance as yf
import numpy as np
import os
from core.common import (
    load_breadth_data_v2,
    download_price_data,
    get_data_start_date,
    calculate_daily_yield,
    create_trade_log_entry,
    close_trade_log_entry,
    create_historical_record,
    create_cash_pct_df,
    build_strategy_result,
    make_empty_result,
    apply_trading_costs,
    BacktestResult
)

# Default path for breadth data
BREADTH_FILE_PATH = os.path.join('data', 'market_breadth_data.csv')


def run_breadth_backtest(
    initial_capital,
    cash_floor_pct,
    rebalance_pct,
    starting_cash_pct,
    cash_yield_apr,
    ticker,
    benchmark_symbol="SPY",
    slippage_bps=0.0,
    commission=0.0,
    start_date=None,
    end_date=None,
    exchange="Nasdaq"
):
    """
    Run Market Breadth Strategy Backtest.
    
    Args:
        initial_capital: Starting capital
        cash_floor_pct: Cash floor percentage (0-1)
        rebalance_pct: Rebalance percentage (0-1)
        starting_cash_pct: Starting cash percentage (0-1)
        cash_yield_apr: Cash yield APR
        ticker: Stock ticker symbol
        benchmark_symbol: Benchmark ticker symbol
        
    Returns:
        BacktestResult object.
    """
    # Strategy Constants
    SIGNAL_SMA_LEN = 5
    SIGNAL_EMA_LEN = 3
    PRICE_EMA_LEN = 3
    PRICE_SMA_LEN = 200

    # Load breadth data (Nasdaq or NYSE)
    breadth = load_breadth_data_v2(BREADTH_FILE_PATH, exchange=exchange)
    
    if breadth.empty:
        return _create_empty_results(str(pd.Timestamp.now().date()), initial_capital, benchmark_symbol, ticker, "Breadth data file is empty or missing required columns.")
    
    # Use standardized start date if provided to maximize cache hits
    if start_date:
        download_start = get_data_start_date(start_date)
    else:
        download_start = breadth.index.min().strftime('%Y-%m-%d')
        
    start_str = str(download_start)
    
    # Download strategy ticker prices
    prices = download_price_data(ticker, download_start)
    
    if prices.empty:
        # Return empty data if no data available
        return _create_empty_results(start_str, initial_capital, benchmark_symbol, ticker, f"Price download failed for ticker: {ticker}")
    
    prices.rename(columns={"Close": "close"}, inplace=True)

    # Download benchmark prices
    benchmark_prices = download_price_data(benchmark_symbol, start_str)
    
    if benchmark_prices.empty:
        return _create_empty_results(start_str, initial_capital, benchmark_symbol, ticker, f"Price download failed for benchmark: {benchmark_symbol}")
    
    benchmark_col = f"{benchmark_symbol}_Close"
    benchmark_prices.rename(columns={"Close": benchmark_col}, inplace=True)

    # Merge data
    df = pd.merge(breadth, prices, left_index=True, right_index=True, how="inner")
    df = pd.merge(df, benchmark_prices, left_index=True, right_index=True, how="left")
    df[benchmark_col] = df[benchmark_col].ffill()
    df = df.dropna(subset=['close', 'highlowq'])

    # Apply date filtering if requested
    if start_date:
        df = df[df.index >= pd.Timestamp(start_date)]
    if end_date:
        df = df[df.index <= pd.Timestamp(end_date)]

    if len(df) == 0:
        return _create_empty_results(start_str, initial_capital, benchmark_symbol, ticker, "No overlapping data found between breadth and price data for the selected range.")

    # Calculate indicators
    sma5 = df['highlowq'].rolling(SIGNAL_SMA_LEN).mean()
    sma3 = sma5.rolling(3).mean()
    df['highlowqag'] = sma3.ewm(span=SIGNAL_EMA_LEN, adjust=False).mean()
    df['ema3'] = df['close'].ewm(span=PRICE_EMA_LEN, adjust=False).mean()
    df['sma200'] = df['close'].rolling(PRICE_SMA_LEN).mean()
    df = df.dropna()

    if len(df) == 0:
        return _create_empty_results(start_str, initial_capital, benchmark_symbol, ticker, "Not enough data after calculating indicators (SMA/EMA).")

    # Initialize states
    cash = initial_capital * starting_cash_pct
    stock_val = initial_capital * (1 - starting_cash_pct)
    bh_val = initial_capital
    benchmark_val = initial_capital
    max_equity = initial_capital
    daily_yield = calculate_daily_yield(cash_yield_apr)
    currently_long = False
    hist, trade_log, pnl_dollars = [], [], []

    # Track first prices
    first_price = None
    first_benchmark_price = None

    # Backtest loop
    for i in range(len(df)):
        row = df.iloc[i]
        date = row.name
        
        if i == 0:
            # Store first prices
            first_price = row['close']
            first_benchmark_price = row[benchmark_col]
            bh_val = initial_capital
            benchmark_val = initial_capital
        else:
            prev = df.iloc[i-1]
            stock_val *= (row['close'] / prev['close'])
            bh_val = initial_capital * (row['close'] / first_price)
            benchmark_val = initial_capital * (row[benchmark_col] / first_benchmark_price)
            cash *= (1 + daily_yield)

        equity = stock_val + cash
            
        # Update floor monthly (Monthly HWM Tracking)
        is_first_day = (i == 0)
        is_month_change = (i > 0 and date.month != df.index[i-1].month)
        
        if is_first_day or is_month_change:
            if equity > max_equity:
                max_equity = equity
            cash_floor = max_equity * cash_floor_pct
            
            # Refill cash floor if diluted by growth (Cash Drift Protection)
            if currently_long and cash < cash_floor:
                shortfall = cash_floor - cash
                sell_amt = min(stock_val, shortfall)
                if sell_amt > 0:
                    proceeds, cost = apply_trading_costs(-sell_amt, row['close'], slippage_bps, commission)
                    stock_val -= sell_amt
                    cash += abs(proceeds)
                    trade_log.append({
                        'Entry Date': date.date() if hasattr(date, 'date') else date,
                        'Status': 'FLOOR REFILL',
                        'Amt': round(sell_amt, 2),
                        'Entry Price': round(row['close'], 2),
                        'Exit Price': 'N/A',
                        'Profit %': 'N/A'
                    })

        # Trading logic (Use PREVIOUS day's signal to avoid look-ahead bias)
        idx_in_df = df.index.get_loc(date)
        if idx_in_df > 0:
            sig_row = df.iloc[idx_in_df-1]
            trigger_buy = sig_row['highlowqag'] > 0
            trigger_sell = (sig_row['ema3'] < sig_row['sma200'] and sig_row['highlowqag'] < 0) or \
                           (sig_row['ema3'] >= sig_row['sma200'] and sig_row['highlowqag'] < -60)
        else:
            trigger_buy = trigger_sell = False

        if trigger_buy and not currently_long:
            amt = max(0, (cash - cash_floor) * rebalance_pct)
            net_amt, cost = apply_trading_costs(amt, row['close'], slippage_bps, commission)
            cash -= net_amt
            stock_val += amt
            currently_long = True
            trade_log.append(create_trade_log_entry(
                date, row['close'], None, 'OPEN', equity, amount=amt
            ))
        elif trigger_sell and currently_long:
            currently_long = False
            # Find the actual open trade to close (don't just take the last one, which might be a refill)
            trade = next((t for t in reversed(trade_log) if t.get('Status') == 'OPEN'), None)
            if trade:
                close_trade_log_entry(trade, date, row['close'], equity)
            
            # Apply cost on sell
            proceeds, cost = apply_trading_costs(-stock_val, row['close'], slippage_bps, commission)
            cash += abs(proceeds)
            stock_val = 0

        # Record history with full diagnostic data for the interactive chart
        equity = stock_val + cash
        hist.append(create_historical_record(
            date, equity, bh_val, benchmark_symbol, benchmark_val, cash, row['close'], 
            HWM=max_equity,
            highlowq=row['highlowq'],
            highlowqag=row['highlowqag'], 
            EMA3=row['ema3'], 
            SMA200=row['sma200'],
            Position=1 if currently_long else 0,
            Buy_Trigger=1 if trigger_buy else 0,
            Sell_Trigger=1 if trigger_sell else 0
        ))

    return build_strategy_result(
        hist, trade_log, initial_capital, ticker, benchmark_symbol, 
        strategy_name="Market Breadth",
        params=locals()
    )


def _create_empty_results(start_date, initial_capital, benchmark_symbol, ticker, error_msg=None):
    """Create empty results when no data is available."""
    return make_empty_result(start_date, initial_capital, ticker, benchmark_symbol, error=error_msg)
