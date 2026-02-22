"""
Volatility (VIX) Strategy
Based on: vix_ema(3) vs vix_ma(20) and absolute VIX levels.
"""

import pandas as pd
import numpy as np
from common import (
    download_multiple_tickers,
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

def run_volatility_vix_strategy(
    symbol_val,
    start_date_val,
    initial_capital_val,
    cash_floor_pct=0.20,
    cash_yield_apr=0.0,
    vix_ema_len=3,
    vix_ma_len=20,
    vix_threshold=20.0,
    short_multiplier=1.15,
    benchmark_symbol="SPY",
    slippage_bps=0.0,
    commission=0.0,
    end_date=None
):
    """
    Volatility (VIX) Strategy:
    Long: VIX EMA(3) < VIX MA(20) OR VIX EMA(3) <= 20
    Close: VIX EMA(3) > VIX MA(20) * 1.15 AND VIX EMA(3) > 20
    """
    # 1. DATA FETCHING
    start_dt = pd.to_datetime(start_date_val)
    # Standardized buffer (730 days) to ensure indicator stability and cache hits
    download_start = get_data_start_date(start_date_val)
    
    # We need the asset, VIX, and Benchmark
    tickers = [symbol_val, "^VIX", benchmark_symbol]
    all_prices = download_multiple_tickers(tickers, download_start)
    
    if all_prices.empty or symbol_val not in all_prices.columns or "^VIX" not in all_prices.columns:
        return make_empty_result(start_date_val, initial_capital_val, symbol_val, benchmark_symbol)

    df_full = all_prices.copy()
    
    # 2. INDICATORS
    # VIX EMA(3)
    df_full['vix_ema'] = df_full['^VIX'].ewm(span=vix_ema_len, adjust=False).mean()
    # VIX MA (EMA of VIX EMA per script logic)
    df_full['vix_ma'] = df_full['vix_ema'].ewm(span=vix_ma_len, adjust=False).mean()
    
    # Filter backtest period
    df = df_full[df_full.index >= start_dt].copy()
    if end_date:
        df = df[df.index <= pd.Timestamp(end_date)]
    
    if df.empty:
        return make_empty_result(start_date_val, initial_capital_val, symbol_val, benchmark_symbol)

    # 3. INITIALIZATION
    cash = initial_capital_val
    stock_val = 0.0
    pos = 0 
    max_equity = initial_capital_val
    trade_log = []  
    hist = []
    daily_yield = calculate_daily_yield(cash_yield_apr)
    
    first_p = df[symbol_val].iloc[0]
    first_bench = df[benchmark_symbol].iloc[0] if benchmark_symbol in df.columns else first_p
    
    # 4. BACKTEST LOOP
    for i in range(len(df)):
        date = df.index[i]
        p = float(df[symbol_val].iloc[i])
        v_ema = float(df['vix_ema'].iloc[i])
        v_ma = float(df['vix_ma'].iloc[i])
        p_bench = float(df[benchmark_symbol].iloc[i]) if benchmark_symbol in df.columns else p
        
        if i > 0:
            prev_p = float(df[symbol_val].iloc[i-1])
            stock_val *= (p / prev_p)
            cash *= (1 + daily_yield)
            
        total_equity = stock_val + cash
        
        # Monthly HWM and Cash Floor Update
        is_first_day = (i == 0)
        is_month_change = (i > 0 and date.month != df.index[i-1].month)
        
        if is_first_day or is_month_change:
            if total_equity > max_equity:
                max_equity = total_equity
            cash_floor = max_equity * cash_floor_pct
            
            # Refill floor if long and cash is too low
            if pos == 1 and cash < cash_floor:
                shortfall = cash_floor - cash
                sell_amt = min(stock_val, shortfall)
                if sell_amt > 0:
                    proceeds, cost = apply_trading_costs(-sell_amt, p, slippage_bps, commission)
                    stock_val -= sell_amt
                    cash += abs(proceeds)
                    trade_log.append({
                        'Date': date.date() if hasattr(date, 'date') else date,
                        'Status': 'FLOOR REFILL',
                        'Amt': round(sell_amt, 2),
                        'Entry Price': round(p, 2),
                        'Exit Price': 'N/A',
                        'Profit %': 'N/A'
                    })

        # Signal Generation (Use PREVIOUS day's data to avoid look-ahead bias)
        idx_in_full = df_full.index.get_loc(date)
        if idx_in_full > 0:
            sig_v_ema = float(df_full['vix_ema'].iloc[idx_in_full-1])
            sig_v_ma = float(df_full['vix_ma'].iloc[idx_in_full-1])
            
            # Logic: Long := vix < vixMA or vix <= 20
            is_long_sig = (sig_v_ema < sig_v_ma) or (sig_v_ema <= vix_threshold)
            # Logic: Short := vix > vixMA * 1.15 and vix > 20
            is_close_sig = (sig_v_ema > sig_v_ma * short_multiplier) and (sig_v_ema > vix_threshold)
        else:
            is_long_sig = False
            is_close_sig = False

        # Execution
        if pos == 0 and is_long_sig:
            pos = 1
            # Calculate investable cash (respecting floor)
            investable = max(0, cash - (max_equity * cash_floor_pct))
            if investable > 0:
                net_invested, cost = apply_trading_costs(investable, p, slippage_bps, commission)
                stock_val += investable
                cash -= net_invested
                trade_log.append(create_trade_log_entry(date, p, None, 'OPEN', total_equity, amount=investable))
            
        elif pos == 1 and is_close_sig:
            pos = 0
            # Close trade log
            trade = next((t for t in reversed(trade_log) if t.get('Status') == 'OPEN'), None)
            if trade:
                close_trade_log_entry(trade, date, p, total_equity)
            
            proceeds, cost = apply_trading_costs(-stock_val, p, slippage_bps, commission)
            cash += abs(proceeds)
            stock_val = 0.0

        # History
        bh_val = initial_capital_val * (p / first_p)
        bench_val = initial_capital_val * (p_bench / first_bench)
        hist.append(create_historical_record(
            date, total_equity, bh_val, benchmark_symbol, bench_val, cash, p,
            VIX_EMA=v_ema, VIX_MA=v_ma
        ))

    return build_strategy_result(
        hist, trade_log, initial_capital_val, symbol_val, benchmark_symbol,
        strategy_name="Volatility (VIX)",
        params=locals()
    )
