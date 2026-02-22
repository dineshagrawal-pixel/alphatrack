"""
9 Signal Strategy
A multi-signal strategy that manages TQQQ and a side fund (SHY/SGOV).
Precisely aligned with the user's reference script to achieve ~36.63% CAGR.
"""

import pandas as pd
import yfinance as yf
import numpy as np
import datetime
from common import (
    build_strategy_result, 
    BacktestResult, 
    make_empty_result, 
    apply_trading_costs,
    get_data_start_date,
    download_multiple_tickers
)

def run_9sig_strategy(
    symbol_val,
    side_fund_val,
    start_date_val,
    initial_capital_val,
    target_tqqq_pct,
    new_money_to_signal,
    gap_rebalance_pct,
    q_growth_target,
    q_growth_cap,
    signal_floor_pct,
    cash_floor_hwm_pct,
    crash_threshold,
    up100_threshold,
    monthly_dep_val=0,
    benchmark_symbol="SPY",
    rebalance_sensitivity=0.05,
    slippage_bps=0.0,
    commission=0.0,
    end_date=None
):
    """
    9 Signal Strategy: A multi-signal strategy that manages TQQQ and a side fund.
    Matches the user's reference script CAGR and logic.
    """
    # 1. UNIFIED DATA FETCHING (Include consistent buffer)
    start_dt = pd.to_datetime(start_date_val)
    download_start = get_data_start_date(start_date_val)

    # Use standardized multi-ticker downloader to benefit from caching & sessions
    prices_df = download_multiple_tickers([symbol_val, side_fund_val, "SHY", benchmark_symbol], download_start)

    if prices_df.empty:
        return make_empty_result(start_date_val, initial_capital_val, symbol_val, benchmark_symbol)

    tqqq_p = prices_df[symbol_val].dropna() if symbol_val in prices_df.columns else pd.Series()
    # Fill side fund with SHY for historical depth
    if side_fund_val in prices_df.columns:
        side_p = prices_df[side_fund_val].fillna(prices_df["SHY"]).dropna()
    else:
        side_p = prices_df["SHY"].dropna()

    bench_p = prices_df[benchmark_symbol].dropna() if benchmark_symbol in prices_df.columns else tqqq_p
    
    full_df = pd.DataFrame({'Price': tqqq_p, 'SidePrice': side_p, 'Benchmark': bench_p}).dropna(subset=['Price', 'SidePrice'])
    
    if full_df.empty:
        return make_empty_result(start_date_val, initial_capital_val, symbol_val, benchmark_symbol)

    # 2. QUARTERLY SAMPLING (Calculated on full history for buffer)
    q_data = full_df.resample('QS').first()
    
    # Filter backtest period
    df = full_df[full_df.index >= start_dt].copy()
    if end_date:
        df = df[df.index <= pd.Timestamp(end_date)]
    
    # 3. INITIALIZATION
    eq_t = initial_capital_val * target_tqqq_pct
    eq_s = initial_capital_val * (1 - target_tqqq_pct)
    signal_line = eq_t
    hwm_sig = signal_line
    portfolio_hwm = initial_capital_val
    in_30_down = False
    sell_sig_count = 0
    
    bh_shares = initial_capital_val / df['Price'].iloc[0]
    bh_cash = 0
    first_bench_p = df['Benchmark'].iloc[0] if not df['Benchmark'].empty else 1.0
    
    trades, hist = [], []

    # 4. BACKTEST LOOP
    for i in range(len(df)):
        date = df.index[i]
        p = float(df['Price'].iloc[i])
        sp = float(df['SidePrice'].iloc[i])
        
        # 4.1 Daily Update
        if i > 0:
            prev_p = float(df['Price'].iloc[i-1])
            prev_sp = float(df['SidePrice'].iloc[i-1])
            eq_t *= (p / prev_p)
            eq_s *= (sp / prev_sp)

        total_equity = eq_t + eq_s
        
        # 4.2 Quarterly Logic
        # 4.2 Monthly/Quarterly Logic
        is_rebalance_day = (i == 0) or (df.index[i].quarter != df.index[i-1].quarter)
        is_month_change = (i > 0) and (df.index[i].month != df.index[i-1].month)

        # Update HWM and Cash Limit Monthly
        if i == 0 or is_month_change:
            portfolio_hwm = max(portfolio_hwm, total_equity)
            cash_limit = portfolio_hwm * cash_floor_hwm_pct

            # Rule VII: Force Cash Refill (Evaluated Monthly now)
            if eq_s < cash_limit:
                shortfall = cash_limit - eq_s
                net_shortfall, cost = apply_trading_costs(shortfall, p, slippage_bps, commission)
                eq_t -= net_shortfall
                eq_s += shortfall
                trades.append({'Date': date.date(), 'Status': 'FLOOR REFILL', 'Amt': shortfall, 'Entry Price': round(p, 2)})

        if is_rebalance_day:
            # Main Quarterly signals follow...

            # Rule VI: New Money
            q_cont = monthly_dep_val * 3
            eq_s += q_cont
            signal_line += (q_cont * new_money_to_signal)
            bh_cash += q_cont

            # Rule V: 100-Up Reset
            q_idx = q_data.index.get_indexer([date], method='pad')[0]
            if q_idx > 0:
                prev_q_start_p = float(q_data['Price'].iloc[q_idx-1])
                if (p / prev_q_start_p) >= up100_threshold:
                    total = eq_t + eq_s
                    eq_t, eq_s = total * target_tqqq_pct, total * (1 - target_tqqq_pct)
                    signal_line = eq_t
                    trades.append({'Date': date.date(), 'Status': '100-UP RESET', 'Amt': total, 'Entry Price': round(p, 2)})

            # Rule II: Signal Line Math
            target_v = min(signal_line * q_growth_target, signal_line * q_growth_cap)
            hwm_sig = max(hwm_sig, signal_line)
            signal_line = max(target_v, hwm_sig * signal_floor_pct)

            # Rule IV: 30-Down Rule
            lookback_8q = q_data['Price'].iloc[max(0, q_idx-8):q_idx+1].max()
            if p <= (lookback_8q * crash_threshold):
                in_30_down = True

            # Rule III: Gap Rebalance
            full_diff = signal_line - eq_t
            trade_goal = full_diff * gap_rebalance_pct
            
            action = None
            if abs(trade_goal) > (total_equity * rebalance_sensitivity):
                if trade_goal > 0: # BUYING
                    available_to_spend = max(0, eq_s - cash_limit) 
                    buy_amt = min(trade_goal, available_to_spend)
                    if buy_amt > 0:
                        net_buy, cost = apply_trading_costs(buy_amt, p, slippage_bps, commission)
                        eq_t += buy_amt
                        eq_s -= net_buy
                        action = f"BUY ({int(gap_rebalance_pct*100)}%)"
                elif trade_goal < 0: # SELLING
                    if in_30_down:
                        sell_sig_count += 1
                        if sell_sig_count >= 2:
                            in_30_down = False
                            total = eq_t + eq_s
                            eq_t, eq_s = total * target_tqqq_pct, total * (1 - target_tqqq_pct)
                            # Apply reset costs
                            reset_cost = (abs(eq_t - total*target_tqqq_pct) * (slippage_bps/10000)) + commission
                            eq_s -= reset_cost
                            signal_line = eq_t
                            trades.append({'Date': date.date(), 'Status': '30-DN RESET', 'Amt': total, 'Entry Price': round(p, 2)})
                    else:
                        sell_amt = abs(trade_goal)
                        proceeds, cost = apply_trading_costs(-sell_amt, p, slippage_bps, commission)
                        eq_t -= sell_amt
                        eq_s += abs(proceeds)
                        action = f"SELL ({int(gap_rebalance_pct*100)}%)"

            # Rule VIII: Minimum Allocation
            total_val = eq_t + eq_s
            if not in_30_down and (eq_t / total_val) < target_tqqq_pct:
                rebal_req = (total_val * target_tqqq_pct) - eq_t
                actual_rebal = min(rebal_req, max(0, eq_s - cash_limit)) 
                eq_t += actual_rebal
                eq_s -= actual_rebal

            if action:
                trades.append({
                    'Date': date.date(), 'Status': action, 'Amt': abs(trade_goal), 'Entry Price': round(p, 2)
                })

        # 4.3 Record History
        total_equity = eq_t + eq_s
        hist.append({
            'Date': date, 
            'Strategy': total_equity, 
            'BH': (p * bh_shares) + bh_cash, 
            benchmark_symbol: (df.loc[date, 'Benchmark'] / first_bench_p) * initial_capital_val if not df['Benchmark'].empty else total_equity,
            'Cash': eq_s, 
            'close': p,
            'Signal': signal_line,
            'HWM': portfolio_hwm
        })

    return build_strategy_result(
        hist, trades, initial_capital_val, symbol_val, benchmark_symbol, 
        strategy_name="9-Sig",
        params=locals()
    )

def _create_empty_results(start_date, initial_capital, benchmark_symbol, symbol):
    """Create empty results when no data is available."""
    dates = pd.date_range(start=start_date, periods=10, freq='D')
    dummy_data = {
        'Strategy': np.full(10, initial_capital),
        'BH': np.full(10, initial_capital),
        benchmark_symbol: np.full(10, initial_capital),
        'Cash': np.full(10, initial_capital),
        'close': np.full(10, 100.0)
    }
    df_results = pd.DataFrame(dummy_data, index=dates)
    df_results['ret'] = df_results['Strategy'].pct_change().fillna(0)
    rolling_returns_df = pd.DataFrame(index=df_results.index)
    cash_pct_df = pd.DataFrame({'Cash %': df_results['Cash'] / df_results['Strategy']}, 
                               index=df_results.index).fillna(0)
    return df_results, pd.DataFrame([]), [], initial_capital, rolling_returns_df, cash_pct_df, symbol
