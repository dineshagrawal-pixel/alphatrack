you provided # app.py

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(layout="wide", page_title="Backtesting Application")
st.title("Volatility Strategy Backtesting Application")

# --- Sidebar for Inputs ---
st.sidebar.header("Strategy Parameters")
symbol = st.sidebar.text_input("Stock Symbol", "TQQQ")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2010-02-11"))

initial_capital = st.sidebar.number_input("Initial Capital", min_value=1000.0, value=100000.0, step=1000.0)
cash_floor_pct = st.sidebar.slider("Cash Floor Percentage (%)", min_value=0.0, max_value=100.0, value=20.0, step=1.0) / 100
cash_yield_daily = st.sidebar.number_input("Daily Cash Yield (e.g., 1.0001 for 0.01% daily)", min_value=1.0, value=1.0001, format="%.5f")

# Monte Carlo Inputs
st.sidebar.header("Monte Carlo Simulation Parameters")
num_simulations = st.sidebar.number_input("Number of Simulations", min_value=100, value=1000, step=100)
sim_years = st.sidebar.number_input("Simulation Years", min_value=1, value=5, step=1)

# Placeholder for file uploads
st.sidebar.header("Upload Files")
uploaded_file_1 = st.sidebar.file_uploader("Upload Benchmark Data (Optional)", type=["csv", "txt"])
uploaded_file_2 = st.sidebar.file_uploader("Upload Custom Strategy Data (Optional)", type=["csv", "txt"])

def run_volatility_strategy_v5(symbol_val, start_date_val, initial_capital_val, cash_floor_pct_val, cash_yield_daily_val):
    # 1. FETCH DATA
    df = yf.download(symbol_val, start=start_date_val, progress=False, multi_level_index=False)
    if df.empty:
        st.error(f"No data found for {symbol_val} from {start_date_val}. Please check the symbol and date.")
        return None, None, None, None, None, None

    close_col = 'Adj Close' if 'Adj Close' in df.columns else 'Close'
    df = df[[close_col]].rename(columns={close_col: 'close'})

    # 2. INDICATORS
    df['ret'] = df['close'].pct_change()
    df['vol200'] = df['ret'].rolling(200).std() * np.sqrt(252)
    df['vol14'] = df['ret'].rolling(14).std() * np.sqrt(252)
    df['sma200'] = df['close'].rolling(200).mean()
    df = df.dropna().copy()

    # 3. BACKTEST SETTINGS
    cash = initial_capital_val
    stock_val = 0.0
    bh_val = initial_capital_val
    max_equity = initial_capital_val
    
    pos = 0 
    trade_list = [] 
    trade_log = []  
    current_trade = {}
    strat_hist, bh_hist, cash_hist = [], [], []

    # 4. BACKTEST LOOP
    for i in range(len(df)):
        row = df.iloc[i]
        date = df.index[i]
        
        if i > 0:
            prev_close = df.iloc[i-1]['close']
            # Update values based on price movement and interest
            stock_val *= (row['close'] / prev_close)
            cash *= cash_yield_daily_val
            bh_val *= (row['close'] / prev_close)

        total_equity = stock_val + cash
        if total_equity > max_equity:
            max_equity = total_equity
            
        cash_floor = max_equity * cash_floor_pct_val

        is_long = (row['vol14'] < row['vol200']) or (row['close'] > row['sma200'])
        is_short = (row['vol14'] > row['vol200']) and (row['close'] < row['sma200'])

        # Logic: Entry
        if pos == 0 and is_long:
            pos = 1
            # Buy with all cash available ABOVE the floor
            investment_amt = max(0, cash - cash_floor)
            cash -= investment_amt
            stock_val += investment_amt
            
            current_trade = {
                'Entry Date': date.date(),
                'Entry Price': round(row['close'], 2),
                'Entry Equity': total_equity
            }
        
        # Logic: Exit
        elif pos == 1 and is_short:
            pos = 0
            # Sell all stock back to cash
            profit_pct = (total_equity / current_trade['Entry Equity'] - 1) * 100
            trade_log.append({
                'Entry Date': current_trade['Entry Date'],
                'Exit Date': date.date(),
                'Entry Price': current_trade['Entry Price'],
                'Exit Price': round(row['close'], 2),
                'Profit %': f"{profit_pct:.2f}%",
                'Status': 'CLOSED'
            })
            trade_list.append(total_equity - current_trade['Entry Equity'])
            
            cash += stock_val
            stock_val = 0

        strat_hist.append(total_equity)
        bh_hist.append(bh_val)
        cash_hist.append(cash)

    # 5. CAPTURE OPEN TRADE
    if pos == 1:
        last_row = df.iloc[-1]
        total_equity = stock_val + cash
        profit_pct = (total_equity / current_trade['Entry Equity'] - 1) * 100
        trade_log.append({
            'Entry Date': current_trade['Entry Date'],
            'Exit Date': df.index[-1].date(),
            'Entry Price': current_trade['Entry Price'],
            'Exit Price': round(last_row['close'], 2),
            'Profit %': f"{profit_pct:.2f}%",
            'Status': 'OPEN (LIVE)'
        })
        trade_list.append(total_equity - current_trade['Entry Equity'])

    df['Strategy'] = strat_hist
    df['BH'] = bh_hist
    df['Total_Cash'] = cash_hist

    # Calculate rolling returns within the function
    df["Strategy_1Y_Roll_Ret"] = df["Strategy"].pct_change(periods=252)
    df["Strategy_3Y_Roll_Ret"] = df["Strategy"].pct_change(periods=252*3)
    df["Strategy_5Y_Roll_Ret"] = df["Strategy"].pct_change(periods=252*5)

    df["BH_1Y_Roll_Ret"] = df["BH"].pct_change(periods=252)
    df["BH_3Y_Roll_Ret"] = df["BH"].pct_change(periods=252*3)
    df["BH_5Y_Roll_Ret"] = df["BH"].pct_change(periods=252*5)

    # Calculate Cash Metrics
    df["Cash_Pct_of_Strategy"] = (df["Total_Cash"] / df["Strategy"])

    return df, trade_list, trade_log, initial_capital_val, df[[
        "Strategy_1Y_Roll_Ret", "BH_1Y_Roll_Ret",
        "Strategy_3Y_Roll_Ret", "BH_3Y_Roll_Ret",
        "Strategy_5Y_Roll_Ret", "BH_5Y_Roll_Ret"
    ]].dropna(), df[["Cash_Pct_of_Strategy"]].dropna()

# --- Monte Carlo Simulation Function ---
def run_monte_carlo(initial_equity, returns_series, num_simulations, sim_years, risk_free_rate=0.0):
    daily_returns = returns_series.dropna()
    if daily_returns.empty: return pd.DataFrame(), pd.DataFrame()
    
    mu = daily_returns.mean()
    sigma = daily_returns.std()
    
    num_days = sim_years * 252
    
    # Store all simulation paths
    simulation_paths = np.zeros((num_days, num_simulations))
    
    for i in range(num_simulations):
        daily_returns_sim = np.random.normal(mu, sigma, num_days)
        cumulative_returns_sim = (1 + daily_returns_sim).cumprod()
        simulation_paths[:, i] = initial_equity * cumulative_returns_sim

    sim_df = pd.DataFrame(simulation_paths)
    
    # Calculate CAGR and Max Drawdown for each simulation
    sim_cagr = []
    sim_mdd = []
    for col in sim_df.columns:
        final_equity = sim_df[col].iloc[-1]
        cagr = (final_equity / initial_equity)**(1/sim_years) - 1 if initial_equity > 0 else 0
        mdd = (sim_df[col] / sim_df[col].cummax() - 1).min()
        sim_cagr.append(cagr)
        sim_mdd.append(mdd)
        
    mc_results_df = pd.DataFrame({
        'CAGR': sim_cagr,
        'Max Drawdown': sim_mdd
    })
    
    return sim_df, mc_results_df

# Streamlit run button
if st.sidebar.button("Run Backtest"):
    df_results, trade_list_results, trade_log_results, initial_capital_used, rolling_returns_df, cash_pct_df = \
        run_volatility_strategy_v5(symbol, start_date, initial_capital, cash_floor_pct, cash_yield_daily)
    
    if df_results is not None:
        # 6. TABLES GENERATION
        # --- New Metric Calculation Functions ---
        def calculate_sharpe_ratio(returns, risk_free_rate=0.0):
            if returns.std() == 0: return 0.0
            annualized_return = (1 + returns).prod()**(252/len(returns)) - 1 # Annualized geometric return
            annualized_std = returns.std() * np.sqrt(252)
            return (annualized_return - risk_free_rate) / annualized_std

        def calculate_sortino_ratio(returns, risk_free_rate=0.0):
            if len(returns[returns < 0]) == 0: return float('inf')
            downside_returns = returns[returns < 0]
            downside_std = downside_returns.std() * np.sqrt(252)
            if downside_std == 0: return 0.0
            annualized_return = (1 + returns).prod()**(252/len(returns)) - 1
            return (annualized_return - risk_free_rate) / downside_std

        def calculate_calmar_ratio(ser_equity, initial_capital, annual_risk_free_rate=0.0):
            years = (ser_equity.index[-1] - ser_equity.index[0]).days / 365.25
            if years <= 0: return 0.0
            cagr = (ser_equity.iloc[-1] / initial_capital)**(1/years) - 1
            max_drawdown = (ser_equity / ser_equity.cummax() - 1).min()
            if max_drawdown == 0: return float('inf')
            return cagr / abs(max_drawdown)

        def calculate_ulcer_index(ser_equity):
            if ser_equity.empty: return 0.0
            peak = ser_equity.expanding().max()
            drawdown_squared = ((ser_equity - peak) / peak)**2
            return np.sqrt(drawdown_squared.mean())
            
        # --- End New Metric Calculation Functions ---

        # Extended get_metrics function to include new metrics
        def get_all_metrics(ser, pnl_list, initial_cap, returns_series):
            years = (ser.index[-1] - ser.index[0]).days / 365.25
            cagr = (ser.iloc[-1] / initial_cap)**(1/years) - 1 if initial_cap > 0 else 0
            mdd = (ser / ser.cummax() - 1).min()
            win_rate = "0%"
            profit_factor = "0.00"
            if len(pnl_list) > 0:
                wins = [t for t in pnl_list if t > 0]
                losses = [abs(t) for t in pnl_list if t < 0]
                win_rate = f"{(len(wins)/len(pnl_list))*100:.1f}%"
                profit_factor = f"{(sum(wins) / sum(losses)):.2f}" if sum(losses) != 0 else "Inf"
            
            # New Metrics Calculations
            sharpe = calculate_sharpe_ratio(returns_series)
            sortino = calculate_sortino_ratio(returns_series)
            calmar = calculate_calmar_ratio(ser, initial_cap)
            ulcer = calculate_ulcer_index(ser)
            
            # Avg Drawdown %
            drawdowns = (ser / ser.cummax() - 1)
            avg_dd_pct = drawdowns[drawdowns < 0].mean() if not drawdowns[drawdowns < 0].empty else 0.0

            # Avg DD Duration (simplified, actual calculation is more complex and involves identifying distinct drawdown periods)
            # Placeholder for now, will refine later
            avg_dd_duration = "N/A" # Will be implemented later

            # Trade Engine Metrics
            total_trades = len(pnl_list)
            avg_pnl_per_trade = sum(pnl_list) / total_trades if total_trades > 0 else 0.0
            
            wins = [t for t in pnl_list if t > 0]
            losses = [abs(t) for t in pnl_list if t < 0]
            avg_win = sum(wins) / len(wins) if len(wins) > 0 else 0.0
            avg_loss = sum(losses) / len(losses) if len(losses) > 0 else 0.0
            payoff_ratio = avg_win / avg_loss if avg_loss != 0 else float("inf")

            # Max Consecutive Losers
            max_consecutive_losers = 0
            current_consecutive_losers = 0
            for pnl in pnl_list:
                if pnl < 0:
                    current_consecutive_losers += 1
                else:
                    max_consecutive_losers = max(max_consecutive_losers, current_consecutive_losers)
                    current_consecutive_losers = 0
            max_consecutive_losers = max(max_consecutive_losers, current_consecutive_losers)

            # Expectancy Per Trade
            expectancy_per_trade = (payoff_ratio * (len(wins) / total_trades)) - (1 * (len(losses) / total_trades)) if total_trades > 0 else 0.0
            
            return [
                f"{cagr*100:.2f}%",
                f"{mdd*100:.2f}%",
                f"{avg_dd_pct*100:.2f}%",
                f"{ulcer:.2f}",
                f"{sharpe:.2f}",
                f"{sortino:.2f}",
                f"{calmar:.2f}",
                str(total_trades),
                win_rate,
                profit_factor,
                f"{payoff_ratio:.2f}",
                f"{avg_pnl_per_trade:,.2f}",
                str(max_consecutive_losers),
                f"{expectancy_per_trade:.2f}",
                avg_dd_duration,
                f"${ser.iloc[-1]:,.0f}"
            ]

        metrics_df = pd.DataFrame({
            'Portfolio': get_all_metrics(df_results['Strategy'], trade_list_results, initial_capital_used, df_results['ret']),
            'Buy & Hold': get_all_metrics(df_results['BH'], [df_results['BH'].iloc[-1] - df_results['BH'].iloc[0]], initial_capital_used, df_results['ret'])
        }, index=[
            'CAGR',
            'Max Drawdown',
            'Avg Drawdown %',
            'Ulcer Index',
            'Sharpe Ratio',
            'Sortino Ratio',
            'Calmar Ratio',
            'Total Trades',
            '% Profitable',
            'Profit Factor',
            'Payoff Ratio',
            'Avg P&L per Trade',
            'Max Consecutive Losers',
            'Expectancy Per Trade',
            'Avg DD Duration',
            'Ending Value'
        ])

        annual = df_results.resample('YE').last()
        annual['1Y_Ret'] = annual['Strategy'].pct_change() # Store as raw numeric
        annual['Price_Pct_Chg'] = annual['close'].pct_change() # Store as raw numeric
        
        # --- Executive Performance (The "What") ---
        st.header("1. Executive Performance (The \"What\")")

        # Table 1: Performance Overview
        performance_overview_df = metrics_df.loc[[
            'CAGR',
            'Profit Factor',
            'Expectancy Per Trade',
            'Ending Value'
        ]].copy()
        
        # Add Standard Range column
        performance_overview_df['Standard Range'] = [
            '>10% (green), 5-10% (yellow), <5% (red)',
            '>2.5 (green), 1.75-2.5 (yellow), 1.0-1.75 (yellow), ≤1.0 (red)',
            '>0 (green), ≤0 (red)',
            'Higher is better'
        ]
        
        st.subheader("Table 1: Performance Overview")

        def color_performance_overview(row):
            colors = ['' for _ in row]
            
            # For CAGR, higher is better. Target > Benchmark (let's say > 0.1 for 10% for green)
            if 'CAGR' in row.name:
                for i, val_str in enumerate(row.values):
                    try:
                        val = float(val_str.replace('%', '').replace('$', '').replace(',', '')) / 100
                        if val > 0.10: # Example target for Green: > 10% CAGR
                            colors[i] = 'background-color: #28a745' # Darker Green
                        elif val > 0.05: # Example target for Yellow: > 5% CAGR
                            colors[i] = 'background-color: #ffc107' # Darker Yellow
                        else:
                            colors[i] = 'background-color: #dc3545' # Darker Red
                    except ValueError: 
                        colors[i] = ''
            
            # For Profit Factor, Target >2.5 (green), 1.75-2.5 or 1.0-1.75 (yellow), ≤1.0 (red)
            if 'Profit Factor' in row.name:
                for i, val_str in enumerate(row.values):
                    try:
                        val = float(val_str)
                        if val > 2.5: # Green
                            colors[i] = 'background-color: #28a745'
                        elif 1.0 < val <= 2.5: # Yellow (1.75-2.5 or 1.0-1.75)
                            colors[i] = 'background-color: #ffc107'
                        else: # Red (<= 1.0, losing money)
                            colors[i] = 'background-color: #dc3545'
                    except ValueError:
                        colors[i] = ''
            
            # For Expectancy Per Trade, Target Positive
            if 'Expectancy Per Trade' in row.name:
                for i, val_str in enumerate(row.values):
                    try:
                        val = float(val_str)
                        if val > 0: # Green
                            colors[i] = 'background-color: #28a745'
                        else: # Red
                            colors[i] = 'background-color: #dc3545'
                    except ValueError:
                        colors[i] = ''

            return colors
        
        st.dataframe(performance_overview_df.style.apply(color_performance_overview, axis=1))

        # Table 2: Risk-Adjusted Returns
        risk_adjusted_returns_df = metrics_df.loc[[
            'Max Drawdown',
            'Avg Drawdown %',
            'Ulcer Index',
            'Sharpe Ratio',
            'Sortino Ratio',
            'Calmar Ratio',
            'Avg DD Duration' 
        ]].copy()
        
        # Add Standard Range column
        risk_adjusted_returns_df['Standard Range'] = [
            '<20% (green), 20-30% (yellow), >30% (red)',
            '<10% (green), 10-15% (yellow), >15% (red)',
            '<5.0 (green), 5.0-10.0 (yellow), >10.0 (red)',
            '>1.0 (green), 0.5-1.0 (yellow), <0.5 (red)',
            '>1.2 (green), 0.8-1.2 (yellow), <0.8 (red)',
            '>1.5 (green), 1.0-1.5 (yellow), <1.0 (red)',
            'N/A'
        ]
        
        st.subheader("Table 2: Risk-Adjusted Returns")
        
        def color_risk_adjusted_returns(row):
            colors = ['' for _ in row]
            # Max Drawdown %: Target < 20% (Green)
            if 'Max Drawdown' in row.name:
                for i, val_str in enumerate(row.values):
                    try:
                        val = float(val_str.replace('%', '').replace(',', ''))
                        if val < 20: # Green
                            colors[i] = 'background-color: #28a745'
                        elif 20 <= val < 30: # Yellow
                            colors[i] = 'background-color: #ffc107'
                        else: # Red
                            colors[i] = 'background-color: #dc3545'
                    except ValueError:
                        colors[i] = ''
            
            # Avg Drawdown %: Target < 10% (Green)
            if 'Avg Drawdown %' in row.name:
                for i, val_str in enumerate(row.values):
                    try:
                        val = float(val_str.replace('%', '').replace(',', ''))
                        if val < 10: # Green
                            colors[i] = 'background-color: #28a745'
                        elif 10 <= val < 15: # Yellow
                            colors[i] = 'background-color: #ffc107'
                        else: # Red
                            colors[i] = 'background-color: #dc3545'
                    except ValueError:
                        colors[i] = ''

            # Ulcer Index: Target < 5.0 (Green)
            if 'Ulcer Index' in row.name:
                for i, val_str in enumerate(row.values):
                    try:
                        val = float(val_str)
                        if val < 5.0: # Green
                            colors[i] = 'background-color: #28a745'
                        elif 5.0 <= val < 10.0: # Yellow
                            colors[i] = 'background-color: #ffc107'
                        else: # Red
                            colors[i] = 'background-color: #dc3545'
                    except ValueError:
                        colors[i] = ''
            
            # Sharpe Ratio: Target > 1.0 (Green)
            if 'Sharpe Ratio' in row.name:
                for i, val_str in enumerate(row.values):
                    try:
                        val = float(val_str)
                        if val > 1.0: # Green
                            colors[i] = 'background-color: #28a745'
                        elif 0.5 <= val <= 1.0: # Yellow
                            colors[i] = 'background-color: #ffc107'
                        else: # Red
                            colors[i] = 'background-color: #dc3545'
                    except ValueError:
                        colors[i] = ''

            # Sortino Ratio: Target > 1.2 (Green)
            if 'Sortino Ratio' in row.name:
                for i, val_str in enumerate(row.values):
                    try:
                        val = float(val_str)
                        if val > 1.2: # Green
                            colors[i] = 'background-color: #28a745'
                        elif 0.8 <= val <= 1.2: # Yellow
                            colors[i] = 'background-color: #ffc107'
                        else: # Red
                            colors[i] = 'background-color: #dc3545'
                    except ValueError:
                        colors[i] = ''
            
            # Calmar Ratio: Target > 1.5 (Green)
            if 'Calmar Ratio' in row.name:
                for i, val_str in enumerate(row.values):
                    try:
                        val = float(val_str)
                        if val > 1.5: # Green
                            colors[i] = 'background-color: #28a745'
                        elif 1.0 <= val <= 1.5: # Yellow
                            colors[i] = 'background-color: #ffc107'
                        else: # Red
                            colors[i] = 'background-color: #dc3545'
                    except ValueError:
                        colors[i] = ''

            return colors

        st.dataframe(risk_adjusted_returns_df.style.apply(color_risk_adjusted_returns, axis=1))
        
        # --- The Trade Engine (The "How") ---
        st.header("2. The Trade Engine (The \"How\")")

        # Table 3: Trade Statistics
        trade_statistics_df = metrics_df.loc[[
            'Total Trades',
            '% Profitable',
            'Payoff Ratio',
            'Avg P&L per Trade',
            'Max Consecutive Losers',
            'Expectancy Per Trade'
        ]].copy()
        
        # Add Standard Range column
        trade_statistics_df['Standard Range'] = [
            'Higher is better',
            '35-55% (green), 25-35% or 55-65% (yellow), else (red)',
            '>2.0 (green), 1.5-2.0 (yellow), <1.5 (red)',
            'Higher is better',
            '<5 (green), 5-10 (yellow), >10 (red)',
            '>0 (green), ≤0 (red)'
        ]
        
        st.subheader("Table 3: Trade Statistics")

        def color_trade_statistics(row):
            colors = ['' for _ in row]
            
            # Win % (Hit Rate): 35% – 55% is healthy
            if '% Profitable' in row.name:
                for i, val_str in enumerate(row.values):
                    try:
                        val = float(val_str.replace('%', '').replace(',', ''))
                        if 35 <= val <= 55: # Green
                            colors[i] = 'background-color: #28a745'
                        elif 25 <= val < 35 or 55 < val <= 65: # Yellow
                            colors[i] = 'background-color: #ffc107'
                        else: # Red
                            colors[i] = 'background-color: #dc3545'
                    except ValueError:
                        colors[i] = ''
            
            # Payoff Ratio: Aim for > 2.0
            if 'Payoff Ratio' in row.name:
                for i, val_str in enumerate(row.values):
                    try:
                        val = float(val_str)
                        if val > 2.0: # Green
                            colors[i] = 'background-color: #28a745'
                        elif 1.5 <= val <= 2.0: # Yellow
                            colors[i] = 'background-color: #ffc107'
                        else: # Red
                            colors[i] = 'background-color: #dc3545'
                    except ValueError:
                        colors[i] = ''
            
            # Avg P&L (per trade): Positive
            if 'Avg P&L per Trade' in row.name:
                for i, val_str in enumerate(row.values):
                    try:
                        val = float(val_str.replace(',', '')) # Remove comma for conversion
                        if val > 0: # Green
                            colors[i] = 'background-color: #28a745'
                        else: # Red
                            colors[i] = 'background-color: #dc3545'
                    except ValueError:
                        colors[i] = ''
            
            # Max Consecutive Losers: Lower is better. Target < 5 is Green.
            if 'Max Consecutive Losers' in row.name:
                for i, val_str in enumerate(row.values):
                    try:
                        val = int(val_str)
                        if val < 5: # Green
                            colors[i] = 'background-color: #28a745'
                        elif 5 <= val <= 10: # Yellow
                            colors[i] = 'background-color: #ffc107'
                        else: # Red
                            colors[i] = 'background-color: #dc3545'
                    except ValueError:
                        colors[i] = ''
            
            # Expectancy Per Trade: Positive
            if 'Expectancy Per Trade' in row.name:
                for i, val_str in enumerate(row.values):
                    try:
                        val = float(val_str)
                        if val > 0: # Green
                            colors[i] = 'background-color: #28a745'
                        else: # Red
                            colors[i] = 'background-color: #dc3545'
                    except ValueError:
                        colors[i] = ''

            return colors
        
        st.dataframe(trade_statistics_df.style.apply(color_trade_statistics, axis=1))

        # Table 5: Periodic & Rolling Analysis
        st.subheader("Table 5: Periodic & Rolling Analysis")
        
        # Prepare Annual Breakdown
        annual_breakdown_df = annual[['Strategy', 'BH', '1Y_Ret']].copy()
        annual_breakdown_df.rename(columns={'1Y_Ret': 'Strategy 1Y Ret'}, inplace=True)
        
        # Prepare Rolling Returns (1Y, 3Y, 5Y) - sampled every 20 rows, last 10
        rolling_sampled = rolling_returns_df.iloc[::20].tail(10).copy()
        
        # Prepare Cash Metrics - sampled every 20 rows, last 10
        cash_sampled = cash_pct_df.iloc[::20].tail(10).copy()
        
        # Combine all three tables into one
        # Rename columns for clarity
        rolling_sampled.columns = ['Strat 1Y Roll', 'BH 1Y Roll', 'Strat 3Y Roll', 'BH 3Y Roll', 'Strat 5Y Roll', 'BH 5Y Roll']
        cash_sampled.columns = ['Cash %']
        
        # Create a combined dataframe using concat (will align by index)
        combined_df = pd.concat([annual_breakdown_df, rolling_sampled, cash_sampled], axis=1)
        
        # Create a copy for coloring that uses numeric values
        combined_df_numeric = combined_df.copy()
        
        # Color function for combined table - works on numeric values
        def color_combined_table(val):
            try:
                if pd.isna(val):
                    return ''
                # Strategy and BH columns (dollar values) - higher is better
                if isinstance(val, (int, float)) and val > 0:
                    return 'background-color: #28a745'  # Green for positive
                elif isinstance(val, (int, float)) and val < 0:
                    return 'background-color: #dc3545'  # Red for negative
                return ''
            except:
                return ''

        # Apply styling with proper formatting and coloring
        # Define formatting
        formatting_dict = {
            'Strategy': '${:,.0f}', 
            'BH': '${:,.0f}', 
            'Strategy 1Y Ret': '{:.2f}%',
            'Strat 1Y Roll': lambda x: f"{x*100:.2f}%" if pd.notna(x) else '',
            'BH 1Y Roll': lambda x: f"{x*100:.2f}%" if pd.notna(x) else '',
            'Strat 3Y Roll': lambda x: f"{x*100:.2f}%" if pd.notna(x) else '',
            'BH 3Y Roll': lambda x: f"{x*100:.2f}%" if pd.notna(x) else '',
            'Strat 5Y Roll': lambda x: f"{x*100:.2f}%" if pd.notna(x) else '',
            'BH 5Y Roll': lambda x: f"{x*100:.2f}%" if pd.notna(x) else '',
            'Cash %': lambda x: f"{x*100:.2f}%" if pd.notna(x) else ''
        }
        
        # Apply formatting
        styled_df = combined_df.style.format(formatting_dict)
        
        # Apply coloring to percentage columns (returns only - not cash %)
        pct_columns = ['Strategy 1Y Ret', 'Strat 1Y Roll', 'BH 1Y Roll', 'Strat 3Y Roll', 'BH 3Y Roll', 'Strat 5Y Roll', 'BH 5Y Roll']
        for col in pct_columns:
            if col in combined_df.columns:
                # For returns: positive = green, negative = red
                styled_df = styled_df.apply(lambda x: ['background-color: #28a745' if pd.notna(v) and v > 0 
                                                        else 'background-color: #dc3545' if pd.notna(v) and v < 0
                                                        else '' for v in x], subset=[col])
        
        # Apply coloring to dollar columns (Strategy, BH)
        dollar_columns = ['Strategy', 'BH']
        for col in dollar_columns:
            if col in combined_df.columns:
                styled_df = styled_df.apply(lambda x: ['background-color: #28a745' if pd.notna(v) and v >= 0 
                                                        else 'background-color: #dc3545' for v in x], subset=[col])

        st.dataframe(styled_df)
        
        # Add color configuration note
        st.markdown("""
        **Color Legend:**
        - 🟢 **Green**: Positive returns / positive values
        - 🔴 **Red**: Negative returns / negative values
        """)
        
        # --- Trade Log (The "Evidence") ---
        st.header("4. Trade Log (The \"Evidence\")")

        st.subheader("Table 6: Trade Log")
        trade_log_df = pd.DataFrame(trade_log_results)
        
        # Apply coloring to Trade Log - Profit % column
        if not trade_log_df.empty and 'Profit %' in trade_log_df.columns:
            def color_profit_pct(val):
                try:
                    val_float = float(str(val).replace('%', ''))
                    if val_float > 0:
                        return 'background-color: #28a745'
                    elif val_float < 0:
                        return 'background-color: #dc3545'
                    else:
                        return ''
                except:
                    return ''
            st.dataframe(trade_log_df.style.format({
                'Entry Price': '${:.2f}',
                'Exit Price': '${:.2f}',
                'Entry Equity': '${:,.2f}'
            }).map(color_profit_pct, subset=['Profit %']))
        else:
            st.dataframe(trade_log_df)

        # --- Stress Testing & Stability (The "Survival") ---
        st.header("3. Stress Testing & Stability (The \"Survival\")")

        # Table 7: Monte Carlo Probability Distribution - Placeholder for now
        st.subheader("Table 7: Monte Carlo Probability Distribution")
        st.info("Monte Carlo simulation will be implemented in a later step.")

        # Table 8: Survival & Discipline - Placeholder for now
        st.subheader("Table 8: Survival & Discipline")
        st.info("Survival & Discipline metrics will be implemented in a later step.")

        # Table 9: Stability (Small Sample Check) - Placeholder for now
        st.subheader("Table 9: Stability (Small Sample Check)")
        st.info("Stability metrics will be implemented in a later step.")

        # 8. PLOT
        fig, ax = plt.subplots(figsize=(12,7))
        ax.plot(df_results.index, df_results["Strategy"], label="Strategy Equity (with 20% Floor)", color="blue", lw=2)
        ax.plot(df_results.index, df_results["BH"], label="TQQQ Buy & Hold", color="gray", alpha=0.4)
        ax.set_yscale("log")
        ax.set_title(f"200SMA + Volatility with 20% Cash Floor (HWM): {symbol}")
        ax.legend()
        ax.grid(True, which="both", alpha=0.3)
        st.pyplot(fig)
