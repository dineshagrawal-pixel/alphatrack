"""
Reporting Module - Reusable reporting functions for backtesting results
Based on Portfolio Visualizer Tactical Asset Allocation Model
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, FuncFormatter
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from .common import BacktestResult


# ==========================================
# CONSTANTS & BRANDING (Portfolio Visualizer Style)
# ==========================================

PV_COLORS = {
    'Strategy': '#1f4e9c',      # Deep Blue
    'BH': '#27ae60',            # Teal / Mint Green
    'Benchmark': '#7fb3d3',     # Light Blue
    'Cash': '#dcdcdc',          # Grey
    'Drawdown': '#dc3545',      # Red
    'Text': '#333333'
}

# Set global matplotlib style for a professional look
plt.rcParams.update({
    'axes.facecolor': 'white',
    'axes.edgecolor': '#e0e0e0',
    'grid.color': '#f0f0f0',
    'font.size': 10,
    'legend.frameon': False,
    'axes.spines.top': False,
    'axes.spines.right': False
})


# ==========================================
# CONSTANTS
# ==========================================

STRESS_PERIODS = [
    ("Covid-19 Crash", "2020-02-19", "2020-03-23"),
    ("2022 Bear Market", "2022-01-03", "2022-10-12"),
    ("2018 Correction", "2018-09-20", "2018-12-24"),
    ("Aug 2024 Carry Trade Unwind", "2024-07-16", "2024-08-05")
]

# ==========================================
# STYLING UTILITIES
# ==========================================

def apply_conditional_styling(df, only_strategy=False):
    """
    Apply professional color coding to metrics tables to highlight strategy effectiveness.
    Colors:
    - Excellent: Dark Green (#1b5e20)
    - Good: Light Green (#4caf50)
    - Moderate: Amber/Yellow (#ffc107)
    - Poor: Red (#d32f2f)
    """
    if df.empty:
        return df.style

    # If the identifier is in the index, move it to a column
    potential_id_cols = ['Metric', 'Type', 'Roll Period', 'Year']
    if df.index.name in potential_id_cols:
        df = df.reset_index()

    # Identify the row label column
    id_col = next((c for c in potential_id_cols if c in df.columns), df.columns[0])

    def get_row_styles(row):
        row_metric_name = str(row[id_col]).lower()
        styles = [''] * len(row)
        
        for i in range(len(row)):
            col_name = str(row.index[i]).lower()
            # Skip the identifier column and the description column
            if col_name in [str(id_col).lower(), 'metric description']:
                continue
            
            # If only_strategy is requested, skip formatting for non-strategy columns
            if only_strategy and 'strategy' not in col_name:
                continue
                
            val = row.iloc[i]
            if pd.isna(val) or val == 'N/A' or val == 'Inf' or val == '-' or val == 'N/A%':
                continue
                
            # Combine row metric and col name for keyword matching
            metric_context = f"{row_metric_name} {col_name}"
            
            try:
                # Clean and parse numeric value
                clean_val = str(val).replace('%', '').replace('$', '').replace(',', '').strip()
                if ' out of ' in clean_val:
                    if '(' in clean_val and '%' in clean_val:
                        clean_val = clean_val.split('(')[1].split('%')[0]
                
                num = float(clean_val)
                
                # Detect scale context
                is_monthly = 'monthly' in metric_context
                is_ratio = any(x in metric_context for x in ['sharpe', 'sortino', 'ratio', 'measure', 'expectancy', 'information', 'factor', 'beta', 'correlation', 'r2', 'alpha'])
                
                # Higher is Better Metrics keywords
                hib = ['return', 'cagr', 'alpha', 'profit', 'sharpe', 'sortino', 'treynor', 'calmar', 
                       'expectancy', 'win %', 'positive', 'm2 measure', 'gain/loss', 'upside capture',
                       'ending balance', 'analyzed trades', 'winning trades', 'information ratio',
                       'active return', 'ratio', 'rolling', 'mean', 'success', 'average', 'high', 'low', 'r2', 'skewness', 'swr', 'pwr']
                
                # Lower is Better Metrics keywords (lib)
                lib = ['drawdown', 'standard deviation', 'volatility', 'tracking error', 'ulcer', 
                       'loss', 'consecutive losers', 'cost impact', 'stdev', 'downside deviation', 
                       'var', 'cvar', 'kurtosis', 'downside capture']
                
                # Handle Conflicts: downside capture is lib, upside capture is hib
                if 'downside capture' in metric_context:
                    match_lib = True
                    match_hib = False
                elif any(x in metric_context for x in hib):
                    match_hib = True
                    match_lib = False
                elif any(x in metric_context for x in lib):
                    match_hib = False
                    match_lib = True
                else:
                    match_hib = False
                    match_lib = False

                if match_hib:
                    if is_ratio:
                        # Ratio scale (Sharpe, Sortino, Information, Calmar, Profit Factor)
                        if num >= 2.0: styles[i] = 'background-color: #1b5e20; color: white'
                        elif num >= 1.0: styles[i] = 'background-color: #4caf50; color: white'
                        elif num < 0: styles[i] = 'background-color: #d32f2f; color: white'
                    elif 'alpha' in metric_context:
                        # Alpha scale (Annualized excess return in %)
                        if num >= 5.0: styles[i] = 'background-color: #1b5e20; color: white'
                        elif num >= 2.0: styles[i] = 'background-color: #4caf50; color: white'
                        elif num < 0: styles[i] = 'background-color: #d32f2f; color: white'
                    elif is_monthly:
                        # Monthly percentage scale
                        if num >= 1.5: styles[i] = 'background-color: #1b5e20; color: white'
                        elif num >= 0.8: styles[i] = 'background-color: #4caf50; color: white'
                        elif num < 0: styles[i] = 'background-color: #d32f2f; color: white'
                    elif 'win %' in metric_context or 'profitable' in metric_context:
                        # Win Rate scale
                        if num >= 65: styles[i] = 'background-color: #1b5e20; color: white'
                        elif num >= 55: styles[i] = 'background-color: #4caf50; color: white'
                        elif num < 45: styles[i] = 'background-color: #d32f2f; color: white'
                    else:
                        # Annual percentage scale (CAGR, Returns)
                        if num >= 20: styles[i] = 'background-color: #1b5e20; color: white'
                        elif num >= 10: styles[i] = 'background-color: #4caf50; color: white'
                        elif num < 0: styles[i] = 'background-color: #d32f2f; color: white'

                elif match_lib:
                    num_abs = abs(num)
                    if 'drawdown' in metric_context:
                        # Drawdown scale
                        if num_abs <= 10: styles[i] = 'background-color: #1b5e20; color: white'
                        elif num_abs <= 20: styles[i] = 'background-color: #4caf50; color: white'
                        elif num_abs > 40: styles[i] = 'background-color: #d32f2f; color: white'
                    elif 'kurtosis' in metric_context:
                        # Kurtosis (Tail risk)
                        if num > 3.0: styles[i] = 'background-color: #ffc107; color: black' # Warning for fat tails
                    elif is_monthly:
                        # Monthly Volatility
                        if num_abs <= 3: styles[i] = 'background-color: #1b5e20; color: white'
                        elif num_abs <= 6: styles[i] = 'background-color: #4caf50; color: white'
                        elif num_abs > 12: styles[i] = 'background-color: #d32f2f; color: white'
                    else:
                        # Annualized volatility etc.
                        if num_abs <= 10: styles[i] = 'background-color: #1b5e20; color: white'
                        elif num_abs <= 20: styles[i] = 'background-color: #4caf50; color: white'
                        elif num_abs > 40: styles[i] = 'background-color: #d32f2f; color: white'
            except:
                pass
        return styles

    return df.style.apply(get_row_styles, axis=1)


def apply_transposed_metrics_styling(df):
    """
    Specifically for tables where Metrics are Columns and Portfolios are Rows.
    Commonly used for Summary and Risk tables to allow tooltips on headers.
    """
    if df.empty:
        return df.style

    # Identify portfolio column
    id_col = next((c for c in ['Portfolio', 'Type', 'Strategy'] if c in df.columns), df.columns[0])

    def get_cell_styles(val, col_name):
        col_name_lower = str(col_name).lower()
        if pd.isna(val) or val == 'N/A' or val == 'Inf' or val == '-' or val == 'N/A%':
            return ''
            
        try:
            clean_val = str(val).replace('%', '').replace('$', '').replace(',', '').strip()
            if ' out of ' in clean_val:
                if '(' in clean_val and '%' in clean_val:
                    clean_val = clean_val.split('(')[1].split('%')[0]
            num = float(clean_val)
            
            hib = ['return', 'cagr', 'alpha', 'profit', 'sharpe', 'sortino', 'treynor', 'calmar', 
                   'expectancy', 'win %', 'positive', 'm2 measure', 'gain/loss', 'capture ratio',
                   'ending balance', 'analyzed trades', 'winning trades', 'information ratio',
                   'active return', 'ratio', 'rolling', 'mean', 'success', 'skewness']
            lib = ['drawdown', 'standard deviation', 'volatility', 'tracking error', 'ulcer', 
                   'loss', 'consecutive losers', 'cost impact', 'stdev', 'cv', 'consistency']
            
            is_ratio = any(x in col_name_lower for x in ['sharpe', 'sortino', 'ratio', 'calmar', 'measure', 'expectancy', 'alpha', 'information', 'factor', 'beta', 'correlation', 'r2'])
            
            # 1. Special case: Small-scale statistical metrics (CV, Consistency, Skewness, Kurtosis)
            if 'skewness' in col_name_lower:
                if num >= 0.5: return 'background-color: #1b5e20; color: white'
                elif num >= 0: return 'background-color: #4caf50; color: white'
                else: return 'background-color: #d32f2f; color: white'
            
            if 'kurtosis' in col_name_lower:
                if num > 3.0: return 'background-color: #ffc107; color: black'
                return ''

            if any(x in col_name_lower for x in ['cv', 'consistency']):
                if num_abs <= 0.15: return 'background-color: #1b5e20; color: white'
                elif num_abs <= 0.30: return 'background-color: #4caf50; color: white'
                elif num_abs > 0.60: return 'background-color: #d32f2f; color: white'
                return ''

            # 2. General Higher is Better (HIB)
            if any(x in col_name_lower for x in hib):
                if is_ratio:
                    # Ratio scale
                    if num >= 2.0: return 'background-color: #1b5e20; color: white'
                    elif num >= 1.0: return 'background-color: #4caf50; color: white'
                    elif num < 0: return 'background-color: #d32f2f; color: white'
                elif 'alpha' in col_name_lower:
                    # Alpha scale
                    if num >= 5.0: return 'background-color: #1b5e20; color: white'
                    elif num >= 2.0: return 'background-color: #4caf50; color: white'
                    elif num < 0: return 'background-color: #d32f2f; color: white'
                elif 'win %' in col_name_lower or 'profitable' in col_name_lower:
                    # Win Rate scale
                    if num >= 65: return 'background-color: #1b5e20; color: white'
                    elif num >= 55: return 'background-color: #4caf50; color: white'
                    elif num < 45: return 'background-color: #d32f2f; color: white'
                else:
                    # Annual percentage scale
                    if num >= 20: return 'background-color: #1b5e20; color: white'
                    elif num >= 10: return 'background-color: #4caf50; color: white'
                    elif num < 0: return 'background-color: #d32f2f; color: white'
            
            # 3. General Lower is Better (LIB)
            elif any(x in col_name_lower for x in lib):
                num_abs = abs(num)
                if 'drawdown' in col_name_lower:
                    if num_abs <= 10: return 'background-color: #1b5e20; color: white'
                    elif num_abs <= 20: return 'background-color: #4caf50; color: white'
                    elif num_abs > 40: return 'background-color: #d32f2f; color: white'
                elif any(x in col_name_lower for x in ['stdev', 'volatility', 'tracking error']):
                    if num_abs <= 10: return 'background-color: #1b5e20; color: white'
                    elif num_abs <= 20: return 'background-color: #4caf50; color: white'
                    elif num_abs > 35: return 'background-color: #d32f2f; color: white'
                else:
                    if num_abs <= 10: return 'background-color: #1b5e20; color: white'
                    elif num_abs <= 20: return 'background-color: #4caf50; color: white'
                    elif num_abs > 35: return 'background-color: #d32f2f; color: white'
        except:
            pass
        return ''

    # Apply style by applying to each column (excluding the ID column)
    styler = df.style
    for col in df.columns:
        if col != id_col:
            styler = styler.map(lambda v, c=col: get_cell_styles(v, c), subset=[col])
    return styler


# ==========================================
# METRIC GLOSSARY & HELP
# ==========================================

METRIC_DEFINITIONS = {
    "CAGR": "Compound Annual Growth Rate: The geometric progression ratio that provides a constant rate of return over the time period.",
    "Max Drawdown": "The maximum observed loss from a peak to a trough of a portfolio, before a new peak is attained.",
    "Sharpe Ratio": "Measure of risk-adjusted return, calculated by dividing the excess return of the portfolio by its standard deviation.",
    "Sortino Ratio": "A variation of the Sharpe ratio that only considers downside volatility (returns below 0).",
    "Volatility": "Annualized Standard Deviation of daily returns. Measures the 'nervousness' of the portfolio.",
    "Alpha": "The excess return of an investment relative to the return of a benchmark index.",
    "Beta": "A measure of the volatility, or systematic risk, of a security or a portfolio in comparison to the market as a whole.",
    "Capture Ratios": "Upside capture measures performance in up-markets; Downside capture measures performance in down-markets (lower is better).",
    "Tail Risk": "Metrics like VaR (Value at Risk) and CVaR (Conditional VaR) that measure the probability of extreme losses.",
    "Rolling Returns": "Returns calculated over overlapping periods (e.g., 3-year rolling) to show consistency of performance.",
    "Monte Carlo": "Simulation technique used to understand the impact of risk and uncertainty in financial models by running thousands of scenarios.",
    "Information Ratio": "A measure of portfolio returns above the returns of a benchmark, usually an index, to the volatility of those returns.",
    "Calmar Ratio": "The ratio of the annualized return to the maximum drawdown over a specific period.",
    "Gain/Loss Ratio": "The ratio of the average profit from winning trades to the average loss from losing trades.",
    "Time in Market": "The percentage of time the strategy had capital invested in the market vs. sitting in cash.",
    "Ulcer Index": "A measure of the depth and duration of drawdowns in a portfolio; a lower value indicates a more comfortable ride.",
    "Standard Deviation": "Annualized Standard Deviation of returns. A common proxy for total risk.",
    "Best Year": "The return of the single best calendar year in the backtest period.",
    "Worst Year": "The return of the single worst calendar year in the backtest period.",
    "R2": "R-Squared: The percentage of a portfolio's movement that can be explained by movements in its benchmark index.",
    "Treynor Ratio": "Risk-adjusted return based on systematic risk (Beta).",
    "Active Return": "The return of the strategy relative to its benchmark (Strategy - Benchmark).",
    "Tracking Error": "The standard deviation of the active return. Measures how closely a portfolio follows its benchmark.",
    "VaR": "Value at Risk: The maximum potential loss over a month with 95% confidence.",
    "CVaR": "Conditional Value at Risk (Expected Shortfall): The average loss in the worst 5% of cases.",
    "Win %": "Percentage of trades that were profitable.",
    "Profit Factor": "The total profit divided by the total loss from all trades.",
    "Payoff Ratio": "The ratio of average profit per winning trade to average loss per losing trade (Average Win / Average Loss).",
    "Total Trades": "Count of all opening and closing transactions.",
    "Annualized Return": "Geometric mean of returns scaled to a 1-year period."
}

TRADE_STATUS_DEFINITIONS = {
    "OPEN": "The trade or position is currently active and has not been closed.",
    "CLOSED": "The trade has been fully exited and the profit/loss is realized.",
    "BUY": "A simple purchase of an asset to establish or increase a position.",
    "SELL": "A sale of an asset to reduce or exit a position.",
    "REBALANCE": "A portfolio adjustment to return to target weights (e.g., 60% Stock / 40% Bonds).",
    "100-UP RESET": "Strategy Rebalance triggered by a 100% gain in a short period (Rule V). Resets allocation to target weights and recalibrates signal line to prevent 'over-buying' at peak prices.",
    "30-DN RESET": "Strategy Rebalance triggered by a significant market crash (Rule IV). Occurs after a >30% decline where the strategy protects capital by ignoring small sell signals and then resetting once stability is detected.",
    "GAP REBAL": "A proportional rebalance used to close the gap between the target signal line and current holdings.",
}

def get_metric_help(section_name):
    """Return a combined help string for a reporting section based on contained metrics."""
    relevant_keys = []
    if "Performance Summary" in section_name: relevant_keys = ["CAGR", "Volatility", "Sharpe Ratio", "Max Drawdown"]
    elif "Risk and Return" in section_name: relevant_keys = ["Alpha", "Beta", "Sharpe Ratio", "Information Ratio", "Capture Ratios", "Tail Risk"]
    elif "Rolling" in section_name: relevant_keys = ["Rolling Returns"]
    elif "Stress Testing" in section_name: relevant_keys = ["Monte Carlo", "Tail Risk"]
    elif "Management" in section_name: relevant_keys = ["Time in Market"]
    elif "Trading Performance" in section_name: relevant_keys = ["Gain/Loss Ratio"]
    
    help_text = "  \n\n".join([f"**{k}**: {METRIC_DEFINITIONS.get(k, '')}" for k in relevant_keys])
    return help_text if help_text else f"Detailed statistical analysis of {section_name.lower()}."


def render_market_stress_analysis(df_results, benchmark_symbol):
    """
    Render performance during historical market stress periods.
    """
    st.subheader("Performance during Market Stress Periods")
    
    stress_periods = STRESS_PERIODS
    
    rows = []
    for period_name, start_date, end_date in stress_periods:
        start_ts = pd.Timestamp(start_date)
        end_ts = pd.Timestamp(end_date)
        
        # Check if backtest covers this period
        if start_ts < df_results.index[0] or end_ts > df_results.index[-1]:
            continue
            
        period_data = df_results.loc[start_ts:end_ts]
        if len(period_data) < 2: continue
        
        def calc_ret(col):
            return (period_data[col].iloc[-1] / period_data[col].iloc[0]) - 1
            
        def calc_max_dd(col):
            sub = period_data[col]
            return (sub / sub.cummax() - 1).min()

        rows.append({
            'Stress Period': period_name,
            'Dates': f"{start_date} to {end_date}",
            'Strategy': calc_ret('Strategy'),
            'Buy & Hold': calc_ret('BH'),
            benchmark_symbol: calc_ret(benchmark_symbol) if benchmark_symbol in df_results.columns else None,
            'Strat Max DD': calc_max_dd('Strategy')
        })

    if rows:
        stress_df = pd.DataFrame(rows)
        
        # Heatmap for Strategy column
        vmax = stress_df['Strategy'].abs().max()
        if pd.isna(vmax) or vmax == 0: vmax = 10.0
        
        styler = stress_df.style.hide(axis='index')
        styler = styler.background_gradient(
            cmap='RdYlGn', subset=['Strategy'],
            vmin=-vmax, vmax=vmax
        ).highlight_null(color=None)
        
        styler = styler.format({
            'Strategy': "{:.2f}%", 'Buy & Hold': "{:.2f}%", benchmark_symbol: "{:.2f}%", 'Strat Max DD': "{:.2f}%"
        }, na_rep="N/A")
        
        st.dataframe(styler, use_container_width=True)
        
        st.caption("Recovery status is based on whether the portfolio returned to the pre-peak value after the stress period.")
    else:
        st.info("Backtest period does not overlap with historical stress periods.")


def run_monte_carlo(initial_equity, returns_series, num_simulations, sim_years, risk_free_rate=0.0):
    """Run Monte Carlo simulation"""
    daily_returns = returns_series.dropna()
    if returns_series is None or len(returns_series) < 2:
        return pd.DataFrame(), pd.DataFrame()
    
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


def calculate_sharpe_ratio(returns, risk_free_rate=0.0, periods=252):
    """Calculate Sharpe Ratio with flexible frequency"""
    if returns is None or len(returns) < 2:
        return 0.0
    std = returns.std()
    if std == 0 or np.isnan(std):
        return 0.0
        
    # Annualized return using geometric mean consistency
    total_ret = (1 + returns).prod() - 1
    num_years = (returns.index[-1] - returns.index[0]).days / 365.25
    if num_years <= 0: return 0.0
    ann_return = (1 + total_ret)**(1/num_years) - 1
    
    annualized_std = std * np.sqrt(periods)
    return (ann_return - risk_free_rate) / annualized_std


def calculate_sortino_ratio(returns, risk_free_rate=0.0, periods=252):
    """Calculate Sortino Ratio with flexible frequency"""
    if returns is None or len(returns) < 2:
        return 0.0
    
    downside_returns = returns[returns < 0]
    if len(downside_returns) < 2:
        if len(downside_returns) == 0:
            return 100.0 # High value for "infinite" Sortino
        downside_std = abs(downside_returns.iloc[0]) * np.sqrt(periods) # Fallback for 1 down period
    else:
        downside_std = downside_returns.std() * np.sqrt(periods)
        
    if downside_std == 0 or np.isnan(downside_std):
        return 0.0
        
    total_ret = (1 + returns).prod() - 1
    num_years = (returns.index[-1] - returns.index[0]).days / 365.25
    if num_years <= 0: return 0.0
    ann_return = (1 + total_ret)**(1/num_years) - 1
    
    return (ann_return - risk_free_rate) / downside_std


def calculate_calmar_ratio(ser_equity, initial_capital, annual_risk_free_rate=0.0):
    """Calculate Calmar Ratio"""
    years = (ser_equity.index[-1] - ser_equity.index[0]).days / 365.25
    if years <= 0:
        return 0.0
    cagr = (ser_equity.iloc[-1] / initial_capital)**(1/years) - 1
    max_drawdown = (ser_equity / ser_equity.cummax() - 1).min()
    if max_drawdown == 0:
        return float('inf')
    return cagr / abs(max_drawdown)


def calculate_ulcer_index(ser_equity):
    """Calculate Ulcer Index"""
    if ser_equity.empty:
        return 0.0
    peak = ser_equity.expanding().max()
    drawdown_squared = ((ser_equity - peak) / peak)**2
    return np.sqrt(drawdown_squared.mean())


def align_returns(returns, benchmark_returns):
    """Align two return series to have the same length and timezone-naive indices"""
    if returns is None or benchmark_returns is None or len(returns) == 0 or len(benchmark_returns) == 0:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    
    # Ensure Series
    if not isinstance(returns, pd.Series): returns = pd.Series(returns)
    if not isinstance(benchmark_returns, pd.Series): benchmark_returns = pd.Series(benchmark_returns)

    # Ensure timezone-naive indices for proper concatenation/intersection
    if hasattr(returns.index, 'tz_localize') and returns.index.tz is not None:
        returns.index = returns.index.tz_localize(None)
    if hasattr(benchmark_returns.index, 'tz_localize') and benchmark_returns.index.tz is not None:
        benchmark_returns.index = benchmark_returns.index.tz_localize(None)
        
    common_idx = returns.index.intersection(benchmark_returns.index)
    if len(common_idx) == 0:
        return pd.Series(dtype=float, index=pd.DatetimeIndex([])), pd.Series(dtype=float, index=pd.DatetimeIndex([]))
        
    return returns.loc[common_idx], benchmark_returns.loc[common_idx]


def calculate_treynor_ratio(returns, benchmark_returns, risk_free_rate=0.0, periods=252):
    """Calculate Treynor Ratio with flexible frequency"""
    returns, benchmark_returns = align_returns(returns, benchmark_returns)
    if len(returns) < 2 or len(benchmark_returns) < 2:
        return 0.0
    returns = np.asarray(returns)
    benchmark_returns = np.asarray(benchmark_returns)
    if returns.ndim != 1 or benchmark_returns.ndim != 1:
        return 0.0
    covariance = np.cov(returns, benchmark_returns)[0][1]
    benchmark_variance = np.var(benchmark_returns)
    if benchmark_variance == 0:
        return 0.0
    beta = covariance / benchmark_variance
    if beta == 0:
        return 0.0
    annualized_return = (1 + returns).prod()**(periods/len(returns)) - 1
    return (annualized_return - risk_free_rate) / beta


def calculate_beta(returns, benchmark_returns):
    """Calculate Beta"""
    returns, benchmark_returns = align_returns(returns, benchmark_returns)
    # Additional validation
    if len(returns) < 2 or len(benchmark_returns) < 2:
        return 0.0
    returns = np.asarray(returns)
    benchmark_returns = np.asarray(benchmark_returns)
    if returns.ndim != 1 or benchmark_returns.ndim != 1:
        return 0.0
    covariance = np.cov(returns, benchmark_returns)[0][1]
    benchmark_variance = np.var(benchmark_returns)
    if benchmark_variance == 0:
        return 0.0
    return covariance / benchmark_variance


def calculate_alpha(returns, benchmark_returns, risk_free_rate=0.0, periods=252):
    """Calculate Alpha with flexible frequency"""
    returns, benchmark_returns = align_returns(returns, benchmark_returns)
    if len(returns) == 0 or len(benchmark_returns) == 0:
        return 0.0
    strat_ann = (1 + returns).prod()**(periods/len(returns)) - 1
    bench_ann = (1 + benchmark_returns).prod()**(periods/len(benchmark_returns)) - 1
    beta = calculate_beta(returns, benchmark_returns)
    alpha = strat_ann - (risk_free_rate + beta * (bench_ann - risk_free_rate))
    return alpha


def calculate_geometric_mean(returns, period='annualized'):
    """Calculate Geometric Mean"""
    if len(returns) == 0:
        return 0.0
    total_ret = (1 + returns).prod() - 1
    if period == 'monthly':
        num_months = len(returns.resample('ME').prod())
        if num_months == 0: return 0.0
        return (1 + total_ret)**(1/num_months) - 1
    else: # annualized
        num_years = (returns.index[-1] - returns.index[0]).days / 365.25
        if num_years <= 0: return 0.0
        return (1 + total_ret)**(1/num_years) - 1


def calculate_withdrawal_rates(returns_series):
    """
    Calculate Safe Withdrawal Rate (SWR) and Perpetual Withdrawal Rate (PWR).
    """
    if len(returns_series) < 2:
        return 0.0, 0.0
    # Resample to monthly returns
    m_rets = (1 + returns_series).resample('ME').prod() - 1
    num_months = len(m_rets)
    if num_months == 0:
        return 0.0, 0.0
    
    cum_prod = (1 + m_rets).cumprod()
    total_prod = cum_prod.iloc[-1]
    
    sum_compound = 0
    for i in range(num_months):
        term = total_prod / cum_prod.iloc[i]
        sum_compound += term
    
    if sum_compound == 0:
        return 0.0, 0.0
        
    swr = total_prod / sum_compound
    pwr = (total_prod - 1) / sum_compound
    return swr * 12, pwr * 12



def calculate_r_squared(returns, benchmark_returns):
    """Calculate R-squared"""
    returns, benchmark_returns = align_returns(returns, benchmark_returns)
    if len(returns) < 2 or len(benchmark_returns) < 2:
        return 0.0
    returns = np.asarray(returns)
    benchmark_returns = np.asarray(benchmark_returns)
    if returns.ndim != 1 or benchmark_returns.ndim != 1:
        return 0.0
    correlation = np.corrcoef(returns, benchmark_returns)[0][1]
    return correlation ** 2


def calculate_gain_loss_ratio(returns):
    """Calculate Gain/Loss Ratio: Average Gain / Abs(Average Loss)"""
    if returns is None or len(returns) == 0:
        return 0.0
    gains = returns[returns > 0]
    losses = returns[returns < 0]
    avg_gain = gains.mean() if len(gains) > 0 else 0.0
    avg_loss = abs(losses.mean()) if len(losses) > 0 else 0.0
    if avg_loss == 0:
        return 0.0
    return avg_gain / avg_loss


def calculate_correlation(returns, benchmark_returns):
    """Calculate Correlation between strategy and benchmark returns"""
    returns, benchmark_returns = align_returns(returns, benchmark_returns)
    if len(returns) < 2 or len(benchmark_returns) < 2:
        return 0.0
    returns = np.asarray(returns)
    benchmark_returns = np.asarray(benchmark_returns)
    if returns.ndim != 1 or benchmark_returns.ndim != 1:
        return 0.0
    return np.corrcoef(returns, benchmark_returns)[0][1]


def calculate_information_ratio(returns, benchmark_returns, periods=252):
    """Calculate Information Ratio with flexible frequency"""
    returns, benchmark_returns = align_returns(returns, benchmark_returns)
    if len(returns) == 0 or len(benchmark_returns) == 0:
        return 0.0
    active_returns = returns - benchmark_returns
    tracking_error = active_returns.std() * np.sqrt(periods)
    if tracking_error == 0:
        return 0.0
    annualized_active_return = (1 + active_returns).prod()**(periods/len(active_returns)) - 1
    return annualized_active_return / tracking_error


def calculate_upside_capture_ratio(returns, benchmark_returns, periods=252):
    """Calculate Upside Capture Ratio with flexible frequency"""
    returns, benchmark_returns = align_returns(returns, benchmark_returns)
    if len(returns) == 0 or len(benchmark_returns) == 0:
        return 0.0
    up_periods = benchmark_returns > 0
    if up_periods.sum() == 0:
        return 0.0
    strat_up = returns[up_periods]
    bench_up = benchmark_returns[up_periods]
    if len(strat_up) == 0 or len(bench_up) == 0:
        return 0.0
    strat_ann = (1 + strat_up).prod()**(periods/len(strat_up)) - 1
    bench_ann = (1 + bench_up).prod()**(periods/len(bench_up)) - 1
    if bench_ann == 0:
        return 0.0
    return (strat_ann / bench_ann) * 100


def calculate_downside_capture_ratio(returns, benchmark_returns, periods=252):
    """Calculate Downside Capture Ratio with flexible frequency"""
    returns, benchmark_returns = align_returns(returns, benchmark_returns)
    if len(returns) == 0 or len(benchmark_returns) == 0:
        return 0.0
    down_periods = benchmark_returns < 0
    if down_periods.sum() == 0:
        return 0.0
    strat_down = returns[down_periods]
    bench_down = benchmark_returns[down_periods]
    if len(strat_down) == 0 or len(bench_down) == 0:
        return 0.0
    strat_ann = (1 + strat_down).prod()**(periods/len(strat_down)) - 1
    bench_ann = (1 + bench_down).prod()**(periods/len(bench_down)) - 1
    if bench_ann == 0:
        return 0.0
    return (strat_ann / bench_ann) * 100


def calculate_var_historical(returns, confidence=0.95):
    """Calculate Historical Value at Risk"""
    if len(returns) == 0:
        return 0.0
    return np.percentile(returns, (1 - confidence) * 100)


def calculate_cvar(returns, confidence=0.95):
    """Calculate Conditional Value at Risk (Expected Shortfall)"""
    if len(returns) == 0:
        return 0.0
    var = calculate_var_historical(returns, confidence)
    tail_returns = returns[returns <= var]
    if len(tail_returns) == 0:
        return var
    return tail_returns.mean()


def normal_ppf(p):
    """Approximate normal inverse CDF"""
    if p <= 0:
        return -np.inf
    if p >= 1:
        return np.inf
    if p < 0.5:
        return -normal_ppf(1 - p)
    a1 = -3.969683028665376e+01
    a2 = 2.209460984245205e+02
    a3 = -2.759285104469687e+02
    a4 = 1.383577518672690e+02
    a5 = -3.066479806614716e+01
    a6 = 2.506628277459239e+00
    b1 = -5.447609879822406e+01
    b2 = 1.615858368580409e+02
    b3 = -1.556989798598866e+02
    b4 = 6.680131188771972e+01
    b5 = -1.328068155288572e+01
    c1 = -7.784894002430293e-03
    c2 = -3.223964580411365e-01
    c3 = -2.400758277161838e+00
    c4 = -2.549732539343734e+00
    c5 = 4.374664141464968e+00
    c6 = 2.938163982698783e+00
    d1 = 7.784695709041462e-03
    d2 = 3.224671290700398e-01
    d3 = 2.445134137142996e+00
    d4 = 3.754408661907416e+00
    p_low = 0.02425
    p_high = 1 - p_low
    q = np.sqrt(-2 * np.log(1 - p))
    if p < p_low:
        r = p - p_low
        x = (((((c1*q+c2)*q+c3)*q+c4)*q+c5)*q+c6) / ((((d1*q+d2)*q+d3)*q+d4)*q+1)
    elif p <= p_high:
        r = p - 0.5
        r2 = r*r
        x = (((((a1*r2+a2)*r2+a3)*r2+a4)*r2+a5)*r2+a6)*r / (((((b1*r2+b2)*r2+b3)*r2+b4)*r2+b5)*r2+1)
    else:
        r = 1 - p
        x = -(((((c1*q+c2)*q+c3)*q+c4)*q+c5)*q+c6) / ((((d1*q+d2)*q+d3)*q+d4)*q+1)
    return x


def calculate_var_analytical(returns, confidence=0.95):
    """Calculate Analytical VaR without scipy"""
    if len(returns) == 0:
        return 0.0
    z = normal_ppf(1 - confidence)
    return returns.mean() + z * returns.std()


def calculate_gain_loss_ratio(returns):
    """Calculate Gain/Loss Ratio (Average Gain / Average Loss)"""
    if returns is None or len(returns) == 0:
        return 0.0
    up_returns = returns[returns > 0]
    down_returns = returns[returns < 0]
    if len(up_returns) == 0:
        return 0.0
    if len(down_returns) == 0:
        return 100.0 # High value for "infinite" G/L ratio
        
    avg_gain = up_returns.mean()
    avg_loss = abs(down_returns.mean())
    if avg_loss == 0 or np.isnan(avg_loss):
        return 100.0
    return avg_gain / avg_loss


def get_all_metrics(ser, pnl_list, initial_cap, returns_series, benchmark_returns=None, spy_returns=None, pnl_pct_list=None):
    """Calculate all metrics for a strategy"""
    years = (ser.index[-1] - ser.index[0]).days / 365.25
    cagr = (ser.iloc[-1] / initial_cap)**(1/years) - 1 if initial_cap > 0 else 0
    mdd = (ser / ser.cummax() - 1).min()
    
    # (Metrics will be re-calculated more robustly in the trade summary block below)
    
    sharpe = calculate_sharpe_ratio(returns_series)
    sortino = calculate_sortino_ratio(returns_series)
    calmar = calculate_calmar_ratio(ser, initial_cap)
    ulcer = calculate_ulcer_index(ser)
    
    if benchmark_returns is None or len(benchmark_returns) == 0:
        benchmark_returns = returns_series
    
    aligned_returns, aligned_benchmark = align_returns(returns_series, benchmark_returns)
    
    treynor = calculate_treynor_ratio(aligned_returns, aligned_benchmark)
    beta = calculate_beta(aligned_returns, aligned_benchmark)
    alpha = calculate_alpha(aligned_returns, aligned_benchmark)
    r_squared = calculate_r_squared(aligned_returns, aligned_benchmark)
    info_ratio = calculate_information_ratio(aligned_returns, aligned_benchmark)
    upside_capture = calculate_upside_capture_ratio(aligned_returns, aligned_benchmark)
    downside_capture = calculate_downside_capture_ratio(aligned_returns, aligned_benchmark)
    var_hist = calculate_var_historical(returns_series)
    var_analytical = calculate_var_analytical(returns_series)
    cvar = calculate_cvar(returns_series)
    gain_loss = calculate_gain_loss_ratio(returns_series)
    
    skewness = returns_series.skew() if len(returns_series) > 2 else 0
    kurt = returns_series.kurtosis() if len(returns_series) > 3 else 0
    
    downside_returns = returns_series[returns_series < 0]
    downside_dev = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
    
    positive_periods = (returns_series > 0).sum()
    total_periods = len(returns_series)
    positive_pct = (positive_periods / total_periods * 100) if total_periods > 0 else 0
    
    ann_return = (1 + returns_series).prod()**(252/len(returns_series)) - 1 if len(returns_series) > 0 else 0
    ann_std = returns_series.std() * np.sqrt(252)
    
    if len(aligned_returns) > 0 and len(aligned_benchmark) > 0:
        active_return = ann_return - ((1 + aligned_benchmark).prod()**(252/len(aligned_benchmark)) - 1)
    else:
        active_return = 0
    
    if len(aligned_returns) > 0 and len(aligned_benchmark) > 0:
        active_returns = aligned_returns - aligned_benchmark
        tracking_error = active_returns.std() * np.sqrt(252)
    else:
        tracking_error = 0
    
    if len(aligned_benchmark) > 0:
        m2 = sharpe * ann_std + ((1 + aligned_benchmark).prod()**(252/len(aligned_benchmark)) - 1)
    else:
        m2 = 0
    
    drawdowns = (ser / ser.cummax() - 1)
    avg_dd_pct = drawdowns[drawdowns < 0].mean() if not drawdowns[drawdowns < 0].empty else 0.0
    
    # --- Trade Level Metrics Unification ---
    # Determine the best source for trade analysis (favor pnl_list if available, else pnl_pct_list)
    analysis_trades = pnl_list if (pnl_list and len(pnl_list) > 0) else pnl_pct_list
    if not analysis_trades:
        analysis_trades = []

    total_trades = len(analysis_trades)
    wins = [t for t in analysis_trades if t > 0]
    losses = [abs(t) for t in analysis_trades if t < 0]
    
    win_rate = f"{(len(wins)/total_trades)*100:.1f}%" if total_trades > 0 else "0.0%"
    
    # Calculate Profit Factor (Sum of wins / Sum of losses)
    sum_wins = sum(wins)
    sum_losses = sum(losses)
    profit_factor = f"{(sum_wins / sum_losses):.2f}" if sum_losses != 0 else ("Inf" if sum_wins > 0 else "0.00")
    
    # Calculate Payoff Ratio
    # If we have percentages, use the percentage list for better accuracy if available
    if pnl_pct_list and len(pnl_pct_list) > 0:
        p_wins = [t for t in pnl_pct_list if t > 0]
        p_losses = [abs(t) for t in pnl_pct_list if t < 0]
        avg_pct_win = sum(p_wins) / len(p_wins) if len(p_wins) > 0 else 0.0
        avg_pct_loss = sum(p_losses) / len(p_losses) if len(p_losses) > 0 else 0.0
        payoff_ratio = avg_pct_win / avg_pct_loss if avg_pct_loss != 0 else (float("inf") if avg_pct_win > 0 else 0.0)
    else:
        avg_win = sum(wins) / len(wins) if len(wins) > 0 else 0.0
        avg_loss = sum(losses) / len(losses) if len(losses) > 0 else 0.0
        payoff_ratio = avg_win / avg_loss if avg_loss != 0 else (float("inf") if avg_win > 0 else 0.0)

    # Re-assign for return dict clarity
    final_win_rate = win_rate
    final_profit_factor = profit_factor
    final_payoff_ratio = payoff_ratio

    avg_pnl_per_trade = sum(pnl_list) / total_trades if (pnl_list and total_trades > 0) else 0.0
    
    max_consecutive_losers = 0
    current_consecutive_losers = 0
    for t in analysis_trades:
        if t < 0:
            current_consecutive_losers += 1
        else:
            max_consecutive_losers = max(max_consecutive_losers, current_consecutive_losers)
            current_consecutive_losers = 0
    max_consecutive_losers = max(max_consecutive_losers, current_consecutive_losers)
    
    expectancy_per_trade = (payoff_ratio * (len(wins) / total_trades)) - (1 * (len(losses) / total_trades)) if total_trades > 0 else 0.0
    
    # Calculate SPY metrics if available
    spy_cagr = "N/A"
    spy_mdd = "N/A"
    spy_sharpe = "N/A"
    spy_ann_return = "N/A"
    spy_ann_std = "N/A"
    spy_ending_value = "N/A"
    spy_correlation = "N/A"
    
    if spy_returns is not None and len(spy_returns) > 0:
        spy_ann = (1 + spy_returns).prod()**(252/len(spy_returns)) - 1 if len(spy_returns) > 0 else 0
        spy_std = spy_returns.std() * np.sqrt(252)
        spy_sharpe_val = calculate_sharpe_ratio(spy_returns)
        
        # Calculate correlation with SPY
        spy_corr = calculate_correlation(returns_series, spy_returns)
        spy_correlation = f"{spy_corr:.2f}"
        
        # Get SPY equity from df_results - find the SPY series
        # We'll need to pass this separately or calculate from returns
        spy_cagr = f"{spy_ann*100:.2f}%"
        spy_sharpe = f"{spy_sharpe_val:.2f}"
        spy_ann_return = f"{spy_ann*100:.2f}%"
        spy_ann_std = f"{spy_std*100:.2f}%"
    
    # Calculate benchmark correlation (using BH as benchmark)
    benchmark_correlation = calculate_correlation(returns_series, benchmark_returns)
    
    return {
        'CAGR': f"{cagr*100:.2f}%",
        'Max Drawdown': f"{mdd*100:.2f}%",
        'Avg Drawdown %': f"{avg_dd_pct*100:.2f}%",
        'Ulcer Index': f"{ulcer:.2f}",
        'Sharpe Ratio': f"{sharpe:.2f}",
        'Sortino Ratio': f"{sortino:.2f}",
        'Calmar Ratio': f"{calmar:.2f}",
        'Total Trades': str(total_trades),
        '% Profitable': str(final_win_rate),
        'Profit Factor': str(final_profit_factor) if final_profit_factor else "0.00",
        'Payoff Ratio': f"{final_payoff_ratio:.2f}" if isinstance(final_payoff_ratio, (float, int)) else str(final_payoff_ratio),
        'Avg P&L per Trade': f"{avg_pnl_per_trade:,.2f}",
        'Max Consecutive Losers': str(max_consecutive_losers),
        'Expectancy Per Trade': f"{expectancy_per_trade:.2f}",
        'Avg DD Duration': "N/A",
        'Ending Value': f"${ser.iloc[-1]:,.0f}",
        'Annualized Return': f"{ann_return*100:.2f}%",
        'Annualized Std Dev': f"{ann_std*100:.2f}%",
        'Downside Deviation': f"{downside_dev*100:.2f}%",
        'Treynor Ratio': f"{treynor:.4f}",
        'Beta': f"{beta:.2f}",
        'Alpha': f"{alpha*100:.2f}%",
        'R-squared': f"{r_squared*100:.2f}%",
        'Information Ratio': f"{info_ratio:.2f}",
        'Upside Capture Ratio': f"{upside_capture:.2f}%",
        'Downside Capture Ratio': f"{downside_capture:.2f}%",
        'Historical VaR (5%)': f"{var_hist*100:.2f}%",
        'Analytical VaR (5%)': f"{var_analytical*100:.2f}%",
        'CVaR (5%)': f"{cvar*100:.2f}%",
        'Skewness': f"{skewness:.2f}",
        'Excess Kurtosis': f"{kurt:.2f}",
        'Gain/Loss Ratio': f"{gain_loss:.2f}",
        'Positive Periods': f"{positive_periods}/{total_periods} ({positive_pct:.1f}%)",
        'Active Return': f"{active_return*100:.2f}%",
        'Tracking Error': f"{tracking_error*100:.2f}%",
        'M2 Measure': f"{m2*100:.2f}%",
        # SPY metrics
        'SPY CAGR': spy_cagr,
        'SPY Max Drawdown': spy_mdd,
        'SPY Sharpe Ratio': spy_sharpe,
        'SPY Annualized Return': spy_ann_return,
        'SPY Annualized Std Dev': spy_ann_std,
        'SPY Ending Value': spy_ending_value,
        # Correlation metrics
        'Benchmark Correlation': f"{benchmark_correlation:.2f}",
        'SPY Correlation': spy_correlation
    }


def render_mobile_metrics_dashboard(perf_df, metrics_dict, benchmark_symbol):
    """
    Enhanced Mobile-first metrics display with Heatmaps.
    Uses sections and color-coding specifically for the Strategy column.
    """
    # 1. THE "BIG 4" DASHBOARD (Stylized Metrics)
    # Streamlit columns stack on mobile.
    c1, c2, c3, c4 = st.columns(4)
    
    # CAGR Card
    with c1:
        st.metric("CAGR", metrics_dict.get('CAGR', '0.00%'))
    # Sharpe Card
    with c2:
        st.metric("Sharpe", metrics_dict.get('Sharpe Ratio', '0.00'))
    # Max Drawdown Card
    with c3:
        # Note: Max Drawdown is already a negative string like "-15.20%"
        st.metric("Max DD", metrics_dict.get('Max Drawdown', '0.00%'))
    # Win Rate Card
    with c4:
        st.metric("Win %", metrics_dict.get('% Profitable', '0.00%'))

    # 2. SECTIONS WITH HEATMAPS
    # We slice the perf_df (already processed and formatted) into categorical expanders
    
    # Define columns to show in mobile (Metric + Comparison)
    # Filter out 'Metric Description' for mobile to save horizontal space
    cols_to_show = [c for c in perf_df.columns if c != 'Metric Description']
    
    def render_styled_slice(df_slice, title, expanded=False):
        with st.expander(title, expanded=expanded):
            # Apply color coding ONLY to strategy column
            # This makes the strategy's performance "pop" against the clean benchmark columns
            styled = apply_conditional_styling(df_slice[cols_to_show], only_strategy=True)
            st.dataframe(styled, use_container_width=True, hide_index=True)

    # 1. Growth & Returns (Rows 0-5: Start, End, Multiple, Cash %, CAGR, Std Dev)
    render_styled_slice(perf_df.iloc[:6], "📈 Portfolio Growth & Balances", expanded=True)
    
    # 2. Risk & Adjusted Returns (Rows 6-11: Best/Worst Year, Max DD, Sharpe, Sortino, G/L)
    render_styled_slice(perf_df.iloc[6:12], "🛡️ Risk-Adjusted Performance")
    
    # 3. Trade Statistics (Rows 12+: Number of Trades, Win %, Profit Factor, Payoff, etc.)
    render_styled_slice(perf_df.iloc[12:], "⚡ Trade Execution Statistics")


def render_performance_overview(metrics_dict, df_results=None, initial_capital=None, benchmark_symbol="SPY", trade_log=None):
    """Render Performance Overview - Mobile First with heatmap strategy emphasis"""
    
    if df_results is None or len(df_results) < 2:
        st.warning("Insufficient data for Performance Summary.")
        return

    # --- 1. PREPARE DATA (Unified for all views) ---
    strat_rets = df_results['ret'].dropna()
    bh_rets = df_results['BH'].pct_change().dropna()
    bench_rets = bh_rets
    if benchmark_symbol in df_results.columns:
        bench_rets = df_results[benchmark_symbol].pct_change().dropna()

    start_val = float(initial_capital) if initial_capital else float(df_results['Strategy'].iloc[0])
    start_balance_str = f"${start_val:,.0f}"

    # Calculate PV-style metrics for all three columns
    s_m = calculate_pv_metrics(strat_rets, bench_rets, df_results['Strategy'])
    b_m = calculate_pv_metrics(bh_rets, bench_rets, df_results['BH'])
    m_m = calculate_pv_metrics(bench_rets, bench_rets, df_results[benchmark_symbol] if benchmark_symbol in df_results.columns else df_results['BH'])

    # Helper formatters
    def fmt_pct(val):
        try: return f"{float(val)*100:.2f}%" if not (np.isinf(val) or np.isnan(val)) else "0.00%"
        except: return "0.00%"
    def fmt_num(val):
        try: return f"{float(val):.2f}" if not (np.isinf(val) or np.isnan(val)) else "0.00"
        except: return "0.00"
    def get_best_worst(series):
        try:
            ann = series.resample('YE').last().pct_change().dropna()
            if len(ann) > 0: return f"{ann.max()*100:.2f}%", f"{ann.min()*100:.2f}%"
        except: pass
        return 'N/A', 'N/A'

    s_best, s_worst = get_best_worst(df_results['Strategy'])
    b_best, b_worst = get_best_worst(df_results['BH'])
    m_best, m_worst = get_best_worst(df_results[benchmark_symbol] if benchmark_symbol in df_results.columns else df_results['BH'])

    bench_name = benchmark_symbol
    if benchmark_symbol == "SPY": bench_name = "SPDR S&P 500 ETF (SPY)"
    elif benchmark_symbol == "QQQ": bench_name = "Invesco QQQ Trust (QQQ)"

    # Build full metric rows matching the style of Portfolio Visualizer
    metrics_rows = [
        ("Start Balance", start_balance_str, start_balance_str, start_balance_str, "Initial value of the portfolio."),
        ("Ending Balance", f"${df_results['Strategy'].iloc[-1]:,.0f}", f"${df_results['BH'].iloc[-1]:,.0f}", f"${df_results[benchmark_symbol].iloc[-1]:,.0f}" if benchmark_symbol in df_results.columns else "N/A", "Final portfolio value."),
        ("Equity Multiple", f"{(df_results['Strategy'].iloc[-1] / start_val):.2f}x", f"{(df_results['BH'].iloc[-1] / start_val):.2f}x", f"{(df_results[benchmark_symbol].iloc[-1] / start_val):.2f}x" if benchmark_symbol in df_results.columns else "N/A", "Total growth multiple."),
        ("Cash Balance %", f"{(df_results.get('Cash', pd.Series([0])).iloc[-1] / df_results['Strategy'].iloc[-1])*100:.2f}%", "0.00%", "0.00%", "Percentage of value currently in cash."),
        ("CAGR", fmt_pct(s_m.get('GM_A')), fmt_pct(b_m.get('GM_A')), fmt_pct(m_m.get('GM_A')), "Compound Annual Growth Rate."),
        ("Std Dev", fmt_pct(s_m.get('SD_A')), fmt_pct(b_m.get('SD_A')), fmt_pct(m_m.get('SD_A')), "Annualized Volatility."),
        ("Best Year", s_best, b_best, m_best, "Highest calendar year return."),
        ("Worst Year", s_worst, b_worst, m_worst, "Lowest calendar year return."),
        ("Max Drawdown", fmt_pct(s_m.get('MDD')), fmt_pct(b_m.get('MDD')), fmt_pct(m_m.get('MDD')), "Maximum peak-to-trough decline."),
        ("Sharpe Ratio", fmt_num(s_m.get('Sharpe')), fmt_num(b_m.get('Sharpe')), fmt_num(m_m.get('Sharpe')), "Risk-adjusted return."),
        ("Sortino Ratio", fmt_num(s_m.get('Sortino')), fmt_num(b_m.get('Sortino')), fmt_num(m_m.get('Sortino')), "Downside-only risk-adjusted return."),
        ("Avg Monthly G/L", fmt_num(s_m.get('GL')), fmt_num(b_m.get('GL')), fmt_num(m_m.get('GL')), "Average monthly gain vs average monthly loss."),
        ("Number of Trades", str(metrics_dict.get('Total Trades', 'N/A')), "1 (B&H)", "N/A", "Total trade count."),
        ("Win Rate %", metrics_dict.get('% Profitable', "N/A"), "N/A", "N/A", "Percentage of profitable trades."),
        ("Profit Factor", str(metrics_dict.get('Profit Factor', '0.00')), "N/A", "N/A", "Gross Profit / Gross Loss."),
        ("Payoff Ratio", str(metrics_dict.get('Payoff Ratio', "N/A")), "N/A", "N/A", "Average Win / Average Loss."),
        ("Expectancy", str(metrics_dict.get('Expectancy Per Trade', 'N/A')), "N/A", "N/A", "Expected profit per trade."),
    ]

    perf_df = pd.DataFrame(metrics_rows, columns=['Metric', 'Strategy', 'Buy & Hold Portfolio', bench_name, 'Metric Description'])

    # --- 2. RENDER THE VIEWS ---
    
    st.subheader("Performance Snapshot")
    render_mobile_metrics_dashboard(perf_df, metrics_dict, benchmark_symbol)
    
    st.markdown("---")
    with st.expander("📊 Full Side-by-Side Comparison (Desktop View)", expanded=False):
        # Desktop view renders heatmaps for EVERYTHING to allow deep comparison
        full_styled = apply_conditional_styling(perf_df, only_strategy=False)
        st.dataframe(full_styled, use_container_width=True, hide_index=True)





def calculate_pv_metrics(returns, benchmark_returns, equity_curve):
    """
    Calculate comprehensive portfolio metrics for the comparison table.
    Guarantees a complete dictionary with ALL keys to prevent gaps in tables.
    """
    # Initialize with absolute zero defaults for ALL expected keys
    metrics = {
        'GM_A': 0.0, 'GM_M': 0.0, 'AM_A': 0.0, 'AM_M': 0.0,
        'SD_A': 0.0, 'SD_M': 0.0, 'DD_M': 0.0, 'MDD': 0.0, 
        'Corr': 0.0, 'Beta': 0.0, 'Alpha': 0.0, 'R2': 0.0,
        'Sharpe': 0.0, 'Sortino': 0.0, 'Treynor': 0.0, 'Calmar': 0.0,
        'M2': 0.0, 'ActiveRet': 0.0, 'TE': 0.0, 'IR': 0.0,
        'Skew': 0.0, 'Kurt': 0.0, 'VaR_H': 0.0, 'VaR_A': 0.0, 'CVaR': 0.0,
        'UpCap': 0.0, 'DownCap': 0.0, 'SWR': 0.0, 'PWR': 0.0,
        'Pos': 0, 'Total': 0, 'GL': 0.0
    }
    
    if equity_curve is None or len(equity_curve) < 1:
        return metrics

    # Sanitize inputs
    returns = returns.fillna(0.0) if returns is not None else pd.Series(dtype=float)
    benchmark_returns = benchmark_returns.fillna(0.0) if benchmark_returns is not None else pd.Series(dtype=float)
    equity_curve = equity_curve.ffill().fillna(1.0) 
    
    # Absolute Returns / CAGR
    try:
        metrics['GM_A'] = calculate_geometric_mean(returns, 'annualized')
        metrics['GM_M'] = (1 + metrics['GM_A'])**(1/12) - 1 if metrics['GM_A'] > -1 else -1.0
    except: pass

    # Monthly Stats (PV Standard)
    try:
        m_rets = (1 + returns).resample('ME').prod() - 1 if not returns.empty else pd.Series([0.0])
        m_bench = (1 + benchmark_returns).resample('ME').prod() - 1 if not benchmark_returns.empty else pd.Series([0.0])
        
        metrics['AM_M'] = m_rets.mean()
        metrics['AM_A'] = (1 + metrics['AM_M'])**12 - 1
        metrics['SD_M'] = m_rets.std()
        metrics['SD_A'] = metrics['SD_M'] * np.sqrt(12)
        
        # Downside Deviation (monthly)
        m_downside = m_rets[m_rets < 0]
        metrics['DD_M'] = m_downside.std() if len(m_downside) > 1 else 0.0
        
        metrics['GL'] = calculate_gain_loss_ratio(m_rets)
        metrics['Pos'] = int((m_rets > 0).sum())
        metrics['Total'] = len(m_rets)
        
        # Skewness and Kurtosis (monthly)
        metrics['Skew'] = m_rets.skew() if len(m_rets) > 2 else 0.0
        metrics['Kurt'] = m_rets.kurtosis() if len(m_rets) > 3 else 0.0
        
        # Value-at-Risk (monthly)
        metrics['VaR_H'] = calculate_var_historical(m_rets)
        metrics['VaR_A'] = calculate_var_analytical(m_rets)
        metrics['CVaR'] = calculate_cvar(m_rets)
    except: pass

    # Risk-Adjusted Ratios & drawdown
    try: 
        metrics['Sharpe'] = calculate_sharpe_ratio(returns)
        metrics['Sortino'] = calculate_sortino_ratio(returns)
        metrics['Calmar'] = calculate_calmar_ratio(equity_curve, equity_curve.iloc[0])
        peak = equity_curve.cummax()
        metrics['MDD'] = (equity_curve / peak - 1).min() if peak.max() > 0 else 0.0
    except: pass

    # Alignment-dependent metrics (Monthly alignment)
    try:
        ma_rets, ma_bench = align_returns(m_rets, m_bench)
        if not ma_rets.empty:
            metrics['Corr'] = calculate_correlation(ma_rets, ma_bench)
            metrics['Beta'] = calculate_beta(ma_rets, ma_bench)
            metrics['Alpha'] = calculate_alpha(ma_rets, ma_bench, periods=12)
            metrics['R2'] = calculate_r_squared(ma_rets, ma_bench)
            metrics['Treynor'] = calculate_treynor_ratio(ma_rets, ma_bench, periods=12)
            metrics['IR'] = calculate_information_ratio(ma_rets, ma_bench, periods=12)
            metrics['UpCap'] = calculate_upside_capture_ratio(ma_rets, ma_bench, periods=12)
            metrics['DownCap'] = calculate_downside_capture_ratio(ma_rets, ma_bench, periods=12)
            
            # Active Return and Tracking Error
            ann_ret = (1 + metrics['GM_A']) - 1
            ann_bench = (1 + m_bench).prod()**(12/len(m_bench)) - 1 if len(m_bench) > 0 else 0.0
            metrics['ActiveRet'] = ann_ret - ann_bench
            
            active_month_rets = ma_rets - ma_bench
            metrics['TE'] = active_month_rets.std() * np.sqrt(12)
            
            # M2 Measure
            metrics['M2'] = metrics['Sharpe'] * metrics['SD_A'] + ann_bench
    except: pass

    # Withdrawal Rates
    try:
        swr_pwr = calculate_withdrawal_rates(returns)
        metrics['SWR'] = swr_pwr.get('SWR', 0.0)
        metrics['PWR'] = swr_pwr.get('PWR', 0.0)
    except: pass

    # Final sanitization Sweep (No NaN/Inf allowed in the dict)
    for k in metrics:
        v = metrics[k]
        if v is None or (isinstance(v, (float, np.float64, np.float32)) and (np.isnan(v) or np.isinf(v))):
            metrics[k] = 0.0

    return metrics


def render_risk_adjusted_returns(metrics_dict, df_results=None, benchmark_symbol="SPY"):
    """
    Render Risk and Return Metrics - Vertical Table Format (Portfolio Visualizer Style)
    Uses pre-calculated metrics_dict for Strategy and fresh calculations for B&H/Benchmark.
    """
    if df_results is None or len(df_results) < 2:
        st.warning("Insufficient data for Risk and Return Metrics table.")
        return

    # 1. Prepare returns for comparison columns
    bh_rets = df_results['BH'].pct_change().dropna()
    bench_rets = bh_rets
    if benchmark_symbol in df_results.columns:
        bench_rets = df_results[benchmark_symbol].pct_change().dropna()

    # Strategy metrics calculated monthly
    s_m = calculate_pv_metrics(df_results['ret'].dropna(), bench_rets, df_results['Strategy'])
    # B&H metrics calculated monthly
    b_m = calculate_pv_metrics(bh_rets, bench_rets, df_results['BH'])
    # Benchmark metrics calculated monthly
    m_m = calculate_pv_metrics(bench_rets, bench_rets, df_results[benchmark_symbol] if benchmark_symbol in df_results.columns else df_results['BH'])

    # Helper formatting functions
    def fmt(val, na='N/A'):
        if val is None or (isinstance(val, float) and (np.isinf(val) or np.isnan(val))):
            return na
        if isinstance(val, str):
            return val
        return f"{val:.2f}"

    def fmt_pct(val, na='N/A'):
        if val is None or (isinstance(val, float) and (np.isinf(val) or np.isnan(val))):
            return na
        if isinstance(val, str):
            return val
        return f"{val*100:.2f}%"

    def get_pos_str(m):
        pos = m.get('Pos', 0)
        total = m.get('Total', 0)
        pct = (pos / total * 100) if total > 0 else 0
        return f"{pos} out of {total} ({pct:.2f}%)"

    # Strategy metrics from metrics_dict (already formatted as strings in get_all_metrics)
    # We use .get() to avoid KeyErrors if some metrics are missing.
    s = metrics_dict

    metrics_rows = [
        ("Arithmetic Mean (monthly)", fmt_pct(s_m.get('AM_M')), fmt_pct(b_m.get('AM_M')), fmt_pct(m_m.get('AM_M')), "Average monthly return using simple addition."),
        ("Arithmetic Mean (annualized)", fmt_pct(s_m.get('AM_A')), fmt_pct(b_m.get('AM_A')), fmt_pct(m_m.get('AM_A')), "Monthly arithmetic mean scaled to a full year (12 months)."),
        ("Geometric Mean (monthly)", fmt_pct(s_m.get('GM_M')), fmt_pct(b_m.get('GM_M')), fmt_pct(m_m.get('GM_M')), "The compound monthly return that would result in the same final value."),
        ("Geometric Mean (annualized)", fmt_pct(s_m.get('GM_A')), fmt_pct(b_m.get('GM_A')), fmt_pct(m_m.get('GM_A')), "The compound annual growth rate (CAGR) of the portfolio."),
        ("Standard Deviation (monthly)", fmt_pct(s_m.get('SD_M')), fmt_pct(b_m.get('SD_M')), fmt_pct(m_m.get('SD_M')), "Monthly volatility of returns. Measures typical monthly deviation from the mean."),
        ("Standard Deviation (annualized)", fmt_pct(s_m.get('SD_A')), fmt_pct(b_m.get('SD_A')), fmt_pct(m_m.get('SD_A')), "Yearly volatility, calculated as monthly std dev times sqrt(12)."),
        ("Downside Deviation (monthly)", fmt_pct(s_m.get('DD_M')), fmt_pct(b_m.get('DD_M')), fmt_pct(m_m.get('DD_M')), "Volatility of negative returns only (MAR=0). Focuses on harmful risk."),
        ("Maximum Drawdown", fmt_pct(s_m.get('MDD')), fmt_pct(b_m.get('MDD')), fmt_pct(m_m.get('MDD')), "Peak-to-trough decline of the portfolio during the backtest period."),
        ("Benchmark Correlation", fmt(s_m.get('Corr')), fmt(b_m.get('Corr')), "1.00", "Measures how closely the portfolio mirrors the benchmark (1.0 = perfect match)."),
        ("Beta(*)", fmt(s_m.get('Beta')), fmt(b_m.get('Beta')), "1.00", "Systematic risk relative to the market/benchmark (>1.0 is more volatile than market)."),
        ("Alpha (annualized)", fmt_pct(s_m.get('Alpha')), fmt_pct(b_m.get('Alpha')), "0.00%", "The excess return generated above the benchmark's expected return."),
        ("R2", fmt_pct(s_m.get('R2')), fmt_pct(b_m.get('R2')), "100.00%", "Percentage of portfolio movements explained by the benchmark (R-Squared)."),
        ("Sharpe Ratio", fmt(s_m.get('Sharpe')), fmt(b_m.get('Sharpe')), fmt(m_m.get('Sharpe')), "Risk-adjusted return (Return / Volatility). Higher is better."),
        ("Sortino Ratio", fmt(s_m.get('Sortino')), fmt(b_m.get('Sortino')), fmt(m_m.get('Sortino')), "Risk-adjusted return using only downside risk (Return / Downside Deviation)."),
        ("Treynor Ratio (%)", fmt(s_m.get('Treynor', 0)*100), fmt(b_m.get('Treynor', 0)*100), fmt(m_m.get('Treynor', 0)*100), "Return per unit of systematic risk (Beta)."),
        ("Calmar Ratio", fmt(s_m.get('Calmar')), fmt(b_m.get('Calmar')), fmt(m_m.get('Calmar')), "Annualized return divided by the maximum drawdown."),
        ("Modigliani–Modigliani Measure", fmt_pct(s_m.get('M2')), fmt_pct(b_m.get('M2')), fmt_pct(m_m.get('M2')), "Risk-adjusted return compared to benchmark in percentage terms (M2)."),
        ("Active Return", fmt_pct(s_m.get('ActiveRet')), fmt_pct(b_m.get('ActiveRet')), "N/A", "The difference between strategy return and benchmark return."),
        ("Tracking Error", fmt_pct(s_m.get('TE')), fmt_pct(b_m.get('TE')), "N/A", "Standard deviation of active returns; measures drift from benchmark."),
        ("Information Ratio", fmt(s_m.get('IR')), fmt(b_m.get('IR')), "N/A", "Active return divided by tracking error. Measures manager skill."),
        ("Skewness", fmt(s_m.get('Skew')), fmt(b_m.get('Skew')), fmt(m_m.get('Skew')), "Measure of return asymmetry; negative means more frequent losses."),
        ("Excess Kurtosis", fmt(s_m.get('Excess Kurtosis', s_m.get('Kurt'))), fmt(b_m.get('Kurt')), fmt(m_m.get('Kurt')), "Measure of 'fat tails'; high values indicate more extreme events."),
        ("Historical Value-at-Risk (5%)", fmt_pct(abs(s_m.get('VaR_H', 0))), fmt_pct(abs(b_m.get('VaR_H', 0))), fmt_pct(abs(m_m.get('VaR_H', 0))), "Potential loss in the worst 5% of monthly outcomes based on history."),
        ("Analytical Value-at-Risk (5%)", fmt_pct(abs(s_m.get('VaR_A', 0))), fmt_pct(abs(b_m.get('VaR_A', 0))), fmt_pct(abs(m_m.get('VaR_A', 0))), "Potential loss based on a normal distribution of returns."),
        ("Conditional Value-at-Risk (5%)", fmt_pct(abs(s_m.get('CVaR', 0))), fmt_pct(abs(b_m.get('CVaR', 0))), fmt_pct(abs(m_m.get('CVaR', 0))), "Average loss in the worst 5% of outcomes (Expected Shortfall)."),
        ("Upside Capture Ratio (%)", fmt(s_m.get('UpCap')), fmt(b_m.get('UpCap')), "100.00", "Percentage of benchmark gains captured during positive months."),
        ("Downside Capture Ratio (%)", fmt(s_m.get('DownCap')), fmt(b_m.get('DownCap')), "100.00", "Percentage of benchmark losses captured during negative months."),
        ("Safe Withdrawal Rate", fmt_pct(s_m.get('SWR')), fmt_pct(b_m.get('SWR')), fmt_pct(m_m.get('SWR')), "Max % withdrawal to avoid portfolio depletion."),
        ("Perpetual Withdrawal Rate", fmt_pct(s_m.get('PWR')), fmt_pct(b_m.get('PWR')), fmt_pct(m_m.get('PWR')), "% withdrawal that keeps the principal balance constant."),
        ("Positive Periods", get_pos_str(s_m), get_pos_str(b_m), get_pos_str(m_m), "Count and percentage of months with a positive return."),
        ("Gain/Loss Ratio", fmt(s_m.get('GL')), fmt(b_m.get('GL')), fmt(m_m.get('GL')), "Average monthly gain divided by average monthly loss."),
    ]

    bench_name = benchmark_symbol
    if benchmark_symbol == "SPY": bench_name = "SPDR S&P 500 ETF (SPY)"
    elif benchmark_symbol == "QQQ": bench_name = "Invesco QQQ Trust (QQQ)"

    df_perf = pd.DataFrame(metrics_rows, columns=['Metric', 'Strategy', 'Buy & Hold Portfolio', bench_name, 'Metric Description'])

    # Calculate height to remove scrollbar (approx 35px per row + header)
    calc_height = (len(metrics_rows) * 35) + 45

    st.dataframe(
        apply_conditional_styling(df_perf), 
        use_container_width=True, 
        hide_index=True,
        height=calc_height
    )
    st.caption(f"* {benchmark_symbol} is used as the benchmark for calculations. Value-at-risk metrics are monthly values.")


def render_trade_statistics(df_results, trade_log, initial_capital):
    """Render Trade Statistics - PV Style"""
    
    if trade_log is None or len(trade_log) == 0:
        st.info("No trades to analyze.")
        return

    # Filter for closed trades
    closed_trades = [t for t in trade_log if t.get('Status') == 'CLOSED']
    total_trades = len(trade_log)
    analyzed_trades = len(closed_trades)
    
    # Rebalancing trades (placeholder)
    rebalancing_trades = 0
    
    # Win/Loss analysis using Profit % from trade entries
    wins = [t for t in closed_trades if t.get('Profit %', 0) > 0]
    losses = [t for t in closed_trades if t.get('Profit %', 0) < 0]
    winning_trades = len(wins)
    losing_trades = len(losses)
    win_pct = (winning_trades / analyzed_trades * 100) if analyzed_trades > 0 else 0
    
    # Win/Loss amounts (%)
    avg_win = np.mean([t['Profit %'] for t in wins]) if wins else 0
    med_win = np.median([t['Profit %'] for t in wins]) if wins else 0
    avg_loss = abs(np.mean([t['Profit %'] for t in losses])) if losses else 0
    med_loss = abs(np.median([t['Profit %'] for t in losses])) if losses else 0
    
    # Turnover calculation
    years = (df_results.index[-1] - df_results.index[0]).days / 365.25
    # Simplified assumption: each trade is a full reallocation
    total_turnover = analyzed_trades * 100 
    avg_annual_turnover = total_turnover / years if years > 0 else 0
    
    # Trading Returns (Cumulative)
    cum_ret_after = (df_results['Strategy'].iloc[-1] / initial_capital - 1) * 100
    
    # Estimated Cost Impact (assuming 0.05% per side = 0.1% per round trip if not tracked)
    # We'll use a conservative default since the base strategies don't track commissions yet
    cost_per_trade_pct = 0.05 
    cum_cost_impact = analyzed_trades * cost_per_trade_pct
    cum_ret_before = cum_ret_after + cum_cost_impact
    
    # Annualized Returns
    ann_ret_after = ((1 + cum_ret_after/100)**(1/years) - 1) * 100 if years > 0 and (1 + cum_ret_after/100) > 0 else 0
    ann_ret_before = ((1 + cum_ret_before/100)**(1/years) - 1) * 100 if years > 0 and (1 + cum_ret_before/100) > 0 else 0
    ann_cost_impact = ann_ret_before - ann_ret_after
    
    # --- 1. Trade Counts & Frequency ---
    counts_data = [
        ("Total Trades", f"{total_trades}"),
        ("Analyzed Trades", f"{analyzed_trades}"),
        ("Rebalancing Trades", f"{rebalancing_trades}"),
        ("Win %", f"{win_pct:.2f}%"),
        ("Avg Annual Turnover", f"{avg_annual_turnover:.2f}%"),
    ]
    df_counts = pd.DataFrame(counts_data, columns=['Metric', 'Value'])

    # --- 2. Win/Loss Analysis ---
    winloss_data = [
        ("Winning Trades", f"{winning_trades}"),
        ("Losing Trades", f"{losing_trades}"),
        ("Average Win", f"{avg_win:.2f}%"),
        ("Median Win", f"{med_win:.2f}%"),
        ("Average Loss", f"{avg_loss:.2f}%"),
        ("Median Loss", f"{med_loss:.2f}%"),
    ]
    df_winloss = pd.DataFrame(winloss_data, columns=['Metric', 'Value'])

    # --- 3. Return & Cost Analysis ---
    return_cost_data = [
        ("Cumulative Return (Before Costs)", f"{cum_ret_before:,.2f}%"),
        ("Cumulative Return (After Costs)", f"{cum_ret_after:,.2f}%"),
        ("Cumulative Cost Impact", f"{cum_cost_impact:.2f}%"),
        ("Annualized Return (Before Costs)", f"{ann_ret_before:.2f}%"),
        ("Annualized Return (After Costs)", f"{ann_ret_after:.2f}%"),
        ("Annualized Cost Impact", f"{ann_cost_impact:.2f}%"),
    ]
    df_return_cost = pd.DataFrame(return_cost_data, columns=['Metric', 'Value'])

    # Display in columns for space efficiency
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Trade Counts & Frequency**")
        st.dataframe(df_counts, use_container_width=True, hide_index=True)
        
    with col2:
        st.markdown("**Win/Loss Analysis**")
        st.dataframe(df_winloss, use_container_width=True, hide_index=True)
        
    with col3:
        st.markdown("**Return & Cost Analysis**")
        st.dataframe(df_return_cost, use_container_width=True, hide_index=True)


def render_trailing_returns(df_results, benchmark_symbol="SPY"):
    """Render Trailing Returns - Portfolio Visualizer style comparison table"""

    def get_trailing_metrics(series, returns_series, full_df):
        metrics = {}
        
        # Helper for total return
        def calc_total_ret(start_idx):
            if len(series) >= abs(start_idx):
                return (series.iloc[-1] / series.iloc[start_idx] - 1) * 100
            return None

        # Helper for annualized return
        def calc_ann_ret(total_ret_pct, days):
            if total_ret_pct is not None and days > 0:
                years = days / 252
                # total_ret_pct is already * 100, convert back for math
                total_ret = total_ret_pct / 100
                return ((1 + total_ret) ** (1/years) - 1) * 100
            return None

        # Helper for ann std dev
        def calc_ann_std(rets):
            if len(rets) > 1:
                return (rets.std() * np.sqrt(252)) * 100
            return None

        # Periods
        # 3 Month (~63 trading days)
        metrics['3 Month TR'] = calc_total_ret(-63)
        
        # Year To Date (precise: from last close of previous year)
        current_year = full_df.index[-1].year
        prev_year_data = series[series.index.year < current_year]
        if not prev_year_data.empty:
            base_p = prev_year_data.iloc[-1]
            metrics['YTD TR'] = (series.iloc[-1] / base_p - 1) * 100
        else:
            # If start of data is in current year, use first price
            first_p = series[series.index.year == current_year].iloc[0]
            metrics['YTD TR'] = (series.iloc[-1] / first_p - 1) * 100
            
        # 1 Year
        metrics['1 Year TR'] = calc_total_ret(-252)
        
        # 3 Year
        tr_3y = calc_total_ret(-252*3)
        metrics['3 Year Ann'] = calc_ann_ret(tr_3y, 252*3)
        
        # 5 Year
        tr_5y = calc_total_ret(-252*5)
        metrics['5 Year Ann'] = calc_ann_ret(tr_5y, 252*5)
        
        # Full
        total_days = (full_df.index[-1] - full_df.index[0]).days
        years = total_days / 365.25
        tr_full = series.iloc[-1] / series.iloc[0] - 1
        metrics['Full Ann'] = ((1 + tr_full)**(1/years) - 1) * 100 if years > 0 else None

        # Volatility metrics after all returns
        metrics['3 Year Std'] = calc_ann_std(returns_series.iloc[-252*3:]) if len(returns_series) >= 252*3 else None
        metrics['5 Year Std'] = calc_ann_std(returns_series.iloc[-252*5:]) if len(returns_series) >= 252*5 else None
        
        return metrics

    # Calculate for all three
    rows = []
    
    # Strategy
    strat_metrics = get_trailing_metrics(df_results['Strategy'], df_results['ret'], df_results)
    rows.append({
        'Name': 'Strategy',
        **strat_metrics
    })
    
    # Buy & Hold
    bh_rets = df_results['BH'].pct_change().dropna()
    bh_metrics = get_trailing_metrics(df_results['BH'], bh_rets, df_results)
    rows.append({
        'Name': 'Buy & Hold Portfolio',
        **bh_metrics
    })
    
    # Benchmark
    if benchmark_symbol in df_results.columns:
        bench_rets = df_results[benchmark_symbol].pct_change().dropna()
        bench_metrics = get_trailing_metrics(df_results[benchmark_symbol], bench_rets, df_results)
        rows.append({
            'Name': benchmark_symbol,
            **bench_metrics
        })
        
    trailing_df = pd.DataFrame(rows)
    
    # Configure numeric subset for Strategy
    strat_idx = trailing_df[trailing_df['Name'] == 'Strategy'].index
    numeric_cols = [c for c in trailing_df.columns if c != 'Name']
    
    # Symmetric scale for returns, centered at 0
    # Collect only return-based values for the global scale calculation
    ret_cols = [c for c in trailing_df.columns if 'TR' in c or 'Ann' in c]
    
    # Force numeric conversion for safety
    for col in ret_cols:
        trailing_df[col] = pd.to_numeric(trailing_df[col], errors='coerce')
        
    subset_data = trailing_df.loc[strat_idx, ret_cols]
    try:
        # Use numpy for safe vmax calculation
        valid_values = subset_data.values.flatten().astype(float)
        vmax = np.nanmax(np.abs(valid_values))
    except (ValueError, TypeError):
        vmax = 0.1
        
    if pd.isna(vmax) or vmax == 0: vmax = 0.1
    
    styler = trailing_df.style.hide(axis='index')
    
    # Apply heatmap only to Strategy returns
    styler = styler.background_gradient(
        cmap='RdYlGn', 
        subset=(strat_idx, ret_cols),
        vmin=-vmax, vmax=vmax,
        axis=None
    ).highlight_null(color=None)
    
    # Format all numeric columns
    styler = styler.format({c: "{:.2f}%" for c in numeric_cols}, na_rep="N/A")
    
    # Apply column config for headers and tooltips
    col_map = {
        '3 Month TR': 'Total Ret (3M)',
        'YTD TR': 'Total Ret (YTD)',
        '1 Year TR': 'Total Ret (1Y)',
        '3 Year Ann': 'Ann. Ret (3Y)',
        '5 Year Ann': 'Ann. Ret (5Y)',
        'Full Ann': 'Ann. Ret (Full)',
        '3 Year Std': 'Ann. Vol (3Y)',
        '5 Year Std': 'Ann. Vol (5Y)'
    }
    
    column_config = {
        "Name": st.column_config.TextColumn("Portfolio", help="Portfolio name."),
    }
    for k, v in col_map.items():
        if k in trailing_df.columns:
            column_config[k] = st.column_config.TextColumn(v, help=METRIC_DEFINITIONS.get(k, v))

    st.dataframe(styler, use_container_width=True, column_config=column_config)
    try:
        last_date = df_results.index[-1]
        date_str = last_date.strftime('%B %Y') if hasattr(last_date, 'strftime') else str(last_date)[:7]
        st.caption(f"Trailing return and volatility are as of {date_str}")
    except:
        pass



def render_drawdown_analysis(df_results):
    """Render Drawdown Analysis - Portfolio Visualizer Style"""
    
    equity = df_results['Strategy']
    rolling_max = equity.expanding().max()
    drawdowns = (equity - rolling_max) / rolling_max
    
    dd_periods = []
    in_dd = False
    current_dd = {}
    
    equity_list = equity.tolist()
    rolling_max_list = rolling_max.tolist()
    drawdowns_list = drawdowns.tolist()
    dates = equity.index.tolist()
    
    dd_threshold = -0.05
    
    for i, dd in enumerate(drawdowns_list):
        date = dates[i]
        
        if dd < dd_threshold and not in_dd:
            in_dd = True
            peak_value = rolling_max_list[i]
            peak_date = dates[i] # Initialize with current date
            for j in range(i - 1, -1, -1):
                if rolling_max_list[j] == peak_value:
                    peak_date = dates[j] # Move peak_date back to the actual peak start
                else:
                    break # Stop when the value is no longer the peak
            
            current_dd = {
                'Start': peak_date.strftime('%Y-%m') if hasattr(peak_date, 'strftime') else str(peak_date)[:7],
                'Trough': date.strftime('%Y-%m') if hasattr(date, 'strftime') else str(date)[:7],
                'Length': 0,
                'Recovery': 'In Progress',
                'Recovery By': 'N/A',
                'Drawdown': dd * 100
            }
        elif in_dd:
            if dd * 100 < current_dd['Drawdown']:
                current_dd['Trough'] = date.strftime('%Y-%m') if hasattr(date, 'strftime') else str(date)[:7]
                current_dd['Drawdown'] = dd * 100
            
            if dd >= 0:
                end_date = date
                recovery_days = (pd.Timestamp(end_date) - pd.Timestamp(current_dd['Trough'])).days
                recovery_months = recovery_days // 30
                current_dd['Recovery'] = f"{recovery_months} months"
                current_dd['Recovery By'] = end_date.strftime('%Y-%m') if hasattr(end_date, 'strftime') else str(end_date)[:7]
                
                length_days = (pd.Timestamp(current_dd['Trough']) - pd.Timestamp(current_dd['Start'])).days
                current_dd['Length'] = length_days // 30 + 1
                dd_periods.append(current_dd.copy())
                in_dd = False
    
    # Flush current drawdown if still active
    if in_dd:
        dd_periods.append(current_dd)

    # Sort and reorder columns
    if dd_periods:
        dd_df = pd.DataFrame(dd_periods)
        
        # Sort by depth (most negative first)
        dd_df = dd_df.sort_values('Drawdown', ascending=True)
        
        # Reorder columns for logical flow
        display_order = ['Start', 'Trough', 'Recovery By', 'Length', 'Recovery', 'Drawdown']
        display_order = [c for c in display_order if c in dd_df.columns]
        dd_df = dd_df[display_order]
        
        # Add tooltip-enabled headers
        column_config = {
            "Drawdown": st.column_config.TextColumn("Depth", help=METRIC_DEFINITIONS["Max Drawdown"]),
            "Recovery": st.column_config.TextColumn("Recovery Time", help="Time taken to return from trough to the previous peak."),
            "Recovery By": st.column_config.TextColumn("Recovery By", help="The month when the portfolio value returned to its previous peak."),
            "Length": st.column_config.TextColumn("Decline Length", help="Total months from peak to trough."),
        }
        
        try:
            dd_vals = pd.to_numeric(dd_df['Drawdown'], errors='coerce').dropna().values
            vmax = np.nanmax(np.abs(dd_vals)) if len(dd_vals) > 0 else 10.0
        except:
            vmax = 10.0
            
        if pd.isna(vmax) or vmax == 0: vmax = 10.0
        
        styler = dd_df.style.hide(axis='index')
        styler = styler.background_gradient(
            cmap='RdYlGn', subset=['Drawdown'],  # Lower is worse (Red), Zero is better (Green)
            vmin=-vmax, vmax=0
        ).highlight_null(color=None)
        
        styler = styler.format({'Drawdown': "{:.2f}%"}, na_rep="N/A")
        
        st.dataframe(
            styler,
            use_container_width=True,
            column_config=column_config,
            hide_index=True
        )
    else:
        st.info("No significant drawdowns found.")
    
    # Drawdown chart
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.fill_between(drawdowns.index, drawdowns * 100, 0, color=PV_COLORS['Drawdown'], alpha=0.3)
    ax.plot(drawdowns.index, drawdowns * 100, color=PV_COLORS['Drawdown'], linewidth=1)
    ax.set_ylabel('Drawdown (%)', color=PV_COLORS['Text'])
    ax.set_title('Underwater Chart (Drawdowns)', fontsize=12, color=PV_COLORS['Text'])
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def render_rolling_returns_analysis(df_results, benchmark_symbol="SPY"):
    """Render Rolling Returns Analysis - Portfolio Visualizer Style"""
    
    def get_rolling_stats(series, window_years):
        window = window_years * 252
        if len(series) < window:
            return None, None, None
        
        # Calculate annualized rolling returns
        # (Price_t / Price_{t-window})^(1/years) - 1
        rolling_ret = (series / series.shift(window))**(1/window_years) - 1
        rolling_ret = rolling_ret.dropna() * 100
        
        if rolling_ret.empty:
            return None, None, None
            
        return rolling_ret.mean(), rolling_ret.max(), rolling_ret.min()

    periods = [1, 3, 5, 7]
    names = ["Strategy", "Buy & Hold Portfolio", benchmark_symbol]
    columns = ["Strategy", "BH", benchmark_symbol]
    
    # Check which columns exist
    active_cols = [c for c in columns if c in df_results.columns]
    active_names = [names[columns.index(c)] for c in active_cols]
    
    # Build MultiIndex for columns
    iterables = [active_names, ["Average", "High", "Low"]]
    idx = pd.MultiIndex.from_product(iterables)
    
    table_rows = []
    for p in periods:
        row = []
        for col in active_cols:
            avg, high, low = get_rolling_stats(df_results[col], p)
            row.extend([avg, high, low])
        table_rows.append(row)
        
    df_roll = pd.DataFrame(table_rows, index=[f"{p} year{'s' if p>1 else ''}" for p in periods], columns=idx)
    df_roll.index.name = "Roll Period"
    
    # Heatmap only for Strategy Average/High/Low (the first group)
    vmax = df_roll["Strategy"].abs().max().max()
    if pd.isna(vmax) or vmax == 0: vmax = 10.0 # fallback
    
    styler_roll = df_roll.style.background_gradient(
        cmap='RdYlGn', subset=[("Strategy", "Average"), ("Strategy", "High"), ("Strategy", "Low")],
        vmin=-vmax, vmax=vmax, axis=None
    ).highlight_null(color=None).format("{:.2f}%", na_rep="N/A")
    
    st.dataframe(styler_roll, use_container_width=True)

    # Add Annualized Rolling Returns Table for each Year
    st.subheader("Annualized Rolling Returns by Year")
    
    def get_annual_roll(series, window_years):
        window = window_years * 252
        if len(series) < window: return pd.Series(dtype=float)
        roll = (series / series.shift(window))**(1/window_years) - 1
        return roll.resample('YE').last() * 100

    active_p_cols = ["Strategy", "BH", benchmark_symbol]
    active_p_cols = [c for c in active_p_cols if c in df_results.columns]
    active_p_names = ["Strategy", "B&H", benchmark_symbol]
    
    roll_data_list = []
    for years in [1, 3, 5]:
        for col, name in zip(active_p_cols, active_p_names):
            roll_ser = get_annual_roll(df_results[col], years)
            for dt, val in roll_ser.items():
                roll_data_list.append({
                    'Year': dt.year,
                    'Period': f"{years}Y",
                    'Portfolio': name,
                    'Return': val
                })
    
    if roll_data_list:
        df_long = pd.DataFrame(roll_data_list)
        df_ann_roll = df_long.pivot(index='Year', columns=['Period', 'Portfolio'], values='Return')

        p_order = ["1Y", "3Y", "5Y"]
        sorted_p_cols = sorted(df_ann_roll.columns, key=lambda x: (p_order.index(x[0]), active_p_names.index(x[1]) if x[1] in active_p_names else 99))
        df_ann_roll = df_ann_roll[sorted_p_cols]

        # Insert Diff (Strategy - B&H) column immediately after Strategy for each period
        diff_cols = []
        final_col_order = []
        for period in p_order:
            # Add Strategy first
            if (period, "Strategy") in df_ann_roll.columns:
                final_col_order.append((period, "Strategy"))
            # Add Diff column right after Strategy if Benchmark exists
            # The name used in active_p_names for the benchmark is the benchmark_symbol itself
            if (period, "Strategy") in df_ann_roll.columns and (period, benchmark_symbol) in df_ann_roll.columns:
                diff_col = (period, f"vs {benchmark_symbol}")
                df_ann_roll[diff_col] = df_ann_roll[(period, "Strategy")] - df_ann_roll[(period, benchmark_symbol)]
                final_col_order.append(diff_col)
                diff_cols.append(diff_col)
            # Add remaining columns for this period (B&H, benchmark)
            for col in sorted_p_cols:
                if col[0] == period and col[1] != "Strategy" and col not in final_col_order:
                    final_col_order.append(col)

        df_ann_roll = df_ann_roll[[c for c in final_col_order if c in df_ann_roll.columns]]
        df_ann_roll = df_ann_roll.sort_index(ascending=False)
        # Convert index to string to avoid numeric formatting (e.g., 2,024)
        df_ann_roll.index = df_ann_roll.index.map(str)
        df_ann_roll.index.name = "Year"

        # Flatten MultiIndex to string names for Streamlit compatibility (fixes JSON serialization of tuples)
        flattened_cols = [f"{c[0]} | {c[1]}" for c in df_ann_roll.columns]
        
        # Identify strategy and diff columns in flattened format for styling
        strat_cols_str = [f"{c[0]} | {c[1]}" for c in df_ann_roll.columns if c[1] == "Strategy"]
        diff_cols_str = [f"{c[0]} | {c[1]}" for c in df_ann_roll.columns if c[1].startswith("vs ")]
        
        # Create a version with flattened columns for display
        df_display = df_ann_roll.copy()
        df_display.columns = flattened_cols

        vmax_ann = df_display[strat_cols_str].abs().max().max()
        if pd.isna(vmax_ann) or vmax_ann == 0: vmax_ann = 15.0

        diff_vmax_ann = df_display[diff_cols_str].abs().max().max() if diff_cols_str else 0
        if pd.isna(diff_vmax_ann) or diff_vmax_ann == 0: diff_vmax_ann = 10.0

        styler_ann = df_display.style.background_gradient(
            cmap='RdYlGn', subset=strat_cols_str,
            vmin=-vmax_ann, vmax=vmax_ann, axis=None
        )

        if diff_cols_str:
            styler_ann = styler_ann.background_gradient(
                cmap='RdYlGn', subset=diff_cols_str,
                vmin=-diff_vmax_ann, vmax=diff_vmax_ann, axis=None
            )

        styler_ann = styler_ann.format("{:+.2f}%", subset=diff_cols_str, na_rep="None")
        styler_ann = styler_ann.format("{:.2f}%", subset=[c for c in flattened_cols if c not in diff_cols_str], na_rep="None")
        styler_ann = styler_ann.highlight_null(props='background-color: #f8f8f8; color: #bbbbbb;')

        # Prepare column config using the flattened string names
        roll_column_config = {
            col: st.column_config.NumberColumn(
                col, # Use the string name as the label
                format="%+.2f%%" if col in diff_cols_str else "%.2f%%"
            ) for col in flattened_cols
        }

        st.dataframe(styler_ann, use_container_width=True, column_config=roll_column_config)
    
    # Rolling Return Chart (3 Years)
    window_3y = 3 * 252
    if len(df_results) > window_3y:
        st.subheader("Annualized Rolling Return - 3 Years")
        fig, ax = plt.subplots(figsize=(12, 6))
        
        colors = ["#1f4e9c", "#27ae60", "#7fb3d3"]
        for i, col in enumerate(active_cols):
            rolling_3y = (df_results[col] / df_results[col].shift(window_3y))**(1/3) - 1
            rolling_3y = rolling_3y.dropna() * 100
            ax.plot(rolling_3y.index, rolling_3y, label=active_names[i], color=colors[i % len(colors)], linewidth=1.5)
            
        ax.axhline(0, color='black', linewidth=0.8, alpha=0.5)
        ax.set_ylabel("Annualized Return (%)")
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=len(active_cols), frameon=False)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # Rolling Return Chart (5 Years)
    window_5y = 5 * 252
    if len(df_results) > window_5y:
        st.subheader("Annualized Rolling Return - 5 Years")
        fig, ax = plt.subplots(figsize=(12, 6))
        
        colors = ["#1f4e9c", "#27ae60", "#7fb3d3"]
        for i, col in enumerate(active_cols):
            rolling_5y = (df_results[col] / df_results[col].shift(window_5y))**(1/5) - 1
            rolling_5y = rolling_5y.dropna() * 100
            ax.plot(rolling_5y.index, rolling_5y, label=active_names[i], color=colors[i % len(colors)], linewidth=1.5)
            
        ax.axhline(0, color='black', linewidth=0.8, alpha=0.5)
        ax.set_ylabel("Annualized Return (%)")
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=len(active_cols), frameon=False)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)



def render_monthly_returns_heatmap(df_results):
    """Render Monthly Returns Heatmap"""
    
    monthly_returns = df_results['Strategy'].resample('ME').last().pct_change().dropna() * 100
    
    if len(monthly_returns) == 0:
        st.warning("Not enough data for monthly returns")
        return
    
    monthly_df = pd.DataFrame({
        'Year': monthly_returns.index.year,
        'Month': monthly_returns.index.month,
        'Return': monthly_returns.values
    })
    
    monthly_pivot = monthly_df.pivot(index='Year', columns='Month', values='Return')
    monthly_pivot = monthly_pivot.sort_index(ascending=False)
    # Convert index to string to avoid numeric formatting (e.g., 2,024)
    monthly_pivot.index = monthly_pivot.index.map(str)
    monthly_pivot.columns = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][:len(monthly_pivot.columns)]
    
    # Calculate max absolute value for symmetric color scale centered at 0
    try:
        # unstack() to 1D, drop NaNs, then compute abs max
        vals = monthly_pivot.unstack().dropna().values.astype(float)
        vmax = np.nanmax(np.abs(vals)) if len(vals) > 0 else 1.0
    except:
        vmax = 1.0
        
    if pd.isna(vmax) or vmax == 0:
        vmax = 1.0
        
    st.dataframe(
        monthly_pivot.style
        .format('{:.2f}%', na_rep='')
        .background_gradient(cmap='RdYlGn', axis=None, vmin=-vmax, vmax=vmax)
        .highlight_null(color=None),
        use_container_width=True
    )
    
    # Monthly Stats Table
    stats_data = {
        'Metric': ['Best Month', 'Worst Month', 'Median Month', 'Positive Months', 'Negative Months'],
        'Value': [
            f"{monthly_returns.max():.2f}%",
            f"{monthly_returns.min():.2f}%",
            f"{monthly_returns.median():.2f}%",
            f"{(monthly_returns > 0).sum()}",
            f"{(monthly_returns < 0).sum()}"
        ]
    }
    st.table(pd.DataFrame(stats_data))


def render_monthly_cash_heatmap(df_results, cash_floor_pct=20.0):
    """Render Monthly Cash % Heatmap"""
    
    if 'Cash' not in df_results.columns:
        return
        
    cash_pct = (df_results['Cash'] / df_results['Strategy']) * 100
    monthly_cash = cash_pct.resample('ME').last()
    
    if len(monthly_cash) == 0:
        return
        
    monthly_df = pd.DataFrame({
        'Year': monthly_cash.index.year,
        'Month': monthly_cash.index.month,
        'CashPct': monthly_cash.values
    })
    
    monthly_pivot = monthly_df.pivot(index='Year', columns='Month', values='CashPct')
    monthly_pivot = monthly_pivot.sort_index(ascending=False)
    monthly_pivot.index = monthly_pivot.index.map(str)
    
    # Ensure columns exist for labeling
    col_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    monthly_pivot.columns = col_labels[:len(monthly_pivot.columns)]
    
    # Dynamic coloring: Yellow pivot at cash_floor_pct
    # Formula for RdYlGn: pivot at 0.5 center
    # vmin = 2 * pivot - 100, vmax = 100
    pivot = float(cash_floor_pct)
    vmin = 2 * pivot - 100
    
    st.dataframe(
        monthly_pivot.style
        .format('{:.1f}%', na_rep='')
        .background_gradient(cmap='RdYlGn', axis=None, vmin=vmin, vmax=100)
        .highlight_null(color=None),
        use_container_width=True
    )

def render_portfolio_hwm_heatmap(df_results):
    """Render Monthly Portfolio vs HWM vs Cash Heatmap"""
    
    if not all(col in df_results.columns for col in ['Strategy', 'HWM', 'Cash']):
        return
        
    # Get month-end values
    monthly = df_results[['Strategy', 'HWM', 'Cash']].resample('ME').last()
    
    if len(monthly) == 0:
        return
        
    # Prepare DataFrame for pivot
    monthly_df = pd.DataFrame({
        'Year': monthly.index.year,
        'Month': monthly.index.month,
        'Strategy': monthly['Strategy'],
        'HWM': monthly['HWM'],
        'Cash': monthly['Cash']
    })
    
    # 1. Create a "Label" pivot with formatted strings
    p_labels = monthly_df.pivot(index='Year', columns='Month', values='Strategy').sort_index(ascending=False)
    p_hwm = monthly_df.pivot(index='Year', columns='Month', values='HWM').sort_index(ascending=False)
    p_cash = monthly_df.pivot(index='Year', columns='Month', values='Cash').sort_index(ascending=False)
    
    # 2. Coloring logic: Based on logarithmic absolute HWM value to handle compounding
    log_hwm = np.log10(p_hwm.replace(0, 1)) # Avoid log(0)
    vmin_log = log_hwm.min().min()
    vmax_log = log_hwm.max().max()
    
    # Titles
    col_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    # 3. Formatter for the dataframe
    def make_cell_label(row_idx, col_idx):
        try:
            val = p_labels.iloc[row_idx, col_idx]
            hwm = p_hwm.iloc[row_idx, col_idx]
            csh = p_cash.iloc[row_idx, col_idx]
            if pd.isna(val): return ""
            return f"{val/1000:.0f}K | H:{hwm/1000:.0f}K | C:{csh/1000:.0f}K"
        except: return ""

    # Re-index display df
    p_final_labels = pd.DataFrame(index=p_labels.index, columns=p_labels.columns)
    for r in range(len(p_labels)):
        for c in range(len(p_labels.columns)):
            p_final_labels.iloc[r, c] = make_cell_label(r, c)
            
    p_final_labels.columns = col_labels[:len(p_final_labels.columns)]
    p_final_labels.index = p_final_labels.index.map(str)

    def get_hwm_color(val, vmin, vmax):
        if pd.isna(val) or vmax == vmin: return "transparent"
        # Log space value
        log_val = np.log10(max(val, 1))
        # Normalized position 0-1 in log space
        pos = (log_val - vmin) / (vmax - vmin)
        
        # RdYlGn roughly:
        if pos < 0.5:
            # Red to Yellow
            r = 220 + (255 - 220) * (pos * 2)
            g = 53 + (193 - 53) * (pos * 2)
            b = 69 + (7 - 69) * (pos * 2)
        else:
            # Yellow to Green
            r = 255 + (40 - 255) * ((pos - 0.5) * 2)
            g = 193 + (167 - 193) * ((pos - 0.5) * 2)
            b = 7 + (69 - 7) * ((pos - 0.5) * 2)
            
        return f"rgb({int(r)}, {int(g)}, {int(b)})"

    st.dataframe(
        p_final_labels.style
        .apply(lambda x: [f"background-color: {get_hwm_color(v, vmin_log, vmax_log)}; color: black" for v in p_hwm.loc[int(x.name)]], axis=1)
    , use_container_width=True)


def render_trade_log(trade_log_results):
    """Render Trade Log"""
    trade_log_df = pd.DataFrame(trade_log_results)
    
    if not trade_log_df.empty:
        def safe_format_curr(val):
            if isinstance(val, (int, float)) and pd.notna(val):
                return f'${val:,.2f}'
            return str(val)

        def safe_format_pct(val):
            if isinstance(val, (int, float)) and pd.notna(val):
                return f'{val:.2f}%'
            return str(val)

        for col in ['Entry Price', 'Exit Price', 'Amt']:
            if col in trade_log_df.columns:
                trade_log_df[col] = trade_log_df[col].apply(safe_format_curr)
        
        if 'Entry Equity' in trade_log_df.columns:
            trade_log_df['Entry Equity'] = trade_log_df['Entry Equity'].apply(safe_format_curr)

        if 'Profit %' in trade_log_df.columns:
            trade_log_df['Profit %'] = trade_log_df['Profit %'].apply(safe_format_pct)

        def color_profit_pct(val):
            if isinstance(val, str) and val.endswith('%'):
                try:
                    val_float = float(val.replace('%', ''))
                    if val_float > 0:
                        return 'background-color: #28a745; color: white'
                    elif val_float < 0:
                        return 'background-color: #dc3545; color: white'
                except:
                    pass
            return ''

        # Prepare styling
        styler = trade_log_df.style
            
        if 'Profit %' in trade_log_df.columns:
            styler = styler.map(color_profit_pct, subset=['Profit %'])
            
        st.dataframe(styler, use_container_width=True)
    else:
        st.subheader("No trades recorded.")

    # Status Glossary Expander
    with st.expander("📖 Trade Status Glossary", expanded=False):
        st.markdown("Explanation of status codes found in the Transaction History:")
        cols = st.columns(2)
        items = list(TRADE_STATUS_DEFINITIONS.items())
        mid = (len(items) + 1) // 2
        
        with cols[0]:
            for k, v in items[:mid]:
                st.markdown(f"**{k}**: {v}")
        with cols[1]:
            for k, v in items[mid:]:
                st.markdown(f"**{k}**: {v}")


def render_monte_carlo_simulation(mc_results, mc_sim_df, initial_capital, num_simulations):
    """Render Monte Carlo Simulation"""
    st.subheader("Monte Carlo Probability Distribution")
    
    if mc_results.empty:
        st.warning("Unable to run Monte Carlo simulation.")
        return
    
    # Combined Percentile Table
    ps = [0.95, 0.90, 0.75, 0.50, 0.25, 0.10, 0.05]
    p_labels = ["Best Case (95%)", "Top 10% (90%)", "Top 25% (75%)", "Median (50%)", "Bottom 25% (25%)", "Bottom 10% (10%)", "Worst Case (5%)"]
    
    p_data = []
    for p, label in zip(ps, p_labels):
        final_val = float(mc_sim_df.iloc[-1].quantile(p))
        cagr_val = float(mc_results['CAGR'].quantile(p)) * 100
        mdd_val = float(mc_results['Max Drawdown'].quantile(1-p)) * 100 # lower p is worse drawdown
        p_data.append({
            'Percentile': label,
            'Final Balance': final_val,
            'CAGR': cagr_val,
            'Max Drawdown': mdd_val
        })
    
    df_mc = pd.DataFrame(p_data)
    
    # Apply standard styling
    styled_mc = apply_conditional_styling(df_mc)
    
    # Custom tweaks for Monte Carlo (format the columns correctly)
    column_config = {
        "Percentile": st.column_config.TextColumn("Percentile"),
        "Final Balance": st.column_config.NumberColumn("Final Balance", format="$%,.0f"),
        "CAGR": st.column_config.NumberColumn("CAGR", format="%.2f%%"),
        "Max Drawdown": st.column_config.NumberColumn("Max Drawdown", format="%.2f%%"),
    }
    
    # Style the CAGR and Drawdown specifically since they are central to MC outcomes
    # We use apply_conditional_styling which already has the industry bands logic
    st.dataframe(
        styled_mc.format({
            'Final Balance': "${:,.0f}",
            'CAGR': "{:.2f}%",
            'Max Drawdown': "{:.2f}%"
        }), 
        use_container_width=True, 
        hide_index=True, 
        column_config=column_config
    )
    
    fig, ax = plt.subplots(figsize=(12, 6))
    for i in range(min(50, mc_sim_df.shape[1])):
        ax.plot(mc_sim_df.index, mc_sim_df.iloc[:, i], alpha=0.1, color='blue')
    ax.plot(mc_sim_df.index, mc_sim_df.quantile(0.5, axis=1), color='blue', linewidth=2, label='Median')
    ax.plot(mc_sim_df.index, mc_sim_df.quantile(0.1, axis=1), color='red', linewidth=2, label='10th Percentile')
    ax.plot(mc_sim_df.index, mc_sim_df.quantile(0.9, axis=1), color='green', linewidth=2, label='90th Percentile')
    ax.set_title(f'Monte Carlo Simulation ({num_simulations} runs)')
    ax.set_xlabel('Days')
    ax.set_ylabel('Portfolio Value ($)')
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'${x:,.0f}'))
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)


def render_survival_discipline(df_results, trade_list_results, trade_log_results):
    """Render Survival & Discipline"""
    st.subheader("Survival & Discipline")
    
    total_days = len(df_results)
    df_results['Cash_Pct_of_Strategy'] = df_results['Cash'] / df_results['Strategy']
    days_in_market = (df_results['Cash'] < df_results['Strategy']).sum()
    time_in_market = (days_in_market / total_days) * 100 if total_days > 0 else 0
    
    avg_cash_pct = df_results['Cash_Pct_of_Strategy'].mean() * 100 if not df_results['Cash_Pct_of_Strategy'].empty else 0
    max_cash_pct = df_results['Cash_Pct_of_Strategy'].max() * 100 if not df_results['Cash_Pct_of_Strategy'].empty else 0
    
    years = (df_results.index[-1] - df_results.index[0]).days / 365.25
    trades_per_year = len(trade_list_results) / years if years > 0 else 0
    
    if trade_log_results is not None and len(trade_log_results) > 0:
        durations = []
        for trade in trade_log_results:
            if 'Entry Date' in trade and 'Exit Date' in trade:
                try:
                    entry = pd.to_datetime(trade['Entry Date'])
                    exit_dt = pd.to_datetime(trade['Exit Date'])
                    durations.append((exit_dt - entry).days)
                except:
                    pass
        avg_trade_duration = np.mean(durations) if durations else 0
    else:
        avg_trade_duration = 0
    
    survival_df = pd.DataFrame({
        'Strategy': [
            time_in_market,
            avg_cash_pct,
            max_cash_pct,
            trades_per_year,
            avg_trade_duration
        ],
        'Benchmark (BH)': [100.0, 0.0, 0.0, 1.0, 365.0]
    }, index=['Time in Market', 'Average Cash %', 'Maximum Cash %', 'Trades per Year', 'Avg Trade Duration'])

    # Transpose survival
    survival_t = survival_df.T.reset_index().rename(columns={'index': 'Strategy Name'})
    
    st.dataframe(
        apply_transposed_metrics_styling(survival_t), 
        use_container_width=True, 
        width='stretch',
        column_config={
            "Strategy Name": st.column_config.TextColumn("Portfolio"),
            "Time in Market": st.column_config.NumberColumn("Time in Market", help=METRIC_DEFINITIONS["Time in Market"], format="%.1f%%"),
            "Average Cash %": st.column_config.NumberColumn("Avg Cash", format="%.1f%%"),
            "Maximum Cash %": st.column_config.NumberColumn("Max Cash", format="%.1f%%"),
            "Trades per Year": st.column_config.NumberColumn("Trade Freq", help="Annual frequency of trading activity.", format="%.1f"),
            "Avg Trade Duration": st.column_config.NumberColumn("Avg Duration", help="Average number of days a trade was held.", format="%.0f days"),
        },
        hide_index=True,
        height=(len(survival_t) + 1) * 35 + 2
    )


def render_stability_check(trade_list_results, df_results):
    """Render Stability Check"""
    st.subheader("Stability Check")
    
    sample_size = len(trade_list_results)
    
    if len(trade_list_results) > 1:
        cv = np.std(trade_list_results) / np.mean(trade_list_results) if np.mean(trade_list_results) != 0 else float('inf')
    else:
        cv = float('inf')
    
    if len(trade_list_results) >= 4:
        skewness = pd.Series(trade_list_results).skew()
        kurtosis = pd.Series(trade_list_results).kurtosis()
    else:
        skewness = 0
        kurtosis = 0
    
    rolling_vol = df_results['ret'].rolling(20).std() * np.sqrt(252)
    vol_consistency = rolling_vol.std()
    
    # Record metrics as raw numbers for styling
    stability_df = pd.DataFrame({
        'Strategy': [
            sample_size,
            abs(cv) if cv != float('inf') else 0.0,
            skewness,
            kurtosis,
            vol_consistency
        ],
        'Benchmark (BH)': [1, 0.0, 0.0, 0.0, 0.0]
    }, index=['Number of Trades', 'CV (Return Volatility)', 'Skewness', 'Kurtosis', 'Vol Consistency'])
    

    stability_t = stability_df.T.reset_index().rename(columns={'index': 'Strategy Name'})

    st.dataframe(
        apply_transposed_metrics_styling(stability_t), 
        use_container_width=True, 
        width='stretch',
        column_config={
            "Strategy Name": st.column_config.TextColumn("Portfolio"),
            "Number of Trades": st.column_config.NumberColumn("Trades", help="Sample size for statistical analysis.", format="%d"),
            "CV (Return Volatility)": st.column_config.NumberColumn("CV", help="Coefficient of Variation: standardized measure of return dispersion.", format="%.2f"),
            "Skewness": st.column_config.NumberColumn("Skewness", help="Measures return asymmetry (longer left vs right tail).", format="%.2f"),
            "Kurtosis": st.column_config.NumberColumn("Kurtosis", help="Measures tail risk.", format="%.2f"),
            "Vol Consistency": st.column_config.NumberColumn("Vol Consistency", help="Standard deviation of rolling volatility. Lower indicates stable risk.", format="%.2f"),
        },
        hide_index=True,
        height=(len(stability_t) + 1) * 35 + 2
    )


def render_equity_curve_plot(df_results, symbol, benchmark_symbol="SPY"):
    """Render the equity curve plot"""
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(df_results.index, df_results["Strategy"], label="Strategy Equity", color="blue", lw=2)
    ax.plot(df_results.index, df_results["BH"], label="Buy & Hold", color="gray", alpha=0.4)
    # Add benchmark if available
    if benchmark_symbol in df_results.columns:
        ax.plot(df_results.index, df_results[benchmark_symbol], label=f"{benchmark_symbol} (Benchmark)", color="orange", alpha=0.6, linewidth=2)
    ax.set_yscale("log")
    ax.set_title(f"Equity Curve: {symbol}")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    st.pyplot(fig)


def render_portfolio_growth_chart(df_results, benchmark_symbol="SPY"):
    """Render Portfolio Growth Chart"""
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df_results.index, df_results["Strategy"], label="Strategy", color=PV_COLORS['Strategy'], linewidth=2.5, zorder=10)
    ax.plot(df_results.index, df_results["BH"], label="Buy & Hold", color=PV_COLORS['BH'], alpha=0.8, linewidth=2)
    # Add benchmark if available
    if benchmark_symbol in df_results.columns:
        ax.plot(df_results.index, df_results[benchmark_symbol], label=f"{benchmark_symbol} (Benchmark)", color=PV_COLORS['Benchmark'], alpha=1.0, linewidth=2)
    
    # Add Cash line if available
    if "Cash" in df_results.columns:
        ax.plot(df_results.index, df_results["Cash"], label="Cash Balance", color=PV_COLORS['Cash'], alpha=0.8, linewidth=1.5, linestyle='--')

    # Add Stress Periods as shaded regions
    for name, start, end in STRESS_PERIODS:
        s_dt = pd.Timestamp(start)
        e_dt = pd.Timestamp(end)
        if s_dt >= df_results.index[0] and e_dt <= df_results.index[-1]:
            ax.axvspan(s_dt, e_dt, color='gray', alpha=0.1, zorder=0)
            # Add text label at the top of the span
            ax.text(s_dt + (e_dt - s_dt)/2, ax.get_ylim()[1], name, 
                    rotation=90, verticalalignment='top', horizontalalignment='center', 
                    fontsize=8, alpha=0.5, color='gray')

    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'${x:,.0f}'))
    ax.set_ylabel("Portfolio Value ($)", color=PV_COLORS['Text'])
    ax.set_title("Portfolio Growth (Log Scale)", fontsize=14, pad=15, color=PV_COLORS['Text'])
    ax.legend(loc='upper left')
    ax.grid(True, which="both", alpha=0.2)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def render_interactive_growth_chart(df_results, symbol, benchmark_symbol="SPY"):
    """
    Render an interactive Plotly chart with equity curves and strategy-specific signal factors.
    Uses explicit data slicing for perfect Y-axis auto-scaling.
    """
    # UI Controls for the chart view
    view_period = st.segmented_control(
        "View Window",
        options=["1Y", "3Y", "5Y", "10Y", "MAX"],
        default="MAX",
        key=f"view_period_{symbol}",
        label_visibility="collapsed"
    )

    # Slice data based on the selected period. Data slicing is the ONLY reliable way
    # to get Plotly to auto-scale the Y-axis correctly for presets (1Y, 3Y, etc.).
    # The user can pan/zoom within the sliced window; switching a preset rebuilds chart.
    if view_period != "MAX":
        try:
            years = int(view_period.replace("Y", ""))
            cutoff = df_results.index[-1] - pd.DateOffset(years=years)
            df_plot = df_results[df_results.index >= cutoff].copy()
        except:
            df_plot = df_results.copy()
    else:
        df_plot = df_results.copy()

    # Identify technical factors that should be in a separate lower panel (e.g., ROC, Vol, Breadth)
    lower_panel_candidates = ['Vol14', 'Vol200', 'Growth_ROC', 'Defensive_ROC', 'highlowqag', 'Signal']
    
    # Rule of thumb: If it's a 'Signal' but its values are small (like ROC or Breadth scores), put it lower.
    # For 9-sig, 'Signal' is equity-scale, so we keep it in the main panel if values are high.
    lower_cols = []
    main_overlay_cols = []
    
    for c in lower_panel_candidates:
        if c in df_results.columns:
            if c == 'Signal':
                # Heuristic: If Signal is large (near equity value), keep in main panel (9-Sig).
                # If Signal is small (near price value), it's likely an SMA - discard it from 
                # this chart to keep it clean.
                first_strat = df_results['Strategy'].iloc[0] if 'Strategy' in df_results.columns else 1
                first_sig = df_results['Signal'].iloc[0]
                
                if first_sig > (first_strat * 0.1):
                    main_overlay_cols.append(c)
                # else: we just don't add it to lower_cols or main_overlay_cols
            else:
                # Other factors like breadth or vol go to the lower pane
                lower_cols.append(c)
    
    # Removed SMA and other price-based overlays
    
    has_lower = len(lower_cols) > 0
    
    if has_lower:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                           vertical_spacing=0.08, row_heights=[0.7, 0.3])
    else:
        fig = go.Figure()

    # 1. Main Panel Equity Curves
    curves = [('Strategy', PV_COLORS['Strategy'], 3), ('BH', PV_COLORS['BH'], 2)]
    if benchmark_symbol in df_results.columns:
        curves.append((benchmark_symbol, PV_COLORS['Benchmark'], 2))
        
    for col, color, width in curves:
        trace_name = "Buy & Hold" if col == 'BH' else col
        fig.add_trace(go.Scatter(
            x=df_plot.index, y=df_plot[col], name=trace_name,
            line=dict(color=color, width=width),
            hovertemplate='%{y:$.2f}'
        ), row=1 if has_lower else None, col=1 if has_lower else None)

    # 2. Main Panel Overlays (SMAs, etc.)
    for col in main_overlay_cols:
        fig.add_trace(go.Scatter(
            x=df_plot.index, y=df_plot[col], name=f"Signal ({col})",
            line=dict(dash='dash', width=1.5, color='#7f8c8d'),
            opacity=0.7,
            hovertemplate='%{y:$.2f}'
        ), row=1 if has_lower else None, col=1 if has_lower else None)

    # 3. Lower Panel Factors (ROC, Vol, Breadth)
    if has_lower:
        # Check if we have breadth data for special handling
        if 'highlowqag' in lower_cols:
            # 3a. Plot thin background line for raw breadth if available
            if 'highlowq' in df_plot.columns:
                fig.add_trace(go.Scatter(
                    x=df_plot.index, y=df_plot['highlowq'], name="Raw Breadth",
                    line=dict(color='#3498db', width=1.5),
                    opacity=0.4,
                    showlegend=False,
                    hovertemplate='Raw: %{y:.2f}'
                ), row=2, col=1)

            # 3b. Plot thick segmented signal line for highlowqag
            # We split into 3 segments to handle colors without gaps
            # Colors: Green (Long), Blue (Neutral Cash), Red (Bearish Cash)
            pos = df_plot['Position'] if 'Position' in df_plot.columns else pd.Series([0]*len(df_plot), index=df_plot.index)
            val = df_plot['highlowqag']
            
            # Helper to create masked series
            def get_masked(mask):
                m = pd.Series([None] * len(val), index=val.index)
                m[mask] = val[mask]
                # To connect segments, we include the point immediately after a segment if it exists
                # This prevents "gaps" in the plotted line
                shifted_mask = mask.shift(1).fillna(False)
                m[shifted_mask] = val[shifted_mask]
                return m

            # Long = Green
            green_mask = (pos == 1)
            fig.add_trace(go.Scatter(
                x=df_plot.index, y=get_masked(green_mask), name="Signal (Long)",
                line=dict(color='#27ae60', width=4), # Bold Green
                showlegend=False,
                hovertemplate='Signal: %{y:.2f}'
            ), row=2, col=1)

            # Bearish Cash = Red
            red_mask = (pos == 0) & (val < -60)
            fig.add_trace(go.Scatter(
                x=df_plot.index, y=get_masked(red_mask), name="Signal (Bearish)",
                line=dict(color='#e74c3c', width=4), # Bold Red
                showlegend=False,
                hovertemplate='Signal: %{y:.2f}'
            ), row=2, col=1)

            # Neutral Cash = Blue
            blue_mask = (pos == 0) & (val >= -60)
            fig.add_trace(go.Scatter(
                x=df_plot.index, y=get_masked(blue_mask), name="Signal (Neutral)",
                line=dict(color='#3498db', width=4), # Blue
                showlegend=False,
                hovertemplate='Signal: %{y:.2f}'
            ), row=2, col=1)

            # Filter out highlowqag from generic scatter plotting
            remaining_cols = [c for c in lower_cols if c != 'highlowqag']
        else:
            remaining_cols = lower_cols

        # Plot any other factors normally
        for col in remaining_cols:
            fig.add_trace(go.Scatter(
                x=df_plot.index, y=df_plot[col], name=col,
                line=dict(width=1.5),
                hovertemplate='%{y:.4f}'
            ), row=2, col=1)
        
        # Determine appropriate label for lower panel
        lower_label = "Indicators"
        if any('vol' in c.lower() for c in lower_cols):
            lower_label = "Volatility Metrics"
        elif 'highlowqag' in lower_cols:
            lower_label = "Market Breadth"
        elif any('roc' in c.lower() for c in lower_cols):
            lower_label = "Momentum (ROC)"

        fig.update_yaxes(
            title_text=lower_label, 
            row=2, col=1, 
            gridcolor='#f0f0f0',
            zeroline=True,
            zerolinecolor='black',
            zerolinewidth=1.5,
            tickformat='.0f',
            title_font=dict(color='#333333'),
            tickfont=dict(color='#333333')
        )

    # Layout and Styling
    fig.update_layout(
        title=dict(text=f"Interactive Portfolio Growth: {symbol}", x=0.01, font=dict(size=18, color='#333333')),
        yaxis=dict(
            type="log", 
            title="Portfolio Value ($)", 
            gridcolor='#f0f0f0',
            autorange=True,
            fixedrange=False,
            title_font=dict(color='#333333'),
            tickfont=dict(color='#333333')
        ),
        xaxis=dict(
            title="Date", 
            gridcolor='#f0f0f0',
            title_font=dict(color='#333333'),
            tickfont=dict(color='#333333')
        ),
        hovermode="x unified",
        dragmode="pan",
        template="plotly_white",
        plot_bgcolor='white',
        paper_bgcolor='white',
        # uirevision='constant' keeps zoom/pan within the current slice.
        # Switching period rebuilds the chart (new Streamlit run) with fresh slice.
        uirevision='constant',
        height=700 if has_lower else 550,
        legend=dict(
            orientation="h", 
            yanchor="bottom", y=1.02, 
            xanchor="right", x=1,
            font=dict(color='#333333')
        ),
        margin=dict(l=0, r=0, t=100, b=0)
    )

    # TradingView-style spikelines (crosshairs)
    fig.update_xaxes(
        showspikes=True,
        spikemode="across",
        spikesnap="cursor",
        showline=True,
        spikedash="dash",
        spikecolor="#999999",
        spikethickness=1
    )
    fig.update_yaxes(
        showspikes=True,
        spikemode="across",
        spikesnap="cursor",
        spikedash="dash",
        spikecolor="#999999",
        spikethickness=1
    )

    fig.update_xaxes(
        type="date"
    )

    # Using 'scrollZoom' in config
    st.plotly_chart(fig, use_container_width=True, theme=None, config={'scrollZoom': True, 'displaylogo': False})




def render_annual_returns(df_results, initial_capital, benchmark_symbol="SPY"):
    """Render Annual Returns grouped bar chart – Portfolio Visualizer style."""

    # ── build per-year return series for each curve ──────────────────────────
    def annual_pct(col, baseline):
        if col not in df_results.columns:
            return pd.Series(dtype=float)
            
        # Prepend baseline value to capture the first year's return
        first_date = df_results.index[0] - pd.Timedelta(days=1)
        extended = pd.concat([pd.Series({first_date: baseline}), df_results[col]])
        
        return (
            extended
            .resample("YE")
            .last()
            .pct_change()
            .dropna()
            * 100
        )

    strat_ann  = annual_pct("Strategy", initial_capital)
    bh_ann     = annual_pct("BH", initial_capital)
    bench_ann  = annual_pct(benchmark_symbol, initial_capital)

    # align all series on the same year index
    all_years = sorted(
        set(strat_ann.index) | set(bh_ann.index) | set(bench_ann.index)
    )
    if not all_years:
        st.info("Not enough data for annual returns chart.")
        return

    years      = [y.year for y in all_years]
    strat_vals = [strat_ann.get(y, float("nan")) for y in all_years]
    bh_vals    = [bh_ann.get(y,    float("nan")) for y in all_years]
    bench_vals = [bench_ann.get(y,  float("nan")) for y in all_years]

    has_bench = bench_ann.notna().any()
    n_groups  = 3 if has_bench else 2
    bar_w     = 0.26 if has_bench else 0.35
    x         = range(len(years))

    # ── colours matching PV style ──────────────────────────────────
    c_strat = PV_COLORS['Strategy']
    c_bh    = PV_COLORS['BH']
    c_bench = PV_COLORS['Benchmark']

    fig, ax = plt.subplots(figsize=(max(10, len(years) * 1.1), 6))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    offsets = [-bar_w, 0, bar_w] if has_bench else [-bar_w / 2, bar_w / 2]

    def _bars(vals, offset, color, label):
        positions = [xi + offset for xi in x]
        bars = ax.bar(positions, vals, bar_w, color=color, label=label,
                      zorder=3, alpha=0.92)
        return bars

    _bars(strat_vals,  offsets[0], c_strat, "Strategy")
    _bars(bh_vals,     offsets[1], c_bh,    "Buy & Hold Portfolio")
    if has_bench:
        _bars(bench_vals, offsets[2], c_bench, benchmark_symbol)

    # ── zero line & grid ─────────────────────────────────────────────────────
    ax.axhline(0, color="#555555", linewidth=0.8, zorder=4)
    ax.yaxis.grid(True, color="#e0e0e0", linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)

    # ── axes formatting ───────────────────────────────────────────────────────
    ax.set_xticks(list(x))
    ax.set_xticklabels(years, fontsize=9)
    ax.set_xlabel("Year", fontsize=10, labelpad=6)
    ax.set_ylabel("Annual Return", fontsize=10, labelpad=6)
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: f"{v:.0f}%")
    )
    ax.tick_params(axis="both", which="both", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # ── legend at bottom, matching reference ──────────────────────────────────
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=n_groups,
        frameon=False,
        fontsize=9,
        markerscale=1.2,
    )

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def render_annual_returns_table(df_results, initial_capital, benchmark_symbol="SPY"):
    """
    Render Annual Returns table - Portfolio Visualizer style.
    """
    
    unique_years = sorted(df_results.index.year.unique(), reverse=True)
    rows = []
    
    for year in unique_years:
        year_data = df_results[df_results.index.year == year]
        if year_data.empty: continue
        
        end_strat = year_data['Strategy'].iloc[-1]
        end_bh = year_data['BH'].iloc[-1]
        end_bench = year_data[benchmark_symbol].iloc[-1] if benchmark_symbol in df_results.columns else None
        end_cash = year_data['Cash'].iloc[-1] if 'Cash' in year_data.columns else 0
        
        prev_data = df_results[df_results.index.year < year]
        if not prev_data.empty:
            start_strat = prev_data['Strategy'].iloc[-1]
            start_bh = prev_data['BH'].iloc[-1]
            start_bench = prev_data[benchmark_symbol].iloc[-1] if benchmark_symbol in df_results.columns else None
        else:
            # Use initial capital for the very first year
            start_strat = initial_capital
            start_bh = initial_capital
            start_bench = initial_capital
            
        ret_strat = (end_strat / start_strat) - 1 if start_strat != 0 else 0
        ret_bh = (end_bh / start_bh) - 1 if start_bh != 0 else 0
        ret_bench = (end_bench / start_bench) - 1 if start_bench is not None and start_bench != 0 else 0
        
        diff = (ret_strat - ret_bench) * 100

        rows.append({
            'Year': year,
            'Strat Ret': ret_strat * 100,
            'Diff': diff,
            'Strat Bal': end_strat,
            'Cash Bal': end_cash,
            'Cash %': (end_cash / end_strat * 100) if end_strat != 0 else 0,
            'BH Ret': ret_bh * 100,
            'BH Bal': end_bh,
            'Bench Ret': ret_bench * 100,
            'Bench Bal': end_bench
        })
        
    df_table = pd.DataFrame(rows)
    # Convert Year to string for display
    df_table['Year'] = df_table['Year'].map(str)
    # Enforce column order: Diff immediately after Strat Ret
    col_order = ['Year', 'Strat Ret', 'Diff', 'Strat Bal', 'Cash Bal', 'Cash %', 'BH Ret', 'BH Bal', 'Bench Ret', 'Bench Bal']
    df_table = df_table[[c for c in col_order if c in df_table.columns]]
    
    # Calculate scale for Strategy Returns (symmetric around 0)
    try:
        s_ret_vals = pd.to_numeric(df_table['Strat Ret'], errors='coerce').dropna().values
        vmax = np.nanmax(np.abs(s_ret_vals)) if len(s_ret_vals) > 0 else 10.0
        
        diff_vals = pd.to_numeric(df_table['Diff'], errors='coerce').dropna().values
        diff_vmax = np.nanmax(np.abs(diff_vals)) if len(diff_vals) > 0 else 5.0
    except:
        vmax = 10.0
        diff_vmax = 5.0

    if pd.isna(vmax) or vmax == 0: vmax = 10.0
    if pd.isna(diff_vmax) or diff_vmax == 0: diff_vmax = 5.0

    styler = df_table.style.hide(axis='index')

    # Apply heatmap to Strategy Return and Diff columns
    styler = styler.background_gradient(
        cmap='RdYlGn',
        subset=['Strat Ret'],
        vmin=-vmax, vmax=vmax,
        axis=None
    ).background_gradient(
        cmap='RdYlGn',
        subset=['Diff'],
        vmin=-diff_vmax, vmax=diff_vmax,
        axis=None
    ).highlight_null(color=None)
    
    # Formatting
    styler = styler.format({
        'Strat Ret': "{:.2f}%", 'BH Ret': "{:.2f}%", 'Bench Ret': "{:.2f}%",
        'Diff': "{:+.2f}%",
        'Strat Bal': "${:,.2f}", 'BH Bal': "${:,.2f}", 'Bench Bal': "${:,.2f}",
        'Cash Bal': "${:,.2f}", 'Cash %': "{:.2f}%"
    }, na_rep="N/A")
    
    column_config = {
        "Year": st.column_config.TextColumn("Year"),
        "Strat Ret": st.column_config.NumberColumn("Strategy Return"),
        "Strat Bal": st.column_config.NumberColumn("Strategy Balance"),
        "Cash Bal": st.column_config.NumberColumn("Cash Balance"),
        "Cash %": st.column_config.NumberColumn("Cash %"),
        "BH Ret": st.column_config.NumberColumn("B&H Return"),
        "Diff": st.column_config.NumberColumn(f"vs {benchmark_symbol}", help=f"Strategy Return minus {benchmark_symbol} Return. Green = outperformed, Red = underperformed."),
        "BH Bal": st.column_config.NumberColumn("B&H Balance"),
        "Bench Ret": st.column_config.NumberColumn(f"{benchmark_symbol} Return"),
        "Bench Bal": st.column_config.NumberColumn(f"{benchmark_symbol} Balance"),
    }
    
    st.dataframe(styler, use_container_width=True, column_config=column_config)


def render_rolling_correlation(df_results, benchmark_symbol):
    """
    Render 6-month rolling correlation between strategy and benchmark.
    """
    if benchmark_symbol not in df_results.columns or len(df_results) < 126:
        return

    st.subheader(f"Rolling Correlation to {benchmark_symbol}")
    
    # Calculate daily returns
    strat_rets = df_results['ret'].dropna()
    bench_rets = df_results[benchmark_symbol].pct_change().dropna()
    
    # Align
    strat_rets, bench_rets = align_returns(strat_rets, bench_rets)
    if len(strat_rets) < 126:
        st.info("Insufficient data for rolling correlation.")
        return
        
    # 6-month window (~126 trading days)
    rolling_corr = strat_rets.rolling(window=126).corr(bench_rets)
    
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(rolling_corr.index, rolling_corr, color=PV_COLORS['Strategy'], linewidth=1.5)
    ax.fill_between(rolling_corr.index, 0, rolling_corr, color=PV_COLORS['Strategy'], alpha=0.1)
    
    ax.set_ylim(-1.0, 1.0)
    ax.axhline(0, color='black', linewidth=0.8, alpha=0.5)
    ax.set_ylabel("Correlation (6M Rolling)")
    ax.set_title(f"Rolling Correlation: Strategy vs {benchmark_symbol}")
    
    # Add bands for high/low correlation
    ax.axhline(0.7, color='red', linestyle='--', alpha=0.3, label='High Correlation (>0.7)')
    ax.axhline(0.3, color='green', linestyle='--', alpha=0.3, label='Low Correlation (<0.3)')
    
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def render_underwater_chart(df_results, benchmark_symbol):
    """
    Render Underwater (Drawdown) chart for comparison.
    """
    st.subheader("Underwater Equity (Drawdowns)")
    
    def get_dd(ser):
        return (ser / ser.cummax() - 1) * 100

    strat_dd = get_dd(df_results['Strategy'])
    bh_dd = get_dd(df_results['BH'])
    
    fig, ax = plt.subplots(figsize=(12, 5))
    
    # Plot BH and Benchmark drawdowns with lighter fills
    ax.fill_between(bh_dd.index, 0, bh_dd, color=PV_COLORS['BH'], alpha=0.15, label='Buy & Hold')
    if benchmark_symbol in df_results.columns:
        bench_dd = get_dd(df_results[benchmark_symbol])
        ax.plot(bench_dd.index, bench_dd, color=PV_COLORS['Benchmark'], alpha=1.0, linewidth=0.8, label=benchmark_symbol)

    # Plot Strategy drawdown prominently
    ax.fill_between(strat_dd.index, 0, strat_dd, color=PV_COLORS['Strategy'], alpha=0.4, label='Strategy')
    ax.plot(strat_dd.index, strat_dd, color=PV_COLORS['Strategy'], linewidth=1.2)

    ax.set_ylim(bottom=min(strat_dd.min(), bh_dd.min()) * 1.1, top=2)
    ax.axhline(0, color='black', linewidth=1)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0f}%'))
    ax.set_ylabel("Drawdown (%)")
    ax.set_title("Portfolio Drawdown Comparison")
    ax.legend(loc='lower left', fontsize=9)
    ax.grid(True, alpha=0.15)
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def render_risk_management_performance(df_results):
    """Render Risk Management Performance"""
    
    total_days = len(df_results)
    in_market = (df_results['Cash'] < df_results['Strategy']).sum()
    out_market = total_days - in_market
    in_market_pct = (in_market / total_days * 100) if total_days > 0 else 0
    out_market_pct = (out_market / total_days * 100) if total_days > 0 else 0
    
    if in_market > 0:
        in_market_returns = df_results[df_results['Cash'] < df_results['Strategy']]['ret']
        in_market_ret = (1 + in_market_returns).prod() - 1
        in_market_std = in_market_returns.std() * np.sqrt(252)
    else:
        in_market_ret = 0
        in_market_std = 0
    
    if out_market > 0:
        out_market_returns = df_results[df_results['Cash'] >= df_results['Strategy']]['ret']
        out_market_ret = (1 + out_market_returns).prod() - 1 if len(out_market_returns) > 0 else 0
    else:
        out_market_ret = 0
    
    risk_mgmt_df = pd.DataFrame({
        'Condition': ['In Market', 'Out of Market'],
        'Periods #': [in_market, out_market],
        'Periods %': [in_market_pct, out_market_pct],
        'Return': [in_market_ret * 100, out_market_ret * 100],
        'Stdev (Ann)': [in_market_std * 100, 0.0],
        'Sharpe Ratio': [in_market_ret/in_market_std if in_market_std > 0 else 0.0, 0.0]
    })
    
    vmax = risk_mgmt_df['Return'].abs().max()
    if pd.isna(vmax) or vmax == 0: vmax = 10.0
    
    styler = risk_mgmt_df.style.hide(axis='index')
    styler = styler.background_gradient(
        cmap='RdYlGn', subset=['Return'],
        vmin=-vmax, vmax=vmax,
        axis=None
    ).highlight_null(color=None)
    
    styler = styler.format({
        'Periods %': "{:.2f}%",
        'Return': "{:.2f}%",
        'Stdev (Ann)': lambda x: f"{x:.2f}%" if x > 0 else "N/A",
        'Sharpe Ratio': lambda x: f"{x:.2f}" if x > 0 else "N/A"
    })
    
    column_config = {
        "Condition": st.column_config.TextColumn("Market Condition", help="Whether the strategy was fully invested or in cash."),
        "Return": st.column_config.TextColumn("Return", help="Cumulative return achieved in this state."),
        "Sharpe Ratio": st.column_config.TextColumn("Sharpe", help=METRIC_DEFINITIONS.get("Sharpe Ratio", "")),
    }
    
    st.dataframe(
        styler,
        use_container_width=True,
        column_config=column_config
    )


def render_allocation_changes(df_results):
    """Render Allocation Changes chart - In Market vs Out of Market"""
    
    # Calculate allocations
    # Strategy = Stock_Value + Cash
    # Stock_Allocation = (Strategy - Cash) / Strategy
    # Cash_Allocation = Cash / Strategy
    
    strategy = df_results['Strategy']
    cash = df_results['Cash']
    
    # Handle cases where strategy value might be zero to avoid division by zero
    in_market = ((strategy - cash) / strategy).fillna(0) * 100
    out_of_market = (cash / strategy).fillna(0) * 100
    
    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    
    ax.stackplot(df_results.index, in_market, out_of_market, 
                 labels=['In Market', 'Out of Market'],
                 colors=[PV_COLORS['Strategy'], PV_COLORS['Cash']], 
                 alpha=0.8)
    
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1f}%'))
    ax.set_ylabel("Allocation", fontsize=10)
    ax.set_xlabel("Time Period", fontsize=10)
    ax.set_title("Allocation Changes", fontsize=12, pad=15)
    
    # Legend at bottom
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=False, fontsize=9)
    
    ax.grid(True, alpha=0.3, axis='y', color='#e0e0e0')
    for spine in ax.spines.values():
        spine.set_visible(False)
        
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def render_last_trade_status(trade_log, df_results):
    """Render Last Trade Status before performance summary."""
    if not trade_log or len(trade_log) == 0:
        return

    # Find last strategic trade (ignore FLOOR REFILL maintenance)
    filtered_trades = [t for t in trade_log if t.get('Status') != 'FLOOR REFILL']
    if not filtered_trades:
        return
    last_trade = filtered_trades[-1]
    
    # Get current portfolio value and cash % from df_results
    current_portfolio_val = df_results['Strategy'].iloc[-1]
    current_cash_val = df_results['Cash'].iloc[-1]
    current_cash_pct = (current_cash_val / current_portfolio_val) * 100 if current_portfolio_val > 0 else 0
    
    st.subheader("Last Trade Status")
    
    # Calculate Profit % (handle open trades using current price)
    profit_pct = last_trade.get('Profit %')
    if profit_pct is None or (isinstance(profit_pct, float) and np.isnan(profit_pct)):
        if last_trade.get('Status') == 'OPEN' and last_trade.get('Entry Price'):
            current_price = df_results['close'].iloc[-1]
            entry_price = last_trade.get('Entry Price')
            profit_pct = (current_price / entry_price - 1) * 100
    
    def safe_pct(v):
        if isinstance(v, (int, float)) and not np.isnan(v):
            return f"{v:.2f}%"
        return str(v)

    def safe_curr(v):
        if isinstance(v, (int, float)) and not np.isnan(v):
            return f"${v:,.2f}"
        return str(v)

    status = last_trade.get('Status', 'N/A')
    
    # Transform to table format
    data = [{
        'Entry Date': last_trade.get('Entry Date', last_trade.get('Date', 'N/A')),
        'Exit Date': last_trade.get('Exit Date', 'N/A') if status in ['CLOSED', 'FLOOR REFILL'] else 'OPEN',
        'Entry Price': safe_curr(last_trade.get('Entry Price', 0)),
        'Exit Price': safe_curr(last_trade.get('Exit Price', 0)) if status == 'CLOSED' else 'N/A',
        'Profit %': safe_pct(profit_pct),
        'Status': status,
        'Portfolio Value': f"${current_portfolio_val:,.2f}",
        'Cash %': f"{current_cash_pct:.2f}%"
    }]
    
    st.dataframe(pd.DataFrame(data), use_container_width=True, hide_index=True)
    st.markdown("---")



def generate_backtest_report(result: BacktestResult):
    """
    Generate the complete backtest report - Mobile First Design.
    Reorganized into logical tabs for easy navigation on small screens.
    """
    df_results = result.df_results
    trade_log_results = result.trade_log.to_dict('records') if not result.trade_log.empty else []
    initial_capital = result.initial_capital
    symbol = result.symbol
    benchmark_symbol = result.benchmark_symbol

    # 1. PRE-COMPUTE DATA
    from .common import calculate_pnl_from_trades
    strategic_trade_log = [t for t in trade_log_results if t.get('Status') != 'FLOOR REFILL']
    s_pnl_list, s_pnl_pct_list = calculate_pnl_from_trades(strategic_trade_log)

    strategy_returns = df_results['ret'].dropna()
    benchmark_returns = df_results['BH'].pct_change().dropna()
    benchmark_strategy_returns = df_results[benchmark_symbol].pct_change().dropna() if benchmark_symbol in df_results.columns else None

    metrics_dict = get_all_metrics(
        df_results['Strategy'], s_pnl_list, initial_capital,
        strategy_returns, benchmark_returns, benchmark_strategy_returns,
        pnl_pct_list=s_pnl_pct_list
    )

    # Pre-calculate the comparison dataframe once (shared between views)
    strat_rets = strategy_returns
    bh_rets = benchmark_returns
    bench_rets = benchmark_strategy_returns if benchmark_strategy_returns is not None else bh_rets
    
    start_val = float(initial_capital)
    start_balance_str = f"${start_val:,.0f}"

    s_m = calculate_pv_metrics(strat_rets, bench_rets, df_results['Strategy'])
    b_m = calculate_pv_metrics(bh_rets, bench_rets, df_results['BH'])
    m_m = calculate_pv_metrics(bench_rets, bench_rets, df_results[benchmark_symbol] if benchmark_symbol in df_results.columns else df_results['BH'])

    def fmt_pct(val):
        try: return f"{float(val)*100:.2f}%" if not (np.isinf(val) or np.isnan(val)) else "0.00%"
        except: return "0.00%"
    def fmt_num(val):
        try: return f"{float(val):.2f}" if not (np.isinf(val) or np.isnan(val)) else "0.00"
        except: return "0.00"
    def get_best_worst(series):
        try:
            ann = series.resample('YE').last().pct_change().dropna()
            if len(ann) > 0: return f"{ann.max()*100:.2f}%", f"{ann.min()*100:.2f}%"
        except: pass
        return 'N/A', 'N/A'

    s_best, s_worst = get_best_worst(df_results['Strategy'])
    b_best, b_worst = get_best_worst(df_results['BH'])
    m_best, m_worst = get_best_worst(df_results[benchmark_symbol] if benchmark_symbol in df_results.columns else df_results['BH'])

    bench_name = benchmark_symbol
    if benchmark_symbol == "SPY": bench_name = "SPDR S&P 500 ETF (SPY)"
    elif benchmark_symbol == "QQQ": bench_name = "Invesco QQQ Trust (QQQ)"

    metrics_rows = [
        ("Start Balance", start_balance_str, start_balance_str, start_balance_str, "Initial value of the portfolio."),
        ("Ending Balance", f"${df_results['Strategy'].iloc[-1]:,.0f}", f"${df_results['BH'].iloc[-1]:,.0f}", f"${df_results[benchmark_symbol].iloc[-1]:,.0f}" if benchmark_symbol in df_results.columns else "N/A", "Final portfolio value."),
        ("Equity Multiple", f"{(df_results['Strategy'].iloc[-1] / start_val):.2f}x", f"{(df_results['BH'].iloc[-1] / start_val):.2f}x", f"{(df_results[benchmark_symbol].iloc[-1] / start_val):.2f}x" if benchmark_symbol in df_results.columns else "N/A", "Total growth multiple."),
        ("Cash Balance %", f"{(df_results.get('Cash', pd.Series([0])).iloc[-1] / df_results['Strategy'].iloc[-1])*100:.2f}%", "0.00%", "0.00%", "Percentage of value currently in cash."),
        ("CAGR", fmt_pct(s_m.get('GM_A')), fmt_pct(b_m.get('GM_A')), fmt_pct(m_m.get('GM_A')), "Compound Annual Growth Rate."),
        ("Std Dev", fmt_pct(s_m.get('SD_A')), fmt_pct(b_m.get('SD_A')), fmt_pct(m_m.get('SD_A')), "Annualized Volatility."),
        ("Best Year", s_best, b_best, m_best, "Highest calendar year return."),
        ("Worst Year", s_worst, b_worst, m_worst, "Lowest calendar year return."),
        ("Max Drawdown", fmt_pct(s_m.get('MDD')), fmt_pct(b_m.get('MDD')), fmt_pct(m_m.get('MDD')), "Maximum peak-to-trough decline."),
        ("Sharpe Ratio", fmt_num(s_m.get('Sharpe')), fmt_num(b_m.get('Sharpe')), fmt_num(m_m.get('Sharpe')), "Risk-adjusted return."),
        ("Sortino Ratio", fmt_num(s_m.get('Sortino')), fmt_num(b_m.get('Sortino')), fmt_num(m_m.get('Sortino')), "Downside-only risk-adjusted return."),
        ("Avg Monthly G/L", fmt_num(s_m.get('GL')), fmt_num(b_m.get('GL')), fmt_num(m_m.get('GL')), "Average monthly gain vs average monthly loss."),
        ("Number of Trades", str(metrics_dict.get('Total Trades', 'N/A')), "1 (B&H)", "N/A", "Total trade count."),
        ("Win Rate %", metrics_dict.get('% Profitable', "N/A"), "N/A", "N/A", "Percentage of profitable trades."),
        ("Profit Factor", str(metrics_dict.get('Profit Factor', '0.00')), "N/A", "N/A", "Gross Profit / Gross Loss."),
        ("Payoff Ratio", str(metrics_dict.get('Payoff Ratio', "N/A")), "N/A", "N/A", "Average Win / Average Loss."),
        ("Expectancy", str(metrics_dict.get('Expectancy Per Trade', 'N/A')), "N/A", "N/A", "Expected profit per trade."),
    ]
    perf_df = pd.DataFrame(metrics_rows, columns=['Metric', 'Strategy', 'Buy & Hold Portfolio', bench_name, 'Metric Description'])

    # 2. RENDER TOP-LEVEL TABS (Mobile Friendly)
    tabs = st.tabs(["📊 Summary", "📅 Returns", "🛡️ Risk", "⚡ Trading", "🧪 Deep Dive"])

    with tabs[0]: # SUMMARY
        st.caption(f"Asset: **{symbol}** | Benchmark: **{benchmark_symbol}**")
        render_last_trade_status(strategic_trade_log, df_results)
        
        # High level cards and categorical heatmaps
        render_mobile_metrics_dashboard(perf_df, metrics_dict, benchmark_symbol)
        
        # Quick Growth Chart (Responsive)
        st.subheader("Interactive Growth Chart")
        # Optimization: use slightly lower height for mobile viewport compatibility
        render_interactive_growth_chart(df_results, symbol, benchmark_symbol)
        
        st.markdown("---")
        with st.expander("🔍 View Full Desktop Comparison Table"):
            full_styled = apply_conditional_styling(perf_df, only_strategy=False)
            st.dataframe(full_styled, use_container_width=True, hide_index=True)

    with tabs[1]: # RETURNS ANALYSIS
        st.subheader("Annual & Periodic Returns")
        render_annual_returns(df_results, initial_capital, benchmark_symbol)
        render_annual_returns_table(df_results, initial_capital, benchmark_symbol)
        
        st.markdown("---")
        st.subheader("Monthly Returns Heatmap")
        render_monthly_returns_heatmap(df_results)
        
        st.markdown("---")
        st.subheader("Trailing Returns")
        render_trailing_returns(df_results, benchmark_symbol)

        st.markdown("---")
        st.subheader("Rolling Returns Analysis")
        render_rolling_returns_analysis(df_results, benchmark_symbol)

    with tabs[2]: # RISK & VOLATILITY
        st.subheader("Drawdown Analysis")
        render_underwater_chart(df_results, benchmark_symbol)
        render_drawdown_analysis(df_results)
        
        st.markdown("---")
        st.subheader("Risk-Adjusted Metrics")
        render_risk_adjusted_returns(metrics_dict, df_results, benchmark_symbol)
        
        st.markdown("---")
        st.subheader("Allocation & Stability")
        render_allocation_changes(df_results)
        render_rolling_correlation(df_results, benchmark_symbol)

    with tabs[3]: # TRADING LOGS
        st.subheader("Execution Statistics")
        render_trade_statistics(df_results, strategic_trade_log, initial_capital)
        
        st.markdown("---")
        st.subheader("Transaction Log")
        render_trade_log(strategic_trade_log)
        
        with st.expander("🛠️ Show Maintenance/Rebalance Log"):
            refills = [t for t in trade_log_results if t.get('Status') == 'FLOOR REFILL']
            if refills: render_trade_log(refills)
            else: st.info("No maintenance trades recorded.")

    with tabs[4]: # DEEP DIVE / ADVANCED
        st.subheader("Advanced Analytical Deep-Dive")
        
        p = result.params
        floor_val = p.get('cash_floor_pct', p.get('cash_floor_pct_val', 0.20))
        if floor_val <= 1.0: floor_val *= 100
        
        with st.expander("📦 Cash & HWM Analysis", expanded=True):
            render_monthly_cash_heatmap(df_results, cash_floor_pct=floor_val)
            render_portfolio_hwm_heatmap(df_results)
            
        with st.expander("📉 Market Stress Performance"):
            render_market_stress_analysis(df_results, benchmark_symbol)
            render_risk_management_performance(df_results)
            
        with st.expander("🎲 Monte Carlo Stress Testing"):
            mc_sim_df, mc_results = run_monte_carlo(initial_capital, strategy_returns, 200, 10)
            render_monte_carlo_simulation(mc_results, mc_sim_df, initial_capital, 200)
            
        with st.expander("⚖️ Survival & Discipline"):
            render_survival_discipline(df_results, s_pnl_list, strategic_trade_log)
            render_stability_check(s_pnl_list, df_results)


def generate_backtest_report_compat(
    df_results,
    trade_list_results,
    trade_log_results,
    initial_capital_used,
    rolling_returns_df,   # kept for backward-compat, not used
    cash_pct_df,          # kept for backward-compat, not used
    mc_sim_df,
    mc_results,
    num_simulations,
    symbol,
    benchmark_symbol="SPY",
):
    """
    Backward-compatible wrapper: accepts the old positional tuple signature
    used by existing callers, wraps it into a BacktestResult, then delegates
    to generate_backtest_report.

    Migration guide: replace calls like
        generate_backtest_report_compat(df, trades, log, cap, rr, cp, mc_df, mc_res, n, sym)
    with:
        from common import build_strategy_result
        result = build_strategy_result(hist, trade_log, cap, sym, benchmark_symbol)
        generate_backtest_report(result)
    """
    import pandas as pd
    from .common import BacktestResult
    trade_log_df = pd.DataFrame(trade_log_results) if trade_log_results else pd.DataFrame()
    result = BacktestResult(
        df_results=df_results,
        trade_log=trade_log_df,
        pnl_list=trade_list_results,
        initial_capital=initial_capital_used,
        symbol=symbol,
        benchmark_symbol=benchmark_symbol,
    )
    # Delegate to the modern unified reporting function
    generate_backtest_report(result)


