"""
Common utilities for backtesting strategies.
Contains reusable code shared across all strategies.
"""

from dataclasses import dataclass, field
import pandas as pd
import yfinance as yf
import numpy as np
import streamlit as st
from functools import lru_cache


# ---------------------------------------------------------------------------
# Shared contract between strategies and the reporting module.
# Every strategy should return a BacktestResult instead of a raw tuple.
# ---------------------------------------------------------------------------

@dataclass
class BacktestResult:
    """
    Standardised output from any backtest strategy.

    df_results must contain columns:
        Strategy  – daily equity curve
        BH        – buy-and-hold equity curve (same starting capital)
        <benchmark_symbol> – benchmark equity curve
        Cash      – uninvested cash
        close     – underlying asset close price
        ret       – daily strategy returns (pct_change of Strategy)

    All other fields are optional; reporting functions handle missing data
    gracefully via .get() with 'N/A' defaults.
    """
    df_results: pd.DataFrame
    trade_log: pd.DataFrame
    pnl_list: list
    initial_capital: float
    symbol: str
    pnl_pct_list: list = field(default_factory=list)
    benchmark_symbol: str = "SPY"
    # Derived helpers – populated automatically by build_strategy_result()
    rolling_returns_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    cash_pct_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    strategy_name: str = "Unknown Strategy"
    params: dict = field(default_factory=dict)


def build_strategy_result(
    hist: list,
    trade_log: list,
    initial_capital: float,
    symbol: str,
    benchmark_symbol: str = "SPY",
    strategy_name: str = "Unknown Strategy",
    params: dict = None,
) -> BacktestResult:
    """
    Build a BacktestResult from raw history + trade-log lists.

    Handles:
    - Building the df_results DataFrame and adding the 'ret' column
    - Computing the rolling_returns_df from common.create_rolling_returns_df
    - Computing the cash_pct_df from common.create_cash_pct_df
    - Deriving pnl_list from closed trades via common.calculate_pnl_from_trades
    """
    if params is None: 
        params = {}
        
    report_df = create_results_dataframe(hist)
    
    # Calculate daily strategy returns
    if not report_df.empty:
        report_df['ret'] = report_df['Strategy'].pct_change()
        
    rolling_returns_df = create_rolling_returns_df(report_df)
    cash_pct_df = create_cash_pct_df(report_df)
    
    # P&L calculation from trade log
    pnl_list, pnl_pct_list = calculate_pnl_from_trades(trade_log)
    
    return BacktestResult(
        df_results=report_df,
        trade_log=pd.DataFrame(trade_log),
        pnl_list=pnl_list,
        pnl_pct_list=pnl_pct_list,
        initial_capital=initial_capital,
        symbol=symbol,
        benchmark_symbol=benchmark_symbol,
        rolling_returns_df=rolling_returns_df,
        cash_pct_df=cash_pct_df,
        strategy_name=strategy_name,
        params=params
    )


def make_empty_result(
    start_date,
    initial_capital: float,
    symbol: str,
    benchmark_symbol: str = "SPY",
) -> BacktestResult:
    """Return a BacktestResult populated with dummy data (used when data download fails)."""
    df_results = create_empty_results_df(start_date, initial_capital, benchmark_symbol)
    rolling_returns_df = pd.DataFrame(index=df_results.index)
    cash_pct_df = pd.DataFrame(
        {'Cash %': df_results['Cash'] / df_results['Strategy']},
        index=df_results.index,
    ).fillna(0)
    return BacktestResult(
        df_results=df_results,
        trade_log=pd.DataFrame(),
        pnl_list=[],
        pnl_pct_list=[],
        initial_capital=initial_capital,
        symbol=symbol,
        benchmark_symbol=benchmark_symbol,
        rolling_returns_df=rolling_returns_df,
        cash_pct_df=cash_pct_df,
        strategy_name="Empty Result",
    )



@st.cache_data(show_spinner=False, ttl=3600)
def _cached_yf_download(ticker, start_date):
    """Internal cached downloader to prevent repetitive API calls."""
    try:
        data = yf.download(ticker, start=start_date, progress=False)
        return data
    except Exception:
        return pd.DataFrame()

def download_price_data(ticker, start_date, progress=False):
    """
    Download price data for a given ticker.
    Uses 'Adj Close' for accurate split/dividend adjusted returns.
    """
    raw = _cached_yf_download(ticker, start_date)
    
    if raw.empty:
        return pd.DataFrame()
    
    # Handle MultiIndex
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    
    if 'Adj Close' in raw.columns:
        prices = raw[['Adj Close']].copy()
        prices.rename(columns={'Adj Close': 'Close'}, inplace=True)
    else:
        prices = raw[['Close']].copy()
        
    return prices.dropna()


@st.cache_data(show_spinner=False, ttl=3600)
def _cached_yf_download_multi(tickers, start_date):
    """Internal cached downloader for multiple tickers."""
    try:
        # Convert list to tuple for hashability if needed, though streamlit caches handle list
        data = yf.download(tickers, start=start_date, progress=False)
        return data
    except Exception:
        return pd.DataFrame()

def download_multiple_tickers(tickers, start_date, progress=False):
    """
    Download price data for multiple tickers.
    """
    if not tickers:
        return pd.DataFrame()
    
    # Sort tickers to make cache key consistent
    sorted_tickers = sorted(tickers) if isinstance(tickers, list) else tickers
    raw = _cached_yf_download_multi(sorted_tickers, start_date)
    
    if raw.empty:
        return pd.DataFrame()
    
    # Extract only the 'Close' prices. 
    # For multiple tickers, yfinance returns a MultiIndex [Attribute, Ticker]
    if isinstance(raw.columns, pd.MultiIndex):
        if 'Close' in raw.columns.levels[0]:
            prices = raw['Close']
        elif 'Adj Close' in raw.columns.levels[0]:
            prices = raw['Adj Close']
        else:
            # Fallback
            prices = raw
    else:
        # Single ticker case or already flattened
        prices = raw[['Close']] if 'Close' in raw.columns else raw
    
    return prices.dropna()


def merge_with_benchmark(df, benchmark_symbol, start_date):
    """
    Merge strategy data with benchmark data.
    
    Args:
        df: Strategy DataFrame with close prices
        benchmark_symbol: Benchmark ticker symbol
        start_date: Start date for benchmark data
        
    Returns:
        DataFrame merged with benchmark data
    """
    benchmark_prices = yf.download(benchmark_symbol, start=start_date, progress=False)
    
    if benchmark_prices.empty:
        return df
    
    # Handle MultiIndex columns from yfinance
    if isinstance(benchmark_prices.columns, pd.MultiIndex):
        benchmark_prices.columns = benchmark_prices.columns.get_level_values(0)
    
    benchmark_prices = benchmark_prices[["Close"]].dropna()
    benchmark_col = f"{benchmark_symbol}_Close"
    benchmark_prices.rename(columns={"Close": benchmark_col}, inplace=True)
    
    # Merge benchmark data
    df = pd.merge(df, benchmark_prices, left_index=True, right_index=True, how="inner")
    return df


def create_empty_results_df(start_date, initial_capital, benchmark_symbol="SPY"):
    """
    Create an empty results DataFrame with dummy data when no real data is available.
    
    Args:
        start_date: Start date for the DataFrame
        initial_capital: Initial capital value
        benchmark_symbol: Benchmark ticker symbol
        
    Returns:
        DataFrame with dummy data
    """
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
    
    return df_results


def get_price_value(price):
    """
    Extract float value from price data (handles potential MultiIndex/Series).
    
    Args:
        price: Price value (may be scalar, Series, or DataFrame)
        
    Returns:
        Float value of the price
    """
    if hasattr(price, 'iloc'):
        return float(price.iloc[0])
    return float(price)


def calculate_initial_states(initial_capital, starting_cash_pct=1.0):
    """
    Calculate initial states for backtesting.
    
    Args:
        initial_capital: Total initial capital
        starting_cash_pct: Percentage of capital to start in cash (0-1)
        
    Returns:
        Dictionary with initial state values
    """
    return {
        'cash': initial_capital * starting_cash_pct,
        'stock_val': initial_capital * (1 - starting_cash_pct),
        'bh_val': initial_capital,
        'benchmark_val': initial_capital,
        'max_equity': initial_capital
    }


def calculate_daily_yield(cash_yield_apr):
    """
    Calculate daily yield from APR.
    
    Args:
        cash_yield_apr: Annual Percentage Rate
        
    Returns:
        Daily yield factor
    """
    return (1 + cash_yield_apr)**(1/252) - 1


def update_position_values(stock_val, current_price, prev_price, cash, daily_yield):
    """
    Update position values for a time period.
    
    Args:
        stock_val: Current stock position value
        current_price: Current asset price
        prev_price: Previous asset price
        cash: Current cash position
        daily_yield: Daily cash yield
        
    Returns:
        Updated (stock_val, cash)
    """
    if stock_val > 0:
        stock_val = stock_val * (current_price / prev_price)
    cash *= (1 + daily_yield)
    return stock_val, cash


def apply_trading_costs(amount, price, slippage_bps=0.0, commission=0.0):
    """
    Apply slippage and commission to a trade amount.
    
    Args:
        amount: Dollar amount of the trade
        price: Asset price
        slippage_bps: Slippage in basis points (1 bp = 0.01%)
        commission: Fixed commission per trade
        
    Returns:
        net_amount: Amount actually received (selling) or spent (buying)
        cost: Total cost incurred
    """
    if amount == 0:
        return 0, 0
    
    slippage_cost = abs(amount) * (slippage_bps / 10000)
    total_cost = slippage_cost + commission
    
    # If buying (amount > 0), we spend more. If selling (amount < 0), we receive less.
    if amount > 0:
        return amount + total_cost, total_cost
    else:
        return amount - total_cost, total_cost


def update_bh_benchmark(bh_val, current_price, first_price, benchmark_val, 
                       current_benchmark_price, first_benchmark_price, initial_capital):
    """
    Update Buy & Hold and benchmark values.
    
    Args:
        bh_val: Current Buy & Hold value
        current_price: Current asset price
        first_price: First asset price (for normalization)
        benchmark_val: Current benchmark value
        current_benchmark_price: Current benchmark price
        first_benchmark_price: First benchmark price (for normalization)
        initial_capital: Initial capital
        
    Returns:
        Updated (bh_val, benchmark_val)
    """
    bh_val = initial_capital * (current_price / first_price)
    benchmark_val = initial_capital * (current_benchmark_price / first_benchmark_price)
    return bh_val, benchmark_val


def calculate_cash_floor(max_equity, cash_floor_pct):
    """
    Calculate cash floor based on maximum equity.
    
    Args:
        max_equity: Maximum equity achieved
        cash_floor_pct: Cash floor percentage (0-1)
        
    Returns:
        Cash floor value
    """
    return max_equity * cash_floor_pct


def update_max_equity(current_equity, max_equity):
    """
    Update maximum equity if current equity is higher.
    
    Args:
        current_equity: Current total equity
        max_equity: Current maximum equity
        
    Returns:
        Updated max_equity
    """
    return max(max_equity, current_equity)


def create_trade_log_entry(date, entry_price, exit_price, status, equity, amount=None):
    """
    Create a trade log entry.
    
    Args:
        date: Trade date
        entry_price: Entry price
        exit_price: Exit price (None if open)
        status: Trade status ('OPEN' or 'CLOSED')
        equity: Total equity at time of trade
        amount: Actual dollar amount invested in the trade (optional, defaults to equity)
        
    Returns:
        Dictionary with trade log entry
    """
    invested_amount = amount if amount is not None else equity
    return {
        'Entry Date': date.date() if hasattr(date, 'date') else date,
        'Exit Date': None,
        'Entry Price': round(entry_price, 2) if entry_price else None,
        'Exit Price': round(exit_price, 2) if exit_price else None,
        'Profit %': None,
        'Status': status,
        'Entry_Equity': equity,
        'Entry_Value': invested_amount
    }


def close_trade_log_entry(trade, date, exit_price, current_equity):
    """
    Close a trade log entry with exit information.
    
    Args:
        trade: Trade log entry to close
        date: Exit date
        exit_price: Exit price
        current_equity: Current equity
        
    Returns:
        Updated trade with exit information
    """
    trade['Exit Date'] = date.date() if hasattr(date, 'date') else date
    trade['Exit Price'] = round(exit_price, 2)
    trade['Status'] = 'CLOSED'
    
    if trade.get('Entry Price'):
        trade['Profit %'] = (exit_price / trade['Entry Price'] - 1) * 100
    
    return trade


def create_historical_record(date, strategy_val, bh_val, benchmark_symbol, benchmark_val, cash, close_price, **kwargs):
    """
    Create a historical record entry.
    """
    record = {
        'Date': date,
        'Strategy': strategy_val,
        'BH': bh_val,
        benchmark_symbol: benchmark_val,
        'Cash': cash,
        'close': close_price
    }
    record.update(kwargs)
    return record


def create_results_dataframe(hist):
    """
    Create results DataFrame from historical records.
    
    Args:
        hist: List of historical records
        
    Returns:
        DataFrame with results and returns
    """
    if not hist:
        return pd.DataFrame()
    
    report_df = pd.DataFrame(hist).set_index('Date')
    report_df['ret'] = report_df['Strategy'].pct_change().fillna(0)
    
    return report_df


def create_rolling_returns_df(report_df):
    """
    Create rolling returns DataFrame.
    
    Args:
        report_df: Results DataFrame
        
    Returns:
        DataFrame with rolling returns
    """
    return pd.DataFrame({
        '1-Month': report_df['Strategy'].pct_change(20).fillna(0),
        '3-Month': report_df['Strategy'].pct_change(60).fillna(0),
        '6-Month': report_df['Strategy'].pct_change(120).fillna(0),
        '1-Year': report_df['Strategy'].pct_change(252).fillna(0)
    }, index=report_df.index)


def create_cash_pct_df(report_df):
    """
    Create cash percentage DataFrame.
    
    Args:
        report_df: Results DataFrame
        
    Returns:
        DataFrame with cash percentages
    """
    return pd.DataFrame({
        'Cash %': report_df['Cash'] / report_df['Strategy']
    }, index=report_df.index).fillna(0)


def calculate_pnl_from_trades(trade_log):
    """
    Calculate P&L from closed trades.
    
    Args:
        trade_log: List of trade log entries
        
    Returns:
        List of P&L values
    """
    pnl_dollars = []
    pnl_percentages = []
    for trade in trade_log:
        if trade['Status'] == 'CLOSED' and trade.get('Entry Price') is not None:
            entry_price = trade['Entry Price']
            exit_price = trade.get('Exit Price', 0)
            # Use Entry_Value (amount invested) if available, otherwise fallback to Entry_Equity
            invested_val = trade.get('Entry_Value', trade.get('Entry_Equity', 0))
            
            if entry_price > 0:
                pnl = (exit_price - entry_price) * (invested_val / entry_price)
                pnl_dollars.append(pnl)
                pnl_percentages.append((exit_price / entry_price - 1) * 100)
    return pnl_dollars, pnl_percentages


def validate_price_data_and_get_defaults(prices, start_date, initial_capital, benchmark_symbol="SPY"):
    """
    Validate price data and return empty results if invalid.
    
    Args:
        prices: Price DataFrame
        start_date: Start date for data
        initial_capital: Initial capital
        benchmark_symbol: Benchmark symbol
        
    Returns:
        Tuple of (is_valid, df_results, rolling_returns_df, cash_pct_df)
    """
    if prices.empty:
        df_results = create_empty_results_df(start_date, initial_capital, benchmark_symbol)
        rolling_returns_df = pd.DataFrame(index=df_results.index)
        cash_pct_df = pd.DataFrame({'Cash %': df_results['Cash'] / df_results['Strategy']}, 
                                   index=df_results.index).fillna(0)
        return False, df_results, rolling_returns_df, cash_pct_df
    
    return True, None, None, None


# ==========================================
# DATA LOADING UTILITIES
# ==========================================

def load_breadth_data(path):
    """
    Load and process breadth data from CSV file.
    
    Args:
        path: Path to the breadth CSV file
        
    Returns:
        DataFrame with Date index and highlowq column
    """
    raw_df = pd.read_csv(path)
    clean_rows = []
    for _, row in raw_df.iterrows():
        parts = str(row['Message']).split(',')
        if len(parts) >= 3:
            date_str = parts[0].strip()
            highs = float(parts[1].strip())
            lows = float(parts[2].strip())
            diff = highs - lows
            clean_rows.append([date_str, diff])

    df = pd.DataFrame(clean_rows, columns=['Date', 'highlowq'])
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    return df.sort_index()


# ==========================================
# INDICATOR CALCULATION UTILITIES
# ==========================================

def calculate_sma(series, window):
    """
    Calculate Simple Moving Average.
    
    Args:
        series: Price series
        window: Window size
        
    Returns:
        SMA series
    """
    return series.rolling(window).mean()


def calculate_ema(series, span):
    """
    Calculate Exponential Moving Average.
    
    Args:
        series: Price series
        span: Span for EMA
        
    Returns:
        EMA series
    """
    return series.ewm(span=span, adjust=False).mean()


def calculate_volatility(returns, window=20, annualize=True):
    """
    Calculate rolling volatility.
    
    Args:
        returns: Return series
        window: Rolling window size
        annualize: Whether to annualize the volatility
        
    Returns:
        Volatility series
    """
    vol = returns.rolling(window).std()
    if annualize:
        vol = vol * np.sqrt(252)
    return vol


def calculate_rolling_max(series, window=None):
    """
    Calculate rolling maximum (expanding if window is None).
    
    Args:
        series: Price series
        window: Window size (None for expanding)
        
    Returns:
        Rolling max series
    """
    if window is None:
        return series.expanding().max()
    return series.rolling(window).max()


def calculate_drawdown(series):
    """
    Calculate drawdown from peak.
    
    Args:
        series: Price/equity series
        
    Returns:
        Drawdown series
    """
    peak = series.expanding().max()
    return (series - peak) / peak


def calculate_quarterly_change(series, periods=63):
    """
    Calculate quarterly (approximately 63 trading days) percentage change.
    
    Args:
        series: Price series
        periods: Number of periods (63 for quarterly)
        
    Returns:
        Quarterly change series
    """
    return series.pct_change(periods=periods)


def format_readable_number(num):
    """Format large numbers with K, M, B suffixes."""
    if num is None or pd.isna(num):
        return "N/A"
    abs_num = abs(num)
    if abs_num >= 1_000_000_000:
        return f"{num / 1_000_000_000:.2f}B"
    elif abs_num >= 1_000_000:
        return f"{num / 1_000_000:.2f}M"
    elif abs_num >= 1_000:
        return f"{num / 1_000:.2f}K"
    else:
        return f"{num:.2f}"
