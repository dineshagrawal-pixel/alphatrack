import sys
import os
import pandas as pd
import numpy as np

# Add parent directory to path to import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from strategies.moving_average import run_moving_average_strategy
from common import download_price_data, get_data_start_date

symbol = "TQQQ"
start_date = "2010-02-11"
download_start = get_data_start_date(start_date)

print(f"Downloading from {download_start}")
prices = download_price_data(symbol, download_start)
print(f"Prices count: {len(prices)}")
print(f"First date: {prices.index[0]}")
print(f"Last date: {prices.index[-1]}")

result = run_moving_average_strategy(
    symbol_val=symbol,
    start_date_val=pd.to_datetime(start_date).date(),
    initial_capital_val=10000.0,
    lookback_period_months=11,
    cash_yield_apr=0.0,
    eval_frequency="Monthly"
)

df = result.df_results
print(f"Results count: {len(df)}")
print(f"Final Strategy Value: {df['Strategy'].iloc[-1]}")
print(f"Final BH Value: {df['BH'].iloc[-1]}")
