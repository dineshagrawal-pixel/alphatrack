from common import download_price_data, download_multiple_tickers
import pandas as pd

ticker = "TQQQ"
start = "2010-01-01"

print(f"Testing download_price_data for {ticker}...")
try:
    data = download_price_data(ticker, start)
    print(f"Columns: {data.columns}")
    if not data.empty:
        print(f"Success! Data head:\n{data.head(1)}")
    else:
        print("FAILED: Data is empty")
except Exception as e:
    print(f"Error: {e}")

print("\nTesting download_multiple_tickers...")
try:
    data_multi = download_multiple_tickers(["TQQQ", "SPY"], start)
    print(f"Multi Columns: {data_multi.columns}")
    if not data_multi.empty:
        print(f"Success! Multi Data head:\n{data_multi.head(1)}")
    else:
        print("FAILED: Multi Data empty")
except Exception as e:
    print(f"Multi Error: {e}")
