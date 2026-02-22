import yfinance as yf
import pandas as pd

ticker = "TQQQ"
start = "2010-01-01"

print(f"Testing yfinance download for {ticker} (no session)...")
try:
    data = yf.download(ticker, start=start, progress=False, auto_adjust=True)
    print(f"Columns type: {type(data.columns)}")
    print(f"Columns: {data.columns}")
    print(f"Empty: {data.empty}")
    if not data.empty:
        print(f"First few rows:\n{data.head()}")
except Exception as e:
    print(f"Error: {e}")

print("\nTesting multiple tickers (no session)...")
try:
    data_multi = yf.download(["TQQQ", "SPY"], start=start, progress=False, auto_adjust=True)
    print(f"Multi Columns: {data_multi.columns}")
    print(f"Multi Empty: {data_multi.empty}")
except Exception as e:
    print(f"Multi Error: {e}")
