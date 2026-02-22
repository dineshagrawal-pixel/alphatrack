import yfinance as yf
import pandas as pd
import requests

def get_yf_session():
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    })
    return session

ticker = "TQQQ"
start = "2010-01-01"

print(f"Testing yfinance download for {ticker}...")
try:
    data = yf.download(ticker, start=start, progress=False, auto_adjust=True, session=get_yf_session())
    print(f"Columns: {data.columns}")
    print(f"Empty: {data.empty}")
    if not data.empty:
        print(f"First few rows:\n{data.head()}")
except Exception as e:
    print(f"Error: {e}")

print("\nTesting multiple tickers...")
try:
    data_multi = yf.download(["TQQQ", "SPY"], start=start, progress=False, auto_adjust=True, session=get_yf_session())
    print(f"Multi Columns: {data_multi.columns}")
    print(f"Multi Empty: {data_multi.empty}")
except Exception as e:
    print(f"Multi Error: {e}")
