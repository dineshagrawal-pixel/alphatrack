import yfinance as yf
import pandas as pd
import time

def test_download(symbol):
    print(f"Testing download for {symbol}...")
    try:
        data = yf.download(symbol, start="2024-01-01", progress=False)
        print(f"Download success! Shape: {data.shape}")
        if data.empty:
            print("Warning: Data is empty")
        else:
            print("Columns:", data.columns.tolist())
            print(data.tail())
    except Exception as e:
        print(f"Download failed with exception: {e}")

    print("\nTesting Ticker.history...")
    try:
        t = yf.Ticker(symbol)
        hist = t.history(period="5d")
        print(f"History success! Shape: {hist.shape}")
        if hist.empty:
            print("Warning: History is empty")
        else:
            print(hist.tail())
    except Exception as e:
        print(f"History failed with exception: {e}")

if __name__ == "__main__":
    test_download("AAPL")
    test_download("SPY")
