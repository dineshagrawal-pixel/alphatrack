import yfinance as yf
import pandas as pd

ticker = "SPY"
data = yf.download(ticker, start="2024-01-01", progress=False)
idx = data.index[0]
print(f"YF FIRST DATE: {idx}")
print(f"YF TZ: {data.index.tz}")

test_date = "2024-01-02T09:30:00.000-05:00"
dt = pd.to_datetime(test_date, utc=True).tz_convert(None).normalize()
print(f"NORMALIZED BREADTH DATE: {dt}")
print(f"MERGE POSSIBLE: {dt == idx}")
