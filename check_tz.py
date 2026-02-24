import yfinance as yf
import pandas as pd

ticker = "SPY"
data = yf.download(ticker, start="2024-01-01", progress=False)
print("YF Index Type:", type(data.index))
print("YF Index Example:", data.index[0])
print("YF Index Timezone:", data.index.tz)

test_date = "2024-01-02T09:30:00.000-05:00"
dt = pd.to_datetime(test_date, utc=True).dt.tz_convert(None).dt.normalize() if hasattr(pd.Series([test_date]), 'dt') else pd.to_datetime(test_date, utc=True).tz_convert(None).normalize()
print("Normalized Date:", dt)
print("Normalized Date Type:", type(dt))
