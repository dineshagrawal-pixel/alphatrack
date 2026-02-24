import pandas as pd
idx = pd.to_datetime(["2010-02-11"])
print(f"Index: {idx}")
print(f"TZ: {idx.tz}")
try:
    idx2 = idx.tz_localize(None)
    print(f"TZ_LOCALIZE(NONE) SUCCESS: {idx2}")
except Exception as e:
    print(f"TZ_LOCALIZE(NONE) FAILED: {e}")
