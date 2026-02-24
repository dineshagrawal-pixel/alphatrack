import pandas as pd
date_str = "2010-02-11"
dt = pd.to_datetime(date_str, utc=True)
print(f"UTC=True: {dt}")
dt2 = dt.tz_convert(None)
print(f"TZ_CONVERT(NONE): {dt2}")
dt3 = dt2.normalize()
print(f"NORMALIZE: {dt3}")
