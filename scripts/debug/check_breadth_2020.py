from common import load_breadth_data_v2
import pandas as pd

breadth = load_breadth_data_v2('pine-logs-High_Low-data.csv', exchange='Nasdaq')
start = '2020-03-01'
end = '2020-04-01'
print(breadth.loc[start:end])
