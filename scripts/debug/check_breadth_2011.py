from common import load_breadth_data_v2
import pandas as pd

breadth = load_breadth_data_v2('pine-logs-High_Low-data.csv', exchange='Nasdaq')
date_2011 = '2011-03-10'
print("Breadth Neighborhood:")
print(breadth.loc['2011-03-08':'2011-03-12'])
