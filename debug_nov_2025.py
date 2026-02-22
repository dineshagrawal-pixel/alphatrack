import pandas as pd
import numpy as np
from strategies.breadth import run_breadth_backtest
from common import load_breadth_data

BREADTH_FILE_PATH = 'pine-logs-TTS.csv'

def diagnostic():
    # Load breadth data
    breadth = load_breadth_data(BREADTH_FILE_PATH)
    
    # Calculate indicators as done in breadth.py
    SIGNAL_SMA_LEN = 5
    SIGNAL_EMA_LEN = 3
    PRICE_EMA_LEN = 3
    PRICE_SMA_LEN = 200

    df = breadth.copy()
    sma5 = df['highlowq'].rolling(SIGNAL_SMA_LEN).mean()
    sma3 = sma5.rolling(3).mean()
    df['highlowqag'] = sma3.ewm(span=SIGNAL_EMA_LEN, adjust=False).mean()
    
    # Look at Nov 2025
    nov_data = df[df.index.month == 11]
    nov_data = nov_data[nov_data.index.year == 2025]
    
    print("--- Market Breadth Diagnostics: Nov 2025 ---")
    print(nov_data[['highlowq', 'highlowqag']])
    
    # Check for zero crossings
    for i in range(1, len(nov_data)):
        prev_sig = nov_data.iloc[i-1]['highlowqag']
        curr_sig = nov_data.iloc[i]['highlowqag']
        if prev_sig <= 0 and curr_sig > 0:
            print(f"BUY SIGNAL detected on {nov_data.index[i]} (Signal: {curr_sig:.2f}%)")
        if prev_sig >= 0 and curr_sig < 0:
            print(f"SELL SIGNAL (Simple < 0) detected on {nov_data.index[i]} (Signal: {curr_sig:.2f}%)")

if __name__ == "__main__":
    diagnostic()
