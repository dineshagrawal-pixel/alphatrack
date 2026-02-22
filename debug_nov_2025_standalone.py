import pandas as pd
import numpy as np

BREADTH_FILE_PATH = 'pine-logs-TTS.csv'

def load_breadth_data_local(path):
    raw_df = pd.read_csv(path)
    clean_rows = []
    for _, row in raw_df.iterrows():
        parts = str(row['Message']).split(',')
        if len(parts) >= 3:
            date_str = parts[0].strip()
            highs = float(parts[1].strip())
            lows = float(parts[2].strip())
            # Normalize to percentage range [-100, 100]
            total = highs + lows
            diff_pct = ((highs - lows) / total * 100) if total > 0 else 0
            clean_rows.append([date_str, diff_pct])

    df = pd.DataFrame(clean_rows, columns=['Date', 'highlowq'])
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    return df.sort_index()

def diagnostic():
    # Load breadth data
    df = load_breadth_data_local(BREADTH_FILE_PATH)
    
    # Calculate indicators
    SIGNAL_SMA_LEN = 5
    SIGNAL_EMA_LEN = 3

    sma5 = df['highlowq'].rolling(SIGNAL_SMA_LEN).mean()
    sma3 = sma5.rolling(3).mean()
    df['highlowqag'] = sma3.ewm(span=SIGNAL_EMA_LEN, adjust=False).mean()
    
    # Look at Nov 2025
    nov_data = df[df.index.month == 11]
    nov_data = nov_data[nov_data.index.year == 2025]
    
    print("--- Market Breadth Diagnostics: Nov 2025 ---")
    print(nov_data[['highlowq', 'highlowqag']])
    
    # Check for zero crossings
    print("\nSignals:")
    for i in range(1, len(nov_data)):
        prev_sig = nov_data.iloc[i-1]['highlowqag']
        curr_sig = nov_data.iloc[i]['highlowqag']
        if prev_sig <= 0 and curr_sig > 0:
            print(f"BUY SIGNAL detected on {nov_data.index[i]} (Signal: {curr_sig:.2f}%)")
        if prev_sig >= 0 and curr_sig < 0:
            print(f"SELL SIGNAL (Simple < 0) detected on {nov_data.index[i]} (Signal: {curr_sig:.2f}%)")

if __name__ == "__main__":
    diagnostic()
