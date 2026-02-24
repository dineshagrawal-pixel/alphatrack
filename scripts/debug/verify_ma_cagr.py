import yfinance as yf
import pandas as pd
import numpy as np
from strategies.moving_average import run_moving_average_strategy

# SETTINGS FROM USER/PV
SYMBOL = "TQQQ"
START_DATE = "2017-01-03" 
END_DATE = "2024-12-31" # Match PV end date
LOOKBACK_MONTHS = 11
CASH_YIELD = 0.04
BENCHMARK = "SPY"
INIT_CAPITAL = 10000

print(f"Running Moving Average Strategy Discovery for {SYMBOL} since {START_DATE} to {END_DATE}...")

# Run Strategy
report_df, trades, pnl, init, rolling, cash, sym = run_moving_average_strategy(
    symbol_val=SYMBOL,
    start_date_val=START_DATE,
    initial_capital_val=INIT_CAPITAL,
    lookback_period_months=LOOKBACK_MONTHS,
    cash_yield_apr=CASH_YIELD,
    benchmark_symbol=BENCHMARK
)

# Filter report to END_DATE
report_df = report_df[report_df.index <= pd.to_datetime(END_DATE)]

# Calculate CAGR
days = (report_df.index[-1] - report_df.index[0]).days
years = days / 365.25

final_val = report_df['Strategy'].iloc[-1]
cagr = ((final_val / INIT_CAPITAL) ** (1/years) - 1) * 100

bh_final = report_df['BH'].iloc[-1]
bh_cagr = ((bh_final / INIT_CAPITAL) ** (1/years) - 1) * 100

bench_final = report_df[BENCHMARK].iloc[-1]
bench_cagr = ((bench_final / INIT_CAPITAL) ** (1/years) - 1) * 100

print("-" * 30)
print(f"Strategy CAGR:   {cagr:.2f}% (PV Target: 50.31%)")
print(f"Buy & Hold CAGR: {bh_cagr:.2f}% (PV Target: 40.88%)")
print(f"Benchmark CAGR:  {bench_cagr:.2f}% (PV Target: 14.70%)")
print("-" * 30)
