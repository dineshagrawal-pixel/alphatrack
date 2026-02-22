import yfinance as yf
import pandas as pd
import numpy as np
import strategies # This imports the strategies package (strategies/__init__.py)

# ==========================================================
# USER'S SCRIPT CORE LOGIC
# ==========================================================
CONFIG = {
    "SYMBOL": "TQQQ",
    "SIDE_FUND": "SGOV",
    "START_DATE": "2010-02-11",
    "INIT_CASH": 10000,
    "TARGET_TQQQ_PCT": 0.60,
    "GAP_REBALANCE_PCT": 0.50,
    "Q_GROWTH_TARGET": 1.09,
    "Q_GROWTH_CAP": 1.30,
    "SIGNAL_FLOOR_PCT": 0.88,
    "CASH_FLOOR_HWM_PCT": 0.20,
    "CRASH_THRESHOLD": 0.70,
    "100_UP_THRESHOLD": 2.00,
}

def run_user_script_logic(data):
    q_data = data.resample('QS').first()
    eq_t = CONFIG["INIT_CASH"] * CONFIG["TARGET_TQQQ_PCT"]
    eq_s = CONFIG["INIT_CASH"] * (1 - CONFIG["TARGET_TQQQ_PCT"])
    signal_line = eq_t
    hwm_sig = signal_line
    portfolio_hwm = CONFIG["INIT_CASH"]
    in_30_down = False
    sell_sig_count = 0
    hist = []

    for i in range(len(q_data)):
        p, sp = q_data['Price'].iloc[i], q_data['SidePrice'].iloc[i]
        date = q_data.index[i]
        if i > 0:
            eq_t *= (p / q_data['Price'].iloc[i-1])
            eq_s *= (sp / q_data['SidePrice'].iloc[i-1])

        current_total = eq_t + eq_s
        portfolio_hwm = max(portfolio_hwm, current_total)
        cash_limit = portfolio_hwm * CONFIG["CASH_FLOOR_HWM_PCT"]

        # FORCE CASH REFILL
        if eq_s < cash_limit:
            shortfall = cash_limit - eq_s
            eq_t -= shortfall
            eq_s += shortfall

        # 100-UP RESET
        if i > 0 and (p / q_data['Price'].iloc[i-1]) >= CONFIG["100_UP_THRESHOLD"]:
            total = eq_t + eq_s
            eq_t, eq_s = total * CONFIG["TARGET_TQQQ_PCT"], total * (1 - CONFIG["TARGET_TQQQ_PCT"])
            signal_line = eq_t

        # SIGNAL LINE MATH
        target = min(signal_line * CONFIG["Q_GROWTH_TARGET"], signal_line * CONFIG["Q_GROWTH_CAP"])
        hwm_sig = max(hwm_sig, signal_line)
        signal_line = max(target, hwm_sig * CONFIG["SIGNAL_FLOOR_PCT"])

        # 30-DOWN DETECT
        if p <= (q_data['Price'].iloc[max(0, i-8):i+1].max() * CONFIG["CRASH_THRESHOLD"]):
            in_30_down = True

        # GAP REBALANCE
        full_diff = signal_line - eq_t
        trade_goal = full_diff * CONFIG["GAP_REBALANCE_PCT"]
        if trade_goal > 0:
            available_to_spend = max(0, eq_s - cash_limit)
            buy_amt = min(trade_goal, available_to_spend)
            eq_t += buy_amt
            eq_s -= buy_amt
        elif trade_goal < 0:
            if in_30_down:
                sell_sig_count += 1
                if sell_sig_count >= 2:
                    in_30_down = False
                    total = eq_t + eq_s
                    eq_t, eq_s = total * CONFIG["TARGET_TQQQ_PCT"], total * (1 - CONFIG["TARGET_TQQQ_PCT"])
                    signal_line = eq_t
            else:
                sell_amt = abs(trade_goal)
                eq_t -= sell_amt
                eq_s -= sell_amt

        # MIN ALLOCATION
        total_val = eq_t + eq_s
        if not in_30_down and (eq_t / total_val) < CONFIG["TARGET_TQQQ_PCT"]:
            rebal_req = (total_val * CONFIG["TARGET_TQQQ_PCT"]) - eq_t
            actual_rebal = min(rebal_req, max(0, eq_s - cash_limit))
            eq_t += actual_rebal
            eq_s -= actual_rebal

        hist.append({'Date': date, 'Strategy': eq_t + eq_s, 'Signal': signal_line})
    
    return pd.DataFrame(hist).set_index('Date')

# ==========================================================
# DIAGNOSTIC RUN
# ==========================================================
print("Fetching original data for reference...")
raw = yf.download(["TQQQ", "SGOV", "SHY"], start=CONFIG["START_DATE"], end="2026-02-19", progress=False)
adj = 'Adj Close' if 'Adj Close' in raw.columns else 'Close'
tqqq_p = raw[adj]["TQQQ"].dropna()
side_p = raw[adj]["SGOV"].fillna(raw[adj]["SHY"]).dropna()
data = pd.DataFrame({'Price': tqqq_p, 'SidePrice': side_p}).dropna()

user_res = run_user_script_logic(data)

print("\nRunning App Strategy Logic...")
# Call the function from the package
app_df, _, _, _, _, _, _ = strategies.run_9sig_strategy(
    symbol_val=CONFIG["SYMBOL"], 
    side_fund_val=CONFIG["SIDE_FUND"], 
    start_date_val=CONFIG["START_DATE"], 
    initial_capital_val=CONFIG["INIT_CASH"],
    target_tqqq_pct=CONFIG["TARGET_TQQQ_PCT"], 
    new_money_to_signal=0.60, 
    gap_rebalance_pct=CONFIG["GAP_REBALANCE_PCT"], 
    q_growth_target=CONFIG["Q_GROWTH_TARGET"], 
    q_growth_cap=CONFIG["Q_GROWTH_CAP"], 
    signal_floor_pct=CONFIG["SIGNAL_FLOOR_PCT"], 
    cash_floor_hwm_pct=CONFIG["CASH_FLOOR_HWM_PCT"], 
    crash_threshold=CONFIG["CRASH_THRESHOLD"], 
    up100_threshold=CONFIG["100_UP_THRESHOLD"],
    monthly_dep_val=0
)

# CAGR Calculations
def get_cagr(df, col, start_val):
    years = (df.index[-1] - df.index[0]).days / 365.25
    return ((df[col].iloc[-1] / start_val) ** (1/years) - 1) * 100

print("\n--- FINAL VERIFICATION ---")
print(f"User Logic Final Value: ${user_res['Strategy'].iloc[-1]:,.2f}")
print(f"User Logic CAGR:        {get_cagr(user_res, 'Strategy', CONFIG['INIT_CASH']):.2f}%")
print(f"App Logic Final Value:  ${app_df['Strategy'].iloc[-1]:,.2f}")
print(f"App Logic CAGR:         {get_cagr(app_df, 'Strategy', CONFIG['INIT_CASH']):.2f}%")

if abs(get_cagr(user_res, 'Strategy', CONFIG['INIT_CASH']) - get_cagr(app_df, 'Strategy', CONFIG['INIT_CASH'])) < 0.1:
    print("\nSUCCESS: CAGR Discrepancy Resolved!")
else:
    print("\nFAILURE: CAGR still diverges.")
