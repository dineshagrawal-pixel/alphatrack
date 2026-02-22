# Strategy Development Guide: Business Logic

This document provides a non-technical overview of the business logic and investment philosophy for each strategy in the Backtest Lab.

---

## 1. 9 Signal Strategy (Triple-Q Side Fund)
**Philosophy**: A wealth-preservation and growth strategy designed to manage high-volatility assets like TQQQ by using a "Signal Line" to systematically bank profits into a safe-haven side fund.

### Key Logic:
- **The Signal Line**: A mathematical target for your TQQQ position. If TQQQ grows faster than the target, you sell the excess into cash. If it falls below, you use cash to "buy the dip" back up to the line.
- **Home Base**: A preferred allocation (e.g., 60% TQQQ / 40% Side Fund). The strategy naturally drifts back here during extreme moves.
- **30-Down Protection**: If the market crashes (70% drop from 2-year highs), the strategy enters a protective mode, pausing sales to avoid liquidation at the bottom.
- **100-Up Reset**: If TQQQ doubles in a single quarter, the strategy "banks" all excess gains immediately to lock in the windfall.

---

## 2. Market Breadth Strategy
**Philosophy**: Uses "Internal Strength" (Breadth) to determine market health. It follows the principle that a healthy rally requires many stocks to participate, not just a few leaders.

### Key Logic:
- **High-Low Differential**: Measures the number of stocks hitting new 52-week highs minus those hitting new lows.
- **Trend Following**: When the differential is positive and growing (Net Highs), the strategy moves into stocks. When Net Lows begin to dominate, the strategy retreats to cash.
- **Momentum Smoothing**: Uses a moving average of the breadth signal to filter out daily "noise" and capture the broader economic tide.

---

## 3. Volatility-Adjusted Strategy
**Philosophy**: Based on the observation that high volatility usually precedes market downturns. It seeks to own stocks during "calm" periods and exit during "turbulent" periods.

### Key Logic:
- **Vol-Switch**: Compares short-term volatility (last 14 days) to long-term volatility (last 200 days).
- **Risk-Off Rule**: If short-term volatility spikes significantly above the long-term average (indicating "panic"), the strategy moves to cash.
- **Trend Confirmation**: Also uses a long-term moving average (200-day) to ensure it only stays in stocks if the broad primary trend is still upward.

---

## 4. Moving Average Crossover
**Philosophy**: A classic "Trend Following" model. Its goal is to stay invested during long bull markets and avoid the "meat grinder" of prolonged bear markets.

### Key Logic:
- **The SMA Filter**: Uses a Simple Moving Average (typically 11-month) as a line in the sand.
- **Stay-In Rule**: As long as the stock price is above the moving average, the trend is considered "Up," and the strategy stays fully invested.
- **Exit Rule**: If the price closes below the moving average, the trend has broken, and the strategy moves to the safety of cash or bonds.

---

## 5. Dual Momentum (GEM)
**Philosophy**: Based on Gary Antonacci's "Global Equities Momentum." It combines the two strongest forces in finance: Relative Momentum (picking the best performer) and Absolute Momentum (staying positive).

### Key Logic:
- **Relative Power**: Look back at the last 12 months and compare the Growth Asset (Stocks) vs. the Defensive Asset (Bonds). Pick the one with the higher total return.
- **The Stress Test**: Even if Stocks are better than Bonds, if Stocks have a *negative* return over the last 12 months, the strategy moves to Cash entirely.
- **Monthly Rotation**: Only rebalances once a month to reduce trading costs and allow momentum trends time to develop.
