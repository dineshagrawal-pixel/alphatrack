# ⚗️ AlphaLab

AlphaLab is a comprehensive investment analysis platform that combines professional-grade **Tactical Asset Allocation (TAA)** backtesting with a live **Portfolio Manager**.

## 🚀 Features

### 1. 📈 Strategy Lab
Research and backtest advanced quantitative strategies with institutional-grade reporting:
- **Breadth Momentum**: Uses market breadth indicators to time regime shifts.
- **Volatility Targeting**: Dynamically adjusts exposure based on market volatility.
- **Moving Average Crossovers**: Classic trend-following with customizable parameters.
- **9-Sig Strategy**: High-conviction tactical rotation.

### 2. 💼 Live Portfolio Manager
Track your actual holdings and compare them against benchmarks:
- **Real-time Metrics**: Track CAGR, Sharpe Ratio, Max Drawdown, and Volatility.
- **Visual Analytics**: Interactive growth charts with 1Y, 3Y, 5Y, and MAX zoom presets.
- **Correlation Analysis**: Check how your portfolio moves relative to SPY.
- **Performance Heatmaps**: Monthly and annual return visualizations.

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/your-username/alphalab.git

# Navigate to the project directory
cd alphalab

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

## 📊 Reporting Metrics
AlphaLab uses the same metrics as industry-standard tools like Portfolio Visualizer:
- **Risk-Adjusted Returns**: Sharpe, Sortino, Treynor, and Calmar ratios.
- **Downside Analysis**: Ulcer Index, VaR (Historical & Analytical), and Drawdown recovery periods.
- **Captured Gains**: Upside/Downside capture ratios and Payoff ratios.

## 🏗️ Architecture
- **Core**: Python & Streamlit
- **Analysis**: Pandas & NumPy
- **Visualization**: Plotly & Matplotlib
- **Strategies**: Modular design in `strategies/` directory

---
*Built for serious investors who want institutional-grade tools in a simple Streamlit interface.*
