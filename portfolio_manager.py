import streamlit as st
import pandas as pd
import json
import os
import yfinance as yf
from datetime import datetime, timedelta
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import numpy as np
import io

from strategies import STRATEGIES

PORTFOLIO_FILE = 'my_portfolio_pro.json'
CONFIG_FILE    = 'strategy_configs.json'


# ─────────────────────────────────────────────
#  I/O helpers
# ─────────────────────────────────────────────

def load_json(path: str) -> dict:
    if os.path.exists(path):
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def save_json(path: str, data: dict):
    with open(path, 'w') as f:
        json.dump(data, f, indent=4, default=str)


def _create_backup(db: dict):
    """Save a timestamped backup of the portfolio database in the .backups/ folder."""
    try:
        os.makedirs('.backups', exist_ok=True)
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        with open(f'.backups/portfolio_{ts}.json', 'w') as f:
            json.dump(db, f, indent=4, default=str)
    except Exception:
        pass

def _sync_equity_curve(acct, current_val):
    """Purge snapshots not in the ledger and add a fresh point for today."""
    trades = acct.get('trades', [])
    if not trades:
        acct['equity_curve'] = []
        return
    
    # Identify valid dates from ledger and setup
    valid_dates = set(t['date'] for t in trades)
    if 'initial_date' in acct:
        valid_dates.add(str(pd.to_datetime(acct['initial_date']).date()))
    
    # Purge snapshots that don't match or have zero equity (corrupted)
    curve = [p for p in acct.get('equity_curve', []) if p['date'] in valid_dates and p['equity'] > 0]
    
    # Add/Update current point
    today_str = str(datetime.now().date())
    curve = [p for p in curve if p['date'] != today_str]
    curve.append({'date': today_str, 'equity': round(current_val, 2)})
    
    curve.sort(key=lambda x: x['date'])
    acct['equity_curve'] = curve


# ─────────────────────────────────────────────
#  Live price  (cached 5 min)
# ─────────────────────────────────────────────

@st.cache_data(ttl=300)
def get_live_price(ticker: str) -> float:
    if not ticker:
        return 0.0
    try:
        t = yf.Ticker(ticker)
        try:
            p = t.fast_info['last_price']
            if p and not np.isnan(p):
                return float(p)
        except Exception:
            pass
        hist = t.history(period='1d')
        if not hist.empty:
            return float(hist['Close'].iloc[-1])
    except Exception:
        pass
    return 0.0


# ─────────────────────────────────────────────
#  Background backtest for advisor signal
# ─────────────────────────────────────────────

def run_signal(strategy_name: str, profile_name: str):
    """Return a BacktestResult for the given strategy/profile, capped at 5 years."""
    configs = load_json(CONFIG_FILE)
    saved   = configs.get(strategy_name, {}).get(profile_name, {})
    info    = STRATEGIES.get(strategy_name)
    if not info:
        return None

    # Standardize on a sliding 5-year window for signals
    end_date   = datetime.now().date()
    start_date = end_date - timedelta(days=5 * 365)

    params = {}
    for p in info['params']:
        name = p['name']
        val  = saved.get(name, p.get('default', p.get('value')))
        if p['type'] == 'date_input':
            if 'start' in name.lower(): val = start_date
            elif isinstance(val, str):
                try: val = pd.to_datetime(val).date()
                except Exception: pass
        params[name] = p['transform'](val) if 'transform' in p else val

    try:
        res = info['function'](**params)
        res.loaded_at = datetime.now()
        return res
    except Exception:
        return None


# ─────────────────────────────────────────────
#  Performance Engine (Professional Metrics)
# ─────────────────────────────────────────────

def calculate_performance_metrics(equity_curve: pd.DataFrame, initial_date: str = None):
    """Calculate professional metrics from an equity curve DataFrame."""
    if equity_curve.empty or len(equity_curve) < 2:
        return {'CAGR': 0.0, 'Sharpe': 0.0, 'MaxDD': 0.0, 'Volatility': 0.0}
    
    ec = equity_curve.sort_values('date').copy()
    ec['date'] = pd.to_datetime(ec['date'])
    ec.set_index('date', inplace=True)
    
    daily = ec['equity'].resample('D').last().ffill()
    returns = daily.pct_change().dropna()
    
    # Use initial_date for CAGR if older than first data point
    start_dt = daily.index[0]
    if initial_date:
        candidate = pd.to_datetime(initial_date)
        if candidate < start_dt: start_dt = candidate
        
    days = (daily.index[-1] - start_dt).days
    if days <= 0: return {'CAGR': 0.0, 'Sharpe': 0.0, 'MaxDD': 0.0, 'Volatility': 0.0}
    
    total_ret = daily.iloc[-1] / daily.iloc[0]
    cagr = (total_ret ** (365.25 / days)) - 1 if daily.iloc[0] > 0 else 0.0
    
    vol = returns.std() * np.sqrt(252)
    sharpe = (cagr / vol) if vol > 0 else 0.0
    
    peak = daily.cummax()
    dd = (daily / peak - 1).fillna(0)
    max_dd = dd.min()
    
    return {
        'CAGR': cagr, 'Sharpe': sharpe, 'MaxDD': max_dd, 
        'Volatility': vol, 'DrawdownSeries': dd
    }


# ─────────────────────────────────────────────
#  Position Engine (Multi-positions)
# ─────────────────────────────────────────────

def get_account_state(trades: list):
    """Derive current positions, cash, and cost basis from trade history."""
    positions = {}
    cash = 0.0
    
    for t in trades:
        t_type = t.get('type', 'TRADE')
        t_qty = float(t.get('shares', 0.0))
        t_price = float(t.get('price', 0.0))
        t_cash = float(t.get('cash_delta', 0.0))
        ticker = str(t.get('ticker', '')).upper()

        cash += t_cash

        if t_type in ['BUY', 'SELL', 'TRADE'] and ticker:
            if ticker not in positions:
                positions[ticker] = {'shares': 0.0, 'total_cost': 0.0}
            
            p = positions[ticker]
            old_shares = p['shares']
            p['shares'] += t_qty

            if t_qty > 0: # BUY
                p['total_cost'] += (t_qty * t_price)
            elif t_qty < 0 and old_shares > 0: # SELL
                ratio = abs(t_qty) / old_shares
                p['total_cost'] -= (p['total_cost'] * min(ratio, 1.0))
            
            if p['shares'] <= 1e-8:
                p['shares'] = 0.0
                p['total_cost'] = 0.0

        elif t_type == 'SPLIT' and ticker:
            if ticker in positions:
                # 'Price' field is used as the split ratio (e.g. 10.0 for 10-for-1)
                ratio = float(t.get('price', 1.0))
                positions[ticker]['shares'] *= ratio
                # Total cost basis remains the same, but is spread over more shares

        elif t_type in ['BONUS', 'DIV_REINVEST'] and ticker:
            if ticker not in positions:
                positions[ticker] = {'shares': 0.0, 'total_cost': 0.0}
            positions[ticker]['shares'] += t_qty
            # Cost basis doesn't increase for bonus shares, it dilutes the average cost.

        elif t_type == 'CORRECTION':
            if ticker:
                if ticker not in positions:
                    positions[ticker] = {'shares': 0.0, 'total_cost': 0.0}
                positions[ticker]['shares'] += t_qty
                if positions[ticker]['shares'] < 0: positions[ticker]['shares'] = 0.0

    return {
        'positions': {k: v for k, v in positions.items() if v['shares'] > 0},
        'cash': cash
    }


# ─────────────────────────────────────────────
#  Data Migration & Loading
# ─────────────────────────────────────────────

def _normalise_account(acct: dict) -> dict:
    """Ensure account dict has all necessary keys for the redesigned UI."""
    acct.setdefault('trades', [])
    acct.setdefault('equity_curve', [])
    acct.setdefault('linked_profile', 'None')
    acct.setdefault('drift_threshold', 5.0)
    acct.setdefault('initial_date', str(datetime.now().date()))
    
    acct.pop('shares', None)
    acct.pop('cash', None)
    
    return acct


def load_portfolio() -> dict:
    raw = load_json(PORTFOLIO_FILE)
    raw.setdefault('accounts', {})
    for name in raw['accounts']:
        raw['accounts'][name] = _normalise_account(raw['accounts'][name])
    return raw


def save_portfolio(db: dict):
    save_json(PORTFOLIO_FILE, db)


# ─────────────────────────────────────────────
#  Main Rendering function
# ─────────────────────────────────────────────

def render_portfolio_manager(account_name: str):
    try:
        db   = load_portfolio()
        acct = db['accounts'].get(account_name)
        if acct is None:
            st.error(f"Account '{account_name}' not found.")
            return

        uid = account_name.replace(' ', '_')

        # ── State Calculation ──────────────────────────────────────
        state = get_account_state(acct['trades'])
        positions = state['positions']
        curr_cash = state['cash']
        
        total_market_val = 0.0
        pos_details = []
        for tk, pos in positions.items():
            price = get_live_price(tk)
            val = pos['shares'] * price
            total_market_val += val
            gain = val - pos['total_cost']
            pos_details.append({
                'Ticker': tk, 'Shares': pos['shares'], 'Price': price, 
                'Value': val, 'Cost': pos['total_cost'], 'G/L': gain
            })
            
        total_val = total_market_val + curr_cash
        stock_pct = (total_market_val / total_val * 100) if total_val > 0 else 0.0
        total_cost = sum(d['Cost'] for d in pos_details)
        total_gl = total_val - total_cost if total_cost > 0 else 0.0
        
        # Ensure history is synced with the current ledger before rendering
        _sync_equity_curve(acct, total_val)
        
        # ── Resolve Signals ────────────────────────────────────────
        linked = acct.get('linked_profile', 'None')
        has_link = linked and linked != 'None'
        sig_data = None
        target_pct = None
        
        if has_link:
            parts = linked.split(' | ', 1)
            sig_data = run_signal(parts[0], parts[1] if len(parts)>1 else 'default')
            if sig_data:
                last_row = sig_data.df_results.iloc[-1]
                bt_val = float(last_row['Strategy'])
                cash_val = float(last_row.get('Cash', 0.0))
                target_pct = ((bt_val - cash_val) / bt_val * 100) if bt_val > 0 else 0.0

        # ═════════════════════════════════════════════════════════════
        #  UI LAYOUT
        # ═════════════════════════════════════════════════════════════
        st.markdown(f"## 💼 {account_name}")
        
        tab_dash, tab_holdings, tab_ledger, tab_setup = st.tabs([
            "📊 Dashboard", "💼 Holdings & Advisor", "📜 Ledger", "⚙️ Setup"
        ])

        with tab_dash:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Value", f"${total_val:,.2f}")
            m2.metric("Total Return", f"${total_gl:+,.2f}", 
                      delta=f"{(total_gl/total_cost*100):.2f}%" if total_cost > 0 else None)
            m3.metric("Cash", f"${curr_cash:,.2f}")
            m4.metric("Stock %", f"{stock_pct:.1f}%")
            
            st.write("")
            ec_df = pd.DataFrame(acct['equity_curve'])
            perf = calculate_performance_metrics(ec_df, acct.get('initial_date'))
            
            col_chart, col_perf = st.columns([3, 1])
            with col_chart:
                if ec_df.empty:
                    st.info("No history yet.")
                else:
                    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                                        vertical_spacing=0.08, row_heights=[0.7, 0.3])
                    ec_df['date'] = pd.to_datetime(ec_df['date'])
                    ec_df = ec_df.sort_values('date')
                    indexed_p = (ec_df['equity'] / ec_df['equity'].iloc[0]) * 100
                    fig.add_trace(go.Scatter(x=ec_df['date'], y=indexed_p, name="Portfolio", 
                                             line=dict(color='#e67e22', width=3)), row=1, col=1)
                    if sig_data is not None:
                        dr = sig_data.df_results
                        shared_start = max(ec_df['date'].min(), dr.index.min())
                        dr_sync = dr[dr.index >= shared_start]
                        if not dr_sync.empty:
                            indexed_s = (dr_sync['Strategy'] / dr_sync['Strategy'].iloc[0]) * 100
                            fig.add_trace(go.Scatter(x=dr_sync.index, y=indexed_s, name="Strategy", 
                                                     line=dict(color='#1f4e9c', dash='dot')), row=1, col=1)
                    if 'DrawdownSeries' in perf:
                        dd = perf['DrawdownSeries']
                        fig.add_trace(go.Scatter(x=dd.index, y=dd*100, name="Drawdown", fill='tozeroy', 
                                                 fillcolor='rgba(231, 76, 60, 0.3)', line=dict(color='#e74c3c', width=1)), row=2, col=1)
                    fig.update_layout(height=450, margin=dict(t=30, b=30, l=10, r=10), template='plotly_white', hovermode='x unified')
                    st.plotly_chart(fig, use_container_width=True)
            with col_perf:
                st.markdown("##### Performance")
                st.metric("CAGR", f"{perf['CAGR']*100:.2f}%")
                st.metric("Max Drawdown", f"{perf['MaxDD']*100:.1f}%")
                st.metric("Sharpe", f"{perf['Sharpe']:.2f}")

        with tab_holdings:
            st.subheader("Current Holdings")
            if not pos_details: st.info("No active positions.")
            else:
                df_p = pd.DataFrame(pos_details)
                df_p['Weight'] = (df_p['Value'] / total_val * 100).round(1)
                st.dataframe(df_p.style.format({'Shares': '{:.4f}', 'Price': '${:,.2f}', 'Value': '${:,.2f}', 'Weight': '{:.1f}%'}), use_container_width=True)
            st.divider()
            st.subheader("🎯 Advisor")
            if has_link and sig_data:
                symbol = getattr(sig_data, 'symbol', 'UNKNOWN')
                cur_sh = positions.get(symbol, {}).get('shares', 0.0)
                p_price = get_live_price(symbol)
                if target_pct is not None and p_price > 0:
                    ideal_val = total_val * (target_pct / 100)
                    diff = (ideal_val / p_price) - cur_sh
                    drift = stock_pct - target_pct
                    if abs(drift) <= acct.get('drift_threshold', 5.0):
                        st.success(f"✅ Balanced ({drift:+.1f}% drift)")
                    else:
                        verb = "BUY" if diff > 0 else "SELL"
                        st.warning(f"👉 Recommended: **{verb} {abs(diff):,.4f} {symbol}** (~${abs(diff*p_price):,.2f})")

        with tab_ledger:
            l1, l2 = st.columns([1, 4])
            with l1:
                with st.popover("➕ Log Entry", use_container_width=True):
                    e_type = st.selectbox("Type", ["BUY", "SELL", "CASH", "SPLIT", "BONUS", "CORRECTION"])
                    e_tick = st.text_input("Ticker").upper()
                    e_date = st.date_input("Date")
                    
                    if e_type == "SPLIT":
                        st.info("ℹ️ Enter the split ratio in the **Ratio** (Price) field. E.g. For a 10-for-1 split, enter **10.0**.")
                    
                    l_c1, l_c2 = st.columns(2)
                    e_qty = l_c1.number_input("Shares / Amount", format="%.4f", value=0.0)
                    e_val = float(get_live_price(e_tick)) if e_tick and e_type != "SPLIT" else 1.0
                    e_price = l_c2.number_input("Price / Ratio", value=e_val, format="%.4f")
                    
                    if st.button("🚀 Confirm", use_container_width=True):
                        if e_type == "SELL" and e_tick:
                            if e_qty > positions.get(e_tick, {}).get('shares', 0.0):
                                st.error("Insufficient shares!"); st.stop()
                        
                        acct['trades'].append({'date': str(e_date), 'type': e_type, 'ticker': e_tick, 
                                               'shares': e_qty, 'price': e_price, 
                                               'cash_delta': (1 if e_type=="SELL" else -1)*(e_qty*e_price) if e_type in ["BUY","SELL"] else e_qty})
                        
                        # Rebuild state and history to ensure consistency
                        state = get_account_state(acct['trades'])
                        prices = {k: get_live_price(k) for k in state['positions'].keys()}
                        new_mv = sum(p['shares']*prices.get(k, 1.0) for k, p in state['positions'].items())
                        _sync_equity_curve(acct, new_mv + state['cash'])
                        
                        _create_backup(db); save_portfolio(db); st.rerun()
            
            if acct['trades']:
                st.info("💡 **Tip:** To delete entries, select the row(s) and press **'Delete'** on your keyboard, then click **'Save Changes'**.")
                df_h = pd.DataFrame(acct['trades'])
                edited = st.data_editor(df_h, num_rows="dynamic", use_container_width=True)
                if st.button("💾 Save Changes"):
                    acct['trades'] = edited.to_dict('records')
                    # Force sync after editing/deleting trades to remove stale snapshots
                    state = get_account_state(acct['trades'])
                    curr_mv = sum(p['shares']*get_live_price(k) for k, p in state['positions'].items())
                    _sync_equity_curve(acct, curr_mv + state['cash'])
                    _create_backup(db); save_portfolio(db); st.rerun()

        with tab_setup:
            st.subheader("Account Calibration")
            new_name = st.text_input("Name", value=account_name)
            init_d = st.date_input("Initial Date", value=pd.to_datetime(acct.get('initial_date', datetime.now())).date())
            if st.button("Update"):
                if new_name != account_name: db['accounts'][new_name] = db['accounts'].pop(account_name); st.session_state.pm_account = new_name
                acct['initial_date'] = str(init_d); save_portfolio(db); st.rerun()
            st.divider()
            if st.button("🗑️ Delete Account", type="primary"):
                if len(db['accounts']) > 1: del db['accounts'][account_name]; save_portfolio(db); st.rerun()

    except Exception as e:
        import traceback; st.error(f"Error: {e}"); st.code(traceback.format_exc())
