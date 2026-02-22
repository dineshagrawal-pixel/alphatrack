import streamlit as st
import pandas as pd
import json
import os
from datetime import datetime

from strategies import STRATEGIES
from reporting import generate_backtest_report

st.set_page_config(layout="wide", page_title="Strategy Backtester")

# --- Configuration Persistence ---
CONFIG_FILE = "strategy_configs.json"
PORTFOLIO_FILE = "my_portfolio_pro.json"

def load_configs():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r") as f:
                return json.load(f)
        except: pass
    return {}

def save_configs(configs):
    with open(CONFIG_FILE, "w") as f:
        json.dump(configs, f, indent=4, default=str)

def load_portfolio():
    if os.path.exists(PORTFOLIO_FILE):
        try:
            with open(PORTFOLIO_FILE, "r") as f:
                return json.load(f)
        except: pass
    return {"strategies": {}}

def save_portfolio(db):
    with open(PORTFOLIO_FILE, "w") as f:
        json.dump(db, f, indent=4, default=str)

# --- Styling ---
st.markdown("""
<style>
.stButton > button {
    background-color: #1f4e9c !important; color: white !important;
    width: 100% !important; border: none !important;
}
.stButton > button:hover { background-color: #153a75 !important; }
</style>
""", unsafe_allow_html=True)

# =========================================================
# SIDEBAR: NAVIGATION (always visible)
# =========================================================
st.sidebar.header("🎯 Navigation")
nav_selection = st.sidebar.radio(
    "Go To:", ["📈 Strategy Laboratory", "💼 Live Portfolio Manager"], key="nav_sel"
)
st.sidebar.markdown("---")

# =========================================================
# SIDEBAR: STRATEGY LAB PANEL
# =========================================================
if nav_selection == "📈 Strategy Laboratory":
    st.sidebar.header("Strategy Selection")
    selected_strategy = st.sidebar.selectbox(
        "Select Strategy", list(STRATEGIES.keys()), key="selected_strategy"
    )

    configs = load_configs()
    if selected_strategy not in configs:
        configs[selected_strategy] = {"default": {}}

    strat_configs = configs[selected_strategy]
    config_names = list(strat_configs.keys())
    if "default" not in config_names:
        strat_configs["default"] = {}
        config_names = ["default"] + config_names

    # --- Backtest Profile ---
    st.sidebar.header("Backtest Profile")
    if 'active_profile_choice' not in st.session_state:
        st.session_state.active_profile_choice = "default"
    if st.session_state.active_profile_choice not in config_names:
        st.session_state.active_profile_choice = "default"

    current_idx = config_names.index(st.session_state.active_profile_choice)
    active_config = st.sidebar.selectbox(
        "Active Profile", config_names,
        index=current_idx,
        help="Select a saved set of parameters to load."
    )
    st.session_state.active_profile_choice = active_config

    p_rename_input = st.sidebar.text_input(
        "Rename / Save As:",
        placeholder="New profile name...",
        key="profile_rename_input",
        help="Type a name to create a new profile, or leave blank to overwrite current."
    )
    if st.sidebar.button("💾 Save Profile Settings", use_container_width=True):
        target_name = p_rename_input.strip() if p_rename_input.strip() else active_config
        configs[selected_strategy][target_name] = st.session_state.get('current_params', {})
        save_configs(configs)
        st.session_state.active_profile_choice = target_name
        st.sidebar.success(f"Saved to '{target_name}'")
        st.rerun()

    st.sidebar.markdown("---")

    # --- Strategy Parameters ---
    strategy_info = STRATEGIES[selected_strategy]
    strategy_params = {}

    st.sidebar.header(f"Parameters: {active_config}")

    # Load saved profile values into session state when profile changes
    if 'last_active_profile' not in st.session_state or st.session_state.last_active_profile != active_config:
        st.session_state.last_active_profile = active_config
        profile_data = strat_configs.get(active_config, {})
        for param in strategy_info["params"]:
            p_name = param["name"]
            if p_name in profile_data:
                p_val = profile_data[p_name]
                if param["type"] == "date_input" and isinstance(p_val, str):
                    try: p_val = pd.to_datetime(p_val).date()
                    except: pass
                st.session_state[f"{selected_strategy}_{active_config}_{p_name}_input"] = p_val

    def render_strategy_param(param):
        p_name = param["name"]
        p_label = param["label"]
        p_type = param["type"]
        unique_key = f"{selected_strategy}_{active_config}_{p_name}_input"
        default_val = st.session_state.get(unique_key, param.get("default", param.get("value")))
        
        if default_val == "today":
            from datetime import date
            default_val = date.today()

        if p_type == "text_input":
            val = st.text_input(p_label, default_val, key=unique_key)
        elif p_type == "number_input":
            raw_step = param.get("step")
            raw_min = param.get("min")
            raw_max = param.get("max")
            is_float = isinstance(default_val, float) or (raw_step and isinstance(raw_step, float))
            if is_float:
                val = st.number_input(p_label, value=float(default_val),
                    step=float(raw_step) if raw_step is not None else None,
                    min_value=float(raw_min) if raw_min is not None else None,
                    max_value=float(raw_max) if raw_max is not None else None,
                    key=unique_key, format=param.get("format"))
            else:
                val = st.number_input(p_label, value=int(default_val),
                    step=int(raw_step) if raw_step is not None else None,
                    min_value=int(raw_min) if raw_min is not None else None,
                    max_value=int(raw_max) if raw_max is not None else None,
                    key=unique_key, format=param.get("format"))
        elif p_type == "slider":
            raw_min = param.get("min", 0.0)
            raw_max = param.get("max", 100.0)
            raw_step = param.get("step")
            is_float_slider = isinstance(default_val, float) or isinstance(raw_min, float) or isinstance(raw_max, float) or (raw_step is not None and isinstance(raw_step, float))
            if is_float_slider:
                val = st.slider(p_label, 
                                min_value=float(raw_min), 
                                max_value=float(raw_max), 
                                value=float(default_val), 
                                step=float(raw_step) if raw_step is not None else None,
                                key=unique_key)
            else:
                val = st.slider(p_label, 
                                min_value=int(raw_min), 
                                max_value=int(raw_max), 
                                value=int(default_val), 
                                step=int(raw_step) if raw_step is not None else None,
                                key=unique_key)
        elif p_type == "date_input":
            val = st.date_input(p_label, default_val, key=unique_key)
        elif p_type == "selectbox":
            options = param.get("options", [])
            idx = options.index(default_val) if default_val in options else 0
            val = st.selectbox(p_label, options, index=idx, key=unique_key)

        strategy_params[p_name] = param["transform"](val) if "transform" in param else val
        if 'current_params' not in st.session_state: st.session_state.current_params = {}
        st.session_state.current_params[p_name] = val

    main_params = [p for p in strategy_info["params"] if not p.get("advanced")]
    advanced_params = [p for p in strategy_info["params"] if p.get("advanced")]

    with st.sidebar:
        for p in main_params: render_strategy_param(p)
        if advanced_params:
            with st.expander("Advanced Parameters", expanded=False):
                for p in advanced_params: render_strategy_param(p)

    st.sidebar.markdown("---")
    if st.sidebar.button("🚀 Run Backtest", type="primary"):
        with st.spinner("Backtesting..."):
            strategy_func = strategy_info["function"]
            st.session_state.last_result = strategy_func(**strategy_params)

# =========================================================
# SIDEBAR: PORTFOLIO MANAGER PANEL
# =========================================================
else:
    from portfolio_manager import load_portfolio, save_portfolio

    db      = load_portfolio()
    configs = load_configs()
    accts   = db.get('accounts', {})
    acct_names = list(accts.keys())

    # Build profile options list once - ensure all strategies have a 'default' option
    all_profile_opts = ['None']
    # Start with default profiles for ALL strategies
    for strat in STRATEGIES.keys():
        all_profile_opts.append(f"{strat} | default")
    
    # Add any other saved profiles from configs
    for strat, profiles in configs.items():
        for prof in profiles.keys():
            opt = f"{strat} | {prof}"
            if opt not in all_profile_opts:
                all_profile_opts.append(opt)

    st.sidebar.header("Portfolio Account")

    if not acct_names:
        st.sidebar.info("No accounts yet. Create one below.")
        selected_account = None
    else:
        # Persist selection
        if ('pm_account' not in st.session_state
                or st.session_state.pm_account not in acct_names):
            st.session_state.pm_account = acct_names[0]

        selected_account = st.sidebar.selectbox(
            "Account", acct_names,
            index=acct_names.index(st.session_state.pm_account),
            label_visibility="collapsed",
        )
        st.session_state.pm_account = selected_account

        # ── Linked Backtest Profile ────────────────────────────────
        st.sidebar.markdown("---")
        st.sidebar.header("Linked Backtest Profile")
        st.sidebar.caption("Optional — provides rebalancing signals.")

        current_link = accts[selected_account].get('linked_profile', 'None')
        link_idx = (all_profile_opts.index(current_link)
                    if current_link in all_profile_opts else 0)

        chosen_link = st.sidebar.selectbox(
            "Profile", all_profile_opts,
            index=link_idx,
            label_visibility="collapsed",
        )
        if chosen_link != current_link:
            accts[selected_account]['linked_profile'] = chosen_link
            save_portfolio(db)

    # ── New Account popover ────────────────────────────────────────
    st.sidebar.markdown("---")
    with st.sidebar.popover("➕ New Account"):
        st.markdown("**Create Account**")
        st.caption("Accounts are standalone ledgers. Ticker is entered per-trade.")
        new_name = st.text_input("Account Name", placeholder="e.g. Roth IRA",
                                 key="new_acc_name")
        new_date = st.date_input("Starting Date", value=datetime.now().date(),
                                 key="new_acc_date")
        new_link = st.selectbox("Link Profile (optional)", all_profile_opts,
                                key="new_acc_profile")
        
        if st.button("✅ Create", use_container_width=True):
            n = new_name.strip()
            if not n:
                st.warning("Enter an account name.")
            elif n in accts:
                st.error(f"'{n}' already exists.")
            else:
                db.setdefault('accounts', {})[n] = {
                    'initial_date':  str(new_date),
                    'linked_profile': new_link,
                    'drift_threshold': 5.0,
                    'trades':        [],
                    'equity_curve':  []
                }
                save_portfolio(db)
                st.session_state.pm_account = n
                st.success(f"✅ Created '{n}'")
                st.rerun()


# =========================================================
# MAIN CONTENT AREA
# =========================================================
if nav_selection == "📈 Strategy Laboratory":
    st.title(f"📈 {selected_strategy}")
    if 'last_result' in st.session_state and st.session_state.last_result is not None:
        generate_backtest_report(st.session_state.last_result)
    else:
        st.info("### 🧪 Strategy Laboratory\nConfigure parameters in the sidebar and click **Run Backtest** to analyze performance.")

else:
    st.title("💼 Live Portfolio Manager")
    from portfolio_manager import render_portfolio_manager

    if selected_account:
        render_portfolio_manager(account_name=selected_account)
    else:
        st.info("### 👋 Welcome\nCreate your first account using **➕ New Account** in the sidebar.")

