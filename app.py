import streamlit as st
import pandas as pd
import json
import os
from datetime import datetime

from strategies import STRATEGIES
from reporting import generate_backtest_report

st.set_page_config(layout="wide", page_title="AlphaLab", page_icon="⚗️")

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
# SIDEBAR: SHARED NAVIGATION & INFO
# =========================================================
with st.sidebar:
    st.header("⚗️ AlphaLab")
    st.caption("Strategic Multi-Asset Backtesting Panel")
    st.markdown("---")
    
    # Simple Navigation
    nav_selection = st.radio(
        "Navigation", ["📈 Strategy Lab", "💼 Portfolio Manager"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.info("💡 **Tip**: Configure parameters in the top bar of the Strategy Lab for a better mobile experience.")

# =========================================================
# MAIN CONTENT AREA
# =========================================================
if nav_selection == "📈 Strategy Lab":
    st.title("📈 Strategy Lab")
    
    # --- Top Control Bar (Mobile First) ---
    configs = load_configs()
    
    # 1. Strategy Selection
    strat_col, prof_col = st.columns([2, 1])
    selected_strategy = strat_col.selectbox(
        "Strategy", list(STRATEGIES.keys()), key="main_strat_sel", label_visibility="collapsed"
    )
    
    if selected_strategy not in configs:
        configs[selected_strategy] = {"default": {}}
    strat_configs = configs[selected_strategy]
    config_names = list(strat_configs.keys())
    if "default" not in config_names:
        config_names = ["default"] + config_names
        
    # 2. Profile Selection
    if 'active_profile_choice' not in st.session_state:
        st.session_state.active_profile_choice = "default"
        
    active_config = prof_col.selectbox(
        "Profile", config_names, 
        index=config_names.index(st.session_state.active_profile_choice) if st.session_state.active_profile_choice in config_names else 0,
        key="main_prof_sel", label_visibility="collapsed"
    )
    st.session_state.active_profile_choice = active_config

    # --- Strategy Parameters & Config ---
    strategy_info = STRATEGIES[selected_strategy]
    strategy_params = {}
    
    # Parameters in a Popover for space efficiency
    with st.popover("⚙️ Parameters & Profiles", use_container_width=True):
        st.markdown(f"**Configuring**: {selected_strategy}")
        st.caption(f"**Active Profile**: {active_config}")
        
        # Load saved profile values into session state
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
                val = st.number_input(p_label, value=default_val, step=param.get("step"), key=unique_key)
            elif p_type == "slider":
                val = st.slider(p_label, min_value=param.get("min", 0.0), max_value=param.get("max", 100.0), 
                                value=default_val, step=param.get("step"), key=unique_key)
            elif p_type == "date_input":
                if isinstance(default_val, str):
                    default_val = pd.to_datetime(default_val).date()
                val = st.date_input(p_label, default_val, key=unique_key)
            elif p_type == "selectbox":
                options = param.get("options", [])
                val = st.selectbox(p_label, options, index=options.index(default_val) if default_val in options else 0, key=unique_key)

            strategy_params[p_name] = param["transform"](val) if "transform" in param else val
            if 'current_params' not in st.session_state: st.session_state.current_params = {}
            st.session_state.current_params[p_name] = val

        # Render Params
        main_params = [p for p in strategy_info["params"] if not p.get("advanced")]
        adv_params = [p for p in strategy_info["params"] if p.get("advanced")]
        for p in main_params: render_strategy_param(p)
        if adv_params:
            with st.expander("Advanced Optimization"):
                for p in adv_params: render_strategy_param(p)
                
        st.markdown("---")
        # Profile Management
        save_col1, save_col2 = st.columns([2, 1])
        new_prof_name = save_col1.text_input("Profile Name", placeholder="New Name...", label_visibility="collapsed")
        if save_col2.button("💾 Save", use_container_width=True):
            target = new_prof_name.strip() if new_prof_name.strip() else active_config
            configs[selected_strategy][target] = st.session_state.current_params
            save_configs(configs)
            st.session_state.active_profile_choice = target
            st.rerun()

    # --- Run Button ---
    if st.button("🚀 Run Backtest", type="primary", use_container_width=True):
        with st.spinner("Backtesting..."):
            strategy_func = strategy_info["function"]
            st.session_state.last_result = strategy_func(**strategy_params)

    # --- Results Display ---
    if 'last_result' in st.session_state and st.session_state.last_result is not None:
        result = st.session_state.last_result
        
        # Safety check: if an old session has a tuple result, clear it to avoid crashes
        if isinstance(result, tuple):
            st.session_state.last_result = None
            st.rerun()
            
        if hasattr(result, 'strategy_name') and result.strategy_name == "Empty Result":
            st.error("### ⚠️ Data Download Failed\nYahoo Finance is currently rate-limiting this request (common on shared hosting). Please try again in 5-10 minutes or try a different asset/date range.")
        else:
            generate_backtest_report(result)
    else:
        st.info("### 👋 Choose a strategy and profile, then hit Run.")

# =========================================================
# PORTFOLIO MANAGER SECTION
# =========================================================
elif nav_selection == "💼 Portfolio Manager":
    st.title("💼 Portfolio Manager")
    from portfolio_manager import render_portfolio_manager, load_portfolio, save_portfolio
    
    db = load_portfolio()
    accts = db.get('accounts', {})
    acct_names = list(accts.keys())
    
    if not acct_names:
        st.info("Create an account to track your live trades.")
    else:
        # Profile Selection Column
        acc_col, tool_col = st.columns([3, 1])
        selected_account = acc_col.selectbox("Account", acct_names, label_visibility="collapsed")
        
        with tool_col.popover("⚙️ Account Options"):
            st.write(f"Settings for: **{selected_account}**")
            # Account specific settings/deletion could go here
            if st.button("🗑️ Delete Account", type="secondary"):
                del db['accounts'][selected_account]
                save_portfolio(db)
                st.rerun()
        
        render_portfolio_manager(account_name=selected_account)

    # Add Account popover (Mobile Style)
    with tool_col.popover("➕ New Account"):
        st.markdown("**Create Account**")
        from datetime import datetime
        # Load profile options for linking
        cfg_for_link = load_configs()
        link_opts = ['None']
        for s_name in STRATEGIES.keys():
            link_opts.append(f"{s_name} | default")
        for s_name, profs in cfg_for_link.items():
            for pr in profs.keys():
                o = f"{s_name} | {pr}"
                if o not in link_opts: link_opts.append(o)

        n_name = st.text_input("Name", placeholder="Roth IRA")
        n_date = st.date_input("Start Date", value=datetime.now().date())
        n_link = st.selectbox("Link Profile", link_opts)
        
        if st.button("✅ Create Account", use_container_width=True):
            if n_name.strip():
                db.setdefault('accounts', {})[n_name.strip()] = {
                    'initial_date': str(n_date),
                    'linked_profile': n_link,
                    'drift_threshold': 5.0,
                    'trades': [],
                    'equity_curve': []
                }
                save_portfolio(db)
                st.rerun()

