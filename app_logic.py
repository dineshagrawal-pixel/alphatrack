import streamlit as st
import pandas as pd
import json
import os
from datetime import datetime, timedelta
from strategies import STRATEGIES
from reporting import generate_backtest_report

CONFIG_FILE = "strategy_configs.json"

def load_configs():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r") as f:
                return json.load(f)
        except:
            pass
    return {}

def save_configs(configs):
    with open(CONFIG_FILE, "w") as f:
        json.dump(configs, f, indent=4)

def run_strategy_config(strategy_name, config_name, limit_years=None):
    """Run a strategy with a named config or 'default'."""
    configs = load_configs()
    strategy_info = STRATEGIES[strategy_name]
    
    # Get parameters
    params = {}
    saved_params = configs.get(strategy_name, {}).get(config_name, {})
    
    for p in strategy_info["params"]:
        p_name = p["name"]
        # Use saved value if exists, else use default from STRATEGIES
        val = saved_params.get(p_name, p.get("default", p.get("value")))
        
        # Date conversion if needed
        if p["type"] == "date_input" and isinstance(val, str):
            val = pd.to_datetime(val).date()
            
        if limit_years and p["type"] == "date_input" and "start" in p_name.lower():
            # Override start date to 5 years ago
            val = datetime.now().date() - timedelta(days=limit_years * 365)
            
        # Apply transformation if defined in STRATEGIES
        if "transform" in p:
             # Transformation logic needs to be handled carefully here
             # Since we are running 'background', we might need a clean way to apply these
             # Let's assume we pass the transformed values to the function
             params[p_name] = p["transform"](val) if not isinstance(val, (float, int)) or val != p["transform"](val) else val
             # Note: This transformation logic is a bit tricky since we don't know if the saved val 
             # is already transformed. Let's stick to raw values and apply transform at call time.
             params[p_name] = p["transform"](val)
        else:
            params[p_name] = val
            
    strategy_func = strategy_info["function"]
    # Add strategy name to params if supported (it should be now)
    return strategy_func(**params)

# --- APP UI ---

# Sidebar: Presets & Config Management
st.sidebar.header("🎯 Strategy Presets")
all_configs = load_configs()
strat_configs = all_configs.get(st.session_state.get('selected_strategy', list(STRATEGIES.keys())[0]), {"default": {}})
config_names = list(strat_configs.keys())

current_config = st.sidebar.selectbox("Select Named Config", ["default"] + [c for c in config_names if c != "default"])

# Load selected config params into session state for the UI
if 'prev_config' not in st.session_state or st.session_state.prev_config != current_config:
    st.session_state.prev_config = current_config
    if current_config in strat_configs:
        for k, v in strat_configs[current_config].items():
            st.session_state[f"{st.session_state.selected_strategy}_{k}"] = v

# --- REST OF APP.PY LOGIC ... (will be integrated in next step)
