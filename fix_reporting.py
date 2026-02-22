
import re

file_path = r'c:\playground\app\backtest\reporting.py'
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Replace the specific call to render_performance_overview
# We use a flexible regex for whitespace
pattern = r'(render_performance_overview\s*\(\s*metrics_dict\s*,\s*df_results\s*,\s*initial_capital\s*,\s*benchmark_symbol\s*,\s*)result\.trade_log(\s*\))'
replacement = r'\1pd.DataFrame(strategic_trade_log)\2'

new_content = re.sub(pattern, replacement, content)

if new_content != content:
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(new_content)
    print("Successfully updated reporting.py")
else:
    print("Could not find the target line in reporting.py")
