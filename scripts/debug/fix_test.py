
import sys

file_path = r'c:\playground\app\backtest\tests\test_all_strategies_regression.py'
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Remove any weird trailing characters or invalid continuation lines
new_content = content.replace('\\u000a', '').strip() + '\n'

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(new_content)

print("Fixed syntax error in test_all_strategies_regression.py")
