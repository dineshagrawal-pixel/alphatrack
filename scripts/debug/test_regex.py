import re
s = "3,700,3,800,170,1,900"
res = re.sub(r'(\d),(\d{3})(?![0-9])', r'\1\2', s)
print(f"Original: {s}")
print(f"Result: {res}")
print(f"Parts: {res.split(',')}")
