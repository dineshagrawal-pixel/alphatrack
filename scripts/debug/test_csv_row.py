import csv
import io

csv_content = """Date,HIGQ,LOWQ,HIGN,LOWN
2011-03-09,93,25,142,7
2011-03-10,"3,700,3,800,170,1,900"
"""

f = io.StringIO(csv_content)
reader = csv.reader(f)
header = next(reader)
for row in reader:
    print(f"Row: {row} (Len: {len(row)})")
