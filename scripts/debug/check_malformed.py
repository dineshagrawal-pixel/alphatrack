import csv

def check_csv(path):
    malformed = []
    with open(path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        for i, row in enumerate(reader, 2):
            if not row: continue
            if len(row) != 5:
                malformed.append((i, row))
    return malformed

malformed = check_csv('pine-logs-High_Low-data.csv')
print(f"Total Malformed Rows: {len(malformed)}")
for line_num, row in malformed:
    print(f"Line {line_num}: {row}")
