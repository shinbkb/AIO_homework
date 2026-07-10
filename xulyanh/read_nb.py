import json
import sys

nb = json.load(open(r'd:\dut_ai\AIO_code\xulyanh\xulianhp2.ipynb', encoding='utf-8'))
cells = [c for c in nb['cells'] if c['cell_type'] == 'code']
for i, c in enumerate(cells):
    src = ''.join(c['source'])
    print(f'=== Cell {i} ===')
    print(src.encode('ascii', errors='replace').decode('ascii'))
    print()
