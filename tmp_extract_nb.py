import json

with open('src/main.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for i, cell in enumerate(nb['cells']):
    src = ''.join(cell['source'])
    print(f'=== CELL {i} [{cell["cell_type"]}] ===')
    print(src)
    print()
