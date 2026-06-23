import json, sys

path = sys.argv[1]
with open(path, encoding='utf-8') as f:
    nb = json.load(f)

for c in nb['cells']:
    if c['cell_type'] == 'markdown':
        src = c['source'] if isinstance(c['source'], str) else ''.join(c['source'])
        print(src[:600].encode('ascii', errors='replace').decode())
        print('---')
