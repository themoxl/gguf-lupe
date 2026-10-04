import os
import json, sys, time, urllib.parse, urllib.request
B = 'http://127.0.0.1:8766'
Q = os.path.expanduser('~').replace('\\', '/') + '/.lmstudio/models/lmstudio-community/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q4_K_M.gguf'
def get(path_, **q):
    with urllib.request.urlopen(B + path_ + '?' + urllib.parse.urlencode(q), timeout=600) as r: return json.loads(r.read())
job = get('/api/run/start', path=Q, q='Was ist die Hauptstadt von Frankreich?', chat='1', max='20', dev=sys.argv[1] if len(sys.argv) > 1 else 'auto')
while job.get('state') not in ('fertig', 'fehler'): time.sleep(0.5); job = get('/api/job', id=job['id'])
rid = job['result']['run']; info = get('/api/run/info', run=rid); pos = info['n_prompt'] - 1
print('device', info['device'])
for b in (3, 30, 63):
    blk = get('/api/run/block', run=rid, b=b)
    for s in blk['steps']:
        if s['op'] == 'MUL_MAT' and any('w' in x for x in s['ins']):
            for j in (7, 100, 3994):
                e = get('/api/run/explain', run=rid, n=s['i'], p=pos, j=j)
                if e.get('kind') != 'matmul': continue
                err = lambda v: abs(v - e['value']) / (abs(e['pos_sum']) + abs(e['neg_sum']))
                print(f"  {s['name']:18s} j={j:5d} llama {e['value']:+.5f} exact {e['total']:+.5f} ({err(e['total'])*1e4:5.2f}‱) q8/{e.get('q8_block')} {e.get('total_q8', 0):+.5f} ({err(e.get('total_q8', 0))*1e4:5.2f}‱) {e.get('wtype')}")
