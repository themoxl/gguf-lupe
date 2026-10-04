import os
import json, struct, sys, time, urllib.parse, urllib.request
import numpy as np
B = sys.argv[1] if len(sys.argv) > 1 else 'http://127.0.0.1:8766'
Q = os.path.join(os.path.expanduser('~'), '.lmstudio', 'models', 'lmstudio-community', 'Qwen3.8-27B-GGUF', 'Qwen3.8-27B-Q4_K_M.gguf')
def get(path_, **q):
    with urllib.request.urlopen(B + path_ + '?' + urllib.parse.urlencode(q), timeout=600) as r:
        b = r.read()
        return json.loads(b) if r.headers.get('Content-Type', '').startswith('application/json') else b
def unpack(b):
    ml = struct.unpack('<I', b[:4])[0]; meta = json.loads(b[4:4 + ml]); o = 4 + ml; o += o % 2
    return meta, np.frombuffer(b[o:], dtype=np.float16).astype(np.float32)
for _ in range(30):
    try: get('/api/info'); break
    except Exception: time.sleep(1)
t = time.time(); inf = get('/api/llm/info'); print(f'llm/info {time.time() - t:.1f}s', json.dumps({k: v for k, v in inf.items() if k != 'builds'}, ensure_ascii=False)[:400])
q = sys.argv[2] if len(sys.argv) > 2 else 'Wie heißt die Hauptstadt von Deutschland?'
t = time.time(); job = get('/api/run/start', path=Q, q=q, chat='1', max='40', dev=sys.argv[3] if len(sys.argv) > 3 else 'auto')
while job.get('state') not in ('fertig', 'fehler'):
    time.sleep(0.5); job = get('/api/job', id=job['id'])
print(f'run job {job["state"]} in {time.time() - t:.1f}s', job.get('message'), job.get('result'))
rid = job['result']['run']
info = get('/api/run/info', run=rid)
print('answer:', repr(''.join(info['pieces'][info['n_prompt']:])), '| why', info['why'], '| timing', info['timing'], '| stored', info['n_stored'], info['mb_stored'], 'MB')
print('layer kinds:', ''.join({'attn': 'A', 'lin': 'l', 'other': '?'}[k] for k in info['layer_kinds']))
pos = info['n_prompt'] - 1
t = time.time(); meta, A = unpack(get('/api/run/stack', run=rid, p=pos)); print(f'stack {time.time() - t:.2f}s', meta['rows'], meta['cols'], 'rms', [round(x, 2) for x in meta['rms'][::16]])
t = time.time(); lens = get('/api/run/lens', run=rid, p=pos); print(f'lens {time.time() - t:.2f}s check={lens["check"]} gpu={lens["gpu"]}')
for li in (0, 16, 32, 40, 48, 56, 60, 62, 63, 64):
    row = lens['rows'][li]; print(f'   layer {li:2d}: top', [(info['pieces'][0] if False else j, round(pp, 3)) for j, pp in row['top'][:3]], 'p_next', round(row.get('p_next', 0), 4), 'rank', row.get('rank_next'))
blk = get('/api/run/block', run=rid, b=3)
print('block 3 steps:', len(blk['steps']))
for s in blk['steps']:
    print(f"   {s['i']:5d} {s['op']:12s} {s['name']:22s} {s['comp']:6s} stored={s['stored']} ins={[x.get('w') or x.get('name') for x in s['ins']]}")
# explanations
def ex(name, j=7):
    st = next(s for s in blk['steps'] if s['name'] == name)
    t = time.time(); e = get('/api/run/explain', run=rid, n=st['i'], p=pos, j=j)
    keep = {k: e[k] for k in e if k not in ('top', 'curve', 'inputs')}
    print(f'explain {name} ({time.time() - t:.2f}s):', json.dumps(keep, ensure_ascii=False)[:330])
    if 'top' in e: print('      top:', [(x['i'], round(x['p'], 4)) for x in e['top'][:5]])
for nm in ('attn_norm-3', 'Qcur_full-3', 'Qcur-3', 'kq-3', 'kq_soft_max-3', 'kqv-3', 'gate_sigmoid-3', 'attn_output-3', 'attn_residual-3', 'ffn_swiglu-3', 'ffn_out-3', 'l_out-3'):
    try:
        ex(nm, 7 if not nm.startswith('kq') else 3)
    except StopIteration:
        print('   (no step', nm, ')')
t = time.time(); meta, v = unpack(get('/api/run/vec', run=rid, n=blk['steps'][0]['i'], p=pos)); print(f'vec {time.time() - t:.2f}s', meta)
an = info['attn_nodes']; k0 = next(iter(an)); meta, v = unpack(get('/api/run/vec', run=rid, n=an[k0], p=pos)); print('attn vec', meta['shape'], 'row sums', np.round(v.reshape(meta['shape'][1], -1).sum(1)[:4], 3))
