import os
import sys, time, collections
import numpy as np
sys.path.insert(0, '.')
import lupe_infer as I

kind = sys.argv[1] if len(sys.argv) > 1 else 'cuda'
path = sys.argv[2] if len(sys.argv) > 2 else os.path.expanduser('~').replace('\\', '/') + '/.lmstudio/models/lmstudio-community/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q4_K_M.gguf'
devsel = sys.argv[3] if len(sys.argv) > 3 else 'gpu'
builds = I.find_builds()
print('builds:', [(b['kind'], b['dir']) for b in builds])
b = next(x for x in builds if x['kind'] == kind)
lib = I.get_lib(b['dir'])
print('tensor size', I.C.sizeof(I.ggml_tensor), 'overhead', lib.ggml_base.ggml_tensor_overhead())
devs = lib.devices()
for d in devs:
    print('  device', d['index'], d['kind'], d['name'], d['desc'], d['free_gb'], '/', d['total_gb'], 'GB')
gpu = next((d['index'] for d in devs if d['kind'] == 'GPU'), None) if devsel == 'gpu' else None
t = time.time()
e = I.Engine(lib, path, gpu, progress=lambda p: None)
print(f'loaded in {time.time() - t:.1f}s on {e.device}: {e.desc}, n_embd {e.n_embd}, n_layer {e.n_layer}, vocab {e.n_vocab}')
q = 'Was ist die Hauptstadt von Frankreich?'
prompt, how = e.chat_prompt(q)
toks = e.tokenize(prompt)
print('template', how, repr(prompt), len(toks), 'tokens')
t = time.time(); new, why = e.generate(toks, 100 - len(toks)); gen_s = time.time() - t
print(f'generated {len(new)} tokens in {gen_s:.2f}s ({why}):', repr(b''.join(e.piece(x) for x in new).decode('utf-8', 'replace')))
t = time.time(); run = I.Run(e, toks + new, len(toks), why, gen_s)
print(f'recorded pass in {run.rec_s:.2f}s (total Run {time.time() - t:.2f}s): {len(run.graph)} nodes, {len(run.leaves)} leaves, {len(run.store)} stored, '
      f'{sum(a.nbytes for a in run.store.values()) / 1e6:.0f} MB')
ops = collections.Counter(x['op'] for x in run.graph)
print('ops:', ops.most_common(18))
names = collections.Counter(x['base'] for x in run.graph if x['name'] and x['il'] is not None)
print('named per layer:', sorted(names.items(), key=lambda kv: -kv[1])[:40])
print('layer nodes:', run.layer_nodes[:3], '...', run.layer_nodes[-2:], 'missing:', sum(1 for x in run.layer_nodes if x is None))
# consistency: greedy generation vs. record-pass argmax at each answer position
agree = sum(1 for k in range(len(toks) - 1, len(toks) + len(new) - 1) if run.top[k][0][0] == run.toks[k + 1])
print(f'record pass argmax agrees with generated token: {agree}/{len(new)}')
k = len(toks) - 1
print('top-5 after the prompt:', [(e.piece(j).decode('utf-8', 'replace'), round(pp, 3)) for j, pp in run.top[k][:5]])
# norms of the token's vector through the layers
v = [run.vec(i, k).astype(np.float32) for i in run.layer_nodes if i is not None]
print('norm through layers:', [round(float(np.linalg.norm(x)), 1) for x in v[::8]], '... last', round(float(np.linalg.norm(v[-1])), 1))
# a graph excerpt for block 3 (full attention) and block 0 (linear)
for blk in (0, 3):
    print(f'--- block {blk}')
    for i, x in enumerate(run.graph):
        if x['block'] == blk and x['op'] not in I.SHAPE_OPS:
            src = [(run.graph[s]['name'] or run.graph[s]['op']) if s >= 0 else 'W:' + run.leaves[-1 - s]['name'] for s in x['src']]
            print(f"  {i:5d} {x['op']:14s} {x['name'][:28]:28s} {x['ne'][:3]}  <- {src[:3]}{' *' if i in run.store else ''}")
# a detail request: the first MUL_MAT of block 3
mm = next(i for i, x in enumerate(run.graph) if x['block'] == 3 and x['op'] == 'MUL_MAT')
t = time.time(); dd = run.detail([mm]); print(f'detail node {mm} ({run.graph[mm]["name"]}) recomputed in {time.time() - t:.2f}s, shape', dd[mm].shape)
e.close()
