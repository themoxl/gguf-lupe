"""Worker process of the GGUF-Lupe server for the "Durchlauf" tab.

llama.cpp runs here, in its own process: it brings its own OpenMP runtime (which clashes with the one
of PyTorch/MKL in the server), and a crash or an out-of-memory abort inside llama.cpp then only ends
this worker, not the server. The server starts it on demand and talks to it over a local connection.

It keeps the loaded model and the recorded runs, answers questions about them, and explains single
numbers: which inputs and which weights of the GGUF file produced them, recomputed step by step.
"""
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')     # only in this process, before any OpenMP runtime starts
import json
import math
import re
import sys
import struct
import threading
import time
import uuid

import numpy as np
from gguf import GGUFReader, quants

import lupe_infer as I

ST = {'lib': None, 'engine': None, 'runs': {}, 'readers': {}, 'force_build': None}
CONN = None                                              # connection to the server


def progress(p, msg):
    if CONN is not None:
        CONN.send(('progress', (float(p), msg)))


def cancelled():
    """The server sends ('cancel',) while a job runs; nothing else can arrive during a call."""
    try:
        return CONN is not None and CONN.poll() and CONN.recv() == ('cancel',)
    except Exception:
        return False
LOCK = threading.RLock()
MAX_RUNS = 4
COPY_RE = re.compile(r'^[A-Za-z0-9_]+#(.+)#\d+$')        # scheduler copy of a tensor to another backend: "CUDA0#name#0"


# ---------------------------------------------------------------- llama.cpp build and devices
def lib():
    with LOCK:
        if ST['lib'] is None:
            bs = I.find_builds([ST['force_build']] if ST['force_build'] else [])
            if ST['force_build']:
                bs = [b for b in bs if os.path.samefile(b['dir'], ST['force_build'])] or bs
            if not bs:
                return None
            ST['lib'] = I.get_lib(bs[0]['dir'])
            ST['build'] = bs[0]
        return ST['lib']


def info():
    builds = I.find_builds([ST['force_build']] if ST['force_build'] else [])
    out = {'version': I.LLAMA_VERSION, 'builds': [{'kind': b['kind'], 'dir': b['dir']} for b in builds], 'max_tokens': I.Engine.MAX_TOKENS,
           'download': f'https://github.com/ggml-org/llama.cpp/releases/tag/{I.LLAMA_VERSION}'}
    try:
        L = lib()
    except Exception as e:
        out['error'] = f'llama.cpp ließ sich nicht laden: {e}'
        return out
    if L is None:
        return out
    out['build'] = ST['build']['kind']
    out['devices'] = [{'id': str(d['index']) if d['kind'] in ('GPU', 'iGPU') else 'cpu', 'kind': d['kind'], 'name': d['desc'] or d['name'], 'backend': d['name'],
                       'free_gb': d['free_gb'], 'total_gb': d['total_gb']} for d in L.devices() if d['kind'] in ('GPU', 'iGPU', 'CPU')]
    e = ST['engine']
    out['engine'] = {'path': e.path, 'device': e.device, 'dev_id': e.dev_id, 'load_s': round(e.load_s, 1)} if e else None
    return out


def engine(path, dev_id, progress):
    """The model loaded in llama.cpp on the chosen device (reused when it already is)."""
    with LOCK:
        e = ST['engine']
        if e and e.path == path and e.dev_id == dev_id:
            return e
        if e:
            ST['engine'] = None
            for r in list(ST['runs'].values()):
                r.engine = None
            ST['runs'].clear()
            e.close()
        L = lib()
        if L is None:
            raise RuntimeError('Kein llama.cpp gefunden. Siehe README: einmal die passende Version herunterladen.')
        idx = None if dev_id == 'cpu' else int(dev_id)
        e = I.Engine(L, path, idx, progress)
        e.dev_id = dev_id
        ST['engine'] = e
        return e


def job_run(path, question, chat, max_new, dev_id):
    size = os.path.getsize(path) / 1e9
    t0 = time.time()
    e = engine(path, dev_id, lambda p: progress(0.6 * p, f'Modell in llama.cpp laden ({size:.1f} GB)'))
    prompt, how = e.chat_prompt(question) if chat else (question, 'roh')
    toks = e.tokenize(prompt)
    if len(toks) >= I.Engine.MAX_TOKENS - 2:
        raise ValueError(f'Die Eingabe ist {len(toks)} Tokens lang. Zusammen mit der Antwort sind höchstens {I.Engine.MAX_TOKENS} erlaubt.')
    room = min(int(max_new), I.Engine.MAX_TOKENS - len(toks))
    progress(0.6, 'Antwort erzeugen')
    t1 = time.time()
    new, why = e.generate(toks, room, on_token=lambda k, t: progress(0.6 + 0.3 * k / room, f'Antwort erzeugen: {k} Tokens'), stop=cancelled)
    gen_s = time.time() - t1
    progress(0.9, 'Durchlauf aufzeichnen')
    run = I.Run(e, toks + new, len(toks), why, gen_s)
    run.question, run.chat, run.template, run.path = question, bool(chat), how, path
    run.id = uuid.uuid4().hex[:10]
    run.total_s = time.time() - t0
    with LOCK:
        ST['runs'][run.id] = run
        while len(ST['runs']) > MAX_RUNS:
            ST['runs'].pop(next(iter(ST['runs'])))
    threading.Thread(target=reader, args=(path,), daemon=True).start()
    return {'run': run.id}


def get_run(rid):
    r = ST['runs'].get(rid)
    if not r or r.engine is None:
        raise KeyError('Dieser Durchlauf ist nicht mehr im Speicher. Bitte neu rechnen.')
    return r


# ---------------------------------------------------------------- graph helpers
SHAPE = I.SHAPE_OPS


def alias(run, leaf_name):
    """A leaf that is really a node: copies between backends, and the KV cache (written from Kcur/Vcur)."""
    m = COPY_RE.match(leaf_name)
    nm = m.group(1) if m else leaf_name
    m2 = re.match(r'^cache_([kv])_l(\d+)', nm)
    if m2:
        nm = ('Kcur-' if m2.group(1) == 'k' else 'Vcur-') + m2.group(2)
    idx = run.by_name.get(nm)
    return idx


def resolve(run, s):
    """Follow views/reshapes/copies back to the node or weight that holds the numbers."""
    seen = 0
    while seen < 20:
        seen += 1
        if s < 0:
            leaf = run.leaves[-1 - s]
            a = alias(run, leaf['name'])
            if a is None:
                return ('w', leaf['name'], leaf)
            s = a
            continue
        x = run.graph[s]
        if x['op'] in SHAPE and x['src']:
            s = x['src'][0]
            continue
        return ('n', s, x)
    return ('n', s, run.graph[s])


def index_names(run):
    if getattr(run, 'by_name', None) is None:
        run.by_name = {}
        for i, x in enumerate(run.graph):
            if x['name'] and x['op'] not in SHAPE:
                run.by_name[x['name']] = i                  # last one wins: the final value under that name


def tok_axis(x, n):
    """Which ggml dimension runs over the tokens of the text (ne1 for [features, tokens], ne2 for [dim, heads, tokens])."""
    ne, nm = x['ne'], x['name']
    if (nm.startswith('kq') or x['op'] == 'SOFT_MAX') and ne[1] == n:
        return 1
    if ne[2] == n and ne[3] == 1:
        return 2
    if ne[1] == n:
        return 1
    return None


def slice_tok(run, i, a, pos):
    """The numbers of node i for one token, flattened (heads after each other)."""
    x = run.graph[i]
    ax = tok_axis(x, run.n)
    a4 = a.reshape([int(v) for v in x['ne'][::-1]]) if a.size == int(np.prod(x['ne'])) else a
    if ax is None or a4.ndim != 4:
        return a4.reshape(-1)
    return np.take(a4, pos, axis=3 - ax).reshape(-1)


def node_values(run, i):
    """Full array of node i, numpy order [ne3, ne2, ne1, ne0]."""
    a = run.store.get(i)
    if a is None:
        a = run.detail([i])[i]
    if a is None:
        raise ValueError('Diese Zwischenstufe lässt sich nicht auslesen.')
    x = run.graph[i]
    if x['name'].startswith('kq_soft_max') and i in run.store:
        return a.astype(np.float32)                         # stored already cut to the text length
    return a.astype(np.float32).reshape([int(v) for v in x['ne'][::-1]])


def vec(run, i, pos):
    x = run.graph[i]
    a = node_values(run, i)
    if x['name'].startswith('kq_soft_max') and i in run.store:
        return a[:, pos, :run.n].reshape(-1), [run.n, a.shape[0]]        # [heads, keys]
    ax = tok_axis(x, run.n)
    if ax is None:
        return a.reshape(-1), [int(v) for v in x['ne']]
    v = np.take(a, pos, axis=3 - ax)
    shape = [int(s) for s in v.shape[::-1]]                                # back to ggml order
    if x['base'] == 'kq' and x['ne'][0] >= run.n:
        v = v.reshape(-1, x['ne'][0])[:, :run.n]; shape[0] = run.n         # keys beyond the text are padding
    return v.reshape(-1), shape


# ---------------------------------------------------------------- answers for the page
def comp_of(run, x):
    b, op = x['base'], x['op']
    ws = [run.leaves[-1 - s]['name'] for s in x['src'] if s < 0]
    w = ' '.join(ws)
    if b.startswith('ffn') or 'ffn_' in w:
        return 'moe' if '_exps' in w else 'ffn'
    if op in ('RMS_NORM', 'NORM', 'L2_NORM') or b in ('attn_norm', 'attn_post_norm', 'ffn_norm', 'norm', 'result_norm') or '_norm.weight' in w:
        return 'norm'
    if b in ('attn_residual', 'l_out', 'ffn_inp', 'post_ffn'):
        return 'other'
    if op in ('GATED_DELTA_NET', 'SSM_CONV', 'SSM_SCAN', 'RWKV_WKV6', 'RWKV_WKV7', 'GATED_LINEAR_ATTN') or 'ssm_' in w or b.startswith(('linear_attn', 'conv_', 'q_conv', 'k_conv', 'v_conv', 'state', 'new_state', 'final_output', 'beta', 'alpha', 'a_softplus')) or b in ('z', 'gate'):
        return 'lin'
    if b.startswith(('Qcur', 'Kcur', 'Vcur', 'kq', 'attn', 'gate_')) or 'attn_' in w:
        return 'attn'
    if b in ('inp_embd', 'model.input_embed', 'inp_scaled') or 'token_embd' in w:
        return 'emb'
    if b.startswith('result_output') or 'output.weight' in w:
        return 'head'
    return 'other'


def run_info(rid):
    r = get_run(rid)
    index_names(r)
    e = r.engine
    kinds = []
    for il in range(e.n_layer):
        ops = {x['op'] for x in r.graph if x['block'] == il}
        names = {x['base'] for x in r.graph if x['block'] == il}
        kinds.append('attn' if 'kq_soft_max' in names or 'FLASH_ATTN_EXT' in ops else 'lin' if ops & {'GATED_DELTA_NET', 'SSM_SCAN', 'SSM_CONV', 'RWKV_WKV7', 'RWKV_WKV6'} else 'other')
    return {'id': r.id, 'path': r.path, 'question': r.question, 'chat': r.chat, 'template': r.template, 'device': e.device, 'desc': e.desc,
            'n_embd': e.n_embd, 'n_layer': e.n_layer, 'n_vocab': e.n_vocab, 'tokens': r.toks, 'n_prompt': r.n_prompt, 'why': r.why,
            'pieces': [e.piece(t).decode('utf-8', 'replace') for t in r.toks],
            'timing': {'load_s': round(e.load_s, 1), 'gen_s': round(r.gen_s, 2), 'rec_s': round(r.rec_s, 2), 'total_s': round(getattr(r, 'total_s', 0), 1)},
            'p_next': r.p_next, 'rank_next': r.rank_next, 'top': r.top, 'layer_kinds': kinds, 'layer_nodes': r.layer_nodes,
            'n_graph': len(r.graph), 'n_stored': len(r.store), 'mb_stored': round(sum(a.nbytes for a in r.store.values()) / 1e6),
            'attn_nodes': {str(x['block']): i for i, x in enumerate(r.graph) if x['base'] == 'kq_soft_max' and i in r.store}}


def pack(meta, arr):
    mb = json.dumps(meta).encode('utf-8')
    pad = (-(4 + len(mb))) % 2
    return struct.pack('<I', len(mb)) + mb + b' ' * pad + np.ascontiguousarray(arr, dtype=np.float16).tobytes()


def run_stack(rid, pos):
    r = get_run(rid)
    vs = [r.vec(i, pos).astype(np.float32) if i is not None else np.zeros(r.engine.n_embd, np.float32) for i in r.layer_nodes]
    A = np.stack(vs)
    rms = np.sqrt((A * A).mean(1)).tolist()
    meta = {'pos': pos, 'rows': len(vs), 'cols': int(A.shape[1]), 'rms': rms, 'nodes': r.layer_nodes,
            'names': [r.graph[i]['name'] if i is not None else '' for i in r.layer_nodes]}
    return pack(meta, A)


def run_vec(rid, i, pos):
    r = get_run(rid)
    index_names(r)
    v, shape = vec(r, i, pos)
    x = r.graph[i]
    meta = {'node': i, 'name': x['name'], 'op': x['op'], 'shape': shape, 'pos': pos, 'stored': i in r.store,
            'min': float(v.min()), 'max': float(v.max()), 'rms': float(np.sqrt((v * v).mean())), 'n': int(v.size)}
    return pack(meta, v)


def run_block(rid, b):
    r = get_run(rid)
    index_names(r)
    steps = []
    last_out = max((i for i, x in enumerate(r.graph) if x['block'] == b and x['base'] == 'l_out'), default=None)
    for i, x in enumerate(r.graph):
        if x['block'] != b or x['op'] in SHAPE:
            continue
        if x['name'].startswith('cache_') or (x['op'] in ('SCALE', 'GET_ROWS') and all(s >= 0 and r.graph[s]['name'].startswith('cache_') for s in x['src'][:1])):
            continue                                        # bookkeeping of the recurrent state, no computation on the token
        ins = []
        for s in x['src']:
            k = resolve(r, s)
            if k[0] == 'w':
                if not k[1].startswith(('inp_', 'leaf_')) and '#' not in k[1]:
                    ins.append({'w': k[1], 'ne': k[2]['ne'][:2], 'type': k[2]['type']})
            else:
                ins.append({'n': k[1], 'name': k[2]['name']})
        steps.append({'i': i, 'name': x['name'], 'base': x['base'], 'op': x['op'], 'ne': x['ne'][:3], 'stored': i in r.store,
                      'comp': comp_of(r, x), 'ins': ins, 'tok': tok_axis(x, r.n) is not None, 'tail': last_out is not None and i > last_out})
    return {'block': b, 'steps': steps}


# ---------------------------------------------------------------- inputs for the logit lens (computed in the server)
def lens_inputs(rid, pos):
    r = get_run(rid)
    X = np.stack([r.vec(i, pos).astype(np.float32) for i in r.layer_nodes if i is not None])
    nxt = r.toks[pos + 1] if pos + 1 < r.n else (r.top[pos][0][0] if r.top[pos] else None)
    return {'X': X, 'next': nxt, 'top1': int(r.top[pos][0][0]), 'path': r.path}


def ping():
    return {'pid': os.getpid()}


def token_texts(rid, ids):
    """Display text of token numbers (as llama.cpp writes them), for examples that travel without the model file."""
    e = get_run(rid).engine
    return {str(int(i)): e.piece(int(i)).decode('utf-8', 'replace') for i in ids}


# ---------------------------------------------------------------- GGUF access for the explanations
def reader(path):
    with LOCK:
        rd = ST['readers'].get(path)
        if rd is None:
            ST['readers'].clear()
            rd = ST['readers'][path] = GGUFReader(path)
            rd.by_name = {t.name: t for t in rd.tensors}
        return rd


def field(rd, key):
    f = rd.fields.get(key)
    if f is None:
        return None
    v = f.parts[f.data[0]] if f.data else None
    if v is None:
        return None
    if v.dtype == np.uint8:
        return bytes(v).decode('utf-8', 'replace')
    return v.tolist()[0] if v.size == 1 else v.tolist()


def weight_row(path, name, j):
    t = reader(path).by_name.get(name)
    if t is None:
        return None
    d = np.asarray(t.data)
    if d.ndim == 1:
        return d.astype(np.float32)
    row = d.reshape(d.shape[0], -1)[j] if d.ndim == 2 else d.reshape(-1, d.shape[-1])[j]
    if t.tensor_type.name in ('F32', 'F16', 'BF16') and row.dtype != np.uint8:
        return row.astype(np.float32)
    return quants.dequantize(np.ascontiguousarray(row[None]), t.tensor_type).reshape(-1).astype(np.float32)


def weight_vec(path, name):
    t = reader(path).by_name.get(name)
    return None if t is None else np.asarray(t.data, dtype=np.float32).reshape(-1)


def f32(bits):
    return struct.unpack('<f', struct.pack('<i', int(bits)))[0]


# ---------------------------------------------------------------- zooming into one number
def run_explain(rid, i, pos, j):
    r = get_run(rid)
    index_names(r)
    x = r.graph[i]
    v, shape = vec(r, i, pos)
    j = max(0, min(int(j), v.size - 1))
    y = float(v[j])
    srcs = [resolve(r, s) for s in x['src']]
    direct = [direct_src(r, s) for s in x['src']]
    out = {'node': i, 'name': x['name'], 'op': x['op'], 'pos': pos, 'j': j, 'value': y, 'shape': shape,
           'inputs': [({'w': s[1]} if s[0] == 'w' else {'n': s[1], 'name': s[2]['name']}) for s in srcs]}

    def act(s, p=pos):           # values of an input node for one token, flattened
        return vec(r, s[1], p)[0]

    op = x['op']
    try:
        if op == 'MUL_MAT' and srcs and srcs[0][0] == 'w' and len(srcs) > 1 and srcs[1][0] == 'n':
            W = weight_row(r.path, srcs[0][1], j)
            a = act(direct[1])
            if W is not None and W.size == a.size:
                prod = W * a
                out.update(kind='matmul', weight=srcs[0][1], row=j, n_terms=int(prod.size), total=float(prod.sum()), wtype=srcs[0][2]['type'], **terms(prod, W, a))
                if srcs[0][2]['type'] not in ('f32', 'f16', 'bf16'):
                    try:
                        out.update(total_q8=llama_like(r, srcs[0][1], srcs[0][2]['type'], j, W, a), q8_block=256 if r.engine.device == 'CPU' else 32)
                    except Exception:
                        pass
                return out
        if op == 'MUL_MAT' and len(srcs) == 2 and srcs[0][0] == 'n' and srcs[1][0] == 'n':
            return attn_matmul(r, out, x, srcs, pos, j, shape)
        if op in ('MUL', 'ADD', 'SUB', 'DIV') and len(srcs) == 2:
            vals = []
            for s in direct:
                if s[0] == 'w':
                    wv = weight_vec(r.path, s[1]); vals.append(float(wv[j % wv.size]) if wv is not None else None)
                else:
                    av = act(s); vals.append(float(av[j % av.size]))
            if None not in vals:
                sym = {'MUL': '×', 'ADD': '+', 'SUB': '−', 'DIV': '÷'}[op]
                calc = {'MUL': vals[0] * vals[1], 'ADD': vals[0] + vals[1], 'SUB': vals[0] - vals[1], 'DIV': vals[0] / vals[1] if vals[1] else float('nan')}[op]
                out.update(kind='binary', sym=sym, a=vals[0], b=vals[1], total=calc)
                return out
        if op == 'RMS_NORM' and direct and direct[0][0] == 'n':
            a = act(direct[0]); ne0 = int(x['ne'][0]); row = j // ne0
            seg = a[row * ne0:(row + 1) * ne0]; eps = f32(x['op_params'][0])
            ms = float((seg * seg).mean())
            out.update(kind='rmsnorm', x=float(a[j]), mean_sq=ms, eps=eps, rms=math.sqrt(ms + eps), total=float(a[j]) / math.sqrt(ms + eps), n_terms=ne0)
            return out
        if op in ('SWIGLU', 'GEGLU', 'GLU') and len(srcs) >= 2:
            g, u = float(act(direct[0])[j]), float(act(direct[1])[j])
            silu = g / (1 + math.exp(-g))
            out.update(kind='glu', gate=g, up=u, act=silu, total=silu * u)
            return out
        if op in ('SILU', 'SIGMOID', 'SOFTPLUS', 'GELU', 'TANH', 'EXP', 'RELU', 'NEG') and direct and direct[0][0] == 'n':
            a = float(act(direct[0])[j])
            fn = {'SILU': lambda t: t / (1 + math.exp(-t)), 'SIGMOID': lambda t: 1 / (1 + math.exp(-t)), 'SOFTPLUS': lambda t: math.log1p(math.exp(t)) if t < 30 else t,
                  'GELU': lambda t: 0.5 * t * (1 + math.tanh(0.7978845608 * (t + 0.044715 * t ** 3))), 'TANH': math.tanh, 'EXP': math.exp, 'RELU': lambda t: max(0.0, t), 'NEG': lambda t: -t}[op]
            out.update(kind='unary', fn=op.lower(), x=a, total=fn(a))
            return out
        if op == 'SCALE' and direct and direct[0][0] == 'n':
            a = float(act(direct[0])[j]); s_, b_ = f32(x['op_params'][0]), f32(x['op_params'][1])
            out.update(kind='scale', x=a, s=s_, b=b_, total=a * s_ + b_)
            return out
        if op == 'GET_ROWS' and srcs and srcs[0][0] == 'w':
            tok = r.toks[pos]
            W = weight_row(r.path, srcs[0][1], tok)
            out.update(kind='lookup', weight=srcs[0][1], token=tok, total=float(W[j]) if W is not None else None)
            return out
        if op == 'SOFT_MAX' and srcs and srcs[0][0] == 'n':
            return softmax_explain(r, out, x, srcs, pos, j, shape)
        if op == 'ROPE' and direct and direct[0][0] == 'n':
            a = act(direct[0]); n_dims = int(x['op_params'][1]); mode = int(x['op_params'][2]); base = f32(x['op_params'][5]) or 10000.0
            ne0 = int(x['ne'][0]); k = j % ne0; head0 = j - k
            if k >= n_dims:
                out.update(kind='rope', rotated=False, x=float(a[j]), total=float(a[j]))
                return out
            neox = bool(mode & 2) or mode >= 8
            if neox:
                half = n_dims // 2; pi = k % half; partner = head0 + (k + half if k < half else k - half)
            else:
                pi = k // 2; partner = head0 + (k + 1 if k % 2 == 0 else k - 1)
            theta = pos * base ** (-2.0 * pi / n_dims)
            out.update(kind='rope', rotated=True, x=float(a[j]), partner=int(partner - head0), x_partner=float(a[partner]), angle=theta, pos=pos, n_dims=n_dims, base=base)
            return out
    except Exception as ex:          # an explanation must never break the page
        out['note'] = f'Nachrechnen nicht möglich: {ex}'
    out.setdefault('kind', 'generic')
    return out


def direct_src(run, s):
    """The input exactly as the op sees it (a view keeps its offset and strides); copies between backends map to their node."""
    if s >= 0:
        return ('n', s, run.graph[s])
    leaf = run.leaves[-1 - s]
    m = COPY_RE.match(leaf['name'])
    if m and m.group(1) in run.by_name:
        k = run.by_name[m.group(1)]
        return ('n', k, run.graph[k])
    return ('w', leaf['name'], leaf)


def raw_row(path, name, j):
    t = reader(path).by_name.get(name)
    d = np.asarray(t.data)
    return d.reshape(d.shape[0], -1)[j] if d.ndim == 2 else d.reshape(-1, d.shape[-1])[j]


def q4k_parts(raw):
    """Q4_K row -> (A, q, B) with weight = A*q - B, constant A, B per 32 values (llama.cpp's block layout: d, dmin, 12 bytes scales, 128 bytes nibbles)."""
    blk = raw.reshape(-1, 144)
    d = blk[:, 0:2].copy().view(np.float16).astype(np.float32)[:, 0]
    dmin = blk[:, 2:4].copy().view(np.float16).astype(np.float32)[:, 0]
    sc_raw, qs = blk[:, 4:16].astype(np.int32), blk[:, 16:144]
    sc = np.empty((len(blk), 8), np.float32); mn = np.empty((len(blk), 8), np.float32)
    for jj in range(8):
        if jj < 4:
            sc[:, jj] = sc_raw[:, jj] & 63; mn[:, jj] = sc_raw[:, jj + 4] & 63
        else:
            sc[:, jj] = (sc_raw[:, jj + 4] & 0xF) | ((sc_raw[:, jj - 4] >> 6) << 4)
            mn[:, jj] = (sc_raw[:, jj + 4] >> 4) | ((sc_raw[:, jj] >> 6) << 4)
    q = np.empty((len(blk), 256), np.float32)
    for c in range(4):
        part = qs[:, c * 32:(c + 1) * 32]
        q[:, c * 64:c * 64 + 32] = part & 0xF
        q[:, c * 64 + 32:c * 64 + 64] = part >> 4
    A = np.repeat(d[:, None] * sc, 32, axis=1).reshape(-1)
    B = np.repeat(dmin[:, None] * mn, 32, axis=1).reshape(-1)
    return A, q.reshape(-1), B


def llama_like(r, name, wtype, j, W, a):
    """The dot product the way llama.cpp computes it: input rounded to 8 bit first.
    GPU (q8_1, blocks of 32): scale stored as fp16; for weights with a minimum (Q4_K) the minimum part uses the
    unrounded block sum, also stored as fp16. CPU (q8_K, blocks of 256): float scale, both parts with the rounded input."""
    if r.engine.device == 'CPU':
        return float((W * q8(a, 256)).sum())
    n = a.size - a.size % 32
    b = a[:n].reshape(-1, 32)
    dd = np.abs(b).max(1, keepdims=True) / 127.0
    q = np.where(dd > 0, np.round(b / np.where(dd > 0, dd, 1)), 0)
    xq = a.copy(); xq[:n] = (q * dd.astype(np.float16).astype(np.float32)).reshape(-1)
    if wtype == 'q4_K' and a.size % 256 == 0:
        A, qw, Bm = q4k_parts(raw_row(r.path, name, j))
        if np.allclose(A * qw - Bm, W, atol=1e-6, rtol=1e-5):
            bsum = b.sum(1).astype(np.float16).astype(np.float32)          # per 32 values, unrounded, stored as fp16
            return float((A * qw * xq).sum() - (Bm.reshape(-1, 32)[:, 0] * bsum).sum())
    return float((W * xq).sum())


def q8(a, blk):
    """Round like llama.cpp's 8-bit activation formats: per block of blk values, scale = max|x| / 127."""
    n = a.size - a.size % blk
    b = a[:n].reshape(-1, blk)
    d = np.abs(b).max(1, keepdims=True) / 127.0
    d[d == 0] = 1
    out = a.copy()
    out[:n] = (np.round(b / d) * d).reshape(-1)
    return out


def terms(prod, A, B, k=12):
    order = np.argsort(-np.abs(prod))
    top = [{'i': int(t), 'a': float(A[t]), 'b': float(B[t]), 'p': float(prod[t])} for t in order[:k]]
    cs = np.cumsum(prod[order])
    pts = sorted(set([1, 2, 3, 5, 8, 12, 20, 32, 50, 80, 128, 200, 320, 500, 800, 1280, 2000, 3200, 5000, 8000, 12800, len(prod)]))
    curve = [[p, float(cs[p - 1])] for p in pts if p <= len(prod)]
    pos_sum, neg_sum = float(prod[prod > 0].sum()), float(prod[prod < 0].sum())
    return {'top': top, 'curve': curve, 'pos_sum': pos_sum, 'neg_sum': neg_sum}


def attn_matmul(r, out, x, srcs, pos, j, shape):
    """kq (scores) or kqv (weighted sum of V): dot products between tokens."""
    nm = x['base']
    if nm.startswith('kq') and not nm.startswith('kqv'):
        n_kv = shape[0]; h, k = divmod(out['j'], n_kv)
        q_node = srcs[1]; k_node = srcs[0]
        qv, qs = vec(r, q_node[1], pos)            # Q of this token [head_dim, n_head]
        kv, ks = vec(r, k_node[1], k)              # K of token k     [head_dim, n_head_kv]
        hd = qs[0]; nh, nkv = qv.size // hd, kv.size // hd
        hk = h // max(1, nh // nkv)
        a = qv[h * hd:(h + 1) * hd]; b = kv[hk * hd:(hk + 1) * hd]
        prod = a * b
        out.update(kind='score', head=int(h), key=int(k), head_kv=int(hk), n_terms=int(hd), total=float(prod.sum()), **terms(prod, a, b))
        return out
    if nm.startswith('kqv'):
        hd = shape[0]; h, dd = divmod(out['j'], hd)
        p_node, v_node = srcs[1], srcs[0]
        P, ps = vec(r, p_node[1], pos)             # attention weights of this token
        n = r.n
        nh = ps[1] if len(ps) > 1 else 1
        probs = P.reshape(nh, -1)[h][:pos + 1]
        vals, nkv = [], None
        for k in range(pos + 1):
            vv, vs = vec(r, v_node[1], k)
            nkv = max(1, vv.size // hd)
            hk = h // max(1, nh // nkv)
            vals.append(vv[hk * hd + dd])
        vals = np.array(vals, dtype=np.float32)
        prod = probs * vals
        out.update(kind='mix', head=int(h), dim=int(dd), n_terms=int(len(prod)), total=float(prod.sum()),
                   top=[{'i': int(k), 'a': float(probs[k]), 'b': float(vals[k]), 'p': float(prod[k])} for k in np.argsort(-np.abs(prod))[:12]])
        return out
    out['kind'] = 'generic'
    return out


def softmax_explain(r, out, x, srcs, pos, j, shape):
    n_kv = shape[0]; h, k = divmod(out['j'], n_kv)
    s_all, ss = vec(r, srcs[0][1], pos)                        # raw scores Q·K of this token
    scale = f32(x['op_params'][0]) or 1.0
    sc = s_all.reshape(-1, ss[0])[h][:pos + 1] * scale
    m = sc.max(); e = np.exp(sc - m); p = e / e.sum()
    order = np.argsort(-p)[:8]
    out.update(kind='softmax', head=int(h), key=int(k), scale=scale, score=float(sc[k]) if k <= pos else None,
               total=float(p[k]) if k <= pos else 0.0, n_terms=int(pos + 1),
               top=[{'i': int(t), 'a': float(sc[t]), 'p': float(p[t])} for t in order])
    return out


# ---------------------------------------------------------------- process entry
def main():
    global CONN
    from multiprocessing.connection import Client
    port, key = int(sys.argv[1]), bytes.fromhex(sys.argv[2])
    ST['force_build'] = sys.argv[3] if len(sys.argv) > 3 and sys.argv[3] != '-' else None
    CONN = Client(('127.0.0.1', port), authkey=key)
    while True:
        try:
            msg = CONN.recv()
        except (EOFError, OSError):
            break
        if not isinstance(msg, tuple) or len(msg) != 2:
            continue                                      # e.g. a late ('cancel',)
        name, args = msg
        try:
            CONN.send(('ok', globals()[name](*args)))
        except KeyError as e:
            CONN.send(('gone', str(e).strip('"\'')))
        except Exception as e:
            import traceback
            traceback.print_exc()
            CONN.send(('err', str(e) or repr(e)))
    e = ST.get('engine')
    if e:
        e.close()


if __name__ == '__main__':
    main()
