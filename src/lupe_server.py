#!/usr/bin/env python3
"""GGUF-Lupe Server

Serves the Lupe page on http://127.0.0.1:<port>/ and does the heavy work on GPU or CPU:
  - lists the GGUF models on this machine (LM Studio folder and others)
  - streams model files to the page (HTTP Range), nothing is uploaded anywhere
  - keeps the whole embedding table in memory: exact nearest neighbours and
    word arithmetic over ALL tokens in ALL dimensions
  - full word map of the entire vocabulary (exact kNN + UMAP-style layout), cached on disk
  - dimension "layers": any column of the embedding table, raw

Minimum: Python 3.9+, numpy, gguf  (pip install numpy gguf)
Optional: PyTorch (NVIDIA CUDA, Apple MPS or fast CPU)  -> much faster

usage: python lupe_server.py [--port 8765] [--models DIR ...] [--device auto|cuda|mps|cpu] [--no-browser]
"""
import argparse, hashlib, json, os, platform, re, struct, sys, threading, time, traceback, uuid, webbrowser
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import numpy as np
try:
    import torch
    HAVE_TORCH = True
except Exception:
    torch = None
    HAVE_TORCH = False
from gguf import GGUFReader, quants

HERE = os.path.dirname(os.path.abspath(__file__))
PAGE = None  # resolved in main()
CACHE = os.path.join(os.path.expanduser('~'), '.cache', 'gguf-lupe')
os.makedirs(CACHE, exist_ok=True)
MODEL_DIRS = []
LOCK = threading.Lock()          # one heavy job at a time
STORES = {}                      # path -> EmbStore
JOBS = {}                        # id -> dict


# ---------------------------------------------------------------- console language
def system_is_german():
    """German if the system language is German, else English. Windows: the display language; elsewhere LC_ALL / LC_MESSAGES / LANG."""
    if os.name == 'nt':
        try:
            import ctypes
            return (ctypes.windll.kernel32.GetUserDefaultUILanguage() & 0x3FF) == 0x07     # primary language id 0x07 = German
        except Exception:
            pass
    for k in ('LC_ALL', 'LC_MESSAGES', 'LANG'):
        if os.environ.get(k):
            return os.environ[k].lower().startswith('de')
    try:
        import locale
        return (locale.getlocale()[0] or '').lower().startswith(('de', 'german'))
    except Exception:
        return False

CONSOLE_DE = system_is_german()

def say(de, en):
    """Text for the console (print lines, --help) in the system language. Not for the page: its texts stay German and the page maps them."""
    return de if CONSOLE_DE else en


# ---------------------------------------------------------------- hardware
def ram_gb():
    try:
        if platform.system() == 'Windows':
            import ctypes
            class MS(ctypes.Structure):
                _fields_ = [('l', ctypes.c_ulong), ('m', ctypes.c_ulong), ('t', ctypes.c_ulonglong), ('a', ctypes.c_ulonglong),
                            ('tp', ctypes.c_ulonglong), ('ap', ctypes.c_ulonglong), ('tv', ctypes.c_ulonglong), ('av', ctypes.c_ulonglong), ('e', ctypes.c_ulonglong)]
            s = MS(); s.l = ctypes.sizeof(MS); ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(s))
            return round(s.t / 1e9, 1)
        return round(os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') / 1e9, 1)
    except Exception:
        return None

def devices():
    out = [{'id': 'cpu', 'kind': 'CPU', 'name': platform.processor() or platform.machine(), 'threads': os.cpu_count(), 'mem_gb': ram_gb(), 'engine': 'PyTorch' if HAVE_TORCH else 'numpy'}]
    if HAVE_TORCH and torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            out.insert(0, {'id': f'cuda:{i}', 'kind': 'GPU', 'name': p.name, 'mem_gb': round(p.total_memory / 1e9, 1), 'engine': 'PyTorch CUDA'})
    if HAVE_TORCH and getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
        out.insert(0, {'id': 'mps', 'kind': 'GPU', 'name': 'Apple GPU (MPS)', 'mem_gb': None, 'engine': 'PyTorch MPS'})
    return out

def pick_device(want):
    ds = [d['id'] for d in devices()]
    if want and want != 'auto':
        if want == 'gpu':
            return next((d for d in ds if d != 'cpu'), 'cpu')
        if want in ds:
            return want
    return ds[0]


# ---------------------------------------------------------------- models
def default_model_dirs():
    h = os.path.expanduser('~')
    cands = [os.path.join(h, '.lmstudio', 'models'), os.path.join(h, '.cache', 'lm-studio', 'models'),
             os.path.join(h, '.cache', 'huggingface', 'hub'), os.path.join(h, 'models'), os.path.join(h, 'Downloads')]
    return [d for d in cands if os.path.isdir(d)]

def list_models():
    seen, out = set(), []
    for root in MODEL_DIRS:
        for dp, dn, fn in os.walk(root):
            for f in fn:
                if f.lower().endswith('.gguf'):
                    p = os.path.realpath(os.path.join(dp, f))
                    if p in seen:
                        continue
                    seen.add(p)
                    try:
                        st = os.stat(p)
                    except OSError:
                        continue
                    out.append({'path': p, 'name': f, 'folder': os.path.relpath(dp, root), 'root': root, 'size': st.st_size, 'mtime': st.st_mtime})
    out.sort(key=lambda m: (-m['size'] if 'mmproj' not in m['name'].lower() else 0, m['name']))
    return out

def allowed(path):
    p = os.path.realpath(path)
    return p.lower().endswith('.gguf') and os.path.isfile(p) and any(p.startswith(os.path.realpath(r) + os.sep) for r in MODEL_DIRS)


# ---------------------------------------------------------------- embedding store
class EmbStore:
    """token_embd of one model on one device: raw values (for dimension layers) + unit rows (cosine)."""
    def __init__(self, path, device, progress):
        t0 = time.time()
        self.path, self.device = path, device
        progress(0.0, 'Verzeichnis der Datei lesen')
        r = GGUFReader(path)
        emb = next((t for t in r.tensors if t.name == 'token_embd.weight'), None)
        if emb is None:
            raise ValueError('Diese Datei hat keine token_embd-Tabelle.')
        self.n, self.d = int(emb.shape[1]), int(emb.shape[0])
        self.qtype = emb.tensor_type.name
        raw = np.asarray(emb.data).reshape(self.n, -1)
        self.rawT = np.empty((self.d, self.n), dtype=np.float16)   # columns = dimensions, contiguous
        unit = np.empty((self.n, self.d), dtype=np.float32)
        CH = 4096
        def part(s):   # chunks are disjoint, so threads can write side by side (numpy releases the GIL here)
            v = quants.dequantize(np.ascontiguousarray(raw[s:s + CH]), emb.tensor_type).reshape(-1, self.d).astype(np.float32)
            self.rawT[:, s:s + len(v)] = v.T.astype(np.float16)
            nrm = np.linalg.norm(v, axis=1, keepdims=True); nrm[nrm == 0] = 1
            unit[s:s + len(v)] = v / nrm
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max(1, min(4, (os.cpu_count() or 2) - 1))) as ex:
            futs = [ex.submit(part, s) for s in range(0, self.n, CH)]
            for k, f in enumerate(as_completed(futs)):
                f.result(); progress(0.95 * (k + 1) / len(futs), 'Tabelle entpacken')
        self.torch = HAVE_TORCH
        if self.torch:
            self.X = torch.from_numpy(unit).to(device)
            del unit
        else:
            self.X = unit
        self.load_s = time.time() - t0
        progress(1, 'bereit')

    def mem_info(self):
        return {'n': self.n, 'd': self.d, 'qtype': self.qtype, 'device': self.device, 'engine': 'PyTorch' if self.torch else 'numpy', 'load_s': round(self.load_s, 1)}

    def topk_vec(self, q, k, exclude=()):
        """q: unit vector (np); exact cosine over all rows."""
        if self.torch:
            qt = torch.from_numpy(np.ascontiguousarray(q, dtype=np.float32)).to(self.device)
            s = self.X @ qt
            if exclude:
                s[torch.tensor(list(exclude), device=self.device)] = -2
            v, i = s.topk(k)
            return i.cpu().numpy().tolist(), v.cpu().numpy().tolist()
        s = self.X @ q.astype(np.float32)
        for e in exclude:
            s[e] = -2
        i = np.argpartition(-s, k)[:k]; i = i[np.argsort(-s[i])]
        return i.tolist(), s[i].tolist()

    def row(self, i):
        return self.rawT[:, i].astype(np.float32)

    def pca(self, k=8):
        """Principal directions of all rows (each row at length 1, as for the map), from all dimensions.
        Every direction mixes all d columns; ordered by how much the tokens spread along it."""
        if getattr(self, '_pca', None) is not None and self._pca['k'] >= k:
            return self._pca
        t0 = time.time(); n, d = self.n, self.d; q = k + 8
        if self.torch:
            X = self.X
            mean = X.mean(0, keepdim=True)
            torch.manual_seed(0)
            U, S, V = torch.pca_lowrank(X, q=q, center=True, niter=4)    # centring is implicit, no copy of the table
            comps = V[:, :k]
            proj = (X @ comps - mean @ comps).float().cpu().numpy()
            comps = comps.float().cpu().numpy(); S = S[:k].float().cpu().numpy(); m = mean.float().cpu().numpy()[0]
        else:
            X = self.X; m = X.mean(0)
            rng = np.random.default_rng(0)
            Om = rng.standard_normal((d, q)).astype(np.float32)
            Y = X @ Om - (m @ Om)[None, :]
            for _ in range(4):                                             # power iterations on the centred table
                Qy, _ = np.linalg.qr(Y)
                Z = X.T @ Qy - np.outer(m, Qy.sum(0))
                Qz, _ = np.linalg.qr(Z)
                Y = X @ Qz - (m @ Qz)[None, :]
            Qy, _ = np.linalg.qr(Y)
            Bm = Qy.T @ X - np.outer(Qy.sum(0), m)
            _, S, Vt = np.linalg.svd(Bm, full_matrices=False)
            comps = Vt[:k].T.astype(np.float32); S = S[:k]
            proj = X @ comps - (m @ comps)[None, :]
        for c in range(k):                                                 # orientation: largest loading positive
            if comps[np.argmax(np.abs(comps[:, c])), c] < 0:
                comps[:, c] *= -1; proj[:, c] *= -1
        total = float(n - n * float(m @ m))                                # rows have length 1: sum of squares minus the mean part
        ratio = (S.astype(np.float64) ** 2 / total).tolist()
        order = np.argsort(proj, axis=0)
        self._pca = {'k': k, 'ratio': ratio, 'total': total, 'seconds': round(time.time() - t0, 2),
                     'std': proj.std(0).tolist(),
                     'dims': [[[int(j), float(comps[j, c])] for j in np.argsort(-np.abs(comps[:, c]))[:12]] for c in range(k)],
                     'pos': [order[::-1, c][:12].tolist() for c in range(k)], 'neg': [order[:, c][:12].tolist() for c in range(k)],
                     'proj': proj.astype(np.float16)}
        return self._pca

    def knn_all(self, k, progress, stop):
        """exact kNN over all rows (cosine), chunked."""
        n = self.n
        nb = np.empty((n, k), dtype=np.int32); sim = np.empty((n, k), dtype=np.float32)
        B = 2048 if (self.torch and self.device != 'cpu') else 512
        t0 = time.time()
        for s in range(0, n, B):
            if stop():
                raise RuntimeError('abgebrochen')
            if self.torch:
                S = self.X[s:s + B] @ self.X.T
                S[torch.arange(S.shape[0], device=self.device), torch.arange(s, s + S.shape[0], device=self.device)] = -2
                v, i = S.topk(k, dim=1)
                nb[s:s + B] = i.cpu().numpy(); sim[s:s + B] = v.cpu().numpy()
            else:
                S = self.X[s:s + B] @ self.X.T
                S[np.arange(S.shape[0]), np.arange(s, s + S.shape[0])] = -2
                i = np.argpartition(-S, k, axis=1)[:, :k]
                vv = np.take_along_axis(S, i, 1); o = np.argsort(-vv, axis=1)
                nb[s:s + B] = np.take_along_axis(i, o, 1); sim[s:s + B] = np.take_along_axis(vv, o, 1)
            done = min(n, s + B) / n; el = time.time() - t0
            progress(done, f'alle Paare vergleichen ({done * 100:.0f} %, noch ca. {el / done * (1 - done):.0f} s)')
        return nb, sim


def get_store(path, device, progress=lambda p, m: None):
    st = STORES.get(path)
    if st and st.device == device:
        return st
    if st:  # other device: free and reload
        STORES.pop(path, None); del st
        if HAVE_TORCH and torch.cuda.is_available():
            torch.cuda.empty_cache()
    for p in list(STORES):   # keep only one table in memory (5+ GB each)
        if p != path:
            STORES.pop(p, None)
    st = EmbStore(path, device, progress)
    STORES[path] = st
    return st


# ---------------------------------------------------------------- full map
def layout_torch(nb, sim, n, device, epochs, neg, progress, k_graph=8):
    dev = device
    nbt = torch.from_numpy(nb[:, :k_graph].astype(np.int64)).to(dev)
    dist = (1 - torch.from_numpy(sim[:, :k_graph]).to(dev)).clamp_min(0)
    rho = dist[:, :1]; target = float(np.log2(k_graph))
    lo = torch.zeros(n, 1, device=dev); hi = torch.full((n, 1), 1e4, device=dev); sig = torch.ones(n, 1, device=dev)
    for _ in range(64):
        s = torch.exp(-(dist - rho).clamp_min(0) / sig).sum(1, keepdim=True); big = s > target
        hi = torch.where(big, sig, hi); lo = torch.where(big, lo, sig); sig = (lo + hi) / 2
    w = torch.exp(-(dist - rho).clamp_min(0) / sig)
    rows = torch.arange(n, device=dev).repeat_interleave(k_graph)
    # symmetrise on CPU-friendly COO (sparse ops are not available on every backend)
    W = torch.sparse_coo_tensor(torch.stack([rows, nbt.reshape(-1)]).cpu(), w.reshape(-1).cpu(), (n, n)).coalesce(); Wt = W.t().coalesce()
    P = (W + Wt - W * Wt).coalesce(); ei, ej = P.indices(); ew = P.values(); keep = ei < ej
    ei, ej, ew = ei[keep].to(dev), ej[keep].to(dev), ew[keep].to(dev)
    g = torch.Generator(device='cpu').manual_seed(11)
    Y = (torch.rand(n, 2, generator=g) * 20 - 10).to(dev)
    aa, bb = 1.577, 0.895; prob = ew / ew.max()
    for ep in range(epochs):
        alpha = 1.0 - ep / epochs
        m = torch.rand(prob.shape, device=dev) < prob; i, j = ei[m], ej[m]
        d = Y[i] - Y[j]; d2 = (d * d).sum(1, keepdim=True)
        gr = (-2 * aa * bb * d2.clamp_min(1e-12) ** (bb - 1)) / (1 + aa * d2 ** bb); gd = (gr * d).clamp(-4, 4) * alpha
        Y.index_add_(0, i, gd); Y.index_add_(0, j, -gd)
        ii = i.repeat(neg); kk = torch.randint(0, n, ii.shape, device=dev)
        d = Y[ii] - Y[kk]; d2 = (d * d).sum(1, keepdim=True)
        gr = (2 * bb) / ((0.001 + d2) * (1 + aa * d2 ** bb)); Y.index_add_(0, ii, (gr * d).clamp(-4, 4) * alpha)
        if ep % 20 == 0:
            progress(ep / epochs, f'Karte auslegen ({ep}/{epochs})')
    return Y.cpu().numpy().astype(np.float32)

def layout_numpy(nb, sim, n, epochs, neg, progress, k_graph=8):
    rng = np.random.default_rng(11)
    dist = np.clip(1 - sim[:, :k_graph], 0, None); rho = dist[:, :1]; target = np.log2(k_graph)
    lo = np.zeros((n, 1)); hi = np.full((n, 1), 1e4); sig = np.ones((n, 1))
    for _ in range(64):
        s = np.exp(-np.clip(dist - rho, 0, None) / sig).sum(1, keepdims=True); big = s > target
        hi = np.where(big, sig, hi); lo = np.where(big, lo, sig); sig = (lo + hi) / 2
    w = np.exp(-np.clip(dist - rho, 0, None) / sig)
    a = np.repeat(np.arange(n), k_graph); b = nb[:, :k_graph].reshape(-1); ww = w.reshape(-1)
    lo_, hi_ = np.minimum(a, b), np.maximum(a, b); key = lo_.astype(np.int64) * n + hi_
    order = np.argsort(key); key, ww = key[order], ww[order]
    uk, start = np.unique(key, return_index=True)
    # fuzzy union of both directions: w1 + w2 - w1*w2 (pairs appear once or twice)
    cnt = np.diff(np.append(start, len(key)))
    first = ww[start]; second = np.where(cnt > 1, ww[np.minimum(start + 1, len(ww) - 1)], 0)
    ew = first + second - first * second
    ei = (uk // n).astype(np.int64); ej = (uk % n).astype(np.int64)
    Y = rng.random((n, 2)) * 20 - 10; aa, bb = 1.577, 0.895; prob = ew / ew.max()
    for ep in range(epochs):
        alpha = 1.0 - ep / epochs
        m = rng.random(len(prob)) < prob; i, j = ei[m], ej[m]
        d = Y[i] - Y[j]; d2 = (d * d).sum(1, keepdims=True)
        gr = (-2 * aa * bb * np.maximum(d2, 1e-12) ** (bb - 1)) / (1 + aa * d2 ** bb); gd = np.clip(gr * d, -4, 4) * alpha
        for ax in (0, 1):
            Y[:, ax] += np.bincount(i, gd[:, ax], n) - np.bincount(j, gd[:, ax], n)
        ii = np.repeat(i, neg); kk = rng.integers(0, n, len(ii))
        d = Y[ii] - Y[kk]; d2 = (d * d).sum(1, keepdims=True)
        rep_ = np.clip((2 * bb) / ((0.001 + d2) * (1 + aa * d2 ** bb)) * d, -4, 4) * alpha
        for ax in (0, 1):
            Y[:, ax] += np.bincount(ii, rep_[:, ax], n)
        if ep % 5 == 0:
            progress(ep / epochs, f'Karte auslegen ({ep}/{epochs})')
    return Y.astype(np.float32)

def map_cache_path(path, k, epochs):
    st = os.stat(path)
    h = hashlib.sha1(f'{os.path.realpath(path)}|{st.st_size}|{int(st.st_mtime)}|cos|{k}|{epochs}'.encode()).hexdigest()[:16]
    return os.path.join(CACHE, f'{os.path.basename(path)}.{h}.lupe-map.bin')

def map_epochs(device):
    # measured on Qwen3.8 (248k tokens): recall@10 of the 2-D layout 0.286 (GPU, 1000), 0.290 (torch CPU, 600), 0.280 (numpy, 600; 400 is close)
    return 1000 if (HAVE_TORCH and device != 'cpu') else 600 if HAVE_TORCH else 400

def cached_map(path):
    # exact match (path, size, date, settings); the most epochs = best layout
    for e in (1000, 600, 400, 300):
        fn = map_cache_path(path, 10, e)
        if os.path.exists(fn):
            return fn
    return None

_RATE = {}
def mac_rate(device):
    # multiply-accumulates per second for "rows x all rows" on this device, measured once
    key = (device, HAVE_TORCH)
    if key in _RATE:
        return _RATE[key]
    d, n = 4096, 16384
    if HAVE_TORCH:
        X = torch.randn(n, d, device=device); Q = X[:512 if device == 'cpu' else 2048]
        sync = (lambda: torch.cuda.synchronize()) if str(device).startswith('cuda') else (lambda: None)
        (Q @ X.T).topk(10, dim=1); sync()
        t = time.time()
        for _ in range(3):
            (Q @ X.T).topk(10, dim=1)
        sync(); dt = (time.time() - t) / 3
    else:
        X = np.random.default_rng(0).random((n, d), dtype=np.float32); Q = X[:512]
        t = time.time(); S = Q @ X.T; np.argpartition(-S, 10, axis=1); dt = time.time() - t
    _RATE[key] = len(Q) * n * d / dt
    return _RATE[key]

def estimate_map(path, device, n, d):
    st = STORES.get(path)
    load = 0 if (st and st.device == device) else n * d / 4e7          # dequantising, ~40 M values/s
    knn = n * n * d / mac_rate(device) * 1.1
    per = 8e-8 if (HAVE_TORCH and device != 'cpu') else 1.1e-6 if HAVE_TORCH else 4.4e-6   # seconds per token and epoch (measured)
    return round(load + knn + per * n * map_epochs(device))

def write_map(fn, hdr, Y, nb):
    hb = json.dumps(hdr).encode('utf-8')
    with open(fn, 'wb') as f:
        f.write(b'LUPEMAP1'); f.write(struct.pack('<I', len(hb))); f.write(hb); f.write(b' ' * ((-(12 + len(hb))) % 4))
        f.write(Y.astype(np.float16).tobytes()); f.write(nb.astype(np.int32).tobytes())

def job_map(job, path, device, k=10, epochs=None):
    gpu = device != 'cpu'
    epochs = epochs or map_epochs(device)
    neg = 20   # fewer negative samples blur the clusters (numpy with 10: recall 0.23 instead of 0.28)
    fn = map_cache_path(path, k, epochs)
    if os.path.exists(fn):
        job['result'] = {'cache': fn}; return
    def P(lo, hi):
        return lambda p, m: job.update(progress=lo + (hi - lo) * p, message=m)
    st = get_store(path, device, P(0, 0.15))
    t0 = time.time()
    nb, sim = st.knn_all(k, P(0.15, 0.8), lambda: job.get('cancel'))
    t_knn = time.time() - t0
    if HAVE_TORCH:
        Y = layout_torch(nb, sim, st.n, device if device != 'mps' else 'cpu', epochs, neg, P(0.8, 0.99))
    else:
        Y = layout_numpy(nb, sim, st.n, epochs, neg, P(0.8, 0.99))
    hdr = {'magic': 'LUPEMAP1', 'model': os.path.basename(path), 'n': st.n, 'dims': st.d, 'k': k, 'epochs': epochs, 'neg': neg,
           'similarity': 'Kosinus der rohen Zeilen, exakt über alle Dimensionen', 'device': device, 'engine': 'PyTorch' if HAVE_TORCH else 'numpy',
           'knn_s': round(t_knn, 1), 'total_s': round(time.time() - t0 + st.load_s, 1), 'created': time.strftime('%Y-%m-%d %H:%M')}
    write_map(fn, hdr, Y, nb)
    job['result'] = {'cache': fn}


# ---------------------------------------------------------------- jobs
def start_job(kind, fn, *args):
    jid = uuid.uuid4().hex[:12]
    job = {'id': jid, 'kind': kind, 'state': 'wartet', 'progress': 0.0, 'message': 'wartet auf freie Rechenzeit', 'started': time.time()}
    JOBS[jid] = job
    def run():
        with LOCK:
            job['state'] = 'läuft'; job['message'] = 'läuft'
            try:
                fn(job, *args); job['state'] = 'fertig'; job['progress'] = 1.0
            except Exception as e:
                traceback.print_exc(); job['state'] = 'fehler'; job['message'] = str(e)
            job['seconds'] = round(time.time() - job['started'], 1)
    threading.Thread(target=run, daemon=True).start()
    return job


# ---------------------------------------------------------------- HTTP
class H(BaseHTTPRequestHandler):
    server_version = 'GGUF-Lupe'

    def log_message(self, fmt, *a):
        # only problems; every Range read and poll would flood the console
        if len(a) > 1 and str(a[1])[:1] in '23':
            return
        sys.stderr.write('  ' + (fmt % a) + '\n')

    def send(self, code, body, ctype='application/json; charset=utf-8', extra=None):
        if isinstance(body, (dict, list)):
            body = json.dumps(body).encode('utf-8')
        elif isinstance(body, str):
            body = body.encode('utf-8')
        self.send_response(code)
        self.send_header('Content-Type', ctype); self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers(); self.wfile.write(body)

    def err(self, code, msg):
        self.send(code, {'error': msg})

    def do_GET(self):
        try:
            self.route()
        except BrokenPipeError:
            pass
        except Exception as e:
            traceback.print_exc()
            try:
                self.err(500, str(e))
            except Exception:
                pass

    def route(self):
        u = urlparse(self.path); q = {k: v[0] for k, v in parse_qs(u.query).items()}; p = u.path
        if p in ('/', '/index.html', '/gguf-lupe.html'):
            with open(PAGE, 'rb') as f:
                return self.send(200, f.read(), 'text/html; charset=utf-8')
        if p.startswith('/examples/'):                       # recorded examples next to the page
            name = p[len('/examples/'):]
            fn = os.path.join(os.path.dirname(PAGE), 'examples', name)
            if not re.fullmatch(r'[a-z0-9-]+\.js', name) or not os.path.isfile(fn):
                return self.send(404, '// kein Beispiel', 'text/javascript; charset=utf-8')
            with open(fn, 'rb') as f:
                return self.send(200, f.read(), 'text/javascript; charset=utf-8')
        if p == '/favicon.ico':
            self.send_response(204); self.end_headers(); return
        if p == '/api/info':
            return self.send(200, {'server': 'GGUF-Lupe', 'version': 1, 'python': platform.python_version(), 'numpy': np.__version__,
                                   'torch': torch.__version__ if HAVE_TORCH else None, 'devices': devices(), 'default_device': pick_device(ARGS.device),
                                   'model_dirs': MODEL_DIRS, 'cache_dir': CACHE,
                                   'loaded': [s.mem_info() | {'path': s.path} for s in STORES.values()]})
        if p == '/api/models':
            return self.send(200, list_models())
        if p == '/api/job':
            j = JOBS.get(q.get('id', ''))
            return self.send(200, j) if j else self.err(404, 'unbekannter Job')
        if p == '/api/job/cancel':
            j = JOBS.get(q.get('id', ''))
            if j: j['cancel'] = True
            return self.send(200, {'ok': True})
        if p.startswith(('/api/llm', '/api/run')):
            return self.route_run(p, q)
        path = q.get('path', '')
        if p.startswith('/api/') and not allowed(path):
            return self.err(403, 'Nur .gguf-Dateien aus den freigegebenen Modell-Ordnern.')
        dev = pick_device(q.get('device'))
        if p == '/api/file':
            return self.file_range(path)
        if p == '/api/load':
            st = STORES.get(path)
            if st and st.device == dev:
                return self.send(200, {'state': 'fertig', 'info': st.mem_info()})
            return self.send(200, start_job('load', lambda job: get_store(path, dev, lambda pr, m: job.update(progress=pr, message=m)) and None))
        if p in ('/api/nn', '/api/analogy', '/api/dim', '/api/row', '/api/pca'):
            st = STORES.get(path)
            if not st:
                return self.err(409, 'Tabelle noch nicht geladen (/api/load).')
            k = min(500, int(q.get('k', 24)))
            if p == '/api/nn':
                i = int(q['id']); v = st.row(i); v /= (np.linalg.norm(v) or 1)
                t = time.time(); ids, sims = st.topk_vec(v, k + 1, exclude=(i,))
                return self.send(200, {'ids': ids[:k], 'sims': sims[:k], 'ms': round((time.time() - t) * 1000, 1), 'device': st.device})
            if p == '/api/analogy':
                a, b, c = [int(x) for x in q['ids'].split(',')]
                un = lambda x: x / (np.linalg.norm(x) or 1)
                v = un(un(st.row(a)) - un(st.row(b)) + un(st.row(c)))
                t = time.time(); ids, sims = st.topk_vec(v, k, exclude=(a, b, c))
                return self.send(200, {'ids': ids, 'sims': sims, 'ms': round((time.time() - t) * 1000, 1), 'device': st.device})
            if p == '/api/dim':
                i = int(q['i']); col = st.rawT[i]
                f = col.astype(np.float32); order = np.argsort(f)
                meta = {'i': i, 'min': float(f.min()), 'max': float(f.max()), 'mean': float(f.mean()), 'std': float(f.std()),
                        'top': order[::-1][:15].tolist(), 'bottom': order[:15].tolist()}
                mb = json.dumps(meta).encode('utf-8')
                body = struct.pack('<I', len(mb)) + mb + b' ' * ((-(4 + len(mb))) % 2) + col.tobytes()
                return self.send(200, body, 'application/octet-stream')
            if p == '/api/row':
                return self.send(200, st.row(int(q['id'])).tobytes(), 'application/octet-stream')
            if p == '/api/pca':
                r = st.pca(min(16, max(3, int(q.get('k', 8)))))
                meta = {x: r[x] for x in r if x != 'proj'} | {'n': st.n, 'd': st.d}
                mb = json.dumps(meta).encode('utf-8')
                body = struct.pack('<I', len(mb)) + mb + b' ' * ((-(4 + len(mb))) % 2) + np.ascontiguousarray(r['proj']).tobytes()
                return self.send(200, body, 'application/octet-stream')
        if p == '/api/map/start':
            return self.send(200, start_job('map', job_map, path, dev))
        if p == '/api/map':
            fn = q.get('cache', '')
            if not (fn.startswith(CACHE) and fn.endswith('.lupe-map.bin') and os.path.isfile(fn)):
                return self.err(404, 'Karte nicht gefunden')
            with open(fn, 'rb') as f:
                return self.send(200, f.read(), 'application/octet-stream')
        if p == '/api/map/cached':
            fn = cached_map(path)
            if fn:
                return self.send(200, {'cache': fn})
            est = None
            if q.get('n') and q.get('d'):
                try:
                    est = estimate_map(path, dev, int(q['n']), int(q['d']))
                except Exception:
                    traceback.print_exc()
            return self.send(200, {'cache': None, 'estimate_s': est, 'device': dev})
        return self.err(404, 'unbekannt')

    def route_run(self, p, q):
        # "Durchlauf": the model runs in llama.cpp (worker process), every intermediate step is recorded
        import lupe_run as R
        try:
            if p == '/api/llm/info':
                return self.send(200, R.call('info'))
            if p == '/api/llm/get':
                return self.send(200, start_job('llama', R.job_get_llama, q.get('kind', 'vulkan')))
            if p == '/api/run/start':
                path = q.get('path', '')
                if not allowed(path):
                    return self.err(403, 'Nur .gguf-Dateien aus den freigegebenen Modell-Ordnern.')
                dev = q.get('dev') or 'auto'
                if dev == 'auto':
                    ds = R.call('info').get('devices') or []
                    gpus = sorted([d for d in ds if d['kind'] == 'GPU'], key=lambda d: -d['total_gb']) or [d for d in ds if d['kind'] == 'iGPU']
                    dev = gpus[0]['id'] if gpus else 'cpu'
                text = q.get('q', '').strip()
                if not text:
                    return self.err(400, 'Bitte eine Frage oder einen Text eingeben.')
                return self.send(200, start_job('run', R.job_run, path, text, q.get('chat', '1') == '1', int(q.get('max', 60)), dev))
            rid = q.get('run', '')
            if p == '/api/run/info':
                return self.send(200, R.call('run_info', rid))
            if p == '/api/run/stack':
                return self.send(200, R.call('run_stack', rid, int(q['p'])), 'application/octet-stream')
            if p == '/api/run/vec':
                return self.send(200, R.call('run_vec', rid, int(q['n']), int(q['p'])), 'application/octet-stream')
            if p == '/api/run/block':
                return self.send(200, R.call('run_block', rid, int(q['b'])))
            if p == '/api/run/lens':
                return self.send(200, R.run_lens(rid, int(q['p'])))
            if p == '/api/run/explain':
                return self.send(200, R.call('run_explain', rid, int(q['n']), int(q['p']), int(q['j'])))
            if p == '/api/run/export':                   # optional lang=de|en; without it the language is guessed from the question
                import lupe_export
                return self.send(200, start_job('export', lupe_export.job_export, rid, (q.get('title') or '').strip() or None, q.get('lang')))
        except R.Gone as e:
            return self.err(410, str(e).strip('"\''))
        return self.err(404, 'unbekannt')

    def file_range(self, path):
        size = os.path.getsize(path); rng = self.headers.get('Range')
        a, b = 0, size - 1
        if rng and rng.startswith('bytes='):
            s, e = rng[6:].split('-'); a = int(s) if s else 0; b = int(e) if e else size - 1
        b = min(b, size - 1); n = b - a + 1
        self.send_response(206 if rng else 200)
        self.send_header('Content-Type', 'application/octet-stream'); self.send_header('Content-Length', str(n))
        self.send_header('Accept-Ranges', 'bytes'); self.send_header('X-File-Size', str(size))
        if rng: self.send_header('Content-Range', f'bytes {a}-{b}/{size}')
        self.end_headers()
        with open(path, 'rb') as f:
            f.seek(a); left = n
            while left > 0:
                chunk = f.read(min(1 << 20, left))
                if not chunk: break
                self.wfile.write(chunk); left -= len(chunk)


def main():
    global ARGS, PAGE, MODEL_DIRS
    ap = argparse.ArgumentParser(description='GGUF-Lupe Server')
    ap.add_argument('--port', type=int, default=8765)
    ap.add_argument('--models', nargs='*', default=[], help=say('weitere Ordner mit .gguf-Dateien', 'more folders with .gguf files'))
    ap.add_argument('--device', default='auto', help='auto | gpu | cuda | mps | cpu')
    ap.add_argument('--page', default=None, help=say('Pfad zur gguf-lupe.html', 'path to gguf-lupe.html'))
    ap.add_argument('--no-browser', action='store_true')
    ap.add_argument('--engine', default='auto', help=say('auto | numpy (erzwingt reine CPU-Rechnung ohne PyTorch)', 'auto | numpy (forces plain CPU computing without PyTorch)'))
    ap.add_argument('--llama', default=None, help=say('Ordner einer llama.cpp-Version (b11388) für den Durchlauf; sonst wird gesucht',
                                                      'folder of a llama.cpp release (b11388) for the forward pass; searched for if not given'))
    ARGS = ap.parse_args()
    global HAVE_TORCH
    if ARGS.engine == 'numpy':
        HAVE_TORCH = False
    if ARGS.llama:
        import lupe_run
        lupe_run.FORCE_BUILD = os.path.abspath(ARGS.llama)
    PAGE = ARGS.page or next((p for p in [os.path.join(HERE, 'gguf-lupe.html'), os.path.join(HERE, '..', 'gguf-lupe.html')] if os.path.isfile(p)), None)
    if not PAGE:
        sys.exit(say('gguf-lupe.html nicht gefunden (gehört in den Ordner über src/, oder --page angeben).',
                     'gguf-lupe.html not found (it belongs in the folder above src/, or use --page).'))
    MODEL_DIRS = [os.path.realpath(d) for d in (ARGS.models + default_model_dirs()) if os.path.isdir(d)]
    url = f'http://127.0.0.1:{ARGS.port}/'
    devs, dflt, n_models = ', '.join(d['kind'] + ' ' + str(d['name']) for d in devices()), pick_device(ARGS.device), len(list_models())
    print(f'GGUF-Lupe Server  {url}')
    print(say(f'  Rechnen: {devs}  (Standard: {dflt})', f'  Compute: {devs}  (default: {dflt})'))
    print(say(f'  Modell-Ordner: {"; ".join(MODEL_DIRS) or "keine gefunden (--models DIR)"}', f'  Model folders: {"; ".join(MODEL_DIRS) or "none found (--models DIR)"}'))
    print(say(f'  Gespeicherte Karten: {CACHE}', f'  Saved maps: {CACHE}'))
    print(say(f'  {n_models} GGUF-Dateien gefunden. Beenden mit Strg+C.', f'  {n_models} GGUF files found. Stop with Ctrl+C.'))
    srv = ThreadingHTTPServer(('127.0.0.1', ARGS.port), H)
    if not ARGS.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print(say('beendet', 'stopped'))


if __name__ == '__main__':
    main()
