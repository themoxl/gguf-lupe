"""Server side of the "Durchlauf" tab: talks to the llama.cpp worker process and computes the logit lens.

The worker (lupe_worker.py) is started on first use with the same Python. Calls go over a local,
authenticated connection, one at a time. If llama.cpp crashes, the next call starts a fresh worker.
"""
import os
import secrets
import subprocess
import sys
import threading
from multiprocessing.connection import Listener

import numpy as np
from gguf import GGUFReader, quants

try:
    import torch
except Exception:
    torch = None

HERE = os.path.dirname(os.path.abspath(__file__))
FORCE_BUILD = None


class Gone(KeyError):
    """The run is no longer in the worker's memory."""


class Worker:
    def __init__(self):
        self.lock = threading.Lock()
        self.proc = self.conn = None

    def _start(self):
        key = secrets.token_bytes(16)
        lst = Listener(('127.0.0.1', 0), authkey=key)
        port = lst.address[1]
        self.proc = subprocess.Popen([sys.executable, os.path.join(HERE, 'lupe_worker.py'), str(port), key.hex(), FORCE_BUILD or '-'], cwd=HERE)
        try:
            lst._listener._socket.settimeout(90)          # do not wait forever if the worker cannot start
        except Exception:
            pass
        try:
            self.conn = lst.accept()
        except OSError:
            self.proc.kill(); self.proc = None
            raise RuntimeError('Der llama.cpp-Prozess ist nicht gestartet (siehe Server-Fenster).')
        finally:
            lst.close()

    def alive(self):
        return self.proc is not None and self.proc.poll() is None

    def call(self, name, *args, progress=None, cancel=None):
        with self.lock:
            if not self.alive():
                self._start()
            try:
                self.conn.send((name, args))
                while True:
                    if cancel is not None and cancel():
                        self.conn.send(('cancel',)); cancel = None
                    if not self.conn.poll(0.25):
                        if not self.alive():
                            raise EOFError
                        continue
                    kind, payload = self.conn.recv()
                    if kind == 'progress':
                        if progress:
                            progress(*payload)
                        continue
                    if kind == 'ok':
                        return payload
                    if kind == 'gone':
                        raise Gone(payload)
                    raise RuntimeError(payload)
            except (EOFError, OSError, ConnectionError):
                code = self.proc.poll() if self.proc else None
                self.proc = self.conn = None
                raise RuntimeError(f'Der llama.cpp-Prozess ist beendet worden (Code {code}). Häufigste Ursache: zu wenig Grafikspeicher. '
                                   'Beim nächsten Versuch wird er neu gestartet; sonst „Rechnen auf: CPU“ wählen.')

    def stop(self):
        with self.lock:
            if self.alive():
                self.proc.terminate()
            self.proc = self.conn = None


WORKER = Worker()


def call(name, *args, **kw):
    return WORKER.call(name, *args, **kw)


# ---------------------------------------------------------------- logit lens: final norm + output head on any layer's vector
class LensHead:
    def __init__(self, path, progress=lambda p, m: None):
        rd = GGUFReader(path)
        def field(key):
            f = rd.fields.get(key)
            if f is None or not f.data:
                return None
            v = f.parts[f.data[0]]
            return bytes(v).decode('utf-8', 'replace') if v.dtype == np.uint8 else v.tolist()[0]
        arch = field('general.architecture')
        self.eps = float(field(f'{arch}.attention.layer_norm_rms_epsilon') or field(f'{arch}.attention.layer_norm_epsilon') or 1e-6)
        self.softcap = field(f'{arch}.final_logit_softcapping')
        tens = {t.name: t for t in rd.tensors}
        nw, nb = tens.get('output_norm.weight'), tens.get('output_norm.bias')
        self.norm_w = np.asarray(nw.data, dtype=np.float32).reshape(-1) if nw is not None else None
        self.norm_b = np.asarray(nb.data, dtype=np.float32).reshape(-1) if nb is not None else None
        self.layernorm = nb is not None
        self.tied = 'output.weight' not in tens
        W = tens['token_embd.weight'] if self.tied else tens['output.weight']
        n, d = int(W.shape[1]), int(W.shape[0])
        raw = np.asarray(W.data).reshape(n, -1)
        out = np.empty((n, d), dtype=np.float16)
        CH = 4096
        from concurrent.futures import ThreadPoolExecutor, as_completed
        def part(s):
            v = raw[s:s + CH]
            out[s:s + len(v)] = (quants.dequantize(np.ascontiguousarray(v), W.tensor_type).reshape(-1, d) if v.dtype == np.uint8 else v).astype(np.float16)
        with ThreadPoolExecutor(max(1, min(4, (os.cpu_count() or 2) - 1))) as ex:
            fs = [ex.submit(part, s) for s in range(0, n, CH)]
            for k, f in enumerate(as_completed(fs)):
                f.result(); progress((k + 1) / len(fs), 'Ausgabe-Kopf entpacken')
        self.gpu = False
        self.W = out
        if torch is not None and torch.cuda.is_available():
            try:
                self.W = torch.from_numpy(out).cuda(); self.gpu = True
            except Exception:
                self.W = out
        self.n, self.d, self.path = n, d, path

    def logits(self, X):
        X = X.astype(np.float32)
        if self.layernorm:
            X = (X - X.mean(1, keepdims=True)) / np.sqrt(X.var(1, keepdims=True) + self.eps)
        else:
            X = X / np.sqrt((X * X).mean(1, keepdims=True) + self.eps)
        if self.norm_w is not None:
            X = X * self.norm_w
        if self.norm_b is not None:
            X = X + self.norm_b
        if self.gpu:
            L = (torch.from_numpy(X).cuda().half() @ self.W.T).float().cpu().numpy()
        else:
            L = np.empty((len(X), self.n), dtype=np.float32)
            for s in range(0, self.n, 16384):
                L[:, s:s + 16384] = X @ self.W[s:s + 16384].astype(np.float32).T
        if self.softcap:
            L = self.softcap * np.tanh(L / self.softcap)
        return L


_LENS = {}
_LENS_LOCK = threading.Lock()


def lens_head(path, progress=lambda p, m: None):
    with _LENS_LOCK:
        h = _LENS.get(path)
        if h is None:
            _LENS.clear()
            h = _LENS[path] = LensHead(path, progress)
        return h


def job_run(job, path, question, chat, max_new, dev_id):
    res = call('job_run', path, question, chat, max_new, dev_id,
               progress=lambda p, m: job.update(progress=0.95 * p, message=m), cancel=lambda: job.get('cancel'))
    lens_head(path, lambda p, m: job.update(progress=0.95 + 0.05 * p, message='Ausgabe-Kopf für die Schicht-Vorschau entpacken (nur beim ersten Mal)'))
    job['result'] = res


def run_lens(rid, pos, k=5):
    inp = call('lens_inputs', rid, pos)
    h = lens_head(inp['path'])
    L = h.logits(inp['X'])
    L -= L.max(1, keepdims=True); P = np.exp(L); P /= P.sum(1, keepdims=True)
    top = np.argsort(-P, axis=1)[:, :k]
    nxt = inp['next']
    rows = []
    for li in range(len(P)):
        row = {'top': [[int(j), float(P[li, j])] for j in top[li]]}
        if nxt is not None:
            row['p_next'] = float(P[li, nxt]); row['rank_next'] = int((P[li] > P[li, nxt]).sum()) + 1
        rows.append(row)
    return {'pos': pos, 'next': nxt, 'rows': rows, 'check': int(top[-1][0]) == inp['top1'], 'gpu': h.gpu, 'tied': h.tied}


# ---------------------------------------------------------------- getting llama.cpp (official release, same version as the code view)
def assets(kind):
    import platform
    v, mach = 'b11388', platform.machine().lower()
    if sys.platform == 'win32':
        arm = 'arm64' in mach
        a = {'cpu': [f'llama-{v}-bin-win-cpu-{"arm64" if arm else "x64"}.zip'], 'vulkan': [f'llama-{v}-bin-win-vulkan-x64.zip'],
             'cuda': [f'llama-{v}-bin-win-cuda-13.4-x64.zip', 'cudart-llama-bin-win-cuda-13.4-x64.zip']}
    elif sys.platform == 'darwin':
        a = {k: [f'llama-{v}-bin-macos-{"arm64" if "arm" in mach else "x64"}.tar.gz'] for k in ('cpu', 'vulkan', 'cuda')}
    else:
        arm = 'aarch64' in mach or 'arm64' in mach
        a = {'cpu': [f'llama-{v}-bin-ubuntu-{"arm64" if arm else "x64"}.tar.gz'], 'vulkan': [f'llama-{v}-bin-ubuntu-vulkan-{"arm64" if arm else "x64"}.tar.gz'],
             'cuda': [f'llama-{v}-bin-ubuntu-cuda-13.4-{"arm64" if arm else "x64"}.tar.gz', f'cudart-llama-{v}-bin-ubuntu-cuda-13.4-{"arm64" if arm else "x64"}.tar.gz']}
    return a[kind]


def job_get_llama(job, kind):
    import tarfile, urllib.request, zipfile
    if kind not in ('cpu', 'vulkan', 'cuda'):
        raise ValueError('unbekannte Variante')
    root = os.path.join(os.path.expanduser('~'), '.cache', 'gguf-lupe', 'llama.cpp-b11388', kind)
    os.makedirs(root, exist_ok=True)
    files = assets(kind)
    for k, name in enumerate(files):
        url = f'https://github.com/ggml-org/llama.cpp/releases/download/b11388/{name}'
        dst = os.path.join(root, name)
        with urllib.request.urlopen(url, timeout=60) as r, open(dst, 'wb') as f:
            total, got = int(r.headers.get('Content-Length') or 0), 0
            while True:
                b = r.read(1 << 20)
                if not b:
                    break
                f.write(b); got += len(b)
                if job.get('cancel'):
                    raise RuntimeError('abgebrochen')
                job.update(progress=(k + (got / total if total else 0.5)) / len(files) * 0.95, message=f'{name}: {got / 1e6:.0f} MB')
        job.update(message=f'{name} entpacken')
        if name.endswith('.zip'):
            with zipfile.ZipFile(dst) as z:
                z.extractall(root)
        else:
            with tarfile.open(dst) as t:
                t.extractall(root)
        os.remove(dst)
    WORKER.stop()                        # the next call starts a worker that sees the new build
    job['result'] = {'dir': root}
