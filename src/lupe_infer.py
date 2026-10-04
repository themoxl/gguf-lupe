"""GGUF-Lupe recorder: a real forward pass with llama.cpp, every intermediate result kept.

Uses the official llama.cpp release DLLs (version b11388, the same source the "Rechenweg" tab shows)
through ctypes. llama.cpp calls back for every node of its compute graph (cb_eval); for the nodes we
ask for, it hands over the computed tensor, which we copy out. That gives two things:

  1. the real compute graph as executed (op, shape, inputs, which weight table from the GGUF file)
  2. the real numbers of every intermediate step, for every token of the text

A run: tokenize the question, generate the answer (greedy, at most 100 tokens in total), then compute
the whole text once more in a single batch with recording switched on. Anything not kept in that
overview pass (e.g. the FFN internals) is recomputed on request by repeating the same pass.
"""
import ctypes as C
import glob
import os
import sys
import threading
import time

import numpy as np

LLAMA_VERSION = 'b11388'

# ---------------------------------------------------------------- C structs (llama.h / ggml.h at b11388)
GGML_MAX_DIMS, GGML_MAX_SRC, GGML_MAX_OP_PARAMS, GGML_MAX_NAME = 4, 10, 64, 64


class ggml_tensor(C.Structure):
    _fields_ = [('type', C.c_int), ('buffer', C.c_void_p),
                ('ne', C.c_int64 * GGML_MAX_DIMS), ('nb', C.c_size_t * GGML_MAX_DIMS),
                ('op', C.c_int), ('op_params', C.c_int32 * (GGML_MAX_OP_PARAMS // 4)), ('flags', C.c_int32),
                ('src', C.c_void_p * GGML_MAX_SRC), ('view_src', C.c_void_p), ('view_offs', C.c_size_t),
                ('data', C.c_void_p), ('name', C.c_char * GGML_MAX_NAME), ('extra', C.c_void_p), ('padding', C.c_char * 8)]


EVAL_CB = C.CFUNCTYPE(C.c_bool, C.POINTER(ggml_tensor), C.c_bool, C.c_void_p)
LOG_CB = C.CFUNCTYPE(None, C.c_int, C.c_char_p, C.c_void_p)
PROGRESS_CB = C.CFUNCTYPE(C.c_bool, C.c_float, C.c_void_p)


class llama_model_params(C.Structure):
    _fields_ = [('devices', C.c_void_p), ('tensor_buft_overrides', C.c_void_p),
                ('n_gpu_layers', C.c_int32), ('split_mode', C.c_int), ('load_mode', C.c_int), ('lazy_mode', C.c_int),
                ('main_gpu', C.c_int32), ('tensor_split', C.c_void_p),
                ('progress_callback', PROGRESS_CB), ('progress_callback_user_data', C.c_void_p),
                ('kv_overrides', C.c_void_p),
                ('vocab_only', C.c_bool), ('check_tensors', C.c_bool), ('use_extra_bufts', C.c_bool),
                ('no_host', C.c_bool), ('no_alloc', C.c_bool), ('load_mtp', C.c_bool)]


class llama_context_params(C.Structure):
    _fields_ = [('n_ctx', C.c_uint32), ('n_batch', C.c_uint32), ('n_ubatch', C.c_uint32), ('n_seq_max', C.c_uint32),
                ('n_rs_seq', C.c_uint32), ('n_outputs_max', C.c_uint32), ('n_outputs_max_per_seq', C.c_uint32),
                ('n_threads', C.c_int32), ('n_threads_batch', C.c_int32),
                ('ctx_type', C.c_int), ('rope_scaling_type', C.c_int), ('pooling_type', C.c_int),
                ('attention_type', C.c_int), ('flash_attn_type', C.c_int),
                ('rope_freq_base', C.c_float), ('rope_freq_scale', C.c_float), ('yarn_ext_factor', C.c_float),
                ('yarn_attn_factor', C.c_float), ('yarn_beta_fast', C.c_float), ('yarn_beta_slow', C.c_float),
                ('yarn_orig_ctx', C.c_uint32), ('defrag_thold', C.c_float),
                ('cb_eval', EVAL_CB), ('cb_eval_user_data', C.c_void_p),
                ('type_k', C.c_int), ('type_v', C.c_int),
                ('abort_callback', C.c_void_p), ('abort_callback_data', C.c_void_p),
                ('embeddings', C.c_bool), ('offload_kqv', C.c_bool), ('no_perf', C.c_bool), ('op_offload', C.c_bool),
                ('swa_full', C.c_bool), ('kv_unified', C.c_bool),
                ('samplers', C.c_void_p), ('n_samplers', C.c_size_t), ('ctx_other', C.c_void_p)]


class llama_batch(C.Structure):
    _fields_ = [('n_tokens', C.c_int32), ('token', C.POINTER(C.c_int32)), ('embd', C.POINTER(C.c_float)),
                ('pos', C.POINTER(C.c_int32)), ('n_seq_id', C.POINTER(C.c_int32)),
                ('seq_id', C.POINTER(C.POINTER(C.c_int32))), ('logits', C.POINTER(C.c_int8))]


class llama_chat_message(C.Structure):
    _fields_ = [('role', C.c_char_p), ('content', C.c_char_p)]


GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_BF16 = 0, 1, 30
DEV_CPU, DEV_GPU, DEV_IGPU = 0, 1, 2
SHAPE_OPS = {'NONE', 'VIEW', 'RESHAPE', 'PERMUTE', 'TRANSPOSE', 'CPY', 'CONT', 'DUP', 'SET_ROWS'}


# ---------------------------------------------------------------- finding a llama.cpp build
def find_builds(extra=()):
    """Folders that contain llama.dll / libllama.so / libllama.dylib of a llama.cpp release."""
    here = os.path.dirname(os.path.abspath(__file__))
    roots = list(extra) + [os.path.join(os.path.dirname(here), 'llama.cpp'), os.path.join(here, 'llama.cpp'), os.path.join(os.path.expanduser('~'), '.cache', 'gguf-lupe', f'llama.cpp-{LLAMA_VERSION}')]
    names = ['llama.dll'] if sys.platform == 'win32' else ['libllama.dylib'] if sys.platform == 'darwin' else ['libllama.so']
    out = []
    def dirs(r, depth=3):                     # release archives unpack with or without sub-folders
        if not os.path.isdir(r):
            return
        yield r
        if depth:
            for d in sorted(glob.glob(os.path.join(r, '*'))):
                if os.path.isdir(d):
                    yield from dirs(d, depth - 1)
    for r in roots:
        for d in dirs(r):
            if any(os.path.isfile(os.path.join(d, n)) for n in names):
                kind = 'cuda' if glob.glob(os.path.join(d, '*ggml-cuda*')) else 'vulkan' if glob.glob(os.path.join(d, '*ggml-vulkan*')) else \
                       'metal' if sys.platform == 'darwin' else 'rocm' if glob.glob(os.path.join(d, '*ggml-hip*')) else 'cpu'
                if not any(os.path.samefile(d, o['dir']) for o in out):
                    out.append({'dir': d, 'kind': kind})
    order = {'cuda': 0, 'metal': 1, 'rocm': 2, 'vulkan': 3, 'cpu': 9}
    return sorted(out, key=lambda b: order.get(b['kind'], 5))


# ---------------------------------------------------------------- the library
class Lib:
    """One llama.cpp build per process (the DLL names are the same in every build)."""

    def __init__(self, build_dir):
        self.dir = os.path.abspath(build_dir)
        if sys.platform == 'win32':
            os.environ['PATH'] = self.dir + os.pathsep + os.environ.get('PATH', '')
            os.add_dll_directory(self.dir)
            ext, pre = '.dll', ''
        else:
            ext, pre = ('.dylib' if sys.platform == 'darwin' else '.so'), 'lib'
        mode = getattr(C, 'RTLD_GLOBAL', 0)
        self.ggml_base = C.CDLL(os.path.join(self.dir, f'{pre}ggml-base{ext}'), mode=mode)
        self.ggml = C.CDLL(os.path.join(self.dir, f'{pre}ggml{ext}'), mode=mode)
        self.llama = C.CDLL(os.path.join(self.dir, f'{pre}llama{ext}'), mode=mode)
        self.logs = []
        self._setup()
        self._log_cb = LOG_CB(self._on_log)
        self.llama.llama_log_set(self._log_cb, None)
        self.ggml.ggml_backend_load_all_from_path(self.dir.encode())
        self.llama.llama_backend_init()
        assert self.ggml_base.ggml_tensor_overhead() == 32 + C.sizeof(ggml_tensor), 'ggml_tensor-Layout passt nicht zu dieser llama.cpp-Version'

    def _on_log(self, level, text, ud):
        try:
            self.logs.append(text.decode('utf-8', 'replace'))
            del self.logs[:-400]
        except Exception:
            pass

    def _setup(self):
        L, G, B = self.llama, self.ggml, self.ggml_base
        vp, i32, u32, f32, b, cs, sz = C.c_void_p, C.c_int32, C.c_uint32, C.c_float, C.c_bool, C.c_char_p, C.c_size_t
        def f(lib, name, res, *args):
            fn = getattr(lib, name); fn.restype = res; fn.argtypes = list(args); return fn
        f(L, 'llama_log_set', None, LOG_CB, vp)
        f(L, 'llama_backend_init', None)
        f(L, 'llama_model_default_params', llama_model_params)
        f(L, 'llama_context_default_params', llama_context_params)
        f(L, 'llama_model_load_from_file', vp, cs, llama_model_params)
        f(L, 'llama_model_free', None, vp)
        f(L, 'llama_init_from_model', vp, vp, llama_context_params)
        f(L, 'llama_free', None, vp)
        f(L, 'llama_model_get_vocab', vp, vp)
        f(L, 'llama_model_n_embd', i32, vp)
        f(L, 'llama_model_n_layer', i32, vp)
        f(L, 'llama_model_n_head', i32, vp)
        f(L, 'llama_model_desc', i32, vp, C.c_char_p, sz)
        f(L, 'llama_model_chat_template', cs, vp, cs)
        f(L, 'llama_vocab_n_tokens', i32, vp)
        f(L, 'llama_vocab_is_eog', b, vp, i32)
        f(L, 'llama_tokenize', i32, vp, cs, i32, C.POINTER(i32), i32, b, b)
        f(L, 'llama_token_to_piece', i32, vp, i32, C.c_char_p, i32, i32, b)
        f(L, 'llama_chat_apply_template', i32, cs, C.POINTER(llama_chat_message), sz, b, C.c_char_p, i32)
        f(L, 'llama_batch_init', llama_batch, i32, i32, i32)
        f(L, 'llama_batch_free', None, llama_batch)
        f(L, 'llama_decode', i32, vp, llama_batch)
        f(L, 'llama_get_logits_ith', C.POINTER(f32), vp, i32)
        f(L, 'llama_get_memory', vp, vp)
        f(L, 'llama_memory_clear', None, vp, b)
        f(L, 'llama_n_ctx', u32, vp)
        f(L, 'llama_synchronize', None, vp)
        f(G, 'ggml_backend_load_all_from_path', None, cs)
        f(G, 'ggml_backend_dev_count', sz)
        f(G, 'ggml_backend_dev_get', vp, sz)
        self.devlib = G
        for lib in (B, G):
            if hasattr(lib, 'ggml_backend_dev_name'):
                f(lib, 'ggml_backend_dev_name', cs, vp); f(lib, 'ggml_backend_dev_description', cs, vp)
                f(lib, 'ggml_backend_dev_type', C.c_int, vp); f(lib, 'ggml_backend_dev_memory', None, vp, C.POINTER(sz), C.POINTER(sz))
                self.devlib = lib
                break
        f(B, 'ggml_backend_tensor_get', None, C.POINTER(ggml_tensor), vp, sz, sz)
        f(B, 'ggml_nbytes', sz, C.POINTER(ggml_tensor))
        f(B, 'ggml_op_desc', cs, C.POINTER(ggml_tensor))
        f(B, 'ggml_op_name', cs, C.c_int)
        f(B, 'ggml_type_name', cs, C.c_int)
        f(B, 'ggml_tensor_overhead', sz)

    def devices(self):
        out, D = [], self.devlib
        for i in range(self.ggml.ggml_backend_dev_count()):
            d = self.ggml.ggml_backend_dev_get(i)
            free, total = C.c_size_t(0), C.c_size_t(0)
            D.ggml_backend_dev_memory(d, C.byref(free), C.byref(total))
            t = D.ggml_backend_dev_type(d)
            out.append({'index': i, 'ptr': d, 'name': D.ggml_backend_dev_name(d).decode(), 'desc': D.ggml_backend_dev_description(d).decode(),
                        'kind': {DEV_CPU: 'CPU', DEV_GPU: 'GPU', DEV_IGPU: 'iGPU'}.get(t, 'other'), 'free_gb': round(free.value / 1e9, 1), 'total_gb': round(total.value / 1e9, 1)})
        return out


_LIB = None
_LIB_LOCK = threading.Lock()


def get_lib(build_dir):
    global _LIB
    with _LIB_LOCK:
        if _LIB is None:
            _LIB = Lib(build_dir)
        elif os.path.abspath(build_dir) != _LIB.dir:
            raise RuntimeError(f'Es ist schon eine andere llama.cpp-Version geladen ({_LIB.dir}). Server neu starten, um zu wechseln.')
        return _LIB


# ---------------------------------------------------------------- a loaded model with its recorder
class Engine:
    N_CTX = 256          # room for the 100-token limit (llama.cpp pads the KV cache)
    MAX_TOKENS = 100

    def __init__(self, lib, path, device_index=None, progress=None):
        """device_index: index from lib.devices() of a GPU, or None for CPU only."""
        self.lib, self.path = lib, path
        L = lib.llama
        mp = L.llama_model_default_params()
        if device_index is None:
            self._devs = (C.c_void_p * 1)(None)                     # no offload device: pure CPU
            mp.n_gpu_layers = 0
            self.device = 'CPU'
        else:
            d = [x for x in lib.devices() if x['index'] == device_index][0]
            self._devs = (C.c_void_p * 2)(d['ptr'], None)
            mp.n_gpu_layers = -1
            self.device = f"{d['desc']} ({d['name']})"
        mp.devices = C.cast(self._devs, C.c_void_p)
        mp.load_mtp = False
        self._progress = PROGRESS_CB(lambda p, ud: (progress(float(p)) if progress else None) or True)
        mp.progress_callback = self._progress
        t0 = time.time()
        self.model = L.llama_model_load_from_file(path.encode('utf-8'), mp)
        if not self.model:
            raise RuntimeError('llama.cpp konnte das Modell nicht laden: ' + ' '.join(x.strip() for x in lib.logs[-6:]))
        self.load_s = time.time() - t0
        self.vocab = L.llama_model_get_vocab(self.model)
        self.n_vocab = L.llama_vocab_n_tokens(self.vocab)
        self.n_embd = L.llama_model_n_embd(self.model)
        self.n_layer = L.llama_model_n_layer(self.model)
        buf = C.create_string_buffer(256); L.llama_model_desc(self.model, buf, 256); self.desc = buf.value.decode('utf-8', 'replace')
        cp = L.llama_context_default_params()
        cp.n_ctx = cp.n_batch = cp.n_ubatch = self.N_CTX
        cp.n_seq_max = 1
        cp.flash_attn_type = 0              # off: the attention weights (softmax) exist as a tensor only without flash attention
        cp.no_perf = True
        if device_index is None:
            cp.op_offload = False
        nt = max(1, (os.cpu_count() or 4))
        cp.n_threads, cp.n_threads_batch = max(1, nt // 2), nt
        self._cb = EVAL_CB(self._on_eval)
        cp.cb_eval = self._cb
        self.ctx = L.llama_init_from_model(self.model, cp)
        if not self.ctx:
            L.llama_model_free(self.model)
            raise RuntimeError('llama.cpp konnte keinen Rechen-Kontext anlegen: ' + ' '.join(x.strip() for x in lib.logs[-6:]))
        self.mem = L.llama_get_memory(self.ctx)
        self.lock = threading.Lock()
        self.rec = None                     # recording state while a pass runs

    def close(self):
        L = self.lib.llama
        if getattr(self, 'ctx', None):
            L.llama_free(self.ctx); self.ctx = None
        if getattr(self, 'model', None):
            L.llama_model_free(self.model); self.model = None

    # -------------------------------------------------- vocabulary
    def tokenize(self, text, special=True):
        L, b = self.lib.llama, text.encode('utf-8')
        n = len(b) + 16
        arr = (C.c_int32 * n)()
        k = L.llama_tokenize(self.vocab, b, len(b), arr, n, True, special)
        if k < 0:
            arr = (C.c_int32 * -k)(); k = L.llama_tokenize(self.vocab, b, len(b), arr, -k, True, special)
        return list(arr[:k])

    def piece(self, tok):
        buf = C.create_string_buffer(256)
        n = self.lib.llama.llama_token_to_piece(self.vocab, tok, buf, 256, 0, True)
        return buf.raw[:max(0, n)]

    def chat_prompt(self, question):
        """The question wrapped in the model's own chat template; thinking switched off where the template has it."""
        L = self.lib.llama
        t = L.llama_model_chat_template(self.model, None)
        tmpl = t.decode('utf-8', 'replace') if t else ''
        if '<|im_start|>' in tmpl:
            s = f'<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n'
            if '<think>' in tmpl:
                s += '<think>\n\n</think>\n\n'
            return s, 'chatml'
        if '<|turn>' in tmpl:                                   # Gemma 4: own turn markers, thinking off = empty thought channel
            s = f'<|turn>user\n{question}<turn|>\n<|turn>model\n'
            if '<|channel>thought' in tmpl:
                s += '<|channel>thought\n<channel|>'
            return s, 'gemma4'
        if '<start_of_turn>' in tmpl:
            return f'<start_of_turn>user\n{question}<end_of_turn>\n<start_of_turn>model\n', 'gemma'
        if tmpl:
            msg = (llama_chat_message * 1)(llama_chat_message(b'user', question.encode('utf-8')))
            buf = C.create_string_buffer(8192)
            n = L.llama_chat_apply_template(tmpl.encode('utf-8'), msg, 1, True, buf, 8192)
            if 0 < n < 8192:
                return buf.raw[:n].decode('utf-8', 'replace'), 'template'
        return question, 'roh'

    # -------------------------------------------------- the eval callback
    def _on_eval(self, tp, ask, ud):
        r = self.rec
        if r is None:
            return False
        try:
            if ask:
                i = r['i']; r['i'] += 1
                t = tp.contents
                if r['graph'] is not None:
                    r['ptr'][C.addressof(t)] = i
                    srcs = []
                    for p in t.src:
                        if not p:
                            continue
                        if p in r['ptr']:
                            srcs.append(r['ptr'][p])
                        else:
                            if p not in r['leaf']:
                                s = ggml_tensor.from_address(p)
                                r['leaf'][p] = len(r['leaves'])
                                r['leaves'].append({'name': s.name.decode('utf-8', 'replace'), 'ne': list(s.ne), 'type': self.lib.ggml_base.ggml_type_name(s.type).decode()})
                            srcs.append(-1 - r['leaf'][p])
                    r['graph'].append({'name': t.name.decode('utf-8', 'replace'), 'op': self.lib.ggml_base.ggml_op_desc(tp).decode(), 'ne': list(t.ne),
                                       'type': t.type, 'src': srcs, 'op_params': list(t.op_params[:8])})
                want = r['want'](i, t)
                if want:
                    r['pending'] = i
                return bool(want)
            i = r['pending']
            t = tp.contents
            n = self.lib.ggml_base.ggml_nbytes(tp)
            raw = np.empty(n, dtype=np.uint8)
            self.lib.ggml_base.ggml_backend_tensor_get(tp, raw.ctypes.data_as(C.c_void_p), 0, n)
            r['got'][i] = self._to_array(raw, t)
            return True
        except Exception as e:      # never let an exception cross into C
            r['error'] = repr(e)
            return True

    @staticmethod
    def _to_array(raw, t):
        dt = {GGML_TYPE_F32: np.float32, GGML_TYPE_F16: np.float16}.get(t.type)
        ne, nb = [int(x) for x in t.ne], [int(x) for x in t.nb]
        if dt is None:
            if t.type == GGML_TYPE_BF16:
                u = raw.view(np.uint16).astype(np.uint32) << 16
                raw, dt, nb = u.view(np.float32).view(np.uint8), np.float32, [x * 2 for x in nb]
            else:
                return None
        a = np.lib.stride_tricks.as_strided(raw.view(dt) if raw.size % np.dtype(dt).itemsize == 0 else raw[: raw.size // np.dtype(dt).itemsize * np.dtype(dt).itemsize].view(dt),
                                            shape=ne[::-1], strides=nb[::-1])
        return np.ascontiguousarray(a, dtype=np.float32)      # numpy order: [ne3, ne2, ne1, ne0]

    # -------------------------------------------------- decoding
    def _batch(self, toks, start_pos, all_logits):
        L = self.lib.llama
        n = len(toks)
        b = L.llama_batch_init(n, 0, 1)
        for k, tok in enumerate(toks):
            b.token[k] = tok; b.pos[k] = start_pos + k; b.n_seq_id[k] = 1; b.seq_id[k][0] = 0
            b.logits[k] = 1 if (all_logits or k == n - 1) else 0
        b.n_tokens = n
        return b

    def _decode(self, toks, start_pos, all_logits):
        L = self.lib.llama
        b = self._batch(toks, start_pos, all_logits)
        try:
            rc = L.llama_decode(self.ctx, b)
        finally:
            L.llama_batch_free(b)
        if rc != 0:
            raise RuntimeError(f'llama_decode meldet Fehler {rc}: ' + ' '.join(x.strip() for x in self.lib.logs[-4:]))

    def logits(self, i):
        p = self.lib.llama.llama_get_logits_ith(self.ctx, i)
        return np.ctypeslib.as_array(p, shape=(self.n_vocab,)).copy()

    def generate(self, prompt_toks, max_new, on_token=None, stop=None):
        """Greedy: always the most probable next token. Returns the new tokens (without the end token) and why it stopped."""
        L = self.lib.llama
        with self.lock:
            self.rec = None
            L.llama_memory_clear(self.mem, True)
            self._decode(prompt_toks, 0, False)
            out, pos, why = [], len(prompt_toks), 'Länge'
            for _ in range(max_new):
                lg = self.logits(-1)
                nxt = int(np.argmax(lg))
                if L.llama_vocab_is_eog(self.vocab, nxt):
                    why = 'Ende'; break
                out.append(nxt)
                if on_token:
                    on_token(len(out), nxt)
                if stop and stop():
                    why = 'abgebrochen'; break
                if len(out) >= max_new:
                    break
                self._decode([nxt], pos, False); pos += 1
            return out, why

    def record(self, toks, want, graph=False):
        """One pass over all tokens with logits for every position; want(i, tensor) picks the nodes to copy out."""
        L = self.lib.llama
        with self.lock:
            r = {'i': 0, 'want': want, 'got': {}, 'graph': [] if graph else None, 'ptr': {}, 'leaf': {}, 'leaves': [], 'pending': None, 'error': None}
            L.llama_memory_clear(self.mem, True)
            self.rec = r
            t0 = time.time()
            try:
                self._decode(toks, 0, True)
                L.llama_synchronize(self.ctx)
            finally:
                self.rec = None
            r['seconds'] = time.time() - t0
            if r['error']:
                raise RuntimeError('Aufzeichnung fehlgeschlagen: ' + r['error'])
            r['logits'] = np.stack([self.logits(k) for k in range(len(toks))])
            return r


# ---------------------------------------------------------------- a recorded run
class Run:
    """Everything the page asks about one question: tokens, graph, stored tensors, next-token probabilities."""

    def __init__(self, engine, toks, n_prompt, why, gen_s):
        self.engine, self.toks, self.n_prompt, self.why, self.gen_s = engine, toks, n_prompt, why, gen_s
        self.n = len(toks)
        e = engine
        n, d = self.n, e.n_embd

        def overview(i, t):
            nm = t.name.decode('utf-8', 'replace')
            if not nm or ' (' in nm or nm.rpartition('-')[0] == 'norm':
                return False                                  # views, and the unweighted norm (recomputed on request)
            ne = list(t.ne)
            if ne[0] == d and ne[1] == n and ne[2] == 1 and ne[3] == 1 and t.type == GGML_TYPE_F32:
                return True                                   # a vector of model width for every token
            return nm.startswith('kq_soft_max')               # attention weights
        r = e.record(toks, overview, graph=True)
        self.graph, self.leaves, self.rec_s = r['graph'], r['leaves'], r['seconds']
        self.store = {}
        for i, a in r['got'].items():
            if a is None:
                continue
            a = a.reshape(-1, a.shape[-2], a.shape[-1]) if a.ndim >= 2 else a
            if r['graph'][i]['name'].startswith('kq_soft_max'):
                a = a[:, :, :n]                               # keys beyond the text are padding of the KV cache
            self.store[i] = a.astype(np.float16)
        lg = r['logits'].astype(np.float32)
        m = lg.max(1, keepdims=True); p = np.exp(lg - m); p /= p.sum(1, keepdims=True)
        top = np.argsort(-p, axis=1)[:, :20]
        self.top = [[(int(j), float(p[k, j])) for j in top[k]] for k in range(n)]
        self.p_next = [float(p[k, toks[k + 1]]) if k + 1 < n else None for k in range(n)]
        self.rank_next = [int((p[k] > p[k, toks[k + 1]]).sum()) + 1 if k + 1 < n else None for k in range(n)]
        self.logits16 = lg.astype(np.float16)
        self._detail = {}                                     # node index -> array, recomputed on request
        self._index()

    def _index(self):
        """Layer of each node (from the '-N' suffix llama.cpp gives named tensors) and the residual-stream chain."""
        g, d, n = self.graph, self.engine.n_embd, self.n
        cur = -1
        for i, x in enumerate(g):
            nm = x['name']
            base, _, tail = nm.rpartition('-')
            il = int(tail) if base and tail.isdigit() and ' ' not in tail else None
            if il is not None:
                cur = il
            x['il'] = il
            x['block'] = cur
            x['base'] = base if il is not None else nm
        # vector of every token before block 0 and after each block
        self.layer_nodes = []
        first = next((i for i in sorted(self.store) if self.store[i].shape[-1] == d), None)
        self.layer_nodes.append(first)
        n_layer = self.engine.n_layer
        for il in range(n_layer):
            cand = [i for i, x in enumerate(self.graph) if x['il'] == il and x['base'] == 'l_out' and i in self.store]
            if not cand:      # fall back to the last stored model-width vector of that block
                cand = [i for i, x in enumerate(self.graph) if x['block'] == il and i in self.store and self.store[i].shape[-1] == d]
            self.layer_nodes.append(cand[-1] if cand else None)

    def vec(self, i, pos):
        a = self.store.get(i)
        if a is None:
            a = self.detail([i])[i]
        if a.ndim == 3 and a.shape[0] == 1:
            return a[0, pos]
        return a[..., pos, :] if a.ndim >= 2 else a

    def detail(self, idx):
        """Recompute the same pass and copy out the requested nodes."""
        need = [i for i in idx if i not in self.store and i not in self._detail]
        if need:
            s = set(need)
            r = self.engine.record(self.toks, lambda i, t: i in s)
            for i, a in r['got'].items():
                if a is not None:
                    self._detail[i] = a.reshape(-1, a.shape[-2], a.shape[-1]) if a.ndim >= 2 else a
            while sum(a.nbytes for a in self._detail.values()) > 1.5e9 and len(self._detail) > len(need):
                self._detail.pop(next(iter(self._detail)))
        return {i: (self.store[i].astype(np.float32) if i in self.store else self._detail.get(i)) for i in idx}
