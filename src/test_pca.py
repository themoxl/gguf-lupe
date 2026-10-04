import os
import sys, time, numpy as np
sys.path.insert(0, '.')
import lupe_server as L
eng = sys.argv[1] if len(sys.argv) > 1 else 'torch'
if eng == 'numpy': L.HAVE_TORCH = False
dev = 'cuda:0' if eng == 'torch' else 'cpu'
st = L.EmbStore(os.path.expanduser('~').replace('\\', '/') + '/.lmstudio/models/lmstudio-community/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q4_K_M.gguf', dev, lambda p, m: None)
print('table loaded', round(st.load_s, 1), 's on', dev, 'engine', 'torch' if st.torch else 'numpy')
r = st.pca(8)
print('pca', r['seconds'], 's')
print('variance share per direction (%):', [round(x * 100, 2) for x in r['ratio']], ' sum of 3:', round(sum(r['ratio'][:3]) * 100, 2), ' sum of 8:', round(sum(r['ratio']) * 100, 2))
# a sanity check: the share of the best 3 directions must exceed the share of 3 random raw columns
X = st.X if not st.torch else st.X.cpu().numpy()
m = X.mean(0); cols = np.random.default_rng(1).choice(st.d, 3, replace=False)
raw3 = float(((X[:, cols] - m[cols]) ** 2).sum()) / r['total']
print('3 random raw columns carry (%):', round(raw3 * 100, 3), '| average column share:', round(100 / st.d, 3))
from gguf import GGUFReader
rd = GGUFReader(st.path); f = rd.fields['tokenizer.ggml.tokens']
toks = [bytes(f.parts[i]).decode('utf-8', 'replace') for i in f.data]
for c in range(3):
    print(f'PC{c + 1}: +', [toks[i] for i in r['pos'][c][:10]])
    print(f'      -', [toks[i] for i in r['neg'][c][:10]])
    print('      dims', [(d, round(w, 3)) for d, w in r['dims'][c][:6]])
