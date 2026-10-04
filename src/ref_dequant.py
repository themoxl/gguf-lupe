import numpy as np, json
from gguf import quants
from gguf.constants import GGMLQuantizationType as T, GGML_QUANT_SIZES as S
rng = np.random.default_rng(42)
out = {}
for t in T:
    if t in (T.I8, T.I16, T.I32, T.I64, T.F64): continue
    blk, ts = S[t]
    nb = 512
    raw = rng.integers(0, 256, size=(nb, ts), dtype=np.uint8)
    # keep fp16/bf16/fp32 scale fields finite: clear the exponent-all-ones pattern by masking top bits of every 2nd byte is type-specific,
    # so instead just compare NaN/Inf-aware on the JS side.
    try:
        y = quants.dequantize(raw.reshape(-1), t).astype(np.float32).reshape(-1)
    except Exception as e:
        out[t.name] = 'ERR ' + type(e).__name__ + ': ' + str(e)[:80]; continue
    raw.tofile(f'dq/{t.name}.bin'); y.tofile(f'dq/{t.name}.f32')
    out[t.name] = [int(t.value), blk, ts, nb, int(y.size)]
json.dump(out, open('dq/index.json', 'w'))
print(out)
