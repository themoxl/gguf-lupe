import sys, numpy as np, json
from gguf import GGUFReader, quants
r = GGUFReader(sys.argv[1])
want = ['blk.0.ffn_up.weight', 'output.weight', 'blk.40.attn_qkv.weight', 'blk.63.ffn_down.weight', 'blk.3.attn_q.weight', 'output_norm.weight', 'blk.0.ssm_a', 'token_embd.weight']
res = {}
for t in r.tensors:
    if t.name in want:
        rows = 3
        ne0 = int(t.shape[0])
        data = np.asarray(t.data)
        # t.data is (rows, bytes_per_row) for quantized or (rows, ne0) for float types
        flat = data.reshape(-1).view(np.uint8) if t.tensor_type.name in ('F32','F16','BF16') else data.reshape(-1)
        bpr = flat.size // int(np.prod(t.shape[1:])) if len(t.shape) > 1 else flat.size
        part = flat[: bpr * min(rows, int(np.prod(t.shape[1:])) if len(t.shape)>1 else 1)]
        y = quants.dequantize(np.ascontiguousarray(part), t.tensor_type).astype(np.float32).reshape(-1)
        res[t.name] = {'type': t.tensor_type.name, 'offset': int(t.data_offset), 'shape': [int(s) for s in t.shape], 'first': y[:5].tolist(), 'sum': float(np.float64(y).sum()), 'n': int(y.size)}
json.dump(res, open('real_ref.json', 'w'), indent=1); print(json.dumps(res)[:1500])
