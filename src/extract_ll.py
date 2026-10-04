"""Pack the llama.cpp sources the Lupe needs for its "Rechenweg" tab.

Input:  llsrc/  (llama.cpp tag b11388: src/models/*.cpp, src/llama-arch.cpp, src/llama-model.cpp, src/llama-graph.cpp)
Output: ll.b64  (gzip + base64 of one JSON object)

  archs    "qwen35" -> "LLM_ARCH_QWEN35"            (llama-arch.cpp, LLM_ARCH_NAMES)
  factory  "LLM_ARCH_QWEN35" -> "llama_model_qwen35" (llama-model.cpp, switch in llama_model_create)
  parents  "llama_model_clef" -> "llama_model_qwen35" (models/models.h)
  tensors  "LLM_TENSOR_ATTN_NORM" -> "blk.%d.attn_norm"
  kv       "LLM_KV_FULL_ATTENTION_INTERVAL" -> "%s.full_attention_interval"
  files    file name -> source text (all model files + llama-graph.cpp)
"""
import base64, gzip, json, os, re

D = 'llsrc'
rd = lambda p: open(os.path.join(D, p), encoding='utf-8').read()
arch_cpp, model_cpp, models_h = rd('llama-arch.cpp'), rd('llama-model.cpp'), rd('models/models.h')

archs = {name: enum for enum, name in re.findall(r'\{\s*(LLM_ARCH_\w+)\s*,\s*"([^"]+)"\s*\}', arch_cpp)}
tensors = dict(re.findall(r'\{\s*(LLM_TENSOR_\w+)\s*,\s*"([^"]+)"\s*\}', arch_cpp))
kv = dict(re.findall(r'\{\s*(LLM_KV_\w+)\s*,\s*"([^"]+)"\s*\}', arch_cpp))

factory, pending = {}, []
for line in model_cpp.splitlines():
    m = re.match(r'\s*case\s+(LLM_ARCH_\w+)\s*:', line)
    if m:
        pending.append(m.group(1)); continue
    m = re.match(r'\s*return\s+new\s+(llama_model_\w+)\s*\(', line)
    if m and pending:
        for e in pending: factory[e] = m.group(1)
        pending = []
    elif line.strip() and not line.strip().startswith('//'):
        pending = []

parents = dict(re.findall(r'struct\s+(llama_model_\w+)\s*:\s*public\s+(llama_model_\w+)', models_h))
alias, cur = {}, None
for line in models_h.splitlines():
    m = re.match(r'struct\s+(llama_model_\w+)\b', line)
    if m: cur = m.group(1)
    m = re.search(r'using\s+graph\s*=\s*(llama_model_\w+)::graph\s*;', line)
    if m and cur: alias[cur] = m.group(1)

files = {}
for f in sorted(os.listdir(os.path.join(D, 'models'))):
    if f.endswith('.cpp'):
        files[f] = rd('models/' + f)
files['llama-graph.cpp'] = rd('llama-graph.cpp')

out = {'version': 'b11388', 'repo': 'https://github.com/ggml-org/llama.cpp/blob/b11388/src/', 'archs': archs, 'factory': factory,
       'parents': parents, 'alias': alias, 'license': rd('LICENSE'), 'tensors': tensors, 'kv': kv, 'files': files}
raw = json.dumps(out, ensure_ascii=False, separators=(',', ':')).encode('utf-8')
gz = gzip.compress(raw, 9)
open('ll.b64', 'w').write(base64.b64encode(gz).decode())
print(f'archs {len(archs)}, factory {len(factory)}, parents {len(parents)}, alias {len(alias)}, tensors {len(tensors)}, kv {len(kv)}, files {len(files)}')
print(f'raw {len(raw) / 1e6:.2f} MB, gzip {len(gz) / 1e6:.2f} MB, base64 {len(gz) * 4 / 3 / 1e6:.2f} MB')
miss = [a for a, e in archs.items() if e not in factory]
print('archs without factory entry:', miss[:20], len(miss))
for a in ['qwen35', 'gemma4', 'llama', 'qwen3', 'gemma3']:
    print(a, archs.get(a), factory.get(archs.get(a)), parents.get(factory.get(archs.get(a), ''), '-'))
