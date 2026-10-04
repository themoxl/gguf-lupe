import re, json
src = open('llamacpp/llama-vocab.cpp', encoding='utf-8').read()

# 1) regex table inside the llm_tokenizer_bpe constructor
start = src.index('struct llm_tokenizer_bpe : llm_tokenizer {')
end = src.index('std::vector<std::string> regex_exprs;', start)
body = src[start:end]


def cstr(s):
    """Unescape a C++ string literal body (\\\\ -> \\, \\" -> ", \\n -> newline, \\xNN bytes)."""
    out = bytearray()
    i = 0
    raw = s.encode('utf-8')
    while i < len(raw):
        c = raw[i:i + 1]
        if c == b'\\':
            n = raw[i + 1:i + 2]
            if n == b'x':
                j = i + 2
                h = b''
                while j < len(raw) and chr(raw[j]) in '0123456789abcdefABCDEF':
                    h += raw[j:j + 1]; j += 1
                out.append(int(h, 16)); i = j; continue
            out += {b'n': b'\n', b'r': b'\r', b't': b'\t', b'\\': b'\\', b'"': b'"', b"'": b"'"}.get(n, b'\\' + n)
            i += 2; continue
        out += c; i += 1
    return out.decode('utf-8')


table = {}
case_re = re.compile(r'((?:\s*(?:case LLAMA_VOCAB_PRE_TYPE_\w+|default)\s*:\s*)+)(.*?)\bbreak;', re.S)
lit_re = re.compile(r'^\s*"((?:[^"\\]|\\.)*)"\s*,?\s*$', re.M)
for m in case_re.finditer(body):
    labels = re.findall(r'case LLAMA_VOCAB_PRE_TYPE_(\w+)', m.group(1))
    if 'default' in m.group(1):
        labels.append('DEFAULT')
    block = m.group(2)
    rx = re.search(r'regex_exprs\s*=\s*\{(.*?)\};', block, re.S)
    if not rx:
        continue
    exprs = [cstr(l) for l in lit_re.findall(rx.group(1))]
    be = 'byte_encode = false' not in block
    for L in labels:
        table[L] = {'re': exprs, 'byte': be}

# 2) tokenizer_pre name -> pre type
names = {}
name_re = re.compile(r'if\s*\(\s*((?:tokenizer_pre\s*==\s*"[^"]+"\s*(?:\|\|)?\s*)+)\)\s*\{\s*pre_type\s*=\s*LLAMA_VOCAB_PRE_TYPE_(\w+);')
for m in name_re.finditer(src):
    for n in re.findall(r'"([^"]+)"', m.group(1)):
        names[n] = m.group(2)

json.dump({'types': table, 'names': names}, open('pre_table.json', 'w', encoding='utf-8'), ensure_ascii=False)
print(len(table), 'regex types;', len(names), 'pre names')
print({k: names.get(k) for k in ['qwen35', 'qwen2', 'llama3', 'llama-bpe', 'gpt-4o', 'gemma4', 'default', 'gpt2']})
print('QWEN35', table.get('QWEN35'))
print('GEMMA4', table.get('GEMMA4'))
print('DEFAULT', table.get('DEFAULT'))
print('types without regex entry:', sorted(set(names.values()) - set(table)))
