"""Save a recorded run as an example that anyone can open later, without server and graphics card.

Works for any model: it takes whatever run is in memory and writes examples/<id>.js next to the page:
    lupeBeispiel("<base64 of gzip of: u32 header length, JSON header, f16 arrays>")
and lists it in examples/index.js (lupeBeispielListe([...])), which the page reads to offer the examples.
The JSONP names lupeBeispiel / lupeBeispielListe stay as they are: the page and older example files use them.
Every example has a language, 'de' or 'en' (lang): given by the page, else guessed from the question.

What goes in: tokens, probabilities and the real compute steps of every block; the way through all layers
(and the logit lens) for the token that starts the answer, the first answer tokens and the two longest words
of the question; attention weights and memory gates for every token; every step of three blocks for the
main token, each with a few worked-out numbers.
"""
import base64, gzip, json, os, re, struct, time

import numpy as np

import lupe_run as R

HERE = os.path.dirname(os.path.abspath(__file__))
# examples live next to the page (gguf-lupe.html), which sits in the folder above src/
ROOT = next((d for d in (HERE, os.path.dirname(HERE)) if os.path.isfile(os.path.join(d, 'gguf-lupe.html'))), os.path.dirname(HERE))
DIR = os.path.join(ROOT, 'examples')
SPECIAL = re.compile(r'^<\|.*\|>$|^<.*>$')
DE_WORDS = re.compile(r'\b(?:ist|der|die|das|und|was|wie|welche[mnrs]?|hat)\b')
EN_WORDS = re.compile(r'\b(?:the|a|is|are|what|who|how|which|where|when|why|and|of|to|do|does|did|can|you|it|this|that)\b')   # "was", "die", "hat" are English words too


def guess_lang(text):
    """'de' or 'en' when the page does not say: umlauts/ß, or more very common German words than English ones."""
    t = (text or '').lower()
    if re.search('[äöüß]', t):
        return 'de'
    return 'de' if len(DE_WORDS.findall(t)) > len(EN_WORDS.findall(t)) else 'en'


def slug(text):
    t = text.lower()
    for a, b in (('ä', 'ae'), ('ö', 'oe'), ('ü', 'ue'), ('ß', 'ss')):
        t = t.replace(a, b)
    t = re.sub(r'[^a-z0-9]+', '-', t).strip('-')
    return (t[:40].rstrip('-') or 'example')


def unpack(b):
    ml = struct.unpack('<I', b[:4])[0]; meta = json.loads(b[4:4 + ml]); o = 4 + ml; o += o % 2
    return meta, np.frombuffer(b[o:], dtype=np.float16)


def job_export(job, rid, title=None, lang=None, n_answer=8):
    P = lambda p, m: job.update(progress=p, message=m)
    info = R.call('run_info', rid)
    lang = lang if lang in ('de', 'en') else guess_lang(info['question'])
    n, n_prompt, n_layer = len(info['tokens']), info['n_prompt'], info['n_layer']
    main = n_prompt - 1
    pieces = info['pieces']
    words = [k for k in range(n_prompt - 1) if not SPECIAL.match(pieces[k]) and re.search(r'[^\W\d_]{3,}', pieces[k])]
    words = sorted(sorted(words, key=lambda k: -len(pieces[k].strip()))[:2])
    stack_pos = sorted(set(words + [main] + list(range(n_prompt, min(n, n_prompt + n_answer)))))
    first_attn = next((i for i, k in enumerate(info['layer_kinds']) if k == 'attn'), None)
    detail = sorted({0, n_layer - 1} | ({first_attn} if first_attn is not None else set()))
    model = os.path.basename(info['path'])
    hdr = {'v': 1, 'question': info['question'], 'title': title or info['question'], 'lang': lang, 'model': model, 'n_vocab': info['n_vocab'], 'recorded': time.strftime('%Y-%m-%d'),
           'device': info['device'], 'llama': 'b11388', 'mainPos': main, 'stackPos': stack_pos, 'detailBlocks': detail,
           'pieces': pieces, 'stacks': {}, 'lens': {}, 'blocks': {}, 'vecs': {}, 'explains': {}}
    arrays = []
    add = lambda key, v: arrays.append((key, np.ascontiguousarray(v, dtype=np.float16)))
    for k, p in enumerate(stack_pos):
        P(0.05 + 0.25 * k / len(stack_pos), f'Weg durch die Schichten: Token {k + 1} von {len(stack_pos)}')
        meta, v = unpack(R.call('run_stack', rid, p))
        add(f'stack:{p}', v); hdr['stacks'][str(p)] = meta
        hdr['lens'][str(p)] = R.run_lens(rid, p)
    P(0.3, 'Rechenschritte der Blöcke')
    for b in range(n_layer):
        hdr['blocks'][str(b)] = R.call('run_block', rid, b)

    def vec(node, p):
        key = f'{node}:{p}'
        if key in hdr['vecs']:
            return None
        meta, v = unpack(R.call('run_vec', rid, node, p))
        add('vec:' + key, v); hdr['vecs'][key] = meta
        return v
    nodes = list(info['attn_nodes'].values()) + [s['i'] for b in range(n_layer) for s in hdr['blocks'][str(b)]['steps'] if s['base'] in ('beta_sigmoid', 'gate')]
    for k, node in enumerate(nodes):
        P(0.32 + 0.3 * k / max(1, len(nodes)), 'Aufmerksamkeit und Speicher aller Tokens')
        for p in range(n):
            vec(node, p)
    steps = [s for b in detail for s in hdr['blocks'][str(b)]['steps'] if s['tok']]
    for k, s in enumerate(steps):
        P(0.62 + 0.36 * k / max(1, len(steps)), f'Schritte im Detail: {k + 1} von {len(steps)}')
        v = vec(s['i'], main)
        if v is None:
            continue
        vf = v.astype(np.float32)
        js = {int(np.argmax(np.abs(vf))), 7 % vf.size, vf.size // 2}
        if s['base'] == 'result_output':
            js = {int(info['tokens'][main + 1]) if main + 1 < n else int(info['top'][main][0][0]), int(info['top'][main][1][0])}
        for j in sorted(js):
            hdr['explains'][f"{s['i']}:{main}:{j}"] = R.call('run_explain', rid, s['i'], main, j)
    ids = set(info['tokens']) | {t[0] for row in info['top'] for t in row}
    for L in hdr['lens'].values():
        ids |= {t[0] for row in L['rows'] for t in row['top']}
    ids |= {x['token'] for x in hdr['explains'].values() if x.get('kind') == 'lookup' and x.get('token') is not None}
    hdr['tok'] = R.call('token_texts', rid, sorted(int(i) for i in ids))
    info = {k: v for k, v in info.items() if k not in ('path', 'pieces')}
    ident = slug(hdr['title'])
    os.makedirs(DIR, exist_ok=True)
    base, k = ident, 2
    while os.path.exists(os.path.join(DIR, ident + '.js')):
        ident = f'{base}-{k}'; k += 1
    info['id'] = 'beispiel:' + ident
    hdr['id'], hdr['info'] = ident, info
    off, blob = 0, []
    hdr['arrays'] = {}
    for key, v in arrays:
        hdr['arrays'][key] = [off, int(v.size)]; blob.append(v.tobytes()); off += v.size
    hj = json.dumps(hdr, ensure_ascii=False, separators=(',', ':')).encode('utf-8')
    raw = struct.pack('<I', len(hj)) + hj + b' ' * ((-(4 + len(hj))) % 2) + b''.join(blob)
    gz = gzip.compress(raw, 9)
    fn = os.path.join(DIR, ident + '.js')
    with open(fn, 'w', encoding='ascii') as f:
        f.write('// GGUF-Lupe: aufgezeichneter Durchlauf (' + hdr['model'].encode('ascii', 'replace').decode() + ')\n')
        f.write('lupeBeispiel("' + base64.b64encode(gz).decode() + '");\n')
    entry = {'id': ident, 'title': hdr['title'], 'question': hdr['question'], 'lang': lang, 'model': hdr['model'], 'recorded': hdr['recorded'], 'device': hdr['device'],
             'tokens': n, 'n_prompt': n_prompt, 'mb': round(os.path.getsize(fn) / 1e6, 1), 'file': f'examples/{ident}.js'}
    write_index(entry)
    job['result'] = entry


def read_index():
    fn = os.path.join(DIR, 'index.js')
    if not os.path.exists(fn):
        return []
    s = open(fn, encoding='utf-8').read()
    m = re.search(r'lupeBeispielListe\((.*)\);\s*$', s, re.S)
    try:
        return json.loads(m.group(1)) if m else []
    except Exception:
        return []


def write_index(entry=None, drop=None):
    items = [x for x in read_index() if x.get('id') not in ({entry['id']} if entry else set()) | ({drop} if drop else set())]
    if entry:
        items.append(entry)
    items = [x for x in items if os.path.exists(os.path.join(ROOT, x['file']))]
    with open(os.path.join(DIR, 'index.js'), 'w', encoding='utf-8') as f:
        f.write('// GGUF-Lupe: list of recorded examples (rewritten whenever an example is saved)\n')
        f.write('lupeBeispielListe(' + json.dumps(items, ensure_ascii=False, indent=1) + ');\n')
    return items
