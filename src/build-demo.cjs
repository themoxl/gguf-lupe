// Builds demo.json for the built-in example: gzipped raw GGUF header, selected weight rows,
// and precomputed nearest neighbours (same algorithm as the page).  node build-demo.cjs <model.gguf>
const fs = require('fs'), zlib = require('zlib'); const G = require('./gguf-core.js'); const MC = require('./map-core.js');
const f = process.argv[2]; const fd = fs.openSync(f, 'r'); const size = fs.fstatSync(fd).size;
const hb = Buffer.alloc(24 << 20); fs.readSync(fd, hb, 0, hb.length, 0);
const h = G.parse(hb.buffer.slice(hb.byteOffset, hb.byteOffset + hb.length), size);
const read = (pos, n) => { const b = Buffer.alloc(n); fs.readSync(fd, b, 0, n, pos); return b; };
const T = name => h.tensors.find(t => t.name === name);
const tok = new G.Tokenizer(h.meta); tok.init();
const enc = s => Array.from(Buffer.from(s, 'utf8'), b => G.B2U[b]).join('');
const lookup = w => tok.t2id.get(enc(' ' + w)) ?? tok.t2id.get(enc(w)) ?? -1;
const dec = raw => { try { return new TextDecoder('utf-8', { fatal: true }).decode(Uint8Array.from(Array.from(raw, c => G.U2B.get(c)))); } catch { return raw; } };
const texts = ['Die Wirtschaftsprüfung prüft den Jahresabschluss.', 'The auditor checks the annual financial statements.', 'Umsatz 2025: 1.234.567,89 € (Vorjahr: 987.654,32 €)',
  'Donaudampfschifffahrtsgesellschaftskapitänsmütze', 'Prüfung bestanden ✅🎉', 'def summe(a, b):\n    return a + b', '<|im_start|>user\nWas ist ein Token?<|im_end|>\n<|im_start|>assistant\n'];
const NN_WORDS = ['König', 'Paris', 'Geld', 'audit', 'Hund', 'Bilanz', 'Wirtschaft', 'Prüfung'];
const ids = new Set();
for (const t of texts) tok.tokenize(t).ids.forEach(i => ids.add(i));
for (const w of [...NN_WORDS, 'King', 'Queen', 'Berlin', 'Germany', 'Deutschland', 'dog', 'cat', 'money']) { const id = lookup(w); if (id >= 0) ids.add(id); }
for (let i = 0; i < 64; i++) ids.add(i);
const rowsets = {};
function addRows(name, list) {
  const t = T(name); const sorted = [...new Set(list)].sort((a, b) => a - b), runs = [];
  for (const r of sorted) { const L = runs[runs.length - 1]; if (L && r === L[0] + L[1]) L[1]++; else runs.push([r, 1]); }
  rowsets[name] = runs.map(([s, n]) => [s, n, read(t.absOffset + s * t.rowBytes, n * t.rowBytes).toString('base64')]);
}
addRows('token_embd.weight', [...ids]);
const range = n => Array.from({ length: n }, (_, i) => i);
addRows('blk.3.attn_q.weight', range(48));
addRows('blk.0.attn_qkv.weight', range(24));
addRows('blk.0.ffn_down.weight', range(8));
addRows('output_norm.weight', [0]);
addRows('blk.0.attn_norm.weight', [0]);
addRows('blk.3.attn_q_norm.weight', range(1));
addRows('blk.0.ssm_a', [0]);
addRows('blk.0.ssm_conv1d.weight', range(64));

// --- nearest neighbours for example words + words of the default sentence
const E = T('token_embd.weight'), d = E.dims[0], V = E.dims[1], qk = G.TYPES[E.type][1];
const rowVec = id => G.dequant(E.type, new Uint8Array(read(E.absOffset + id * E.rowBytes, E.rowBytes)), d / qk);
const qIds = [...new Set([...NN_WORDS.map(lookup), ...tok.tokenize(texts[0]).ids.filter(id => /\p{L}{2}/u.test(dec(h.meta['tokenizer.ggml.tokens'].items[id])))])].filter(id => id >= 0);
const unit = v => { let n = 0; for (const x of v) n += x * x; n = Math.sqrt(n) || 1; return Float32Array.from(v, x => x / n); };
const qs = qIds.map(id => unit(rowVec(id)));
const CALC = [['King', 'man', 'woman'], ['brother', 'man', 'woman'], ['Vater', 'Mann', 'Frau'], ['walking', 'walk', 'swim'], ['Paris', 'France', 'Germany'], ['König', 'Mann', 'Frau']];
const calcKeys = [];
for (const [a, b, c] of CALC) { const ids3 = [a, b, c].map(lookup); if (ids3.some(x => x < 0)) continue; const [A, B, C] = ids3.map(i => unit(rowVec(i))); qs.push(unit(A.map((x, i) => x - B[i] + C[i]))); calcKeys.push(ids3); qIds.push(-1 - calcKeys.length); }
const K = 40, tops = qs.map(() => []);
const CH = 1024, buf = Buffer.alloc(CH * E.rowBytes), out = new Float32Array(CH * d), t0 = Date.now();
for (let s = 0; s < V; s += CH) {
  const n = Math.min(CH, V - s); fs.readSync(fd, buf, 0, n * E.rowBytes, E.absOffset + s * E.rowBytes);
  const all = G.dequant(E.type, new Uint8Array(buf.buffer, buf.byteOffset, n * E.rowBytes), n * d / qk, out);
  for (let r = 0; r < n; r++) {
    const o = r * d; let nn = 0; for (let i = 0; i < d; i++) nn += all[o + i] * all[o + i]; nn = Math.sqrt(nn) || 1;
    for (let q = 0; q < qs.length; q++) {
      if (s + r === qIds[q]) continue;
      const qv = qs[q]; let dot = 0; for (let i = 0; i < d; i++) dot += qv[i] * all[o + i];
      const c = dot / nn, Tq = tops[q];
      if (Tq.length < K || c > Tq[Tq.length - 1][0]) { Tq.push([c, s + r]); Tq.sort((a, b) => b[0] - a[0]); if (Tq.length > K) Tq.pop(); }
    }
  }
}
function pca2(vs) { // identical to the page
  const n = vs.length, dd = vs[0].length;
  const X = vs.map(v => { let s = 0; for (let i = 0; i < dd; i++) s += v[i] * v[i]; s = Math.sqrt(s) || 1; const o = new Float64Array(dd); for (let i = 0; i < dd; i++) o[i] = v[i] / s; return o; });
  const mean = new Float64Array(dd); for (const x of X) for (let i = 0; i < dd; i++) mean[i] += x[i] / n;
  for (const x of X) for (let i = 0; i < dd; i++) x[i] -= mean[i];
  const Gm = X.map(a => X.map(b => { let s = 0; for (let k = 0; k < dd; k++) s += a[k] * b[k]; return s; }));
  const eig = Mx => { let v = Array.from({ length: n }, (_, i) => Math.sin(i + 1) + 0.5), lam = 0; for (let it = 0; it < 300; it++) { const w = v.map((_, i) => Mx[i].reduce((s, x, j) => s + x * v[j], 0)); lam = Math.hypot(...w) || 1; v = w.map(x => x / lam); } return [lam, v]; };
  const [l1, v1] = eig(Gm), G2 = Gm.map((row, i) => row.map((x, j) => x - l1 * v1[i] * v1[j])), [l2, v2] = eig(G2);
  return v1.map((_, i) => [v1[i] * Math.sqrt(l1), v2[i] * Math.sqrt(Math.max(0, l2))]);
}
const nn = {}, calc = {};
calcKeys.forEach((ids3, k) => { const q = qIds.length - calcKeys.length + k; calc[ids3.join(',')] = tops[q].map(([c, j]) => [j, +c.toFixed(4)]); });
qIds.forEach((id, q) => {
  if (id < 0) return;
  const top = tops[q].slice(0, 24).map(([c, j]) => [j, +c.toFixed(4)]);
  const xy = pca2([rowVec(id), ...top.map(([j]) => rowVec(j))]).map(([x, y]) => [+x.toFixed(4), +y.toFixed(4)]);
  nn[id] = { top, xy };
});
console.log('neighbours for', qIds.length, 'words in', ((Date.now() - t0) / 1000).toFixed(1), 's:', qIds.map(id => dec(h.meta['tokenizer.ggml.tokens'].items[id])).join(' | '));
console.log('  König ->', nn[lookup('König')].top.slice(0, 8).map(([j, c]) => dec(h.meta['tokenizer.ggml.tokens'].items[j]) + ':' + c.toFixed(2)).join(' '));

for (const [key, top] of Object.entries(calc)) { key.split(',').forEach(i => ids.add(+i)); top.slice(0, 15).forEach(([j]) => ids.add(j)); }
addRows('token_embd.weight', [...ids]);
// --- word map (same selection + algorithm as the page)
{
  const t1 = Date.now(), sel = MC.select({ n: 2600, size: V, text: i => dec(h.meta['tokenizer.ggml.tokens'].items[i]), isNormal: i => h.meta['tokenizer.ggml.token_type'].items[i] === 1, lookup });
  const vecs = sel.ids.map(rowVec), { U, n } = MC.prep(vecs), Qm = MC.toInt8(U, n, d), KM = 10;
  const { nb, sim } = MC.knnRows(Qm, n, d, KM, 0, n);
  MC.layout(nb, sim, n, KM, MC.pca2(U, n, d), {}, null).then(Y => {
    const lab = MC.regions(Y, n, 8);
    demoMap = { ids: sel.ids, Y: Array.from(Y, x => +x.toFixed(3)), lab: Array.from(lab), nb: Array.from(nb), K: KM, curated: [...sel.curated] };
    console.log('map', n, 'words in', ((Date.now() - t1) / 1000).toFixed(1), 's');
    finish();
  });
}
let demoMap = null;
function finish() {
const headerGz = zlib.gzipSync(read(0, h.headerEnd), { level: 9 }).toString('base64');
let fullmap = null;
if (process.argv[3]) { // full map from the Lupe-Server pipeline: f16 coords + top-4 exact neighbours (24-bit ids)
  const b = fs.readFileSync(process.argv[3]), hl = b.readUInt32LE(8), hdr = JSON.parse(b.subarray(12, 12 + hl).toString('utf8'));
  let o = 12 + hl; o += (4 - o % 4) % 4;
  const n = hdr.n, k = hdr.k, KK = 4, coords = b.subarray(o, o + n * 4), nbAll = new Int32Array(b.buffer.slice(b.byteOffset + o + n * 4, b.byteOffset + o + n * 4 + n * k * 4));
  const packed = Buffer.alloc(n * KK * 3);
  for (let i = 0; i < n; i++) for (let j = 0; j < KK; j++) { const v = nbAll[i * k + j], q = (i * KK + j) * 3; packed[q] = v & 255; packed[q + 1] = (v >> 8) & 255; packed[q + 2] = (v >> 16) & 255; }
  fullmap = { n, k: KK, hdr: { ...hdr, k: KK }, coords: Buffer.from(coords).toString('base64'), nb: packed.toString('base64') };
  console.log('full map embedded:', n, 'tokens,', (fullmap.coords.length + fullmap.nb.length) / 1e6, 'MB base64');
}
const demo = { fileName: f.split('/').pop(), fileSize: size, headerGz, rows: rowsets, nn, calc, map: demoMap, fullmap };
const json = JSON.stringify(demo).replace(/</g, '\\u003c');
fs.writeFileSync('demo.json', json);
console.log('demo.json', (json.length / 1e6).toFixed(2), 'MB; embedding rows', ids.size);
}
