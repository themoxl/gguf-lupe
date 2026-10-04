// Adds precomputed nearest neighbours for more example words to an existing demo.json (same algorithm as build-demo.cjs),
// so the built-in example also works with the English word list.   node add-demo-nn.cjs <model.gguf> [demo.json]
const fs = require('fs'); const G = require('./gguf-core.js');
const f = process.argv[2], out = process.argv[3] || 'demo.json';
const demo = JSON.parse(fs.readFileSync(out, 'utf8'));
const fd = fs.openSync(f, 'r'), size = fs.fstatSync(fd).size;
if (size !== demo.fileSize) throw new Error('demo.json was built from a different file');
const hb = Buffer.alloc(24 << 20); fs.readSync(fd, hb, 0, hb.length, 0);
const h = G.parse(hb.buffer.slice(hb.byteOffset, hb.byteOffset + hb.length), size);
const read = (pos, n) => { const b = Buffer.alloc(n); fs.readSync(fd, b, 0, n, pos); return b; };
const tok = new G.Tokenizer(h.meta); tok.init();
const enc = s => Array.from(Buffer.from(s, 'utf8'), b => G.B2U[b]).join('');
const lookup = w => tok.t2id.get(enc(' ' + w)) ?? tok.t2id.get(enc(w)) ?? -1;
const items = h.meta['tokenizer.ggml.tokens'].items;
const dec = raw => { try { return new TextDecoder('utf-8', { fatal: true }).decode(Uint8Array.from(Array.from(raw, c => G.U2B.get(c)))); } catch { return raw; } };
const EN_WORDS = ['King', 'Paris', 'money', 'audit', 'dog', 'Germany', 'auditor', 'financial'];
const EN_TEXT = 'The auditor checks the annual financial statements.';
const want = [...new Set([...EN_WORDS.map(lookup), ...tok.tokenize(EN_TEXT).ids.filter(id => /\p{L}{2}/u.test(dec(items[id])))])].filter(id => id >= 0 && !demo.nn[id]);
console.log('new words:', want.map(id => JSON.stringify(dec(items[id]))).join(' ') || '(none)');
// embedding rows for the English sample texts of the Tokens tab (rows only, no neighbours)
const EN_SAMPLES = ['Revenue 2025: $1,234,567.89 (prior year: $987,654.32)', 'Exam passed ✅🎉', 'def total(a, b):\n    return a + b', '<|im_start|>user\nWhat is a token?<|im_end|>\n<|im_start|>assistant\n'];
const rowIds = new Set(want); for (const t of EN_SAMPLES) tok.tokenize(t).ids.forEach(i => rowIds.add(i));
const E = h.tensors.find(t => t.name === 'token_embd.weight'), d = E.dims[0], V = E.dims[1], qk = G.TYPES[E.type][1];
const rowVec = id => G.dequant(E.type, new Uint8Array(read(E.absOffset + id * E.rowBytes, E.rowBytes)), d / qk);
const unit = v => { let n = 0; for (const x of v) n += x * x; n = Math.sqrt(n) || 1; return Float32Array.from(v, x => x / n); };
const qs = want.map(id => unit(rowVec(id))), K = 40, tops = qs.map(() => []);
const CH = 1024, buf = Buffer.alloc(CH * E.rowBytes), outv = new Float32Array(CH * d), t0 = Date.now();
for (let s = 0; s < V; s += CH) {
  const n = Math.min(CH, V - s); fs.readSync(fd, buf, 0, n * E.rowBytes, E.absOffset + s * E.rowBytes);
  const all = G.dequant(E.type, new Uint8Array(buf.buffer, buf.byteOffset, n * E.rowBytes), n * d / qk, outv);
  for (let r = 0; r < n; r++) {
    const o = r * d; let nn = 0; for (let i = 0; i < d; i++) nn += all[o + i] * all[o + i]; nn = Math.sqrt(nn) || 1;
    for (let q = 0; q < qs.length; q++) {
      if (s + r === want[q]) continue;
      const qv = qs[q]; let dot = 0; for (let i = 0; i < d; i++) dot += qv[i] * all[o + i];
      const c = dot / nn, Tq = tops[q];
      if (Tq.length < K || c > Tq[Tq.length - 1][0]) { Tq.push([c, s + r]); Tq.sort((a, b) => b[0] - a[0]); if (Tq.length > K) Tq.pop(); }
    }
  }
}
function pca2(vs) { // identical to build-demo.cjs / the page
  const n = vs.length, dd = vs[0].length;
  const X = vs.map(v => { let s = 0; for (let i = 0; i < dd; i++) s += v[i] * v[i]; s = Math.sqrt(s) || 1; const o = new Float64Array(dd); for (let i = 0; i < dd; i++) o[i] = v[i] / s; return o; });
  const mean = new Float64Array(dd); for (const x of X) for (let i = 0; i < dd; i++) mean[i] += x[i] / n;
  for (const x of X) for (let i = 0; i < dd; i++) x[i] -= mean[i];
  const Gm = X.map(a => X.map(b => { let s = 0; for (let k = 0; k < dd; k++) s += a[k] * b[k]; return s; }));
  const eig = Mx => { let v = Array.from({ length: n }, (_, i) => Math.sin(i + 1) + 0.5), lam = 0; for (let it = 0; it < 300; it++) { const w = v.map((_, i) => Mx[i].reduce((s, x, j) => s + x * v[j], 0)); lam = Math.hypot(...w) || 1; v = w.map(x => x / lam); } return [lam, v]; };
  const [l1, v1] = eig(Gm), G2 = Gm.map((row, i) => row.map((x, j) => x - l1 * v1[i] * v1[j])), [l2, v2] = eig(G2);
  return v1.map((_, i) => [v1[i] * Math.sqrt(l1), v2[i] * Math.sqrt(Math.max(0, l2))]);
}
want.forEach((id, q) => {
  const top = tops[q].slice(0, 24).map(([c, j]) => [j, +c.toFixed(4)]);
  const xy = pca2([rowVec(id), ...top.map(([j]) => rowVec(j))]).map(([x, y]) => [+x.toFixed(4), +y.toFixed(4)]);
  demo.nn[id] = { top, xy };
  console.log(' ', dec(items[id]).trim(), '->', top.slice(0, 6).map(([j, c]) => dec(items[j]).trim() + ':' + c.toFixed(2)).join(' '));
});
// make sure the query words themselves have their embedding row in the demo (needed for the table and the compare box)
const er = demo.rows['token_embd.weight'], have = new Set(); for (const [s0, n] of er) for (let i = 0; i < n; i++) have.add(s0 + i);
let added = 0; for (const id of rowIds) if (!have.has(id)) { er.push([id, 1, read(E.absOffset + id * E.rowBytes, E.rowBytes).toString('base64')]); added++; }
console.log('embedding rows added:', added);
er.sort((a, b) => a[0] - b[0]);
const json = JSON.stringify(demo).replace(/</g, '\\u003c');
fs.writeFileSync(out, json);
console.log(want.length, 'words added in', ((Date.now() - t0) / 1000).toFixed(1), 's ->', out, (json.length / 1e6).toFixed(2), 'MB');
