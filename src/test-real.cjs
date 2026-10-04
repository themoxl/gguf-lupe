const fs = require('fs'); const G = require('./gguf-core.js');
const f = process.argv[2]; const fd = fs.openSync(f, 'r'); const size = fs.fstatSync(fd).size;
const hb = Buffer.alloc(16 << 20); fs.readSync(fd, hb, 0, hb.length, 0);
const h = G.parse(hb.buffer.slice(hb.byteOffset, hb.byteOffset + hb.length), size);
const ref = JSON.parse(fs.readFileSync('real_ref.json'));
for (const [name, r] of Object.entries(ref)) {
  const t = h.tensors.find(x => x.name === name);
  const rows = Math.min(3, t.dims.slice(1).reduce((a, b) => a * b, 1));
  const tr = G.TYPES[t.type]; const nBytes = t.rowBytes * rows;
  const b = Buffer.alloc(nBytes); fs.readSync(fd, b, 0, nBytes, t.absOffset);
  const y = G.dequant(t.type, new Uint8Array(b.buffer, b.byteOffset, nBytes), nBytes / tr[2]);
  let s = 0; for (const v of y) s += v;
  const ok = t.absOffset === r.offset && Math.abs(s - r.sum) <= 1e-4 * Math.max(1, Math.abs(r.sum)) && y.length === r.n;
  console.log((ok ? 'OK  ' : 'FAIL'), name.padEnd(24), t.typeName.padEnd(5), 'off', t.absOffset, (t.absOffset === r.offset ? '=' : '!= ' + r.offset), 'dims', JSON.stringify(t.dims), 'sum js', s.toFixed(6), 'py', r.sum.toFixed(6), 'first', Array.from(y.slice(0, 3)).map(v => v.toFixed(5)).join(','));
}
