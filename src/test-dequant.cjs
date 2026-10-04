const fs = require('fs'); const G = require('./gguf-core.js');
const idx = JSON.parse(fs.readFileSync('dq/index.json'));
let allOk = true;
for (const [name, v] of Object.entries(idx)) {
  if (typeof v === 'string') { console.log(name.padEnd(8), 'ref n/a:', v.slice(0, 60)); continue; }
  const [type, blk, ts, nb] = v;
  const raw = new Uint8Array(fs.readFileSync(`dq/${name}.bin`));
  const refB = fs.readFileSync(`dq/${name}.f32`); const ref = new Float32Array(refB.buffer, refB.byteOffset, refB.length / 4);
  const y = G.dequant(type, raw, nb);
  if (!y) { console.log(name.padEnd(8), 'not implemented in JS (canDequant=' + G.canDequant(type) + ')'); continue; }
  let bad = 0, firstBad = -1, maxRel = 0, finite = 0;
  for (let i = 0; i < ref.length; i++) {
    const a = y[i], b = ref[i];
    if (Number.isNaN(a) && Number.isNaN(b)) continue;
    if (!Number.isFinite(b) || !Number.isFinite(a)) { if (a !== b) { bad++; if (firstBad < 0) firstBad = i; } continue; }
    finite++;
    const rel = Math.abs(a - b) / Math.max(1e-30, Math.abs(b), Math.abs(a));
    if (Math.abs(a - b) > 1e-6 * Math.max(1, Math.abs(b)) && rel > 1e-5) { bad++; if (firstBad < 0) firstBad = i; }
    maxRel = Math.max(maxRel, Math.abs(a - b) > 0 ? rel : 0);
  }
  const ok = bad === 0; if (!ok) allOk = false;
  console.log(name.padEnd(8), ok ? 'OK  ' : 'FAIL', `n=${ref.length} finite=${finite} bad=${bad} maxRel=${maxRel.toExponential(2)}` + (ok ? '' : ` first@${firstBad}: js=${y[firstBad]} ref=${ref[firstBad]}`));
}
console.log(allOk ? 'ALL IMPLEMENTED TYPES MATCH' : 'MISMATCHES');
