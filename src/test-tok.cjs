const fs = require('fs'), { execFileSync } = require('child_process'); const G = require('./gguf-core.js');
const LM = require('os').homedir().replace(/\\/g, '/') + '/.lmstudio/models/lmstudio-community/';
const models = { qwen: LM + 'Qwen3.8-27B-GGUF/Qwen3.8-27B-Q4_K_M.gguf', gemma: LM + 'gemma-4-12B-it-QAT-GGUF/gemma-4-12B-it-QAT-Q4_0.gguf' };
const cases = JSON.parse(fs.readFileSync('tok_cases.json', 'utf8'));
// reference: llama-tokenize of llama.cpp b11388 (the Lupe server downloads it into ~/.cache/gguf-lupe), or set LLAMA_TOKENIZE
const TOKENIZE = process.env.LLAMA_TOKENIZE || ['cpu', 'vulkan', 'cuda'].map(k => require('path').join(require('os').homedir(), '.cache', 'gguf-lupe', 'llama.cpp-b11388', k, process.platform === 'win32' ? 'llama-tokenize.exe' : 'llama-tokenize')).find(f => fs.existsSync(f)) || 'llama-tokenize';
let fails = 0, total = 0;
for (const [name, f] of Object.entries(models)) {
  const fd = fs.openSync(f, 'r'), b = Buffer.alloc(32 << 20); fs.readSync(fd, b, 0, b.length, 0);
  const h = G.parse(b.buffer.slice(b.byteOffset, b.byteOffset + b.length), 0);
  const T = new G.Tokenizer(h.meta);
  let t0 = performance.now(); T.init(); const initMs = performance.now() - t0;
  console.log(`\n== ${name}: model=${T.model} pre=${T.pre} -> ${T.preType} byte=${T.byteEncode} escWs=${T.escapeWs} ignoreMerges=${T.ignoreMerges} addBos=${T.addBos} specials=${T.specials.length} init ${initMs.toFixed(0)} ms`);
  for (const [i, text] of cases.entries()) {
    for (const parseSpecial of [true, false]) {
      fs.writeFileSync('case.txt', text);
      let ref;
      try {
        const args = ['-m', f, '-f', 'case.txt', '--ids', '--log-disable']; if (!parseSpecial) args.push('--no-parse-special');
        const out = execFileSync(TOKENIZE, args, { encoding: 'utf8', stdio: ['ignore', 'pipe', 'ignore'] });
        ref = JSON.parse(out.trim().split('\n').pop());
      } catch (e) { console.log(`  case ${i}: reference failed (${String(e.message).slice(0, 60)})`); continue; }
      const t1 = performance.now(); const mine = T.tokenize(text, { parseSpecial }).ids; const ms = performance.now() - t1;
      total++;
      const ok = JSON.stringify(mine) === JSON.stringify(ref);
      if (!ok) { fails++; console.log(`  FAIL case ${i} special=${parseSpecial} ${JSON.stringify(text).slice(0, 60)}\n     ref ${JSON.stringify(ref)}\n     js  ${JSON.stringify(mine)}`); }
      else if (parseSpecial) console.log(`  ok  case ${i} (${ref.length} tok, ${ms.toFixed(1)} ms) ${JSON.stringify(text).slice(0, 50)}`);
    }
  }
}
console.log(`\n${total - fails}/${total} identical to llama.cpp b11388`);
