'use strict';
(() => {
const $ = s => document.querySelector(s);
// ---------- language: German or English, chosen by the browser, switchable (stored per browser)
let LANG = (() => { try { const s = localStorage.getItem('lupe-lang'); if (s === 'de' || s === 'en') return s; } catch {} return /^de\b/i.test(navigator.language || '') ? 'de' : 'en'; })();
const isEN = () => LANG === 'en';
const tl = (de, en) => LANG === 'en' ? en : de;          // every visible text: tl('Deutsch', 'English')
const LOC = () => LANG === 'en' ? 'en-US' : 'de-DE';
// texts from the Lupe server arrive in German; SRV_EN (parts/info_srv.js) maps them for the English page
const srvMsg = s => { if (!isEN() || !s || typeof SRV_EN === 'undefined') return s; for (const [re, en] of SRV_EN) if (re.test(s)) return s.replace(re, en); return s; };
const nf = (x, d = 0) => x.toLocaleString(LOC(), { minimumFractionDigits: d, maximumFractionDigits: d });
const fmtInt = x => Math.round(x).toLocaleString(LOC());
const sig = (x, n = 3) => !isFinite(x) ? String(x) : x === 0 ? '0' : x.toLocaleString(LOC(), { maximumSignificantDigits: n });
const esc = s => String(s).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
function fmtBytes(b, prec) {
  if (!isFinite(b)) return '–';
  if (b >= 1e9) return nf(b / 1e9, 2) + ' GB';
  if (b >= 1e6) return nf(b / 1e6, prec ?? (b >= 1e8 ? 0 : 1)) + ' MB';
  if (b >= 1e3) return nf(b / 1e3, b >= 1e5 ? 0 : 1) + ' KB';
  return fmtInt(b) + ' B';
}
function fmtParams(n) {
  if (n >= 1e9) return nf(n / 1e9, 2) + tl(' Mrd.', ' B');
  if (n >= 1e6) return nf(n / 1e6, 1) + tl(' Mio.', ' M');
  if (n >= 1e4) return nf(n / 1e3, 1) + tl(' Tsd.', ' K');
  return fmtInt(n);
}
const pct = (a, b) => { const p = a / b * 100; return (p < 0.1 ? nf(p, 3) : p < 10 ? nf(p, 1) : nf(p, 0)) + ' %'; };
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const isDark = () => css('--scheme') === 'dark';
const hex2rgb = h => { h = h.replace('#', ''); if (h.length === 3) h = h.replace(/./g, c => c + c); const n = parseInt(h, 16); return [n >> 16 & 255, n >> 8 & 255, n & 255]; };
const bpw = type => { const t = GGUF.TYPES[type]; return t ? t[2] * 8 / t[1] : NaN; };
const arrLen = v => v.n ?? v.items.length;
const fmtBits = x => nf(x, 4).replace(/[.,]?0+$/, '');
function niceStep(raw) { const p = 10 ** Math.floor(Math.log10(raw)), f = raw / p; return (f <= 1 ? 1 : f <= 2 ? 2 : f <= 2.5 ? 2.5 : f <= 5 ? 5 : 10) * p; }
const niceInt = raw => { for (const k of [1, 2, 5, 10, 20, 25, 50, 100, 200, 250, 500, 1000, 2000, 2500, 5000, 10000, 20000, 50000]) if (k >= raw) return k; return 100000; };
const subOf = tn => tn.endsWith('_K') ? ({ Q2_K: 16, Q3_K: 16, Q6_K: 16 }[tn] || 32) : 0;

// ---------- components ----------
const COMPS = {
  emb:   { get label() { return tl('Embedding', 'Embedding'); }, v: '--c-emb' },
  ffn:   { get label() { return tl('FFN / MLP', 'FFN / MLP'); }, v: '--c-ffn' },
  moe:   { get label() { return tl('MoE-Experten', 'MoE experts'); }, v: '--c-moe' },
  lin:   { get label() { return tl('Lineare Attention / SSM', 'Linear attention / SSM'); }, v: '--c-lin' },
  attn:  { get label() { return tl('Attention', 'Attention'); }, v: '--c-attn' },
  mtp:   { get label() { return tl('MTP-Kopf', 'MTP head'); }, v: '--c-mtp' },
  head:  { get label() { return tl('Ausgabe-Kopf', 'Output head'); }, v: '--c-head' },
  norm:  { get label() { return tl('Normierung', 'Normalization'); }, v: '--c-norm' },
  other: { get label() { return tl('Sonstiges', 'Other'); }, v: '--c-other' },
};
const STACK_ORDER = ['emb', 'ffn', 'moe', 'lin', 'attn', 'mtp', 'head', 'other', 'norm'];
const LAYER_RE = /^(?:([a-z]+)\.)?blk\.(\d+)\.(.+)$/;
function classify(t, layerHasSsm) {
  const m = LAYER_RE.exec(t.name), rest = m ? m[3] : t.name;
  if (/^token_embd|^(position|patch|tok)_embd|^token_types/.test(t.name) || /^(?:[a-z]+\.)?(patch_embd|position_embd)/.test(t.name)) return 'emb';
  if (/norm/.test(rest)) return 'norm';
  if (!m && /^output\./.test(t.name)) return 'head';
  if (/^nextn\./.test(rest)) return 'mtp';
  if (/_exps|_shexp|gate_inp|exp_probs/.test(rest)) return 'moe';
  if (/^ffn_/.test(rest)) return 'ffn';
  if (/^ssm_|^time_mix|^channel_mix|^shortconv/.test(rest)) return 'lin';
  if (layerHasSsm && /^attn_(qkv|gate)/.test(rest)) return 'lin';
  if (/^attn_/.test(rest)) return 'attn';
  return 'other';
}

// ---------- token text ----------
const utf8fatal = new TextDecoder('utf-8', { fatal: true });
function tokenText(raw, style) {
  if (raw == null) return { text: '', bytes: false };
  if (style === 'bpe') {
    const bytes = [];
    for (const ch of raw) { const b = GGUF.U2B.get(ch); if (b === undefined) return { text: raw, bytes: false }; bytes.push(b); }
    try { return { text: utf8fatal.decode(new Uint8Array(bytes)), bytes: false }; }
    catch { return { text: bytes.map(b => b.toString(16).toUpperCase().padStart(2, '0')).join(' '), bytes: true }; }
  }
  if (style === 'spm') return /^<0x[0-9A-F]{2}>$/.test(raw) ? { text: raw.slice(3, 5), bytes: true } : { text: raw.replace(/▁/g, ' '), bytes: false };
  return { text: raw, bytes: false };
}
function tokStyle(model, raws) {
  if (model === 'gpt2') return 'bpe';
  if (model === 'llama') return 'spm';
  let g = 0, u = 0, n = 0;
  for (const r of raws) { if (++n > 20000) break; if (r.charCodeAt(0) === 0x120) g++; else if (r.charCodeAt(0) === 0x2581) u++; }
  return u > g && u > 50 ? 'spm' : g > 50 ? 'bpe' : 'raw';
}
const wsHTML = s => esc(s).replace(/ /g, '<span class="ws">␣</span>').replace(/\n/g, '<span class="ws">↵</span>').replace(/\r/g, '<span class="ws">⏎</span>').replace(/\t/g, '<span class="ws">⇥</span>');
const wsPlain = s => s.replace(/ /g, '␣').replace(/\n/g, '↵').replace(/\r/g, '⏎').replace(/\t/g, '⇥');
const TOKTYPE = {
  get 1() { return tl('normal', 'normal'); }, get 2() { return tl('unbekannt', 'unknown'); }, get 3() { return tl('Steuer-Token', 'control token'); },
  get 4() { return tl('benutzerdefiniert', 'user-defined'); }, get 5() { return tl('reserviert', 'reserved'); }, get 6() { return tl('Byte', 'byte'); },
};

// ---------- byte sources ----------
class FileSource {
  constructor(file) { this.file = file; this.kind = 'file'; this.weightBytes = 0; }
  async read(pos, len) { const b = new Uint8Array(await this.file.slice(pos, pos + len).arrayBuffer()); this.weightBytes += b.length; return b; }
  async readRowRange(t, start, count) {
    const rb = t.rowBytes, buf = await this.read(t.absOffset + start * rb, count * rb);
    return Array.from({ length: count }, (_, i) => buf.subarray(i * rb, (i + 1) * rb));
  }
  has() { return true; }
}
const b64 = s => { const bin = atob(s), u = new Uint8Array(bin.length); for (let i = 0; i < bin.length; i++) u[i] = bin.charCodeAt(i); return u; };
class DemoSource {
  constructor(d) {
    this.kind = 'demo'; this.sets = {};
    for (const [name, ranges] of Object.entries(d.rows)) this.sets[name] = ranges.map(([s, n, b]) => ({ s, n, bytes: b64(b) }));
  }
  has(t) { return !!this.sets[t.name]; }
  row(t, r) {
    const rs = this.sets[t.name]; if (!rs) return null;
    for (const g of rs) if (r >= g.s && r < g.s + g.n) return g.bytes.subarray((r - g.s) * t.rowBytes, (r - g.s + 1) * t.rowBytes);
    return null;
  }
  async readRowRange(t, start, count) { return Array.from({ length: count }, (_, i) => this.row(t, start + i)); }
}
async function readRows(src, t, rows) {
  const out = [];
  for (let i = 0; i < rows.length;) {
    let j = i; while (j + 1 < rows.length && rows[j + 1] === rows[j] + 1) j++;
    out.push(...await src.readRowRange(t, rows[i], j - i + 1));
    i = j + 1;
  }
  return out;
}
const decodeRow = (t, raw) => raw ? GGUF.dequant(t.type, raw, t.dims[0] / GGUF.TYPES[t.type][1]) : null;
const canRead = t => GGUF.TYPES[t.type] && isFinite(t.rowBytes) && GGUF.canDequant(t.type);

// ---------- state ----------
const DEFAULT_TEXTS = { de: 'Die Wirtschaftsprüfung prüft den Jahresabschluss.', en: 'The auditor checks the annual financial statements.' };
const S = { vf: { cat: null, len: null, ws: false }, vsort: { key: 'id', dir: 1 }, mf: { bucket: null }, msort: { key: 'rank', dir: 1 }, tf: { comp: null, type: null }, eSub: 'table', spSort: { key: 'id', dir: 1 }, metaSort: { key: 'file', dir: 1 }, nnId: null, flowBlock: null, tourStep: 0, tourNN: null, model: null, tab: 'overview', layerMode: 'comp', tokSub: 'split', wSub: 'table', rowMode: 'head', histLog: false, sel: null, sort: { key: 'fileOrder', dir: 1 }, text: DEFAULT_TEXTS[LANG], tokRes: null, tokSel: null, wordSel: null, embMode: 'text', embA: null, embB: null, blk: 0 };

function buildModel(h, source, info) {
  const tensors = h.tensors, layers = new Map(), swaArr = h.meta[(h.meta['general.architecture'] || '') + '.attention.sliding_window_pattern'], swaPat = swaArr && swaArr.items;
  for (const t of tensors) {
    const m = LAYER_RE.exec(t.name);
    t.layerKey = m ? (m[1] ? m[1] + '.' : '') + m[2] : null;
    if (t.layerKey != null) { if (!layers.has(t.layerKey)) layers.set(t.layerKey, { key: t.layerKey, prefix: m[1] || '', idx: +m[2], tensors: [] }); layers.get(t.layerKey).tensors.push(t); }
  }
  for (const L of layers.values()) {
    const rests = L.tensors.map(t => t.name.replace(LAYER_RE, '$3'));
    const ssm = rests.some(r => /^ssm_|^time_mix|^shortconv/.test(r));
    for (const t of L.tensors) t.comp = classify(t, ssm);
    L.kind = rests.some(r => r.startsWith('nextn.')) ? 'mtp' : ssm ? 'lin' : rests.some(r => r.startsWith('attn_')) ? 'attn' : 'other';
    L.moe = rests.some(r => /_exps/.test(r));
    if (L.kind === 'attn' && swaPat && swaPat[L.idx]) L.kind = 'swa';
    L.bytes = L.tensors.reduce((a, t) => a + t.nBytes, 0);
  }
  for (const t of tensors) if (t.comp == null) t.comp = classify(t, false);
  const byComp = {}, byType = {};
  let params = 0, bytes = 0;
  for (const t of tensors) {
    params += t.nElements; bytes += t.nBytes;
    (byComp[t.comp] ||= { bytes: 0, params: 0, n: 0 }); byComp[t.comp].bytes += t.nBytes; byComp[t.comp].params += t.nElements; byComp[t.comp].n++;
    (byType[t.typeName] ||= { bytes: 0, params: 0, n: 0, type: t.type }); byType[t.typeName].bytes += t.nBytes; byType[t.typeName].params += t.nElements; byType[t.typeName].n++;
  }
  const meta = h.meta, toks = meta['tokenizer.ggml.tokens'], types = meta['tokenizer.ggml.token_type'];
  const tokModel = meta['tokenizer.ggml.model'] || '';
  const M = {
    h, source, info, tensors, byComp, byType, params, bytes, meta,
    arch: meta['general.architecture'] || '', tokModel,
    tokens: toks ? toks.items : [], types: types ? types.items : null,
    sorted: tensors.slice().sort((a, b) => a.absOffset - b.absOffset),
    layers: [...layers.values()].sort((a, b) => a.prefix.localeCompare(b.prefix) || a.idx - b.idx),
    byName: new Map(tensors.map(t => [t.name, t])),
  };
  M.vocabSize = M.tokens.length;
  M.tokStyle = tokStyle(tokModel, M.tokens);
  M.tok = M.vocabSize ? new GGUF.Tokenizer(meta) : null;
  const E = M.byName.get('token_embd.weight');
  M.E = E && E.dims.length === 2 && E.dims[1] === M.vocabSize && canRead(E) ? E : null;
  return M;
}
const tokType = id => S.model.types ? S.model.types[id] : 1;
function disp(id) {
  const M = S.model;
  if (M._disp) return M._disp[id];
  return tokenText(M.tokens[id], M.tokStyle).text;
}
function ensureDisp() {
  const M = S.model; if (M._disp) return;
  const n = M.vocabSize, d = new Array(n), partial = new Uint8Array(n);
  for (let i = 0; i < n; i++) { const r = tokenText(M.tokens[i], M.tokStyle); d[i] = r.text; if (r.bytes) partial[i] = 1; }
  M._disp = d; M._partial = partial;
}
function lookupWord(w) {
  const M = S.model;
  if (!M.tok || !M.tok.supported) { ensureDisp(); const i = M._disp.indexOf(' ' + w); return i >= 0 ? i : M._disp.indexOf(w); }
  M.tok.init();
  const style = M.tokStyle, enc = s => style === 'bpe' ? Array.from(new TextEncoder().encode(s), b => GGUF.B2U[b]).join('') : style === 'spm' ? s.replace(/ /g, '▁') : s;
  for (const c of [' ' + w, w]) { const id = M.tok.t2id.get(enc(c)); if (id !== undefined) return id; }
  return -1;
}

// ---------- tooltip ----------
const tip = $('#tip');
function showTip(html, e) {
  tip.innerHTML = html; tip.hidden = false;
  const r = tip.getBoundingClientRect(), pad = 14;
  let x = e.clientX + pad, y = e.clientY + pad;
  if (x + r.width > innerWidth - 8) x = e.clientX - r.width - pad;
  if (y + r.height > innerHeight - 8) y = e.clientY - r.height - pad;
  tip.style.left = Math.max(8, x) + 'px'; tip.style.top = Math.max(8, y) + 'px';
}
const hideTip = () => { tip.hidden = true; };
const tensorTip = t => { const p = fmtParams(t.nElements), b = fmtBytes(t.nBytes), o = fmtInt(t.absOffset); return `<b><code>${esc(t.name)}</code></b><br>${COMPS[t.comp].label} · ${esc(t.typeName)} · [${t.dims.map(fmtInt).join(' × ')}]<br><span class="m">${tl(`${p} Parameter · ${b} · ab Byte ${o}`, `${p} parameters · ${b} · at byte ${o}`)}</span>`; };

// ---------- colour helpers ----------
let THEME = null;
function theme() {
  if (THEME) return THEME;
  const n = hex2rgb(css('--div-neg')), m = hex2rgb(css('--div-mid')), p = hex2rgb(css('--div-pos')), lut = new Uint8ClampedArray(256 * 3);
  for (let i = 0; i < 256; i++) { const x = i / 255 * 2 - 1, a = x < 0 ? n : p, f = Math.abs(x); for (let c = 0; c < 3; c++) lut[i * 3 + c] = m[c] + (a[c] - m[c]) * f; }
  const g = k => css(k);
  return (THEME = { lut, surface: g('--surface'), sunken: g('--sunken'), ink: g('--ink'), ink2: g('--ink-2'), muted: g('--muted'), line: g('--line'), line2: g('--line-2'), surfRGB: hex2rgb(g('--surface')), sunkRGB: hex2rgb(g('--sunken')), lineRGB: hex2rgb(g('--line-2')), mono: g('--font-mono'), ui: g('--font-ui'), dark: isDark() });
}
function percentileAbs(arrs, p) {
  let n = 0; for (const a of arrs) n += a.length;
  const stride = Math.max(1, Math.floor(n / 200000)), s = [];
  let k = 0; for (const a of arrs) for (let i = 0; i < a.length; i++, k++) if (k % stride === 0) s.push(Math.abs(a[i]));
  s.sort((a, b) => a - b); return s[Math.min(s.length - 1, Math.floor(p * s.length))] || 1e-9;
}
const colorIdx = (v, s) => Math.max(0, Math.min(255, Math.round((v / s * 0.5 + 0.5) * 255))) * 3;

// =====================================================================
// MatrixView: a scrollable, zoomable table of numbers, rows read lazily
// =====================================================================
const HH = 24;
const ZOOMS = [{ k: 'fit', get label() { return tl('Überblick', 'Overview'); } }, { k: 'near', get label() { return tl('Nah', 'Near'); } }, { k: 'cells', get label() { return tl('Zellen', 'Cells'); } }, { k: 'num', get label() { return tl('Zahlen', 'Numbers'); } }];
class MatrixView {
  constructor(root, opt = {}) {
    this.root = root; this.opt = opt;
    root.innerHTML = `<div class="mv-bar"><div class="seg" role="group">${ZOOMS.map(z => `<button type="button" data-z="${z.k}" aria-pressed="false"></button>`).join('')}</div><span class="mv-pos"></span></div>`
      + `<div class="mv-vp"><canvas></canvas><div class="mv-sc" tabindex="0"><div class="mv-sp"></div></div></div>`
      + `<div style="display:flex;flex-wrap:wrap;gap:8px 24px;align-items:center"><div class="scale" style="flex:0 1 420px"><span class="lo"></span><div class="ramp"></div><span class="hi"></span></div><span class="count mv-note"></span></div>`;
    this.vp = root.querySelector('.mv-vp'); this.cv = root.querySelector('canvas'); this.sc = root.querySelector('.mv-sc'); this.sp = root.querySelector('.mv-sp');
    this.pos = root.querySelector('.mv-pos'); this.note = root.querySelector('.mv-note');
    this.cache = new Map(); this.inflight = new Set(); this.queue = []; this.busy = 0; this.zoom = opt.zoom || 'fit'; this.hover = null; this.gen = 0;
    this.relang();
    root.querySelector('.seg').onclick = e => { const b = e.target.closest('[data-z]'); if (b) this.setZoom(b.dataset.z); };
    this.sc.addEventListener('scroll', () => this.draw(), { passive: true });
    this.sc.addEventListener('mousemove', e => this.onMove(e));
    this.sc.addEventListener('mouseleave', () => { this.hover = null; hideTip(); this.draw(); });
    this.sc.addEventListener('click', e => this.onClick(e));
    if (window.ResizeObserver) new ResizeObserver(() => { if (this.d && this.root.offsetParent) this.layout(); }).observe(this.vp);
  }
  // texts in opt.aria and in the data (note, missingText, cornerText, rowWord, colWord): a string, a [de, en] pair or a function
  tx(v) { return typeof v === 'function' ? v() : Array.isArray(v) ? tl(v[0], v[1]) : v; }
  // language switch: rewrite the static labels, then repaint (canvas, scale numbers, position line, note)
  relang() {
    const seg = this.root.querySelector('.seg');
    seg.setAttribute('aria-label', tl('Zoomstufe', 'Zoom level'));
    seg.querySelectorAll('[data-z]').forEach(b => { b.textContent = ZOOMS.find(z => z.k === b.dataset.z).label; });
    this.sc.setAttribute('aria-label', this.tx(this.opt.aria) || tl('Zahlentabelle', 'Number table'));
    this.shownScale = undefined; this.draw();
  }
  setData(d) {
    this.d = d; this.cache.clear(); this.inflight.clear(); this.queue = []; this.busy = 0; this.gen++;
    this.scale = d.scale || null; this.hl = new Set(d.highlight || []); this.selRow = d.selRow ?? null; this.hover = null;
    this.LW = d.label ? Math.round(Math.min(220, Math.max(150, (this.sc.clientWidth || 900) * 0.2))) : 76;
    if (d.zoom) this.zoom = d.zoom;
    this.maxRows = Math.max(300, Math.floor(64e6 / (d.nCols * 4)));
    this.updateScale();
    this.layout(true);
  }
  n() { return this.d.rows ? this.d.rows.length : this.d.nRows; }
  rowAt(di) { return this.d.rows ? this.d.rows[di] : di; }
  maxVpH() { return Math.max(300, Math.min(innerHeight * 0.66, 620)); }
  cellSize() {
    const vw = Math.max(50, (this.sc.clientWidth || 900) - this.LW), C = this.d.nCols, n = this.n();
    if (this.zoom === 'fit') {
      const maxH = this.maxVpH() - HH - 18;
      return [Math.min(40, vw / C), n * 26 <= maxH ? 26 : n <= 60 ? Math.max(6, Math.floor(maxH / n)) : 3];
    }
    return this.zoom === 'near' ? [4, 4] : this.zoom === 'cells' ? [14, 14] : [64, 24];
  }
  layout(resetScroll) {
    if (!this.d) return;
    [this.cw, this.ch] = this.cellSize();
    const n = this.n(), C = this.d.nCols;
    this.vp.style.height = Math.round(Math.min(this.maxVpH(), HH + n * this.ch + 18)) + 'px';
    this.sp.style.width = Math.ceil(this.LW + C * this.cw) + 'px';
    this.sp.style.height = Math.ceil(HH + n * this.ch) + 'px';
    if (resetScroll) { this.sc.scrollTop = 0; this.sc.scrollLeft = 0; }
    this.root.querySelectorAll('[data-z]').forEach(b => b.setAttribute('aria-pressed', String(b.dataset.z === this.zoom)));
    this.draw();
  }
  setZoom(z) {
    if (!this.d) return;
    const gw = this.sc.clientWidth - this.LW, gh = this.sc.clientHeight - HH;
    const focus = this.hover ? { di: this.hover.di, c: this.hover.c } : { di: (this.sc.scrollTop + gh / 2) / this.ch, c: (this.sc.scrollLeft + gw / 2) / this.cw };
    if (this.selRow != null && !this.hover) { const di = this.d.rows ? this.d.rows.indexOf(this.selRow) : this.selRow; if (di >= 0) focus.di = di; }
    this.zoom = z; this.layout();
    const gw2 = this.sc.clientWidth - this.LW, gh2 = this.sc.clientHeight - HH;
    this.sc.scrollLeft = Math.max(0, (focus.c + 0.5) * this.cw - gw2 / 2);
    this.sc.scrollTop = Math.max(0, (focus.di + 0.5) * this.ch - gh2 / 2);
    this.draw();
  }
  scrollToRow(di, col) {
    const gh = this.sc.clientHeight - HH, gw = this.sc.clientWidth - this.LW;
    this.sc.scrollTop = Math.max(0, (di + 0.5) * this.ch - gh / 2);
    if (col != null) this.sc.scrollLeft = Math.max(0, (col + 0.5) * this.cw - gw / 2);
    this.draw();
  }
  draw() { if (!this.raf) this.raf = requestAnimationFrame(() => { this.raf = 0; this.paint(); }); }
  ensure(d0, d1) {
    const need = new Set();
    for (let i = d0; i < d1; i++) { const r = this.rowAt(i); if (!this.cache.has(r) && !this.inflight.has(r)) need.add(r); }
    const uniq = [...need].sort((a, b) => a - b), runs = [];
    for (const r of uniq) { const L = runs[runs.length - 1]; if (L && r === L.s + L.n && L.n < 192) L.n++; else runs.push({ s: r, n: 1 }); }
    this.queue = runs; this.pump();
  }
  pump() {
    const gen = this.gen;
    while (this.busy < 2 && this.queue.length) {
      const run = this.queue.shift();
      for (let k = 0; k < run.n; k++) this.inflight.add(run.s + k);
      this.busy++;
      this.d.fetch(run.s, run.n)
        .then(arr => { if (gen === this.gen) arr.forEach((v, k) => this.cache.set(run.s + k, v || null)); })
        .catch(() => { if (gen === this.gen) for (let k = 0; k < run.n; k++) this.cache.set(run.s + k, null); })
        .finally(() => {
          if (gen !== this.gen) return;
          for (let k = 0; k < run.n; k++) this.inflight.delete(run.s + k);
          this.busy--;
          if (!this.scale) this.autoScale();
          if (this.cache.size > this.maxRows) { let drop = this.cache.size - Math.floor(this.maxRows * 0.8); for (const k of this.cache.keys()) { if (drop-- <= 0) break; this.cache.delete(k); } }
          this.draw(); this.pump();
          this.opt.onData && this.opt.onData();
        });
    }
  }
  autoScale() {
    const vs = []; for (const v of this.cache.values()) { if (v) vs.push(v); if (vs.length >= 48) break; }
    if (!vs.length) return;
    this.scale = percentileAbs(vs, 0.99); this.updateScale();
  }
  updateScale() {
    const s = this.scale; this.shownScale = s;
    this.root.querySelector('.lo').textContent = s ? '−' + sig(s) : '';
    this.root.querySelector('.hi').textContent = s ? '+' + sig(s) : '';
  }
  rowData(r) { const v = this.cache.get(r); if (v !== undefined) { this.cache.delete(r); this.cache.set(r, v); } return v; }
  paint() {
    const d = this.d; if (!d || !this.root.offsetParent) return;
    const T = theme(), dpr = window.devicePixelRatio || 1;
    const W = this.sc.clientWidth, H = this.sc.clientHeight; if (!W || !H) return;
    if (this.cv.width !== Math.round(W * dpr) || this.cv.height !== Math.round(H * dpr)) {
      this.cv.width = Math.round(W * dpr); this.cv.height = Math.round(H * dpr); this.cv.style.width = W + 'px'; this.cv.style.height = H + 'px';
    }
    const ctx = this.cv.getContext('2d'); ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.fillStyle = T.surface; ctx.fillRect(0, 0, W, H);
    const { cw, ch, LW } = this, n = this.n(), C = d.nCols, st = this.sc.scrollTop, sl = this.sc.scrollLeft;
    const gw = W - LW, gh = H - HH;
    const d0 = Math.max(0, Math.floor(st / ch)), d1 = Math.min(n, Math.ceil((st + gh) / ch));
    this.ensure(d0, d1);
    const pooled = cw < 1, base = this.scale || 0.05, lut = T.lut;
    // pooling keeps the largest |value| per pixel, so widen the colour scale by the expected max of k samples
    const s = pooled ? base * Math.max(1, Math.sqrt(2 * Math.log(2 / cw)) / 1.6) : base;
    if (this.shownScale !== s) { this.shownScale = s; this.root.querySelector('.lo').textContent = '−' + sig(s); this.root.querySelector('.hi').textContent = '+' + sig(s); }
    if (this.zoom === 'num') {
      ctx.font = `11px ${T.mono}`; ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
      const c0 = Math.floor(sl / cw), c1 = Math.min(C, Math.ceil((sl + gw) / cw));
      for (let di = d0; di < d1; di++) {
        const y = HH + di * ch - st, v = this.rowData(this.rowAt(di));
        for (let c = c0; c < c1; c++) {
          const x = LW + c * cw - sl;
          if (v === undefined) { ctx.fillStyle = T.sunken; ctx.fillRect(x, y, cw - 1, ch - 1); continue; }
          if (v === null) { this.hatch(ctx, x, y, cw - 1, ch - 1, T); continue; }
          const k = colorIdx(v[c], s), r = lut[k], g = lut[k + 1], b = lut[k + 2];
          ctx.fillStyle = `rgb(${r},${g},${b})`; ctx.fillRect(x, y, cw - 1, ch - 1);
          ctx.fillStyle = (0.299 * r + 0.587 * g + 0.114 * b) > 150 ? '#0f1520' : '#ffffff';
          ctx.fillText(sig(v[c], 3), x + cw - 7, y + ch / 2);
        }
      }
    } else if (gw > 0 && gh > 0) {
      const IW = Math.ceil(gw), IH = Math.ceil(gh);
      const img = ctx.createImageData(IW, IH), px = img.data, line = new Uint8ClampedArray(IW * 4);
      const gapC = cw >= 6, gapR = ch >= 6, surf = T.surfRGB;
      const fillRow = (o, rgb) => { for (let x = 0; x < IW; x++) { const q = o + x * 4; px[q] = rgb[0]; px[q + 1] = rgb[1]; px[q + 2] = rgb[2]; px[q + 3] = 255; } };
      let lastDi = -1, lastState = 0;
      for (let y = 0; y < IH; y++) {
        const yy = y + st, di = Math.floor(yy / ch), o = y * IW * 4;
        if (di >= n || (gapR && (yy % ch) >= ch - 1)) { fillRow(o, surf); continue; }
        if (di !== lastDi) {
          lastDi = di;
          const v = this.rowData(this.rowAt(di));
          lastState = v === undefined ? 1 : v === null ? 2 : 0;
          if (!lastState) {
            for (let x = 0; x < IW; x++) {
              const q = x * 4, xx = x + sl; let val;
              if (pooled) {
                const a = Math.floor(xx / cw); if (a >= C) { line[q] = surf[0]; line[q + 1] = surf[1]; line[q + 2] = surf[2]; line[q + 3] = 255; continue; }
                const b = Math.min(C, Math.max(a + 1, Math.floor((xx + 1) / cw))); val = v[a];
                for (let c = a + 1; c < b; c++) if (Math.abs(v[c]) > Math.abs(val)) val = v[c];
              } else {
                const c = Math.floor(xx / cw);
                if (c >= C || (gapC && (xx % cw) >= cw - 1)) { line[q] = surf[0]; line[q + 1] = surf[1]; line[q + 2] = surf[2]; line[q + 3] = 255; continue; }
                val = v[c];
              }
              const k = colorIdx(val, s); line[q] = lut[k]; line[q + 1] = lut[k + 1]; line[q + 2] = lut[k + 2]; line[q + 3] = 255;
            }
          }
        }
        if (lastState === 1) fillRow(o, T.sunkRGB);
        else if (lastState === 2) { for (let x = 0; x < IW; x++) { const q = o + x * 4, rgb = ((x + yy) % 10 < 2) ? T.lineRGB : T.sunkRGB; px[q] = rgb[0]; px[q + 1] = rgb[1]; px[q + 2] = rgb[2]; px[q + 3] = 255; } }
        else px.set(line, o);
      }
      if (!this.off) this.off = document.createElement('canvas');
      this.off.width = IW; this.off.height = IH;
      this.off.getContext('2d').putImageData(img, 0, 0);
      ctx.imageSmoothingEnabled = false;
      ctx.drawImage(this.off, LW, HH, IW, IH);
    }
    // quantization block lines
    let noteTxt = this.tx(d.note) || '';
    if (d.qk > 1 && !pooled && cw * d.qk >= 10) {
      const sub = d.sub || 0, unit = sub && cw * sub >= 8 ? sub : d.qk;
      const c0 = Math.floor(sl / cw), c1 = Math.min(C, Math.ceil((sl + gw) / cw)), yEnd = Math.min(H, HH + n * ch - st);
      ctx.save(); ctx.beginPath(); ctx.rect(LW, HH, gw, gh); ctx.clip();
      for (let c = Math.ceil(c0 / unit) * unit; c <= c1; c += unit) {
        if (c === 0) continue;
        const major = c % d.qk === 0, x = Math.round(LW + c * cw - sl) - 0.5;
        ctx.strokeStyle = major ? T.ink : T.ink2; ctx.globalAlpha = major ? 0.6 : 0.3; ctx.lineWidth = major ? 1.5 : 1;
        ctx.beginPath(); ctx.moveTo(x, HH); ctx.lineTo(x, yEnd); ctx.stroke();
      }
      ctx.restore();
      const qk = fmtInt(d.qk), thin = unit === sub;
      noteTxt = tl(`Senkrechte Linien: ${d.typeName}-Blöcke à ${qk} Zahlen, je eine Skala${thin ? ` · dünn: Unterblöcke à ${sub}` : ''}`,
        `Vertical lines: ${d.typeName} blocks of ${qk} numbers, one scale each${thin ? ` · thin: sub-blocks of ${sub}` : ''}`);
    }
    this.note.textContent = noteTxt;
    // hover crosshair + selected row
    ctx.save(); ctx.beginPath(); ctx.rect(LW, HH, gw, gh); ctx.clip();
    const rowRect = (di, color, w) => { const y = HH + di * ch - st; if (y + ch < HH || y > H) return; ctx.strokeStyle = color; ctx.lineWidth = w; ctx.strokeRect(LW + 1, y + 0.5, Math.max(1, Math.min(gw, C * cw - sl) - 2), Math.max(1, ch - 1)); };
    if (this.hover) {
      rowRect(this.hover.di, T.ink2, 1);
      if (!pooled) { const x = LW + this.hover.c * cw - sl; ctx.strokeStyle = T.ink2; ctx.lineWidth = 1; ctx.strokeRect(x + 0.5, HH, Math.max(1, cw - 1), Math.min(gh, n * ch - st)); }
    }
    if (this.selRow != null) for (let di = d0; di < d1; di++) if (this.rowAt(di) === this.selRow) rowRect(di, T.ink, 2);
    ctx.restore();
    // header
    ctx.fillStyle = T.sunken; ctx.fillRect(LW, 0, W - LW, HH);
    ctx.font = `11px ${T.ui}`; ctx.fillStyle = T.muted; ctx.textBaseline = 'middle'; ctx.textAlign = 'center';
    if (pooled) {
      const step = niceInt(C / Math.max(1, gw / 90));
      for (let c = 0; c < C; c += step) { const x = LW + c * cw; ctx.textAlign = c === 0 ? 'left' : 'center'; ctx.fillText(fmtInt(c), c === 0 ? LW + 4 : x, HH / 2); }
    } else {
      const step = niceInt(56 / cw), c0 = Math.floor(sl / cw), c1 = Math.min(C, Math.ceil((sl + gw) / cw));
      for (let c = Math.ceil(c0 / step) * step; c < c1; c += step) { const x = LW + (c + 0.5) * cw - sl; if (x > LW + 12) ctx.fillText(fmtInt(c), x, HH / 2); }
    }
    // gutter
    ctx.fillStyle = T.surface; ctx.fillRect(0, HH, LW, H - HH);
    const labelEvery = ch >= 13 ? 1 : Math.ceil(16 / ch);
    for (let di = Math.ceil(d0 / labelEvery) * labelEvery; di < d1; di += labelEvery) {
      const r = this.rowAt(di), top = HH + di * ch - st, y = top + ch / 2;
      if (y < HH + 6) continue;
      const lab = d.label && ch >= 13 ? d.label(r, di) : null, isSel = r === this.selRow;
      if (isSel && ch >= 13) { ctx.fillStyle = T.sunken; ctx.fillRect(0, top, LW, ch); }
      ctx.textAlign = 'right'; ctx.fillStyle = T.muted; ctx.font = `11px ${T.ui}`;
      const idTxt = fmtInt(r);
      ctx.fillText(idTxt, LW - 8, y);
      if (lab) {
        const idW = ctx.measureText(idTxt).width;
        ctx.textAlign = 'left'; ctx.fillStyle = T.ink; ctx.font = `${isSel ? 650 : 400} 12.5px ${T.mono}`;
        this.fitText(ctx, lab, 8, y, LW - idW - 22);
      }
    }
    ctx.fillStyle = T.sunken; ctx.fillRect(0, 0, LW, HH);
    ctx.fillStyle = T.muted; ctx.textAlign = 'left'; ctx.font = `11px ${T.ui}`;
    ctx.fillText(this.tx(d.cornerText) || tl('Zeile ╲ Spalte', 'Row ╲ column'), 8, HH / 2);
    ctx.strokeStyle = T.line2; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(LW - 0.5, 0); ctx.lineTo(LW - 0.5, H); ctx.moveTo(0, HH - 0.5); ctx.lineTo(W, HH - 0.5); ctx.stroke();
    // status line
    let missing = 0, loading = 0; for (let di = d0; di < d1; di++) { const v = this.cache.get(this.rowAt(di)); if (v === undefined) loading++; else if (v === null) missing++; }
    const missTxt = missing && this.tx(d.missingText);
    if (!this.hover) this.pos.innerHTML = missTxt ? esc(missTxt) : loading ? tl('lese Zeilen …', 'reading rows …') : this.defaultPos(d0, d1);
  }
  defaultPos(d0, d1) {
    const d = this.d; if (d1 <= d0) return '';
    const k = fmtInt(d1 - d0), a = fmtInt(d0), b = fmtInt(d1 - 1), all = fmtInt(d.rows ? d.rows.length : d.nRows), cols = fmtInt(d.nCols);
    const rows = d.rows ? tl(`Sichtbar: ${k} von ${all} Zeilen`, `Visible: ${k} of ${all} rows`) : tl(`Sichtbar: Zeilen <b>${a}</b>–<b>${b}</b> von ${all}`, `Visible: rows <b>${a}</b>–<b>${b}</b> of ${all}`);
    return rows + (this.cw < 1 ? tl(` · alle ${cols} Spalten verkleinert`, ` · all ${cols} columns shrunk`) : '');
  }
  fitText(ctx, txt, x, y, maxW) {
    if (ctx.measureText(txt).width <= maxW) { ctx.fillText(txt, x, y); return; }
    let lo = 0, hi = txt.length;
    while (lo < hi) { const mid = (lo + hi + 1) >> 1; if (ctx.measureText(txt.slice(0, mid) + '…').width <= maxW) lo = mid; else hi = mid - 1; }
    ctx.fillText(txt.slice(0, lo) + '…', x, y);
  }
  hatch(ctx, x, y, w, h, T) { ctx.fillStyle = T.sunken; ctx.fillRect(x, y, w, h); ctx.save(); ctx.beginPath(); ctx.rect(x, y, w, h); ctx.clip(); ctx.strokeStyle = T.line2; ctx.lineWidth = 1; for (let k = -h; k < w; k += 8) { ctx.beginPath(); ctx.moveTo(x + k, y + h); ctx.lineTo(x + k + h, y); ctx.stroke(); } ctx.restore(); }
  cellAt(e) {
    const b = this.sc.getBoundingClientRect(), x = e.clientX - b.left, y = e.clientY - b.top;
    if (x < this.LW || y < HH || x > this.sc.clientWidth || y > this.sc.clientHeight) return null;
    const di = Math.floor((y - HH + this.sc.scrollTop) / this.ch), c = Math.floor((x - this.LW + this.sc.scrollLeft) / this.cw);
    if (di < 0 || di >= this.n() || c < 0 || c >= this.d.nCols) return null;
    return { di, c, r: this.rowAt(di), x };
  }
  onMove(e) {
    const h = this.cellAt(e);
    if (!h) { if (this.hover) { this.hover = null; this.draw(); } hideTip(); return; }
    const changed = !this.hover || this.hover.di !== h.di || this.hover.c !== h.c;
    this.hover = h;
    const d = this.d, v = this.cache.get(h.r), lab = d.label ? d.label(h.r, h.di) : null;
    let c = h.c, cEnd = c;
    if (this.cw < 1) { cEnd = Math.min(d.nCols - 1, Math.floor((h.x - this.LW + this.sc.scrollLeft + 1) / this.cw) - 1); if (v) { for (let k = c; k <= cEnd; k++) if (Math.abs(v[k]) > Math.abs(v[c])) c = k; } }
    const val = v ? `<b>${sig(v[c], 4)}</b>` : v === null ? tl('nicht im Beispiel', 'not in the example') : tl('wird gelesen …', 'reading …');
    const c0 = fmtInt(h.c), c1 = fmtInt(cEnd);
    const rowName = `${this.tx(d.rowWord) || tl('Zeile', 'Row')} ${fmtInt(h.r)}${lab ? ` <code>${esc(lab)}</code>` : ''}`;
    const colName = `${this.tx(d.colWord) || tl('Spalte', 'Column')} ${fmtInt(c)}${cEnd > h.c ? ` <span class="m">${tl(`(größte aus ${c0}–${c1})`, `(largest of ${c0}–${c1})`)}</span>` : ''}`;
    showTip(`${rowName}<br>${colName}<br>${tl('Wert', 'Value')}: ${val}`, e);
    this.pos.innerHTML = `${rowName} · ${colName} · ${tl('Wert', 'value')} ${val}`;
    if (changed) this.draw();
  }
  onClick(e) { const h = this.cellAt(e); if (!h) return; this.selRow = h.r; this.draw(); this.opt.onRow && this.opt.onRow(h.r, h); }
}

// ---------- overview ----------
const metaNum = (M, key) => { const v = M.meta[M.arch + '.' + key]; return v == null ? null : v; };
function showVal(v) {
  if (v == null) return null;
  if (typeof v === 'object' && v.items) { const a = Array.from(v.items); const mn = Math.min(...a), mx = Math.max(...a); return mn === mx ? fmtInt(mn) : `${fmtInt(mn)}–${fmtInt(mx)}`; }
  return typeof v === 'number' ? fmtInt(v) : String(v);
}
function renderFileBar() {
  const M = S.model, info = M.info;
  $('#fileName').textContent = info.fileName;
  const pill = $('#modePill');
  pill.textContent = info.demo ? tl('Eingebautes Beispiel', 'Built-in example') : M.source.kind === 'server' ? tl('Über Lupe-Server', 'Via Lupe server') : tl('Lokale Datei', 'Local file');
  pill.className = 'pill' + (info.demo ? '' : ' live');
  updateReadStat();
}
function updateReadStat() {
  const M = S.model; if (!M) return;
  const h = M.h;
  if (M.info.demo) { $('#readStat').innerHTML = tl('<strong>Beispiel</strong> · nur ausgewählte Zeilen', '<strong>Example</strong> · selected rows only'); return; }
  const hb = fmtBytes(h.headerEnd, 2), size = fmtBytes(h.fileSize), p = pct(h.headerEnd, h.fileSize), ms = fmtInt(M.info.parseMs), wb = fmtBytes(M.source.weightBytes);
  $('#readStat').innerHTML = tl(`Header <strong>${hb}</strong> von <strong>${size}</strong> (${p}) · gelesen in <strong>${ms} ms</strong> · Gewichte gelesen: <strong>${wb}</strong>`,
    `Header <strong>${hb}</strong> of <strong>${size}</strong> (${p}) · read in <strong>${ms} ms</strong> · weights read: <strong>${wb}</strong>`);
}
function renderOverview() {
  const M = S.model, h = M.h;
  const share = pct(h.fileSize - h.headerEnd, h.fileSize), params = fmtParams(M.params), nT = fmtInt(h.nTensors), hb = fmtBytes(h.headerEnd, 2);
  $('#ovExplain').innerHTML = tl(`Diese Datei <b>ist</b> das Modell: zu ${share} Zahlen, <b>${params} gelernte Gewichte</b> in ${nT} Tabellen (Tensoren). Ganz vorne steht ein kleiner Header (${hb}) mit Bauplan, Wörterbuch und Inhaltsverzeichnis. Beim Öffnen wird nur er gelesen.`,
    `This file <b>is</b> the model: ${share} of it is numbers, <b>${params} learned weights</b> in ${nT} tables (tensors). At the very front sits a small header (${hb}) with the blueprint, vocabulary and table of contents. Opening the file reads only this header.`);
  const ft = M.meta['general.file_type'];
  const blocks = metaNum(M, 'block_count'), nextn = metaNum(M, 'nextn_predict_layers');
  const heads = metaNum(M, 'attention.head_count'), kv = metaNum(M, 'attention.head_count_kv');
  const experts = metaNum(M, 'expert_count'), used = metaNum(M, 'expert_used_count');
  const bits = nf(M.bytes * 8 / M.params, 2), nMtp = fmtInt(nextn || 0), kvN = kv != null ? showVal(kv) : '', usedN = fmtInt(used || 0), nKV = fmtInt(h.nKV), fsz = fmtInt(h.fileSize);
  const facts = [
    [tl('Architektur', 'Architecture'), esc(M.arch || '–'), esc(M.meta['general.name'] || '')],
    [tl('Parameter', 'Parameters'), fmtParams(M.params), M.meta['general.size_label'] ? 'Label: ' + esc(M.meta['general.size_label']) : ''],
    [tl('Quantisierung', 'Quantization'), esc(ft != null ? (GGUF.FILE_TYPES[ft] || tl('Typ ', 'Type ') + ft) : '–'), tl(`Ø ${bits} Bit pro Gewicht`, `avg. ${bits} bits per weight`)],
    [tl('Blöcke (Schichten)', 'Blocks (layers)'), blocks != null ? fmtInt(blocks) : String(M.layers.length), nextn ? tl(`davon ${nMtp} MTP-${nextn === 1 ? 'Block' : 'Blöcke'}`, `incl. ${nMtp} MTP ${nextn === 1 ? 'block' : 'blocks'}`) : ''],
    [tl('Kontextlänge', 'Context length'), showVal(metaNum(M, 'context_length')) ?? '–', tl('Tokens', 'tokens')],
    [tl('Dimensionen', 'Dimensions'), showVal(metaNum(M, 'embedding_length')) ?? '–', tl('Zahlen pro Token-Vektor', 'numbers per token vector')],
    [tl('Attention-Köpfe', 'Attention heads'), heads != null ? showVal(heads) : '–', kv != null ? tl(`${kvN} Key/Value-Köpfe`, `${kvN} key/value heads`) : ''],
    ...(experts ? [[tl('Experten (MoE)', 'Experts (MoE)'), fmtInt(experts), used ? tl(`${usedN} aktiv pro Token`, `${usedN} active per token`) : '']] : []),
    [tl('Wörterbuch', 'Vocabulary'), M.vocabSize ? fmtInt(M.vocabSize) : '–', tl('Tokens', 'tokens')],
    [tl('Tensoren', 'Tensors'), fmtInt(h.nTensors), tl(`${nKV} Metadaten-Einträge`, `${nKV} metadata entries`)],
    [tl('Dateigröße', 'File size'), fmtBytes(h.fileSize), tl(`${fsz} Byte · GGUF v${h.version}`, `${fsz} bytes · GGUF v${h.version}`)],
  ];
  $('#facts').innerHTML = facts.map(([k, v, s]) => `<div><dt>${k}</dt><dd>${v}${s ? `<small>${s}</small>` : ''}</dd></div>`).join('');
  $('#compLegend').innerHTML = STACK_ORDER.filter(c => M.byComp[c]).map(c => `<span><i style="background:var(${COMPS[c].v})"></i>${COMPS[c].label}</span>`).join('') + `<span><i style="background:var(--ink)"></i>Header</span>`;
  if (S.tab === 'overview') { drawStrip(); renderBars(); renderLayers(); }
}
function drawStrip() {
  const M = S.model, cv = $('#strip'), dpr = window.devicePixelRatio || 1;
  const w = cv.clientWidth, hgt = cv.clientHeight; if (!w) return;
  cv.width = Math.round(w * dpr); cv.height = Math.round(hgt * dpr);
  const ctx = cv.getContext('2d'); ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const F = M.h.fileSize, X = b => b / F * w;
  ctx.fillStyle = css('--sunken'); ctx.fillRect(0, 0, w, hgt);
  const col = {}; for (const c in COMPS) col[c] = css(COMPS[c].v);
  let run = null; const runs = [];
  for (const t of M.sorted) {
    if (run && run.comp === t.comp) run.end = t.absOffset + t.nBytes;
    else { run = { comp: t.comp, start: t.absOffset, end: t.absOffset + t.nBytes }; runs.push(run); }
  }
  for (const r of runs) { const x0 = X(r.start), x1 = X(r.end); ctx.fillStyle = col[r.comp]; ctx.fillRect(x0, 0, Math.max(x1 - x0, 0.6), hgt); }
  ctx.fillStyle = css('--surface');
  for (let i = 1; i < runs.length; i++) { const x = X(runs[i].start), pw = X(runs[i - 1].end) - X(runs[i - 1].start); if (pw > 4) ctx.fillRect(x - 0.5, 0, 1, hgt); }
  ctx.fillStyle = css('--ink'); ctx.fillRect(0, 0, Math.max(X(M.h.dataStart), 3), hgt);
  if (S.sel) { const x0 = X(S.sel.absOffset), x1 = Math.max(X(S.sel.absOffset + S.sel.nBytes), x0 + 2); ctx.strokeStyle = css('--ink'); ctx.lineWidth = 2; ctx.strokeRect(x0 - 1, 1, x1 - x0 + 2, hgt - 2); }
  if (S.stripHover != null) { ctx.fillStyle = css('--ink'); ctx.fillRect(S.stripHover - 0.5, 0, 1, hgt); }
  $('#hdrCallout').textContent = `Header ${fmtBytes(M.h.headerEnd, 2)} · ${pct(M.h.headerEnd, F)}`;
  const gb = F / 1e9, step = niceStep(gb / 5), ticks = [];
  for (let v = 0; v <= gb + 1e-9; v += step) ticks.push(v);
  const keep = ticks.filter((v, i) => i === 0 || v / gb * w < w - 94);
  $('#stripAxis').innerHTML = keep.map((v, i) => `<span style="left:${v / gb * 100}%">${i === 0 ? '0' : nf(v, step < 1 ? 1 : 0)} GB</span>`).join('') + `<span style="left:100%;transform:translateX(-100%)">${nf(gb, 2)} GB</span>`;
}
function tensorAt(off) {
  const a = S.model.sorted; let lo = 0, hi = a.length - 1;
  while (lo < hi) { const mid = (lo + hi + 1) >> 1; if (a[mid].absOffset <= off) lo = mid; else hi = mid - 1; }
  return a[lo];
}
function barRows(host, rows, total, fmt = fmtBytes) {
  const max = Math.max(...rows.map(r => r.bytes));
  host.innerHTML = rows.map((r, i) => `<div class="bar-row${r.onClick ? ' clickable' : ''}" data-i="${i}"${r.current ? ' aria-current="true"' : ''}${r.onClick ? ' role="button" tabindex="0"' : ''}><div class="lab"><i style="background:${r.color}"></i>${r.label}${r.sub ? `<small>${r.sub}</small>` : ''}</div><div class="bar-track"><div class="bar" style="width:calc(${r.bytes / max} * (100% - 7.5em));background:${r.color}"></div><span class="bar-val">${fmt(r.bytes)} · ${pct(r.bytes, total)}</span></div></div>`).join('');
  host.onmousemove = e => { const row = e.target.closest('.bar-row'); if (!row || !rows[+row.dataset.i].tip) return hideTip(); showTip(rows[+row.dataset.i].tip, e); };
  host.onmouseleave = hideTip;
  host.onclick = e => { const row = e.target.closest('.bar-row'); const r = row && rows[+row.dataset.i]; if (r && r.onClick) { hideTip(); r.onClick(); } };
  host.onkeydown = e => { if ((e.key === 'Enter' || e.key === ' ') && e.target.classList.contains('clickable')) { e.preventDefault(); e.target.click(); } };
}
let RAMP = [];
function typeColors(M) {
  const ramp = isDark()
    ? ['#184f95', '#1c5cab', '#256abf', '#2a78d6', '#3987e5', '#5598e7', '#6da7ec', '#86b6ef', '#9ec5f4', '#cde2fb']
    : ['#86b6ef', '#6da7ec', '#5598e7', '#3987e5', '#2a78d6', '#256abf', '#1c5cab', '#184f95', '#104281', '#0d366b'];
  const types = Object.keys(M.byType).sort((a, b) => (bpw(M.byType[a].type) || 99) - (bpw(M.byType[b].type) || 99));
  const out = {};
  types.forEach((tn, i) => { out[tn] = ramp[types.length === 1 ? 5 : Math.round(i / (types.length - 1) * (ramp.length - 1))]; });
  RAMP = types; return out;
}
function renderBars() {
  const M = S.model;
  barRows($('#compBars'), Object.entries(M.byComp).sort((a, b) => b[1].bytes - a[1].bytes).map(([c, v]) => {
    const p = fmtParams(v.params), b = fmtBytes(v.bytes), sh = pct(v.bytes, M.bytes), n = fmtInt(v.n);
    return {
      label: COMPS[c].label, sub: tl(`${p} Parameter`, `${p} parameters`), bytes: v.bytes, color: `var(${COMPS[c].v})`,
      tip: `<b>${COMPS[c].label}</b><br>${tl(`${b} · ${sh} der Gewichte<br><span class="m">${n} Tensoren · ${p} Parameter · Klick listet sie auf</span>`, `${b} · ${sh} of weights<br><span class="m">${n} tensors · ${p} parameters · click to list them</span>`)}`,
      onClick: () => showTensorsWhere({ comp: c }),
    };
  }), M.bytes);
  const tc = typeColors(M);
  barRows($('#typeBars'), Object.entries(M.byType).sort((a, b) => b[1].bytes - a[1].bytes).map(([tn, v]) => {
    const ok = isFinite(bpw(v.type)), bits = ok ? fmtBits(bpw(v.type)) : '', b = fmtBytes(v.bytes), sh = pct(v.bytes, M.bytes), n = fmtInt(v.n);
    return {
      label: `<span class="mono">${esc(tn)}</span>`, sub: ok ? tl(`${bits} Bit/Gewicht · ${n} Tensoren`, `${bits} bits/weight · ${n} tensors`) : tl(`${n} Tensoren`, `${n} tensors`), bytes: v.bytes, color: tc[tn],
      tip: `<b>${esc(tn)}</b>${ok ? tl(` · ${bits} Bit pro Gewicht`, ` · ${bits} bits per weight`) : ''}<br>${tl(`${b} · ${sh} der Gewichte<br><span class="m">${n} Tensoren · Klick listet sie auf</span>`, `${b} · ${sh} of weights<br><span class="m">${n} tensors · click to list them</span>`)}`,
      onClick: () => showTensorsWhere({ type: tn }),
    };
  }), M.bytes);
}
const topRound = (x, y, w, h, r) => { r = Math.min(r, w / 2, h); return `M${x},${y + h}V${y + r}Q${x},${y} ${x + r},${y}H${x + w - r}Q${x + w},${y} ${x + w},${y + r}V${y + h}Z`; };
function layerGroups(L, mode) {
  const g = new Map();
  for (const t of L.tensors) { const k = mode === 'comp' ? t.comp : t.typeName; if (!g.has(k)) g.set(k, { key: k, bytes: 0, tensors: [] }); const e = g.get(k); e.bytes += t.nBytes; e.tensors.push(t); }
  const order = mode === 'comp' ? k => STACK_ORDER.indexOf(k) : k => RAMP.indexOf(k);
  return [...g.values()].sort((a, b) => order(a.key) - order(b.key));
}
function renderLayers() {
  const M = S.model, host = $('#layerChart'), Ls = M.layers;
  if (!Ls.length) { host.innerHTML = tl('<p class="note">Keine Blöcke (<span class="mono">blk.N.*</span>) in dieser Datei.</p>', '<p class="note">No blocks (<span class="mono">blk.N.*</span>) in this file.</p>'); $('#layerLede').textContent = ''; $('#layerLegend').innerHTML = ''; $('#layerFoot').textContent = ''; return; }
  const kinds = {}; for (const L of Ls) kinds[L.kind] = (kinds[L.kind] || 0) + 1;
  const KL = Object.fromEntries(['lin', 'swa', 'attn', 'mtp', 'other'].map(k => [k, kindLabel(M, k)]));
  const fai = metaNum(M, 'full_attention_interval');
  const nL = fmtInt(Ls.length), kList = ['lin', 'swa', 'attn', 'mtp', 'other'].filter(k => kinds[k]).map(k => `${fmtInt(kinds[k])} × ${KL[k]}`).join(' · '), every = fmtInt(fai || 0);
  $('#layerLede').textContent = tl(`${nL} Blöcke: ${kList}.`, `${nL} blocks: ${kList}.`)
    + (fai && kinds.lin ? tl(` Jeder ${every}. Block nutzt volle Attention, die übrigen eine lineare Variante mit fester Zustandsgröße statt wachsendem KV-Cache.`, ` One block in ${every} uses full attention; the others use a linear variant with a fixed-size state instead of a growing KV cache.`) : '')
    + (Ls.some(L => L.moe) ? tl(' Die Blöcke enthalten Mixture-of-Experts-FFNs.', ' The blocks contain mixture-of-experts FFNs.') : '');
  const mode = S.layerMode, tc = typeColors(M);
  const colorOf = k => mode === 'comp' ? `var(${COMPS[k].v})` : tc[k];
  const keys = mode === 'comp' ? STACK_ORDER.filter(c => Ls.some(L => L.tensors.some(t => t.comp === c))) : RAMP.filter(tn => Ls.some(L => L.tensors.some(t => t.typeName === tn)));
  $('#layerLegend').innerHTML = keys.map(k => `<span><i style="background:${colorOf(k)}"></i>${mode === 'comp' ? COMPS[k].label : `<span class="mono">${esc(k)}</span> · ${fmtBits(bpw(M.byType[k].type))} ${tl('Bit', 'bits')}`}</span>`).join('');
  const W = Math.max(host.clientWidth || 800, Ls.length * 9 + 70), H = 250;
  const padL = 58, padR = 8, padT = 10, padB = 24, iw = W - padL - padR, ih = H - padT - padB;
  const maxB = Math.max(...Ls.map(L => L.bytes)), stepB = niceStep(maxB / 4), yMax = Math.ceil(maxB / stepB) * stepB;
  const y = b => padT + ih - b / yMax * ih, step = iw / Ls.length, bw = Math.max(2, Math.min(24, step - 2));
  let s = `<svg width="${W}" height="${H}" viewBox="0 0 ${W} ${H}" role="img" aria-label="${tl('Bytes pro Block', 'Bytes per block')}">`;
  for (let v = 0; v <= yMax + 1; v += stepB) s += `<line class="${v ? 'gridline' : 'baseline'}" x1="${padL}" x2="${W - padR}" y1="${y(v)}" y2="${y(v)}"/><text class="chart-text" x="${padL - 8}" y="${y(v) + 4}" text-anchor="end">${v ? fmtBytes(v) : '0'}</text>`;
  const lblEvery = [1, 2, 4, 5, 8, 10, 16, 20, 25, 50, 100].find(k => k * step >= 24) || 100;
  Ls.forEach((L, i) => {
    const x = padL + i * step + (step - bw) / 2, gs = layerGroups(L, mode).filter(g => g.bytes / yMax * ih >= 0.75);
    let acc = 0;
    gs.forEach((g, gi) => {
      const top = gi === gs.length - 1, y0 = y(acc + g.bytes), hh = g.bytes / yMax * ih;
      const yy = top ? y0 : y0 + 2, h2 = top ? hh : Math.max(0.6, hh - 2);
      s += `<path class="hit" d="${top ? topRound(x, yy, bw, h2, 4) : `M${x},${yy}h${bw}v${h2}h${-bw}Z`}" style="fill:${colorOf(g.key)}" data-l="${i}" data-g="${esc(g.key)}"/>`;
      acc += g.bytes;
    });
    if (i % lblEvery === 0) s += `<text class="chart-text" x="${x + bw / 2}" y="${H - 6}" text-anchor="middle">${L.prefix ? esc(L.prefix) + '.' : ''}${L.idx}</text>`;
  });
  host.innerHTML = s + '</svg>';
  const svg = host.firstChild;
  const groupAt = e => { const p = e.target.closest('[data-l]'); if (!p) return null; const L = Ls[+p.dataset.l]; return { L, g: layerGroups(L, mode).find(g => g.key === p.dataset.g) }; };
  svg.onmousemove = e => {
    const r = groupAt(e); if (!r) return hideTip();
    const { L, g } = r, lab = mode === 'comp' ? COMPS[g.key].label : g.key;
    const list = g.tensors.slice().sort((a, b) => b.nBytes - a.nBytes).slice(0, 6).map(t => `<code>${esc(t.name.replace(LAYER_RE, '$3'))}</code> <span class="m">${esc(t.typeName)} · ${fmtBytes(t.nBytes)}</span>`).join('<br>');
    const gb = fmtBytes(g.bytes), lb = fmtBytes(L.bytes), more = g.tensors.length - 6;
    showTip(`<b>Block ${esc(L.key)}</b> <span class="m">(${KL[L.kind] || L.kind})</span><br>${esc(lab)}: ${tl(`${gb} von ${lb}`, `${gb} of ${lb}`)}<br>${list}${more > 0 ? `<br><span class="m">${tl(`+${more} weitere`, `+${more} more`)}</span>` : ''}<br><span class="m">${tl('Klick öffnet den größten Tensor', 'click opens the largest tensor')}</span>`, e);
  };
  svg.onmouseleave = hideTip;
  svg.onclick = e => { const r = groupAt(e); if (r) openTensor(r.g.tensors.slice().sort((a, b) => b.nBytes - a.nBytes)[0]); };
  const out = M.tensors.filter(t => t.layerKey == null);
  const outList = Object.entries(out.reduce((a, t) => { a[t.comp] = (a[t.comp] || 0) + t.nBytes; return a; }, {})).sort((a, b) => b[1] - a[1]).map(([c, b]) => `${COMPS[c].label} ${fmtBytes(b)}`).join(' · ');
  $('#layerFoot').textContent = out.length ? tl(`Außerhalb der Blöcke: ${outList}`, `Outside the blocks: ${outList}`) : '';
}

// =====================================================================
// TOKENS
// =====================================================================
function examples(M) {
  const ok = M.tok && M.tok.supported; if (ok) M.tok.init();
  const has = s => ok && M.tok.t2id.has(s);
  // the built-in demo carries embedding rows for exactly these texts (build-demo.cjs, add-demo-nn.cjs)
  const ex = [
    [tl('Deutsch', 'English'), DEFAULT_TEXTS[LANG]],
    [tl('Englisch', 'German'), DEFAULT_TEXTS[isEN() ? 'de' : 'en']],
    [tl('Zahlen', 'Numbers'), tl('Umsatz 2025: 1.234.567,89 € (Vorjahr: 987.654,32 €)', 'Revenue 2025: $1,234,567.89 (prior year: $987,654.32)')],
    [tl('Langes Wort', 'Long German word'), 'Donaudampfschifffahrtsgesellschaftskapitänsmütze'],
    ['Emoji', tl('Prüfung bestanden ✅🎉', 'Exam passed ✅🎉')],
    ['Code', tl('def summe(a, b):\n    return a + b', 'def total(a, b):\n    return a + b')],
  ];
  const chat = tl('Chat-Format', 'Chat format'), q = tl('Was ist ein Token?', 'What is a token?');
  if (has('<|im_start|>')) ex.push([chat, `<|im_start|>user\n${q}<|im_end|>\n<|im_start|>assistant\n`]);
  else if (has('<|turn>')) ex.push([chat, `<|turn>user\n${q}<turn|>\n<|turn>model\n`]);
  else if (has('<start_of_turn>')) ex.push([chat, `<start_of_turn>user\n${q}<end_of_turn>\n<start_of_turn>model\n`]);
  return ex;
}
function renderTokensTab() {
  const M = S.model;
  $('#nVocab').textContent = M.vocabSize ? fmtInt(M.vocabSize) : '';
  const nV = fmtInt(M.vocabSize);
  $('#tokExplain').innerHTML = M.vocabSize
    ? tl(`Ein Sprachmodell liest keine Buchstaben, sondern <b>Tokens</b>: Wörter und Wortstücke aus einem festen <b>Wörterbuch</b> mit ${nV} Einträgen. Jedes Token hat eine Nummer, die <b>Token-ID</b>; so wird aus jedem Text eine Liste von Zahlen. Die ID ist zugleich die Zeile des Tokens in der <b>Embedding-Tabelle</b>.`,
      `A language model doesn’t read letters but <b>tokens</b>: words and word pieces from a fixed <b>vocabulary</b> of ${nV} entries. Each token has a number, its <b>token ID</b>, so any text becomes a list of numbers. The ID is also the token’s row in the <b>embedding table</b>.`)
    : tl('Diese Datei enthält kein Wörterbuch (zum Beispiel ein reiner Bild-Encoder).', 'This file has no vocabulary (for example, a pure image encoder).');
  const ex = examples(M);
  $('#tokExamples').innerHTML = tl('Beispiele:', 'Examples:') + ' ' + ex.map(([l], i) => `<button type="button" class="ui" data-ex="${i}">${esc(l)}</button>`).join('');
  $('#tokExamples').onclick = e => { const b = e.target.closest('[data-ex]'); if (!b) return; S.text = ex[+b.dataset.ex][1]; $('#tokText').value = S.text; S.tokSel = null; S.wordSel = null; runTokenize(); };
  $('#tokText').value = S.text;
  runTokenize();
}
function runTokenize() {
  const M = S.model, unsup = $('#tokUnsupported');
  if (!M.tok || !M.tok.supported) {
    // same two cases as GGUF.Tokenizer (gguf-core.js) sets its German .reason for; anything else shows .reason as is
    const tk = M.tok, mdl = tk && esc(tk.model || tl('unbekannt', 'unknown'));
    const why = !tk ? '' : tk.model !== 'gpt2' && tk.model !== 'gemma4'
      ? tl(`Live-Zerlegung nur für BPE-Tokenizer (Qwen, Llama 3, Mistral Nemo, Gemma 4 …), dieses Modell nutzt „${mdl}“.`, `Live splitting only for BPE tokenizers (Qwen, Llama 3, Mistral Nemo, Gemma 4 …); this model uses “${mdl}”.`)
      : !tk.merges || !tk.tokens.length ? tl('Die Datei enthält keine Merge-Regeln.', 'The file has no merge rules.') : esc(tk.reason);
    unsup.hidden = false; unsup.innerHTML = tk ? why + tl(' Wörterbuch und Spezial-Tokens funktionieren trotzdem.', ' Vocabulary and special tokens still work.') : tl('Kein Tokenizer in dieser Datei.', 'No tokenizer in this file.');
    $('#tokOut').hidden = true; S.tokRes = null; $('#wordPick').innerHTML = ''; $('#bpeSteps').innerHTML = '';
    if (S.tab === 'embed' || S.tab === 'meaning') renderEmbed();
    return;
  }
  unsup.hidden = !M.tok.approx; if (M.tok.approx) unsup.textContent = tl(`Pre-Tokenizer „${M.tok.pre}“ unbekannt: Zerlegung nur angenähert.`, `Unknown pre-tokenizer “${M.tok.pre}”: the split is approximate.`);
  $('#tokOut').hidden = false;
  const text = S.text.slice(0, 20000);
  const res = M.tok.tokenize(text, { trace: true });
  S.tokRes = res;
  const n = res.ids.length, chars = Array.from(text).length, words = (text.match(/\S+/g) || []).length;
  const cN = fmtInt(chars), tN = fmtInt(n), cpt = n ? nf(chars / n, 1) : '', wN = fmtInt(words);
  $('#tokStats').innerHTML = tl(`<span><span class="big num">${cN}</span> Zeichen</span><span>→</span><span><span class="big num">${tN}</span> Tokens</span>`, `<span><span class="big num">${cN}</span> characters</span><span>→</span><span><span class="big num">${tN}</span> tokens</span>`)
    + (n ? tl(`<span>${cpt} Zeichen pro Token · ${wN} Wörter</span>`, `<span>${cpt} characters per token · ${wN} words</span>`) : '');
  let html = '';
  res.pieces.forEach((p, i) => {
    const tt = tokenText(M.tokens[p.id], M.tokStyle), special = p.kind === 'special' || p.kind === 'bos' || p.kind === 'eos';
    html += `<button type="button" class="tk tk-c${i % 6}${special ? ' special' : ''}" data-i="${i}" aria-pressed="false"><span class="tt">${tt.bytes ? `<span class="ws">${esc(tt.text)}</span>` : wsHTML(tt.text)}</span><small>${fmtInt(p.id)}</small></button>`;
    if (!tt.bytes && /\n/.test(tt.text)) html += '<span class="flow-break"></span>';
  });
  $('#tokFlow').innerHTML = html || `<span class="count">${tl('Leerer Text: keine Tokens.', 'Empty text: no tokens.')}</span>`;
  $('#tokIds').textContent = '[' + res.ids.join(', ') + ']';
  if (S.tokSel == null || S.tokSel >= n) { const k = res.pieces.findIndex(p => !p.kind && p.word != null && res.words[p.word].syms.length > 1); S.tokSel = k >= 0 ? k : Math.max(0, res.pieces.findIndex(p => !p.kind)); }
  markTokSel(); renderTokDetail(); renderWords();
  if (S.tab === 'embed' || S.tab === 'meaning') renderEmbed();
}
function markTokSel() { document.querySelectorAll('#tokFlow .tk').forEach(b => b.setAttribute('aria-pressed', String(+b.dataset.i === S.tokSel))); }
let detailSeq = 0;
async function renderTokDetail() {
  const M = S.model, host = $('#tokDetail'), res = S.tokRes;
  if (!res || !res.pieces.length) { host.innerHTML = ''; host.hidden = true; return; }
  host.hidden = false;
  const p = res.pieces[S.tokSel] || res.pieces[0], id = p.id, raw = M.tokens[id], tt = tokenText(raw, M.tokStyle), E = M.E;
  const kindTxt = p.kind === 'bos' ? tl('automatisch vorangestellt (Textanfang)', 'added automatically (start of text)') : p.kind === 'special' ? tl('Spezial-Token', 'special token') : p.fallback ? tl('Byte-Fallback (Zeichen nicht im Wörterbuch)', 'byte fallback (character not in the vocabulary)') : TOKTYPE[tokType(id)];
  const idN = fmtInt(id), dN = E ? fmtInt(E.dims[0]) : '';
  host.innerHTML = `<h3><span>${tl('Ausgewählt: Token', 'Selected token')}</span><code>${tt.bytes ? esc(tt.text) : wsHTML(tt.text)}</code><span class="count">ID ${idN} · ${tl('Rohform', 'raw form')}: <span class="mono">${esc(raw)}</span> · ${esc(kindTxt)}</span></h3>`
    + (E ? tl(`<p class="cap">Zeile ${idN} der Embedding-Tabelle · ${dN} Zahlen · blau negativ, rot positiv</p>`, `<p class="cap">Row ${idN} of the embedding table · ${dN} numbers · blue negative, red positive</p>`)
      + `<canvas class="vecstrip" id="tdStrip"></canvas><div class="vecnums" id="tdNums">${tl('lese Zeile …', 'reading row …')}</div><div class="quick"><button type="button" class="ui" id="tdGo">${tl(`Zeile ${idN} in der Embedding-Tabelle zeigen →`, `Show row ${idN} in the embedding table →`)}</button></div>` : '');
  if (!E) return;
  $('#tdGo').onclick = () => { S.embMode = 'text'; S.embFocus = S.tokSel; S.eSub = 'table'; setTab('embed'); };
  const seq = ++detailSeq;
  const [rawRow] = await M.source.readRowRange(E, id, 1);
  if (seq !== detailSeq) return;
  updateReadStat();
  const v = decodeRow(E, rawRow);
  if (!v) { $('#tdNums').innerHTML = M.info.demo ? tl('Zeile nicht im eingebauten Beispiel.', 'Row not in the built-in example.') : tl('Zeile konnte nicht gelesen werden.', 'Could not read the row.'); $('#tdStrip').hidden = true; return; }
  drawVecStrip($('#tdStrip'), v);
  const rest = fmtInt(v.length - 14);
  $('#tdNums').textContent = Array.from(v.subarray(0, 14)).map(x => sig(x, 3)).join(' · ') + tl(` · … (${rest} weitere)`, ` · … (${rest} more)`);
}
function drawVecStrip(cv, v, scale) {
  const n = v.length, W = Math.min(n, 1024), T = theme(), s = scale || percentileAbs([v], 0.99);
  cv.width = W; cv.height = 1;
  const ctx = cv.getContext('2d'), img = ctx.createImageData(W, 1);
  for (let x = 0; x < W; x++) {
    const a = Math.floor(x * n / W), b = Math.max(a + 1, Math.floor((x + 1) * n / W)); let best = v[a];
    for (let i = a + 1; i < b; i++) if (Math.abs(v[i]) > Math.abs(best)) best = v[i];
    const k = colorIdx(best, s), o = x * 4; img.data[o] = T.lut[k]; img.data[o + 1] = T.lut[k + 1]; img.data[o + 2] = T.lut[k + 2]; img.data[o + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
  cv.onmousemove = e => { const r = cv.getBoundingClientRect(), i = Math.min(n - 1, Math.floor((e.clientX - r.left) / r.width * n)); showTip(`Dimension ${fmtInt(i)}: <b>${sig(v[i], 4)}</b>`, e); };
  cv.onmouseleave = hideTip;
}
const symHTML = (sym, style, hot) => {
  const t = tokenText(sym, style);
  return `<span class="sym${hot ? ' hot' : ''}${t.bytes ? ' byte' : ''}">${t.bytes ? esc(t.text) : wsHTML(t.text)}</span>`;
};
function renderWords() {
  const M = S.model, res = S.tokRes, host = $('#wordPick');
  const words = res.words.map((w, i) => ({ w, i })).filter(x => x.w.init.length > 1);
  if (!words.length) { host.innerHTML = `<span class="count">${tl('Kein Wort mit mehr als einem Zeichen im Text.', 'No word longer than one character in the text.')}</span>`; $('#bpeSteps').innerHTML = ''; return; }
  if (S.wordSel == null || !words.some(x => x.i === S.wordSel)) S.wordSel = words.slice().sort((a, b) => (b.w.steps?.length || 0) - (a.w.steps?.length || 0))[0].i;
  const style = M.tok.byteEncode ? 'bpe' : 'spm';
  host.innerHTML = tl('Wort wählen:', 'Pick a word:') + ' ' + words.map(({ w, i }) => `<button type="button" data-w="${i}" aria-current="${i === S.wordSel}">${wsHTML(tokenText(w.init.join(''), style).text)}</button>`).join('');
  const w = res.words[S.wordSel];
  const row = (n, rule, syms, hot) => `<div class="step"><span class="n">${n}</span><span class="rule">${rule}</span><span class="syms">${syms.map((s, k) => symHTML(s, style, k === hot)).join('')}</span></div>`;
  const nI = fmtInt(w.init.length);
  let out = row('Start', M.tok.byteEncode ? tl(`${nI} Bytes`, `${nI} bytes`) : tl(`${nI} Zeichen`, `${nI} characters`), w.init, -1);
  if (w.steps && w.steps.length && w.steps[0].whole) out += `<p class="cap">${tl('Ganzes Wort steht im Wörterbuch: keine Schritte', 'Whole word is in the vocabulary: no steps')}</p>` + row(tl('Ende', 'End'), tl('1 Token', '1 token'), w.syms, 0);
  else (w.steps || []).forEach((st, k) => { const r = fmtInt(st.rank); out += row(`${k + 1}.`, tl(`Regel ${r}`, `Rule ${r}`), st.syms, st.at); });
  const ids = w.syms.map(s => M.tok.t2id.get(s)), nS = fmtInt(w.syms.length), one = w.syms.length === 1;
  const idList = ids.map(x => x === undefined ? tl('Byte-Fallback', 'byte fallback') : 'ID ' + fmtInt(x)).join(', ');
  const grey = M.tok.byteEncode && w.init.length > Array.from(tokenText(w.init.join(''), style).text).length;
  out += `<p class="cap" style="margin-top:6px">${tl(`Ergebnis: <b>${nS} ${one ? 'Token' : 'Tokens'}</b> (${idList})`, `Result: <b>${nS} ${one ? 'token' : 'tokens'}</b> (${idList})`)}${grey ? tl(' · grau: einzelne Bytes', ' · gray: single bytes') : ''}${ib('bpeResult')}</p>`;
  $('#bpeSteps').innerHTML = out;
}

// ---------- vocabulary ----------
// [key, label]; the label (index 1) is a getter, so CATS[i][1] follows the current language (part E reads it too)
const CATS = [
  ['lat', 'Lateinisch', 'Latin'], ['han', 'Han (Chinesisch/Japanisch)', 'Han (Chinese/Japanese)'], ['kana', 'Japanisch (Kana)', 'Japanese (kana)'], ['hang', 'Koreanisch', 'Korean'], ['cyr', 'Kyrillisch', 'Cyrillic'], ['arab', 'Arabisch', 'Arabic'],
  ['grk', 'Griechisch', 'Greek'], ['heb', 'Hebräisch', 'Hebrew'], ['ind', 'Indische Schriften', 'Indic scripts'], ['thai', 'Thai', 'Thai'], ['oth', 'Andere Schriften', 'Other scripts'], ['num', 'Zahlen', 'Numbers'],
  ['emo', 'Emoji', 'Emoji'], ['pun', 'Satzzeichen & Symbole', 'Punctuation & symbols'], ['ws', 'Leerzeichen & Einrückung', 'Spaces & indentation'], ['byte', 'Byte-Stücke', 'Byte pieces'], ['spec', 'Spezial-Tokens', 'Special tokens'], ['res', 'Reserviert', 'Reserved'],
].map(([k, de, en]) => Object.defineProperty([k], 1, { get: () => tl(de, en), enumerable: true }));
const nChars = n => tl(`${fmtInt(n)} Zeichen`, `${fmtInt(n)} ${n === 1 ? 'character' : 'characters'}`);
const lenLabel = L => L >= 16 ? tl('16 oder mehr Zeichen', '16 or more characters') : nChars(L);
const CAT_RE = [['lat', /\p{Script=Latin}/u], ['han', /\p{Script=Han}/u], ['kana', /[\p{Script=Hiragana}\p{Script=Katakana}]/u], ['hang', /\p{Script=Hangul}/u], ['cyr', /\p{Script=Cyrillic}/u], ['arab', /\p{Script=Arabic}/u], ['grk', /\p{Script=Greek}/u], ['heb', /\p{Script=Hebrew}/u], ['ind', /[\p{Script=Devanagari}\p{Script=Bengali}\p{Script=Tamil}\p{Script=Telugu}\p{Script=Gujarati}\p{Script=Gurmukhi}\p{Script=Kannada}\p{Script=Malayalam}\p{Script=Oriya}\p{Script=Sinhala}]/u], ['thai', /\p{Script=Thai}/u]];
function buildVocabCats() {
  const M = S.model; if (M._cat) return;
  ensureDisp();
  const n = M.vocabSize, cat = new Uint8Array(n), lens = new Uint16Array(n), bys = new Uint16Array(n), idx = Object.fromEntries(CATS.map(([k], i) => [k, i]));
  const lenHist = new Array(17).fill(0), te = new TextEncoder(); let wordStart = 0, longest = 0, longestId = 0, umlaut = 0;
  for (let i = 0; i < n; i++) {
    const ty = M.types ? M.types[i] : 1, t = M._disp[i], len = Array.from(t).length;
    lens[i] = len; bys[i] = M._partial[i] ? 1 : te.encode(t).length;
    let c;
    if (ty === 3 || ty === 4 || ty === 2) c = 'spec';
    else if (ty === 5) c = 'res';
    else if (ty === 6 || M._partial[i]) c = 'byte';
    else {
      const tr = t.trim();
      if (!tr) c = 'ws';
      else {
        const L = tr.match(/\p{L}/u);
        if (L) { c = 'oth'; for (const [k, re] of CAT_RE) if (re.test(L[0])) { c = k; break; } }
        else if (/\p{Extended_Pictographic}/u.test(tr)) c = 'emo';
        else if (/^\p{N}+$/u.test(tr)) c = 'num';
        else c = 'pun';
      }
      if (t.charCodeAt(0) === 32) wordStart++;
      if (/[äöüßÄÖÜ]/.test(t)) umlaut++;
      lenHist[Math.min(16, len)]++;
      if (len > longest && c !== 'ws' && c !== 'pun') { longest = len; longestId = i; }
    }
    cat[i] = idx[c];
  }
  const counts = new Array(CATS.length).fill(0); for (let i = 0; i < n; i++) counts[cat[i]]++;
  Object.assign(M, { _cat: cat, _len: lens, _bytes: bys, _lenHist: lenHist, _wordStart: wordStart, _longest: longestId, _umlaut: umlaut, _catCount: counts });
}
let vocabList = null, mergeList = null;
class VList {
  constructor(host, opt) {
    this.host = host; this.opt = opt; this.rowH = 30; this.items = []; this.sel = null;
    host.innerHTML = `<div class="vl-head ${opt.cols}"></div><div class="vl-sp" style="position:relative"></div>`;
    this.head = host.querySelector('.vl-head'); this.sp = host.querySelector('.vl-sp');
    this.renderHead();
    host.onscroll = () => this.draw();
    host.onclick = e => {
      const sb = e.target.closest('.vl-head [data-sort]'); if (sb) { opt.onSort && opt.onSort(sb.dataset.sort); return; }
      const r = e.target.closest('.vl-row'); if (r) { this.sel = +r.dataset.k; this.draw(true); opt.onClick && opt.onClick(this.sel); }
    };
  }
  renderHead() {
    const so = this.opt.sort ? this.opt.sort() : null;
    this.head.innerHTML = this.opt.colsDef.map(([k, l, r]) => k
      ? `<span${r ? ' class="r"' : ''} aria-sort="${so && so.key === k ? (so.dir > 0 ? 'ascending' : 'descending') : 'none'}"><button type="button" data-sort="${k}">${l}</button></span>`
      : `<span${r ? ' class="r"' : ''}>${l}</span>`).join('');
  }
  // language switch: rewrite the column heads and the selection bar
  relang() { this.renderHead(); if (this.sel != null && this.opt.onClick) this.opt.onClick(this.sel); }
  setItems(items) { this.items = items; this.sp.style.height = items.length * this.rowH + 'px'; this.host.scrollTop = 0; this.draw(true); }
  draw(force) {
    if (!this.host.offsetParent) return;
    const top = Math.max(0, this.host.scrollTop - 32), h = this.host.clientHeight;
    const i0 = Math.max(0, Math.floor(top / this.rowH) - 4), i1 = Math.min(this.items.length, Math.ceil((top + h) / this.rowH) + 4);
    if (!force && this.i0 === i0 && this.i1 === i1) return;
    this.i0 = i0; this.i1 = i1;
    let html = '';
    for (let i = i0; i < i1; i++) { const k = this.items[i]; html += `<div class="vl-row ${this.opt.cols}" data-k="${k}" style="top:${i * this.rowH}px"${k === this.sel ? ' aria-selected="true"' : ''}>${this.opt.row(k)}</div>`; }
    this.sp.innerHTML = html;
  }
}
function selBar(id, html) {
  let bar = $('#' + id);
  if (!bar) { bar = document.createElement('div'); bar.id = id; bar.className = 'note'; (id === 'vocabSel' ? $('#vocabList') : $('#mergeList')).before(bar); }
  bar.innerHTML = html;
  const go = bar.querySelector('[data-go]'); if (go) go.onclick = () => jumpToRow(+go.dataset.go);
  const nn = bar.querySelector('[data-nn]'); if (nn) nn.onclick = () => openNN(+nn.dataset.nn);
}
function renderVocab() {
  const M = S.model;
  if (!M.vocabSize) { $('#vocabCount').textContent = tl('Kein Wörterbuch in dieser Datei.', 'No vocabulary in this file.'); return; }
  if (M._vocabReady) { renderVocabCharts(); if (vocabList) { vocabList.relang(); vocabList.draw(true); vocabInfo(); } return; }
  buildVocabCats(); M._vocabReady = true;
  S.vf = { cat: null, len: null, ws: false }; S.vsort = { key: 'id', dir: 1 };
  vocabList = new VList($('#vocabList'), {
    cols: 'cols-vocab', get colsDef() { return [['id', 'ID', 1], ['tok', 'Token'], [null, tl('Rohform', 'Raw form')], ['cat', tl('Art', 'Category')], ['len', tl('Zeichen', 'Chars'), 1], ['bytes', 'Bytes', 1]]; }, sort: () => S.vsort,
    onSort: k => { S.vsort = { key: k, dir: S.vsort.key === k ? -S.vsort.dir : (k === 'len' || k === 'bytes' ? -1 : 1) }; vocabList.renderHead(); filterVocab(); },
    row: id => `<span class="r">${fmtInt(id)}</span><span class="mono">${M._partial[id] ? `<span class="ws">${esc(M._disp[id])}</span>` : wsHTML(M._disp[id])}</span><span class="mono dim">${esc(M.tokens[id])}</span><span class="dim">${CATS[M._cat[id]][1]}</span><span class="r">${M._len[id]}</span><span class="r">${M._bytes[id]}</span>`,
    onClick: id => selBar('vocabSel', `<b>Token ${fmtInt(id)}</b>: <span class="mono">${wsHTML(M._disp[id])}</span> · ${CATS[M._cat[id]][1]} · ${nChars(M._len[id])} ` + (M.E ? `<button type="button" class="btn" style="padding:3px 10px;font-size:13px" data-go="${id}">${tl(`Embedding-Zeile ${fmtInt(id)} →`, `Embedding row ${fmtInt(id)} →`)}</button> <button type="button" class="btn" style="padding:3px 10px;font-size:13px" data-nn="${id}">${tl('Ähnliche Wörter', 'Similar words')}</button>` : '')),
  });
  renderVocabCharts(); filterVocab();
}
function setVocabFilter(p) { Object.assign(S.vf, p); $('#vocabWS').setAttribute('aria-pressed', String(S.vf.ws)); renderVocabCharts(); filterVocab(); }
function renderVocabCharts() {
  const M = S.model, n = M.vocabSize, vf = S.vf;
  const order = CATS.map((c, k) => [c, k, M._catCount[k]]).filter(x => x[2] > 0).sort((a, b) => b[2] - a[2]);
  barRows($('#vocabScripts'), order.slice(0, 10).map(([c, k, cnt]) => ({ label: c[1], bytes: cnt, color: 'var(--c-ffn)', current: vf.cat === k, onClick: () => setVocabFilter({ cat: vf.cat === k ? null : k }), tip: tl(`<b>${c[1]}</b><br>${fmtInt(cnt)} Tokens · ${pct(cnt, n)}<br><span class="m">Klick filtert die Liste</span>`, `<b>${c[1]}</b><br>${fmtInt(cnt)} tokens · ${pct(cnt, n)}<br><span class="m">Click to filter the list</span>`) })), n, fmtInt);
  const host = $('#vocabLen'), W = Math.max(280, host.clientWidth || 420), H = 170, padB = 22, padT = 8, B = 16, hist = M._lenHist.slice(1);
  const max = Math.max(...hist), bw = W / B, ih = H - padB - padT;
  let s = `<svg width="${W}" height="${H}" viewBox="0 0 ${W} ${H}" role="img" aria-label="${tl('Tokens nach Länge', 'Tokens by length')}"><line class="baseline" x1="0" x2="${W}" y1="${H - padB}" y2="${H - padB}"/>`;
  hist.forEach((c, i) => {
    const h = Math.max(c ? 1 : 0, c / max * ih), x = i * bw + 1, on = vf.len == null || vf.len === i + 1;
    s += `<rect class="pick" data-i="${i}" x="${x - 1}" y="${padT}" width="${bw}" height="${ih}" style="fill:transparent"/>`;
    if (c) s += `<path class="pick" data-i="${i}" d="${topRound(x, H - padB - h, Math.max(1, bw - 2), h, 3)}" style="fill:var(--c-ffn);fill-opacity:${on ? 1 : 0.3}"/>`;
    if (i % 3 === 0 || i === B - 1) s += `<text class="chart-text" x="${x + bw / 2 - 1}" y="${H - 6}" text-anchor="middle">${i === B - 1 ? '16+' : i + 1}</text>`;
  });
  host.innerHTML = s + '</svg>';
  const svg = host.firstChild;
  svg.onmousemove = e => { const p = e.target.closest('.pick'); if (!p) return hideTip(); const i = +p.dataset.i, L = lenLabel(i + 1); showTip(tl(`${L}: <b>${fmtInt(hist[i])}</b> Tokens<br><span class="m">Klick filtert die Liste</span>`, `${L}: <b>${fmtInt(hist[i])}</b> tokens<br><span class="m">Click to filter the list</span>`), e); };
  svg.onmouseleave = hideTip;
  svg.onclick = e => { const p = e.target.closest('.pick'); if (!p) return; const L = +p.dataset.i + 1; hideTip(); setVocabFilter({ len: vf.len === L ? null : L }); };
  const ws = fmtInt(M._wordStart), um = fmtInt(M._umlaut), ibw = ib('vocabWordStart'), lw = `<span class="mono">${wsHTML(M._disp[M._longest].slice(0, 40))}</span> (${nChars(M._len[M._longest])})`;
  $('#vocabLenNote').innerHTML = tl(`${ws} Tokens beginnen mit Leerzeichen (Wortanfang) ${ibw} · ${um} mit ä, ö, ü oder ß · längstes Wort-Token: ${lw}`, `${ws} tokens start with a space (word start) ${ibw} · ${um} contain ä, ö, ü or ß · longest word token: ${lw}`);
  $('#vocabCats').innerHTML = `<button type="button" data-cat="" aria-pressed="${vf.cat == null}">${tl('Alle', 'All')}<small>${fmtInt(n)}</small></button>` + order.map(([c, k, cnt]) => `<button type="button" data-cat="${k}" aria-pressed="${vf.cat === k}">${c[1]}<small>${fmtInt(cnt)}</small></button>`).join('');
}
let vocabTimer = 0;
function filterVocab() {
  const M = S.model, qRaw = $('#vocabQ').value, q = qRaw.trim(), { cat, len, ws } = S.vf, n = M.vocabSize;
  let items;
  if (/^#\d+$/.test(q)) { const id = +q.slice(1); items = id < n ? [id] : []; }
  else {
    const ql = qRaw.toLowerCase(), out = [];
    for (let i = 0; i < n; i++) {
      if (cat != null && M._cat[i] !== cat) continue;
      if (len != null && (len === 16 ? M._len[i] < 16 : M._len[i] !== len)) continue;
      if (ws && M._disp[i].charCodeAt(0) !== 32) continue;
      if (ql && !M._disp[i].toLowerCase().includes(ql)) continue;
      out.push(i);
    }
    items = out;
  }
  const { key, dir } = S.vsort;
  if (key !== 'id' || dir < 0) {
    const cmp = key === 'tok' ? (a, b) => COLL.compare(M._disp[a], M._disp[b]) || a - b
      : key === 'cat' ? (a, b) => M._cat[a] - M._cat[b] || a - b
      : key === 'len' ? (a, b) => M._len[a] - M._len[b] || a - b
      : key === 'bytes' ? (a, b) => M._bytes[a] - M._bytes[b] || a - b : (a, b) => a - b;
    items.sort((a, b) => cmp(a, b) * dir);
  }
  vocabList.setItems(items);
  vocabInfo();
}
function vocabInfo() {
  const M = S.model, q = $('#vocabQ').value.trim(), { cat, len, ws } = S.vf, n = M.vocabSize, k = vocabList.items.length;
  $('#vocabCount').textContent = tl(`${fmtInt(k)} von ${fmtInt(n)} Tokens`, `${fmtInt(k)} of ${fmtInt(n)} tokens`);
  renderFilterChips($('#vocabFilters'), [
    cat != null && [CATS[cat][1], () => setVocabFilter({ cat: null })],
    len != null && [lenLabel(len), () => setVocabFilter({ len: null })],
    ws && [tl('Nur Wortanfänge', 'Word starts only'), () => setVocabFilter({ ws: false })],
    q && [tl(`Suche „${q}“`, `Search “${q}”`), () => { $('#vocabQ').value = ''; filterVocab(); }],
  ]);
}
function renderFilterChips(host, items) {
  items = items.filter(Boolean);
  host.innerHTML = items.length ? 'Filter: ' + items.map(([l], i) => `<span class="fchip">${esc(l)}<button type="button" data-x="${i}" aria-label="${tl(`Filter ${esc(l)} entfernen`, `Remove filter ${esc(l)}`)}">×</button></span>`).join('') + (items.length > 1 ? ` <button type="button" class="linkbtn" data-x="all">${tl('alle entfernen', 'clear all')}</button>` : '') : '';
  host.onclick = e => { const b = e.target.closest('[data-x]'); if (!b) return; if (b.dataset.x === 'all') items.slice().reverse().forEach(([, f]) => f()); else items[+b.dataset.x][1](); };
}
const COLL = new Intl.Collator('de', { sensitivity: 'base' });
function renderMerges() {
  const M = S.model;
  if (!M.tok || !M.tok.merges) { $('#mergeCount').textContent = tl('Keine Merge-Regeln in dieser Datei (anderer Tokenizer-Typ).', 'No merge rules in this file (different tokenizer type).'); return; }
  if (!M._mergesReady) {
    M._mergesReady = true; M.tok.init(); ensureDisp();
    const mg = M.tok.merges, style = M.tok.byteEncode ? 'bpe' : 'spm', n = mg.length, md = new Array(n), mlen = new Uint16Array(n), mres = new Array(n);
    for (let i = 0; i < n; i++) {
      const w = mg[i], p = w.indexOf(' ', 1), a = w.slice(0, p), b = w.slice(p + 1), ta = tokenText(a, style), tb = tokenText(b, style), id = M.tok.t2id.get(a + b);
      md[i] = [ta, tb, id]; mres[i] = id !== undefined ? M._disp[id] : ta.text + tb.text; mlen[i] = Array.from(mres[i]).length;
    }
    Object.assign(M, { _md: md, _mlen: mlen, _mres: mres });
    S.mf = { bucket: null }; S.msort = { key: 'rank', dir: 1 };
    const show = t => t.bytes ? `<span class="ws">${esc(t.text)}</span>` : wsHTML(t.text);
    mergeList = new VList($('#mergeList'), {
      cols: 'cols-merge', get colsDef() { return [['rank', tl('Regel', 'Rule'), 1], [null, tl('Paar', 'Pair')], ['res', tl('Ergebnis', 'Result')], ['len', tl('Länge', 'Length'), 1], ['id', tl('Token-ID', 'Token ID'), 1]]; }, sort: () => S.msort,
      onSort: k => { S.msort = { key: k, dir: S.msort.key === k ? -S.msort.dir : (k === 'len' ? -1 : 1) }; mergeList.renderHead(); filterMerges(); },
      row: i => { const [a, b, id] = md[i]; return `<span class="r">${fmtInt(i)}</span><span class="mono">${show(a)} <span class="dim">+</span> ${show(b)}</span><span class="mono">${wsHTML(mres[i])}</span><span class="r">${mlen[i]}</span><span class="r">${id !== undefined ? fmtInt(id) : '–'}</span>`; },
      onClick: i => {
        const id = md[i][2]; if (id === undefined) return;
        const pair = `<span class="mono">${wsHTML(md[i][0].text)}</span> + <span class="mono">${wsHTML(md[i][1].text)}</span>`, res = `${fmtInt(id)} <span class="mono">${wsHTML(mres[i])}</span>`;
        selBar('mergeSel', tl(`<b>Regel ${fmtInt(i)}</b>: ${pair} → Token ${res} `, `<b>Rule ${fmtInt(i)}</b>: ${pair} → token ${res} `) + (M.E ? `<button type="button" class="btn" style="padding:3px 10px;font-size:13px" data-go="${id}">${tl(`Embedding-Zeile ${fmtInt(id)} →`, `Embedding row ${fmtInt(id)} →`)}</button>` : ''));
      },
    });
  } else if (mergeList) mergeList.relang();
  renderMergeChart(); filterMerges();
}
const MB = 25;
function renderMergeChart() {
  const M = S.model, n = M._md.length, size = Math.ceil(n / MB), avg = [], ex = [];
  for (let b = 0; b < MB; b++) {
    const lo = b * size, hi = Math.min(n, lo + size); let s = 0;
    for (let i = lo; i < hi; i++) s += M._mlen[i];
    avg.push(hi > lo ? s / (hi - lo) : 0);
    ex.push([lo, lo + Math.floor(size / 3), lo + Math.floor(2 * size / 3)].filter(i => i < hi).map(i => M._mres[i]));
  }
  const host = $('#mergeChart'), W = Math.max(300, host.clientWidth || 700), H = 190, padL = 40, padB = 24, padT = 10, ih = H - padB - padT, iw = W - padL - 6;
  const ymax = Math.ceil(Math.max(...avg) / 2) * 2 || 2, bw = iw / MB, cur = S.mf.bucket;
  let s = `<svg width="${W}" height="${H}" viewBox="0 0 ${W} ${H}" role="img" aria-label="${tl('Durchschnittliche Länge je Regelbereich', 'Average length per rule range')}">`;
  for (let v = 0; v <= ymax; v += 2) { const y = padT + ih - v / ymax * ih; s += `<line class="${v ? 'gridline' : 'baseline'}" x1="${padL}" x2="${W - 6}" y1="${y}" y2="${y}"/><text class="chart-text" x="${padL - 6}" y="${y + 4}" text-anchor="end">${v}</text>`; }
  avg.forEach((a, b) => {
    const h = a / ymax * ih, x = padL + b * bw + 1, on = cur == null || cur === b;
    s += `<rect class="pick" data-b="${b}" x="${x - 1}" y="${padT}" width="${bw}" height="${ih}" style="fill:transparent"/><path class="pick" data-b="${b}" d="${topRound(x, padT + ih - h, Math.max(1, bw - 2), h, 3)}" style="fill:var(--c-ffn);fill-opacity:${on ? 1 : 0.3}"/>`;
    if (b % 5 === 0) s += `<text class="chart-text" x="${x}" y="${H - 6}">${b * size >= 1000 ? fmtInt(Math.round(b * size / 1000)) + tl(' Tsd.', ' K') : b * size}</text>`;
  });
  s += `<text class="chart-text" x="${padL}" y="${padT + 10}" dx="4">${tl('Zeichen', 'characters')}</text></svg>`;
  host.innerHTML = s;
  const svg = host.firstChild;
  svg.onmousemove = e => { const p = e.target.closest('.pick'); if (!p) return hideTip(); const b = +p.dataset.b, r = `${fmtInt(b * size)}–${fmtInt(Math.min(n, (b + 1) * size) - 1)}`, eg = ex[b].map(x => esc(wsPlain(x))).join(' · ');
    showTip(tl(`Regeln ${r}<br>Ø <b>${nf(avg[b], 1)} Zeichen</b> pro Ergebnis<br><span class="m">z. B. ${eg}</span>`, `Rules ${r}<br>avg. <b>${nf(avg[b], 1)} characters</b> per result<br><span class="m">e.g. ${eg}</span>`), e); };
  svg.onmouseleave = hideTip;
  svg.onclick = e => { const p = e.target.closest('.pick'); if (!p) return; hideTip(); const b = +p.dataset.b; S.mf.bucket = cur === b ? null : b; renderMergeChart(); filterMerges(); };
}
function filterMerges() {
  const M = S.model, qRaw = $('#mergeQ').value, q = qRaw.toLowerCase(), md = M._md, n = md.length, size = Math.ceil(n / MB), b = S.mf.bucket, out = [];
  const lo = b != null ? b * size : 0, hi = b != null ? Math.min(n, lo + size) : n;
  for (let i = lo; i < hi; i++) { if (q && !M._mres[i].toLowerCase().includes(q)) continue; out.push(i); }
  const { key, dir } = S.msort;
  if (key !== 'rank' || dir < 0) {
    const cmp = key === 'res' ? (x, y) => COLL.compare(M._mres[x], M._mres[y]) || x - y
      : key === 'len' ? (x, y) => M._mlen[x] - M._mlen[y] || x - y
      : key === 'id' ? (x, y) => (md[x][2] ?? 1e9) - (md[y][2] ?? 1e9) || x - y : (x, y) => x - y;
    out.sort((x, y) => cmp(x, y) * dir);
  }
  mergeList.setItems(out);
  $('#mergeCount').textContent = tl(`${fmtInt(out.length)} von ${fmtInt(n)} Regeln`, `${fmtInt(out.length)} of ${fmtInt(n)} rules`);
  renderFilterChips($('#mergeFilters'), [
    b != null && [tl(`Regeln ${fmtInt(lo)}–${fmtInt(hi - 1)}`, `Rules ${fmtInt(lo)}–${fmtInt(hi - 1)}`), () => { S.mf.bucket = null; renderMergeChart(); filterMerges(); }],
    qRaw.trim() && [tl(`Suche „${qRaw.trim()}“`, `Search “${qRaw.trim()}”`), () => { $('#mergeQ').value = ''; filterMerges(); }],
  ]);
}
// "Wofür" column: [pattern, de, en], resolved with tl() when the table is drawn
const SPECIAL_INFO = [
  [/^<\|endoftext\|>$/, 'Ende eines Dokuments (trennt Texte im Training)', 'End of a document (separates texts in training)'],
  [/^<\|im_start\|>$/, 'Beginn einer Chat-Nachricht, danach die Rolle (system, user, assistant)', 'Start of a chat message, then the role (system, user, assistant)'],
  [/^<\|im_end\|>$/, 'Ende einer Chat-Nachricht: Hier hört das Modell auf', 'End of a chat message: the model stops here'],
  [/^<think>$/, 'Beginn des Nachdenkens (Reasoning) vor der Antwort', 'Start of thinking (reasoning) before the answer'], [/^<\/think>$/, 'Ende des Nachdenkens, danach die Antwort', 'End of thinking, then the answer'],
  [/^<tool_call>$/, 'Beginn eines Werkzeugaufrufs (z. B. Suche, Rechner)', 'Start of a tool call (e.g. search, calculator)'], [/^<\/tool_call>$/, 'Ende des Werkzeugaufrufs', 'End of the tool call'],
  [/^<tool_response>$/, 'Antwort eines Werkzeugs an das Modell', 'Reply from a tool to the model'], [/^<\/tool_response>$/, 'Ende der Werkzeug-Antwort', 'End of the tool reply'],
  [/^<\|vision_start\|>$/, 'Beginn eines Bildes', 'Start of an image'], [/^<\|vision_end\|>$/, 'Ende eines Bildes', 'End of an image'],
  [/^<\|image_pad\|>$/, 'Platzhalter für die Bild-Vektoren', 'Placeholder for the image vectors'], [/^<\|video_pad\|>$/, 'Platzhalter für Video-Bilder', 'Placeholder for video frames'], [/^<\|vision_pad\|>$/, 'Füllplatz im Bildbereich', 'Padding in the image area'],
  [/^<\|fim_(prefix|middle|suffix)\|>$/, 'Fill-in-the-Middle: Code-Lücken ergänzen', 'Fill-in-the-middle: filling gaps in code'], [/^<\|fim_pad\|>$/, 'Füllzeichen für Fill-in-the-Middle', 'Padding for fill-in-the-middle'],
  [/^<\|repo_name\|>$/, 'Name eines Code-Repositorys', 'Name of a code repository'], [/^<\|file_sep\|>$/, 'Trenner zwischen Dateien', 'Separator between files'],
  [/^<\|(object_ref|box|quad)_(start|end)\|>$/, 'Verweis auf eine Bildregion (Objekterkennung)', 'Reference to an image region (object detection)'],
  [/^<\|audio_(start|end)\|>$/, 'Rahmen um eine Audio-Eingabe', 'Start or end of an audio input'], [/^<\|audio_pad\|>$/, 'Platzhalter für Audio-Vektoren', 'Placeholder for audio vectors'], [/^<tts_/, 'Sprachausgabe (Text-to-Speech)', 'Speech output (text-to-speech)'],
  [/^<bos>$|^<s>$|^<\|begin_of_text\|>$/, 'Textanfang', 'Start of text'], [/^<eos>$|^<\/s>$|^<\|end_of_text\|>$/, 'Textende', 'End of text'], [/^<pad>$/, 'Auffüllen auf gleiche Länge', 'Padding to equal length'], [/^<unk>$/, 'Unbekanntes Zeichen', 'Unknown character'], [/^<mask>$/, 'Maskiertes Wort (Training)', 'Masked word (training)'],
  [/^<start_of_turn>$|^<\|turn>$/, 'Beginn eines Gesprächszugs', 'Start of a conversation turn'], [/^<end_of_turn>$|^<turn\|>$/, 'Ende eines Gesprächszugs', 'End of a conversation turn'],
  [/image|vision/i, 'Bildeingabe', 'Image input'], [/audio/i, 'Audioeingabe', 'Audio input'], [/tool/i, 'Werkzeugnutzung', 'Tool use'], [/think|reason/i, 'Nachdenken', 'Thinking'], [/pad/i, 'Füllzeichen', 'Padding'],
];
function renderSpecial() {
  const M = S.model;
  if (!M.types) { $('#specBody').innerHTML = `<tr><td colspan="4">${tl('Keine Token-Typen in dieser Datei.', 'No token types in this file.')}</td></tr>`; return; }
  if (!M._specRows) {
    const rows = []; let reserved = 0, bytes = 0;
    for (let i = 0; i < M.vocabSize; i++) { const t = M.types[i]; if (t === 2 || t === 3 || t === 4) rows.push(i); else if (t === 5) reserved++; else if (t === 6) bytes++; }
    M._specRows = rows; M._specMore = [reserved, bytes];
  }
  const [reserved, bytes] = M._specMore, ns = M._specRows.length, bos = M.meta['tokenizer.ggml.bos_token_id'], eos = M.meta['tokenizer.ggml.eos_token_id'];
  $('#specNote').innerHTML = [tl(`${fmtInt(ns)} Spezial-Tokens`, `${fmtInt(ns)} special tokens`), reserved && tl(`${fmtInt(reserved)} reserviert`, `${fmtInt(reserved)} reserved`), bytes && tl(`${fmtInt(bytes)} Byte-Tokens`, `${fmtInt(bytes)} byte tokens`),
    bos != null && tl(`Textanfang: ID ${fmtInt(bos)}`, `start of text: ID ${fmtInt(bos)}`), eos != null && tl(`Textende: ID ${fmtInt(eos)}`, `end of text: ID ${fmtInt(eos)}`)].filter(Boolean).join(' · ') + (reserved || bytes ? ' ' + ib('specCounts') : '');
  const so = S.spSort || { key: 'id', dir: 1 }, info = s => { const x = SPECIAL_INFO.find(([re]) => re.test(s)); return x ? tl(x[1], x[2]) : ''; };
  const cmp = so.key === 'tok' ? (a, b) => COLL.compare(M.tokens[a], M.tokens[b]) : so.key === 'type' ? (a, b) => M.types[a] - M.types[b] || a - b : (a, b) => a - b;
  const rows = M._specRows.slice().sort((a, b) => cmp(a, b) * so.dir);
  $('#specBody').innerHTML = rows.map(i => `<tr><td class="r">${fmtInt(i)}</td><td>${esc(M.tokens[i])}</td><td>${TOKTYPE[M.types[i]]}</td><td>${esc(info(M.tokens[i])) || '<span class="count">–</span>'}</td></tr>`).join('');
  document.querySelectorAll('#specHead th[aria-sort]').forEach(th => th.setAttribute('aria-sort', th.querySelector('button').dataset.sort === so.key ? (so.dir > 0 ? 'ascending' : 'descending') : 'none'));
}
function setTokSub(sub) {
  S.tokSub = sub;
  document.querySelectorAll('[data-sub]').forEach(b => b.setAttribute('aria-pressed', String(b.dataset.sub === sub)));
  for (const k of ['split', 'vocab', 'merges', 'special']) $('#ts-' + k).hidden = k !== sub;
  const M = S.model; if (!M) return;
  const run = () => { if (sub === 'vocab') renderVocab(); if (sub === 'merges') renderMerges(); if (sub === 'special') renderSpecial(); };
  if ((sub === 'vocab' && !M._vocabReady) || (sub === 'merges' && !M._mergesReady)) { $(sub === 'vocab' ? '#vocabCount' : '#mergeCount').textContent = tl('wird aufbereitet …', 'preparing …'); setTimeout(run, 30); }
  else run();
}

// =====================================================================
// EMBEDDING TABLE
// =====================================================================
let embView = null;
function embFetch(start, count) {
  const M = S.model, E = M.E;
  return M.source.readRowRange(E, start, count).then(raws => { updateReadStat(); return raws.map(r => decodeRow(E, r)); });
}
// status line of the matrix views in the example (demo) file
const hatchTxt = () => S.model.info.demo ? tl('Schraffiert: nicht im Beispiel enthalten', 'Hatched: not included in the example') : '';
function renderEmbed() {
  const M = S.model, E = M.E, msg = $('#embMsg');
  const sub = document.querySelector('#p-embed .subtabs');
  let none = $('#embNone');  // visible one-line note: #embExplain is only shown behind the ⓘ
  if (!E) {
    if (!none) { none = document.createElement('p'); none.id = 'embNone'; none.className = 'note'; sub.before(none); }
    none.hidden = false; none.innerHTML = tl('Keine lesbare Embedding-Tabelle (<span class="mono">token_embd.weight</span>) in dieser Datei.', 'No readable embedding table (<span class="mono">token_embd.weight</span>) in this file.');
    $('#embExplain').innerHTML = tl('Die Embedding-Tabelle fehlt, passt nicht zum Wörterbuch oder nutzt ein Format, das die Lupe nicht dekodieren kann.', 'The embedding table is missing, does not match the vocabulary, or uses a format GGUF-Lupe cannot decode.');
    sub.hidden = true; for (const k of ['table', 'nn', 'calc', 'map', 'cmp']) $('#es-' + k).hidden = true; return;
  }
  if (none) none.hidden = true;
  sub.hidden = false;
  const d = E.dims[0], V = E.dims[1];
  $('#embExplain').innerHTML = tl(`Jede Token-ID ist eine <b>Zeilennummer</b> in dieser Tabelle; jede Zeile hat <b>${fmtInt(d)} Zahlen</b>, eine pro <b>Dimension</b> (Spalte). Diese Zahlen hat das Modell im Training gelernt. Für deinen Text schlägt es eine Zeile pro Token nach (gleiches Token, gleiche Zeile): Das ist die Eingabe fürs Netz.`,
    `Every token ID is a <b>row number</b> in this table; each row has <b>${fmtInt(d)} numbers</b>, one per <b>dimension</b> (column). The model learned these numbers in training. For your text it looks up one row per token (same token, same row): that is the input to the network.`);
  $('#embFacts').innerHTML = [`<span class="chip">Tensor <b class="mono">${esc(E.name)}</b></span>`, tl(`<span class="chip">Form <b>${fmtInt(V)} × ${fmtInt(d)}</b></span>`, `<span class="chip">Shape <b>${fmtInt(V)} × ${fmtInt(d)}</b></span>`),
    tl(`<span class="chip">gespeichert als <b>${esc(E.typeName)}</b> · ${fmtBits(bpw(E.type))} Bit je Zahl</span>`, `<span class="chip">stored as <b>${esc(E.typeName)}</b> · ${fmtBits(bpw(E.type))} bits per number</span>`),
    tl(`<span class="chip"><b>${fmtParams(E.nElements)}</b> Zahlen · <b>${fmtBytes(E.nBytes)}</b></span>`, `<span class="chip"><b>${fmtParams(E.nElements)}</b> numbers · <b>${fmtBytes(E.nBytes)}</b></span>`)].join('');
  document.querySelectorAll('#embModeText,#embModeAll').forEach(b => b.setAttribute('aria-pressed', String((b.id === 'embModeText') === (S.embMode === 'text'))));
  if (!embView) embView = new MatrixView($('#embView'), { get aria() { return tl('Embedding-Tabelle', 'Embedding table'); }, onRow: r => pickEmb(r) });
  const tr = GGUF.TYPES[E.type];
  // getters: the view reads these texts when it draws, so they follow a language switch
  const base = () => ({ t: E, nRows: V, nCols: d, fetch: embFetch, rowWord: 'Token', colWord: 'Dimension', qk: tr[1], sub: subOf(E.typeName), typeName: E.typeName,
    get cornerText() { return tl('Token ╲ Dimension', 'Token ╲ dimension'); }, get missingText() { return hatchTxt(); } });
  const useText = S.embMode === 'text' && S.tokRes && S.tokRes.ids.length;
  if (S.embMode === 'text' && !useText) { msg.hidden = false; msg.textContent = tl('Noch kein Text: Tippe im Tab „Tokens“ einen Satz ein oder wähle „Ganze Tabelle“.', 'No text yet: type a sentence in the “Tokens” tab or show the whole table.'); }
  if (useText) {
    const ids = S.tokRes.ids.slice(0, 2000);
    msg.hidden = false;
    const txt = esc(S.text.length > 70 ? S.text.slice(0, 70) + '…' : S.text), k = fmtInt(ids.length), dd = fmtInt(d);
    msg.innerHTML = tl(`„<b>${txt}</b>“ → ${k} Tokens → <b>${k} × ${dd} Zahlen</b>, die Eingabe fürs Netz`, `“<b>${txt}</b>” → ${k} tokens → <b>${k} × ${dd} numbers</b>, the input to the network`);
    const wasText = embView.d && embView.d.rows;
    embView.setData(Object.assign(base(), { rows: ids, zoom: wasText ? embView.zoom : 'num', selRow: S.embA, label: (r, di) => `${di + 1}. ${wsPlain(disp(r))}` }));
    if (S.embFocus != null) { const di = S.embFocus; S.embFocus = null; embView.selRow = ids[di]; pickEmb(ids[di]); embView.scrollToRow(di); }
  } else if (S.embMode === 'all') {
    msg.hidden = true;
    const wasAll = embView.d && !embView.d.rows;
    embView.setData(Object.assign(base(), { label: r => wsPlain(disp(r)), zoom: wasAll ? embView.zoom : 'cells', selRow: S.embA }));
    if (S.embJumpTo != null) { const r = S.embJumpTo; S.embJumpTo = null; embView.selRow = r; embView.scrollToRow(r); }
  }
  renderEmbSel();
  setESub(S.eSub);
}
function jumpToRow(id) { S.embMode = 'all'; S.embJumpTo = id; S.eSub = 'table'; if (id !== S.embA) { S.embB = S.embA; S.embA = id; } setTab('embed'); }
function pickEmb(r) { if (r !== S.embA) { S.embB = S.embA; S.embA = r; } renderEmbSel(); if (S.eSub === 'cmp') drawCompare(); }
const PRESETS = [['King', 'Queen'], ['König', 'King'], ['Paris', 'Berlin'], ['Germany', 'Deutschland'], ['dog', 'cat'], ['Hund', 'Bilanz']];
function renderPresets() {
  const ok = PRESETS.map(([a, b]) => [lookupWord(a), lookupWord(b), a, b]).filter(x => x[0] >= 0 && x[1] >= 0);
  $('#tokPresets').innerHTML = ok.length ? tl('Beispiele: ', 'Examples: ') + ok.map(([a, b, x, y]) => `<button type="button" class="ui" data-a="${a}" data-b="${b}">${esc(x)} ↔ ${esc(y)}</button>`).join('') : '';
  if (S.embA == null && ok.length) { S.embA = ok[0][0]; S.embB = ok[0][1]; }
}
let cmpSeq = 0;
async function drawCompare() {
  const M = S.model, E = M.E; if (!E) return;
  const seq = ++cmpSeq;
  const get = async id => {
    if (id == null) return null;
    const cached = embView && embView.cache.get(id);
    if (cached) return { id, v: cached };
    const [raw] = await M.source.readRowRange(E, id, 1);
    const v = decodeRow(E, raw);
    return v ? { id, v } : { id, err: tl('Im eingebauten Beispiel nicht enthalten.', 'Not included in the built-in example.') };
  };
  const [a, b] = await Promise.all([get(S.embA), get(S.embB)]);
  if (seq !== cmpSeq) return;
  updateReadStat();
  const vs = [a, b].filter(x => x && x.v).map(x => x.v), sc = vs.length ? percentileAbs(vs, 0.99) : 1, T = theme();
  const one = (host, x, tag) => {
    if (!x) { host.innerHTML = `<div class="who"><span class="tok-tag">${tag}</span> ${tl('Klick auf eine zweite Zeile.', 'Click a second row.')}</div>`; return; }
    let nrm = 0; if (x.v) for (const q of x.v) nrm += q * q;
    const rid = fmtInt(x.id), more = x.v ? tl(` · ${fmtInt(x.v.length)} Zahlen · Länge ${sig(Math.sqrt(nrm), 3)}`, ` · ${fmtInt(x.v.length)} numbers · length ${sig(Math.sqrt(nrm), 3)}`) : '';
    host.innerHTML = `<div class="who"><span class="tok-tag">${tag}</span><b>${wsHTML(disp(x.id))}</b><span>${tl(`Zeile ${rid}`, `Row ${rid}`)}${more}</span></div>` + (x.err ? `<p class="count">${x.err}</p>` : '<canvas></canvas>');
    if (!x.v) return;
    const n = x.v.length, cols = n >= 2048 && n % 128 === 0 ? 128 : n % 64 === 0 ? 64 : Math.ceil(Math.sqrt(n * 3)), rows = Math.ceil(n / cols);
    const cv = host.querySelector('canvas'); cv.width = cols; cv.height = rows; cv.style.aspectRatio = `${cols} / ${rows}`;
    const ctx = cv.getContext('2d'), img = ctx.createImageData(cols, rows);
    for (let i = 0; i < Math.min(n, cols * rows); i++) { const o = i * 4, k = colorIdx(x.v[i], sc); img.data[o] = T.lut[k]; img.data[o + 1] = T.lut[k + 1]; img.data[o + 2] = T.lut[k + 2]; img.data[o + 3] = 255; }
    ctx.putImageData(img, 0, 0);
    cv.onmousemove = e => { const r = cv.getBoundingClientRect(), c = Math.floor((e.clientX - r.left) / r.width * cols), rr = Math.floor((e.clientY - r.top) / r.height * rows), i = rr * cols + c; if (i < 0 || i >= n) return hideTip(); showTip(`Dimension ${fmtInt(i)}: <b>${sig(x.v[i], 4)}</b>`, e); };
    cv.onmouseleave = hideTip;
  };
  one($('#embA'), a, 'A'); one($('#embB'), b, 'B');
  if (a && b && a.v && b.v) {
    let d = 0, x2 = 0, y2 = 0; for (let i = 0; i < a.v.length; i++) { d += a.v[i] * b.v[i]; x2 += a.v[i] ** 2; y2 += b.v[i] ** 2; }
    $('#cosBox').innerHTML = `<span class="big num">${nf(d / Math.sqrt(x2 * y2), 3)}</span><p>${tl('<b>Kosinus-Ähnlichkeit</b> · 1 = gleiche Richtung, 0 = unabhängig', '<b>Cosine similarity</b> · 1 = same direction, 0 = unrelated')} ${ib('embCosine')}</p>`;
  } else $('#cosBox').innerHTML = '';
}

// =====================================================================
// WEIGHTS
// =====================================================================
let wView = null;
const MAX_ROWS = 64;
function wExplain() {
  const M = S.model, n = M.tensors.filter(t => t.dims.length > 1).length;
  $('#wExplain').innerHTML = tl(`Neben der Embedding-Tabelle besteht das Modell aus vielen weiteren Zahlentabellen, den <b>Gewichtsmatrizen</b> (${fmtInt(n)} zweidimensionale in dieser Datei). Beim Rechnen wird jede <b>Zeile</b> mit dem Vektor eines Tokens verrechnet und ergibt eine Zahl der Ausgabe; jede <b>Spalte</b> gehört zu einer Eingangs-Dimension.`,
    `Besides the embedding table, the model consists of many more number tables, the <b>weight matrices</b> (${fmtInt(n)} two-dimensional ones in this file). During computation, each <b>row</b> is combined with a token’s vector and yields one output number; each <b>column</b> belongs to one input dimension.`);
}
function quickPicks() {
  const M = S.model, src = M.source;
  let names;
  if (src.kind === 'demo') names = Object.keys(src.sets).filter(n => M.byName.has(n) && n !== 'token_embd.weight');
  else {
    const pick = re => M.tensors.find(t => re.test(t.name));
    names = [...new Set([pick(/attn_q\.weight$/), pick(/attn_qkv\.weight$/), pick(/ffn_down\.weight$/), pick(/ffn_down_exps\.weight$/), pick(/^output\.weight$/), pick(/^output_norm\.weight$/)].filter(Boolean).map(t => t.name))];
  }
  // aria-current from S.sel, so re-rendering (language switch) keeps the highlight
  $('#quickPicks').innerHTML = (src.kind === 'demo' ? tl('Im Beispiel: ', 'In the example: ') : tl('Schnellwahl: ', 'Quick pick: ')) + names.map(n => `<button type="button" data-name="${esc(n)}" aria-current="${!!S.sel && S.sel.name === n}">${esc(n)}</button>`).join('');
}
function openTensor(t) { if (!t) return; if (t === S.model.E) { jumpToRow(S.embA ?? 0); return; } selectTensor(t); setTab('weights'); }
function tensorChips(t) {
  const ne0 = t.dims[0], R = fmtInt(Math.round(t.nElements / ne0)), C = fmtInt(ne0), full = t.dims.length > 2 ? ` (${t.dims.map(fmtInt).join(' × ')})` : '', bits = isFinite(bpw(t.type)) ? fmtBits(bpw(t.type)) : '';
  $('#lupeInfo').innerHTML = [
    `<span class="chip"><b>${COMPS[t.comp].label}</b></span>`,
    tl(`<span class="chip">Form <b>${R} Zeilen × ${C} Spalten</b>${full}</span>`, `<span class="chip">Shape <b>${R} rows × ${C} columns</b>${full}</span>`),
    tl(`<span class="chip">gespeichert als <b>${esc(t.typeName)}</b>${bits && ` · ${bits} Bit je Zahl`}</span>`, `<span class="chip">stored as <b>${esc(t.typeName)}</b>${bits && ` · ${bits} bits per number`}</span>`),
    tl(`<span class="chip"><b>${fmtParams(t.nElements)}</b> Zahlen · <b>${fmtBytes(t.nBytes)}</b></span>`, `<span class="chip"><b>${fmtParams(t.nElements)}</b> numbers · <b>${fmtBytes(t.nBytes)}</b></span>`),
  ].join('');
}
function selectTensor(t) {
  if (!t) return;
  const M = S.model;
  S.sel = t; S.blk = 0; S.sample = null;
  $('#tensorPick').value = t.name;
  document.querySelectorAll('#tensorTable tbody tr[aria-selected="true"]').forEach(r => r.removeAttribute('aria-selected'));
  const row = document.querySelector(`#tensorTable tbody tr[data-i="${t.idx}"]`); if (row) row.setAttribute('aria-selected', 'true');
  document.querySelectorAll('#quickPicks button').forEach(b => b.setAttribute('aria-current', String(b.dataset.name === t.name)));
  const tr = GGUF.TYPES[t.type], ne0 = t.dims[0], nRows = Math.round(t.nElements / ne0);
  tensorChips(t);
  const msg = $('#lupeMsg');
  const fail = html => { msg.innerHTML = html; msg.hidden = false; for (const k of ['table', 'dist', 'prof', 'store']) $('#ws-' + k).hidden = true; };
  if (!tr || !isFinite(t.rowBytes)) return fail(tl(`Für den Typ <b>${esc(t.typeName)}</b> ist kein Blockformat bekannt.`, `No known block format for type <b>${esc(t.typeName)}</b>.`));
  if (M.source.kind === 'demo' && !M.source.has(t)) {
    const names = Object.keys(M.source.sets).filter(n => M.byName.has(n) && n !== 'token_embd.weight');
    const btns = names.map(n => `<button type="button" class="btn" style="padding:2px 8px;font-size:12px;margin:2px" data-pick="${esc(n)}"><span class="mono">${esc(n)}</span></button>`).join(' ');
    return fail(tl(`<b>Nicht im Beispiel.</b> Für alle Tabellen die echte .gguf-Datei öffnen. Im Beispiel: ${btns}`, `<b>Not in the example.</b> Open the real .gguf file for all tables. In the example: ${btns}`));
  }
  if (!GGUF.canDequant(t.type)) return fail(tl(`<b>${esc(t.typeName)}</b> nutzt Gitter-Codebücher (E8/D4): Dekodieren ist nicht eingebaut.`, `<b>${esc(t.typeName)}</b> uses lattice codebooks (E8/D4): decoding is not built in.`));
  msg.hidden = true;
  if (!wView) wView = new MatrixView($('#wView'), { get aria() { return tl('Gewichtsmatrix', 'Weight matrix'); } });
  const isTok = (t.name === 'output.weight' || t === M.E) && t.dims[1] === M.vocabSize;
  // getters: the view reads these texts when it draws, so they follow a language switch
  wView.setData({
    t, nRows, nCols: ne0, label: isTok ? r => wsPlain(disp(r)) : null,
    get rowWord() { return isTok ? 'Token' : tl('Zeile', 'Row'); }, get colWord() { return tl('Spalte', 'Column'); }, get cornerText() { return isTok ? tl('Token ╲ Spalte', 'Token ╲ column') : tl('Zeile ╲ Spalte', 'Row ╲ column'); },
    fetch: (s, n) => M.source.readRowRange(t, s, n).then(raws => { updateReadStat(); return raws.map(r => decodeRow(t, r)); }),
    qk: tr[1], sub: subOf(t.typeName), typeName: t.typeName, zoom: wView.d ? wView.zoom : 'fit',
    get missingText() { return hatchTxt(); },
  });
  setWSub(S.wSub);
  loadSample();
}
let sampleSeq = 0;
async function loadSample() {
  const M = S.model, t = S.sel, src = M.source, seq = ++sampleSeq;
  if (!t || !canRead(t) || (src.kind === 'demo' && !src.has(t))) return;
  const ne0 = t.dims[0], nRows = Math.round(t.nElements / ne0);
  let R = Math.min(nRows, MAX_ROWS, Math.max(1, Math.floor(32e6 / t.rowBytes))), rows;
  if (src.kind === 'demo') { const g = src.sets[t.name][0]; R = Math.min(R, g.n); rows = Array.from({ length: R }, (_, i) => g.s + i); }
  else rows = S.rowMode === 'spread' && nRows > R ? Array.from({ length: R }, (_, i) => Math.floor(i * (nRows - 1) / (R - 1))) : Array.from({ length: R }, (_, i) => i);
  $('#rowsSpread').disabled = src.kind === 'demo' || nRows <= MAX_ROWS;
  const t0 = performance.now();
  const raws = await readRows(src, t, rows);
  if (seq !== sampleSeq) return;
  const vals = raws.map(r => decodeRow(t, r)).filter(Boolean), ms = performance.now() - t0;
  updateReadStat();
  S.sample = { t, rows, raws, vals, nRows, ms };
  readLine();
  renderWSub();
}
function readLine() {
  const s = S.sample; if (!s || s.t !== S.sel) return;
  const R = fmtInt(s.rows.length), ms = nf(s.ms, 1), by = fmtBytes(s.rows.length * s.t.rowBytes);
  $('#lupeRead').innerHTML = S.model.source.kind === 'demo' ? tl(`Stichprobe: <strong>${R} Zeilen</strong> aus dem Beispiel · ${ms} ms`, `Sample: <strong>${R} rows</strong> from the example · ${ms} ms`)
    : tl(`Stichprobe: <strong>${R} von ${fmtInt(s.nRows)} Zeilen</strong> · ${by} · <strong>${ms} ms</strong>`, `Sample: <strong>${R} of ${fmtInt(s.nRows)} rows</strong> · ${by} · <strong>${ms} ms</strong>`);
}
function setWSub(sub) {
  S.wSub = sub;
  document.querySelectorAll('[data-wsub]').forEach(b => b.setAttribute('aria-pressed', String(b.dataset.wsub === sub)));
  // re-run (tab shown again, language switch): rebuild the texts; a shown message is rebuilt by selectTensor
  if (!$('#lupeMsg').hidden) { if (S.sel) selectTensor(S.sel); return; }
  if (S.sel) tensorChips(S.sel); readLine();
  for (const k of ['table', 'dist', 'prof', 'store']) $('#ws-' + k).hidden = k !== sub;
  if (sub === 'table' && wView) wView.layout();
  renderWSub();
}
function renderWSub() {
  if (!S.sample || S.sample.t !== S.sel || !$('#lupeMsg').hidden || !S.sample.vals.length) return;
  if (S.wSub === 'dist') drawDist(); else if (S.wSub === 'prof') drawProf(); else if (S.wSub === 'store') drawStore();
}
function drawDist() {
  if (!S.sample) return;
  const { vals } = S.sample;
  let n = 0, sum = 0, sq = 0, mn = Infinity, mx = -Infinity, zeros = 0;
  for (const a of vals) for (let i = 0; i < a.length; i++) { const v = a[i]; if (!isFinite(v)) continue; n++; sum += v; sq += v * v; if (v < mn) mn = v; if (v > mx) mx = v; if (v === 0) zeros++; }
  const mean = sum / n, sd = Math.sqrt(Math.max(0, sq / n - mean * mean));
  let out3 = 0; for (const a of vals) for (let i = 0; i < a.length; i++) if (Math.abs(a[i] - mean) > 3 * sd) out3++;
  const distinct = (() => { const set = new Set(); for (let i = 0; i < Math.min(vals[0].length, 4096); i++) set.add(vals[0][i]); return set.size; })();
  const range = vals.length === 1 ? Math.max(Math.abs(mn), Math.abs(mx)) : percentileAbs(vals, 0.999);
  $('#histCap').innerHTML = tl(`<b>Wie oft kommt welcher Wert vor?</b> ${fmtInt(n)} Zahlen aus der Stichprobe`, `<b>How often does each value occur?</b> ${fmtInt(n)} numbers from the sample`) + ' ' + ib('weightHist');
  $('#hist').innerHTML = histogram(vals, range, mn, mx, S.histLog);
  $('#wStats').innerHTML = [[tl('Zahlen', 'Numbers'), fmtInt(n)], ['Minimum', sig(mn, 4)], ['Maximum', sig(mx, 4)], [tl('Mittelwert', 'Mean'), sig(mean, 3)], [tl('Standardabweichung', 'Standard deviation'), sig(sd, 3)], [tl('Exakt 0', 'Exactly 0'), pct(zeros, n)], [tl('Weiter als 3σ vom Mittel', 'More than 3σ from the mean'), pct(out3, n)], [tl('Verschiedene Werte (Zeile 1)', 'Distinct values (row 1)'), fmtInt(distinct)]]
    .map(([k, v]) => `<div><dt>${k}</dt><dd>${v}</dd></div>`).join('');
}
function histogram(vals, range, mn, mx, log) {
  const host = $('#hist'), W = Math.max(280, host.clientWidth || 600), H = 300, padB = 24, padT = 8, B = 121;
  const lo = -range, hi = range, cnt = new Float64Array(B); let outside = 0;
  for (const a of vals) for (let i = 0; i < a.length; i++) { const v = a[i]; if (v < lo || v > hi) { outside++; continue; } cnt[Math.min(B - 1, Math.floor((v - lo) / (hi - lo) * B))]++; }
  const f = log ? x => Math.log10(1 + x) : x => x, max = f(Math.max(...cnt)) || 1, bw = W / B, ih = H - padB - padT;
  let s = `<svg width="${W}" height="${H}" viewBox="0 0 ${W} ${H}" role="img" aria-label="${tl('Histogramm der Werte', 'Histogram of the values')}"><line class="baseline" x1="0" x2="${W}" y1="${H - padB}" y2="${H - padB}"/><line class="baseline" x1="${W / 2}" x2="${W / 2}" y1="${padT}" y2="${H - padB}"/>`;
  for (let i = 0; i < B; i++) { if (!cnt[i]) continue; const h = Math.max(1, f(cnt[i]) / max * ih), x = i * bw + (bw > 4 ? 1 : 0), w = Math.max(1, bw - (bw > 4 ? 2 : 0.5)); s += `<path class="hb" data-i="${i}" d="${topRound(x, H - padB - h, w, h, Math.min(2, w / 2))}" style="fill:var(--c-ffn)"/>`; }
  s += `<text class="chart-text" x="0" y="${H - 6}">−${sig(range)}</text><text class="chart-text" x="${W / 2}" y="${H - 6}" text-anchor="middle">0</text><text class="chart-text" x="${W}" y="${H - 6}" text-anchor="end">+${sig(range)}</text></svg>`;
  setTimeout(() => { const svg = host.querySelector('svg'); if (!svg) return; svg.onmousemove = e => { const p = e.target.closest('.hb'); if (!p) return hideTip(); const i = +p.dataset.i, a0 = lo + i * (hi - lo) / B, a = sig(a0, 3), z = sig(a0 + (hi - lo) / B, 3), c = fmtInt(cnt[i]); showTip(tl(`${a} bis ${z}: <b>${c}</b> Zahlen`, `${a} to ${z}: <b>${c}</b> numbers`), e); }; svg.onmouseleave = hideTip; });
  if (outside) s += `<p class="count">${tl(`${fmtInt(outside)} Werte außerhalb des Bereichs (Min ${sig(mn)}, Max ${sig(mx)})`, `${fmtInt(outside)} values outside the range (min ${sig(mn)}, max ${sig(mx)})`)}</p>`;
  return s;
}
function drawProf() {
  const { vals, t, rows } = S.sample, R = vals.length, ne0 = t.dims[0];
  if (R === 1) {
    $('#profCap').innerHTML = tl(`<b>Alle ${fmtInt(ne0)} Werte</b> dieses Vektors`, `<b>All ${fmtInt(ne0)} values</b> of this vector`) + (/norm/.test(t.name) ? ' ' + ib('weightNorm') : '');
    $('#prof').innerHTML = lineChart(vals[0], { signed: true, W: Math.max($('#prof').clientWidth || 800, 320) });
    bindLineHover($('#prof'), vals[0], 'Dimension', tl('Wert', 'Value'));
    $('#rowProfFig').hidden = true; return;
  }
  const prof = new Float32Array(ne0);
  for (const a of vals) for (let i = 0; i < ne0; i++) prof[i] += Math.abs(a[i]);
  for (let i = 0; i < ne0; i++) prof[i] /= R;
  $('#profCap').innerHTML = tl(`<b>Mittlerer Betrag je Spalte</b> · ${fmtInt(R)} Zeilen · ${fmtInt(ne0)} Eingangs-Dimensionen`, `<b>Mean magnitude per column</b> · ${fmtInt(R)} rows · ${fmtInt(ne0)} input dimensions`) + ' ' + ib('weightProfile');
  $('#prof').innerHTML = lineChart(prof, { signed: false, W: Math.max($('#prof').clientWidth || 800, 320), markMax: true });
  bindLineHover($('#prof'), prof, tl('Spalte', 'Column'), tl('mittl. |w|', 'mean |w|'));
  const rp = new Float32Array(R); vals.forEach((a, k) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * a[i]; rp[k] = Math.sqrt(s); });
  $('#rowProfFig').hidden = false;
  $('#rowProfCap').innerHTML = tl(`<b>Länge jeder Zeile</b> · ${fmtInt(R)} Stichprobenzeilen`, `<b>Length of each row</b> · ${fmtInt(R)} sampled rows`) + ' ' + ib('weightRowLen');
  $('#rowProf').innerHTML = lineChart(rp, { signed: false, W: Math.max($('#rowProf').clientWidth || 800, 320), xLabels: [fmtInt(rows[0]), fmtInt(rows[rows.length - 1])] });
  bindLineHover($('#rowProf'), rp, tl('Stichprobenzeile', 'Sample row'), tl('Länge', 'Length'), k => fmtInt(rows[k]));
}
function lineChart(arr, o) {
  const W = o.W, H = 190, padL = 56, padR = 10, padT = 24, padB = 22, iw = W - padL - padR, ih = H - padT - padB, n = arr.length;
  const P = Math.max(2, Math.min(n, Math.floor(iw))), mins = new Float32Array(P), maxs = new Float32Array(P);
  for (let p = 0; p < P; p++) { const a = Math.floor(p * n / P), b = Math.max(a + 1, Math.floor((p + 1) * n / P)); let lo = Infinity, hi = -Infinity; for (let i = a; i < Math.min(b, n); i++) { if (arr[i] < lo) lo = arr[i]; if (arr[i] > hi) hi = arr[i]; } mins[p] = lo === Infinity ? 0 : lo; maxs[p] = hi === -Infinity ? 0 : hi; }
  let lo = o.signed ? Math.min(0, ...mins) : 0, hi = Math.max(...maxs); if (hi === lo) hi = lo + 1;
  const step = niceStep((hi - lo) / 3); lo = Math.floor(lo / step) * step; hi = Math.ceil(hi / step) * step;
  const X = p => padL + (P === 1 ? 0 : p / (P - 1) * iw), Y = v => padT + ih - (v - lo) / (hi - lo) * ih;
  let s = `<svg width="${W}" height="${H}" viewBox="0 0 ${W} ${H}" role="img" aria-label="${tl('Linienverlauf', 'Line chart')}">`;
  for (let v = lo; v <= hi + step / 2; v += step) s += `<line class="${Math.abs(v) < step / 1e6 ? 'baseline' : 'gridline'}" x1="${padL}" x2="${W - padR}" y1="${Y(v)}" y2="${Y(v)}"/><text class="chart-text" x="${padL - 6}" y="${Y(v) + 4}" text-anchor="end">${sig(v, 3)}</text>`;
  let up = '', dn = '';
  for (let p = 0; p < P; p++) up += `${p ? 'L' : 'M'}${X(p).toFixed(1)},${Y(maxs[p]).toFixed(1)}`;
  for (let p = P - 1; p >= 0; p--) dn += `L${X(p).toFixed(1)},${Y(o.signed ? mins[p] : lo).toFixed(1)}`;
  s += `<path d="${up}${dn}Z" style="fill:var(--c-ffn);fill-opacity:.14"/><path d="${up}" style="fill:none;stroke:var(--c-ffn);stroke-width:${P > 400 ? 1.25 : 2};stroke-linejoin:round"/>`;
  if (o.signed) { let d = ''; for (let p = 0; p < P; p++) d += `${p ? 'L' : 'M'}${X(p).toFixed(1)},${Y(mins[p]).toFixed(1)}`; s += `<path d="${d}" style="fill:none;stroke:var(--c-ffn);stroke-width:1.25"/>`; }
  const xl = o.xLabels || ['0', fmtInt(n - 1)];
  s += `<text class="chart-text" x="${padL}" y="${H - 6}">${xl[0]}</text><text class="chart-text" x="${W - padR}" y="${H - 6}" text-anchor="end">${xl[1]}</text>`;
  if (o.markMax) {
    let bi = 0; for (let i = 1; i < n; i++) if (arr[i] > arr[bi]) bi = i;
    const p = Math.min(P - 1, Math.floor(bi * P / n)), x = X(p), y = Y(arr[bi]), anchor = x > W - 140 ? 'end' : 'start', dx = anchor === 'end' ? -8 : 8;
    s += `<circle cx="${x}" cy="${y}" r="4" style="fill:var(--c-ffn);stroke:var(--surface);stroke-width:2"/><text class="chart-text" x="${x + dx}" y="${Math.max(y - 8, 12)}" text-anchor="${anchor}" style="fill:var(--ink);paint-order:stroke;stroke:var(--surface);stroke-width:4px;stroke-linejoin:round">${tl(`Spalte ${fmtInt(bi)}: ${sig(arr[bi])}`, `Column ${fmtInt(bi)}: ${sig(arr[bi])}`)}</text>`;
  }
  return s + `<line class="xhair" x1="0" x2="0" y1="${padT}" y2="${padT + ih}" style="stroke:var(--ink-2);stroke-width:1" visibility="hidden"/></svg>`;
}
function bindLineHover(host, arr, xl, yl, fmtX = fmtInt) {
  const svg = host.querySelector('svg'); if (!svg) return;
  const W = +svg.getAttribute('width'), padL = 56, padR = 10, iw = W - padL - padR, n = arr.length, xh = svg.querySelector('.xhair');
  svg.onmousemove = e => {
    const b = svg.getBoundingClientRect(), x = (e.clientX - b.left) / b.width * W;
    if (x < padL || x > W - padR) { xh.setAttribute('visibility', 'hidden'); return hideTip(); }
    const f = (x - padL) / iw, P = Math.max(2, Math.min(n, Math.floor(iw))), p = Math.round(f * (P - 1));
    const a = Math.min(n - 1, Math.floor(p * n / P)), bnd = Math.min(n, Math.max(a + 1, Math.floor((p + 1) * n / P)));
    let bi = a; for (let i = a; i < bnd; i++) if (Math.abs(arr[i]) > Math.abs(arr[bi])) bi = i;
    xh.setAttribute('x1', x); xh.setAttribute('x2', x); xh.setAttribute('visibility', 'visible');
    showTip(`${xl} ${bnd - a > 1 ? `${fmtX(a)}–${fmtX(bnd - 1)}` : fmtX(a)}<br>${yl}: <b>${sig(arr[bi], 4)}</b>`, e);
  };
  svg.onmouseleave = () => { xh.setAttribute('visibility', 'hidden'); hideTip(); };
}
// block fields: [name, offset, bytes, [de, en] description]; FORMULA: [de, en]; both resolved with tl() in drawStore
const F_D = ['Skalierung (f16)', 'scale (f16)'], F_SD = ['Super-Skalierung (f16)', 'super-scale (f16)'], F_SM = ['Super-Minimum (f16)', 'super-minimum (f16)'], F_M = ['Minimum (f16)', 'minimum (f16)'],
  F_QH = ['5. Bit jedes Codes', '5th bit of each code'], F_Q4L = ['untere 4 Bit der Codes', 'lower 4 bits of the codes'];
const FIELDS = {
  Q4_0: [['d', 0, 2, F_D], ['qs', 2, 16, ['32 × 4-Bit-Codes, 2 pro Byte', '32 × 4-bit codes, 2 per byte']]],
  Q4_1: [['d', 0, 2, F_D], ['m', 2, 2, F_M], ['qs', 4, 16, ['32 × 4-Bit-Codes', '32 × 4-bit codes']]],
  Q5_0: [['d', 0, 2, F_D], ['qh', 2, 4, F_QH], ['qs', 6, 16, F_Q4L]],
  Q5_1: [['d', 0, 2, F_D], ['m', 2, 2, F_M], ['qh', 4, 4, F_QH], ['qs', 8, 16, F_Q4L]],
  Q8_0: [['d', 0, 2, F_D], ['qs', 2, 32, ['32 × 8-Bit-Codes (int8)', '32 × 8-bit codes (int8)']]],
  Q2_K: [['scales', 0, 16, ['16 × (4-Bit-Skala + 4-Bit-Minimum)', '16 × (4-bit scale + 4-bit minimum)']], ['qs', 16, 64, ['256 × 2-Bit-Codes', '256 × 2-bit codes']], ['d', 80, 2, F_SD], ['dmin', 82, 2, F_SM]],
  Q3_K: [['hmask', 0, 32, ['oberes Bit der 3-Bit-Codes', 'high bit of the 3-bit codes']], ['qs', 32, 64, ['untere 2 Bit der Codes', 'lower 2 bits of the codes']], ['scales', 96, 12, ['16 × 6-Bit-Teilskalen, gepackt', '16 × 6-bit sub-scales, packed']], ['d', 108, 2, F_SD]],
  Q4_K: [['d', 0, 2, F_SD], ['dmin', 2, 2, F_SM], ['scales', 4, 12, ['8 × (6-Bit-Skala + 6-Bit-Minimum), gepackt', '8 × (6-bit scale + 6-bit minimum), packed']], ['qs', 16, 128, ['256 × 4-Bit-Codes, 2 pro Byte', '256 × 4-bit codes, 2 per byte']]],
  Q5_K: [['d', 0, 2, F_SD], ['dmin', 2, 2, F_SM], ['scales', 4, 12, ['8 × (6-Bit-Skala + 6-Bit-Minimum)', '8 × (6-bit scale + 6-bit minimum)']], ['qh', 16, 32, F_QH], ['qs', 48, 128, F_Q4L]],
  Q6_K: [['ql', 0, 128, ['untere 4 Bit der 6-Bit-Codes', 'lower 4 bits of the 6-bit codes']], ['qh', 128, 64, ['obere 2 Bit der Codes', 'upper 2 bits of the codes']], ['scales', 192, 16, ['16 × 8-Bit-Teilskalen (int8)', '16 × 8-bit sub-scales (int8)']], ['d', 208, 2, F_SD]],
  Q8_K: [['d', 0, 4, ['Skalierung (f32)', 'scale (f32)']], ['qs', 4, 256, ['256 × 8-Bit-Codes', '256 × 8-bit codes']], ['bsums', 260, 32, ['Teilsummen (nur für Matmul)', 'partial sums (for matmul only)']]],
  IQ4_NL: [['d', 0, 2, F_D], ['qs', 2, 16, ['32 × 4-Bit-Indizes in eine nichtlineare Wertetabelle', '32 × 4-bit indices into a non-linear value table']]],
  IQ4_XS: [['d', 0, 2, F_SD], ['scales_h', 2, 2, ['obere Skalen-Bits', 'upper scale bits']], ['scales_l', 4, 4, ['untere Skalen-Bits', 'lower scale bits']], ['qs', 8, 128, ['256 × 4-Bit-Indizes (nichtlineare Tabelle)', '256 × 4-bit indices (non-linear table)']]],
  MXFP4: [['e', 0, 1, ['gemeinsamer Exponent (E8M0)', 'shared exponent (E8M0)']], ['qs', 1, 16, ['32 × FP4-Werte (E2M1)', '32 × FP4 values (E2M1)']]],
  NVFP4: [['scales', 0, 4, ['4 × Skala (FP8 E4M3), je 16 Werte', '4 × scale (FP8 E4M3), one per 16 values']], ['qs', 4, 32, ['64 × FP4-Werte (E2M1)', '64 × FP4 values (E2M1)']]],
  TQ2_0: [['qs', 0, 64, ['256 × 2-Bit-Ternärcodes (−1, 0, +1)', '256 × 2-bit ternary codes (−1, 0, +1)']], ['d', 64, 2, F_D]],
};
const FORMULA = {
  Q4_0: ['Gewicht = d · (q − 8), q ist ein 4-Bit-Code (0–15).', 'Weight = d · (q − 8); q is a 4-bit code (0–15).'], Q4_1: ['Gewicht = d · q + m, q ist ein 4-Bit-Code.', 'Weight = d · q + m; q is a 4-bit code.'],
  Q5_0: ['Gewicht = d · (q − 16), q ist ein 5-Bit-Code aus qs + qh.', 'Weight = d · (q − 16); q is a 5-bit code from qs + qh.'], Q5_1: ['Gewicht = d · q + m, q ist ein 5-Bit-Code.', 'Weight = d · q + m; q is a 5-bit code.'],
  Q8_0: ['Gewicht = d · q, q ist eine 8-Bit-Ganzzahl (−128 bis 127).', 'Weight = d · q; q is an 8-bit integer (−128 to 127).'],
  Q2_K: ['16 Unterblöcke à 16 Zahlen. Gewicht = d · Skala · q − dmin · Minimum, q ist 2 Bit groß.', '16 sub-blocks of 16 numbers. Weight = d · scale · q − dmin · minimum; q has 2 bits.'],
  Q3_K: ['16 Unterblöcke à 16 Zahlen. Gewicht = d · (Skala − 32) · q, q ist ein 3-Bit-Code mit Vorzeichen.', '16 sub-blocks of 16 numbers. Weight = d · (scale − 32) · q; q is a signed 3-bit code.'],
  Q4_K: ['8 Unterblöcke à 32 Zahlen. Gewicht = d · Skala · q − dmin · Minimum, q ist ein 4-Bit-Code (0–15): nur 16 mögliche Werte pro Unterblock.', '8 sub-blocks of 32 numbers. Weight = d · scale · q − dmin · minimum; q is a 4-bit code (0–15): only 16 possible values per sub-block.'],
  Q5_K: ['8 Unterblöcke à 32 Zahlen. Gewicht = d · Skala · q − dmin · Minimum, q ist ein 5-Bit-Code.', '8 sub-blocks of 32 numbers. Weight = d · scale · q − dmin · minimum; q is a 5-bit code.'],
  Q6_K: ['16 Unterblöcke à 16 Zahlen. Gewicht = d · Skala · (q − 32), q ist ein 6-Bit-Code aus ql (4 Bit) + qh (2 Bit): 64 mögliche Werte pro Unterblock.', '16 sub-blocks of 16 numbers. Weight = d · scale · (q − 32); q is a 6-bit code from ql (4 bits) + qh (2 bits): 64 possible values per sub-block.'],
  IQ4_NL: ['Gewicht = d · Tabelle[q]: Die 16 Stufen sind der Glockenform der Gewichte angepasst.', 'Weight = d · table[q]: the 16 levels follow the bell shape of the weights.'], IQ4_XS: ['Wie IQ4_NL, aber mit 6-Bit-Teilskalen je 32 Zahlen.', 'Like IQ4_NL, but with 6-bit sub-scales per 32 numbers.'],
  MXFP4: ['Gewicht = 2^(e−128) · FP4-Wert: vier Bit pro Gewicht als winzige Gleitkommazahl.', 'Weight = 2^(e−128) · FP4 value: four bits per weight as a tiny floating-point number.'], NVFP4: ['Gewicht = Skala · FP4-Wert, eine FP8-Skala je 16 Zahlen.', 'Weight = scale · FP4 value, one FP8 scale per 16 numbers.'], TQ2_0: ['Gewicht = d · (q − 1): nur −d, 0 oder +d.', 'Weight = d · (q − 1): only −d, 0 or +d.'],
};
function drawStore() {
  if (!S.sample) return;
  const { t, raws, vals } = S.sample, tn = t.typeName, tr = GGUF.TYPES[t.type], QK = tr[1], BS = tr[2], plain = QK === 1;
  const rowRaw = raws.find(Boolean), rowVals = vals[0]; if (!rowRaw || !rowVals) return;
  const per = plain ? 32 : QK, nBlk = Math.ceil(t.dims[0] / per);
  S.blk = Math.max(0, Math.min(nBlk - 1, S.blk));
  const off = plain ? S.blk * 32 * BS : S.blk * BS, nb = plain ? Math.min(32 * BS, rowRaw.length - off) : BS;
  const bytes = rowRaw.subarray(off, off + nb), v0 = S.blk * per, bv = rowVals.subarray(v0, Math.min(rowVals.length, v0 + per));
  const bk = fmtInt(S.blk + 1), nk = fmtInt(nBlk), rw = fmtInt(S.sample.rows[0]), span = `${fmtInt(v0)}–${fmtInt(v0 + bv.length - 1)}`;
  $('#blkPos').textContent = tl(`Block ${bk} von ${nk} · Zeile ${rw} · Zahlen ${span}`, `Block ${bk} of ${nk} · row ${rw} · numbers ${span}`);
  $('#blkPrev').disabled = S.blk === 0; $('#blkNext').disabled = S.blk >= nBlk - 1;
  const fields = plain ? [[tl('Werte', 'values'), 0, nb, [`${nb / BS} Zahlen à ${BS * 8} Bit (${tn}), direkt gespeichert`, `${nb / BS} numbers of ${BS * 8} bits each (${tn}), stored directly`]]] : (FIELDS[tn] || [[tl('Daten', 'data'), 0, BS, ['Codes und Skalen', 'codes and scales']]]);
  const fOf = i => fields.findIndex(f => i >= f[1] && i < f[1] + f[2]);
  let hx = '';
  for (let r = 0; r < nb; r += 16) {
    hx += `<div class="row"><span class="off">${r.toString(16).padStart(4, '0')}</span>`;
    for (let i = r; i < Math.min(nb, r + 16); i++) { const f = fOf(i); hx += `<span class="b${f >= 0 ? ' f' + (f % 5) : ''}">${bytes[i].toString(16).padStart(2, '0')}</span>`; }
    hx += '</div>';
  }
  $('#hex').innerHTML = hx;
  const q = fmtInt(QK), by = fmtInt(BS), bits = fmtBits(BS * 8 / QK), nbk = fmtInt(t.nElements / QK);
  $('#blockCap').innerHTML = plain
    ? tl(`<b>So liegen die Zahlen in der Datei</b> · ${tn}: ${BS * 8} Bit pro Zahl`, `<b>How the numbers are stored in the file</b> · ${tn}: ${BS * 8} bits per number`)
    : tl(`<b>So passen ${q} Zahlen in ${by} Bytes</b> · <b>${bits} Bit pro Zahl</b> · ${nbk} Blöcke im Tensor`, `<b>How ${q} numbers fit into ${by} bytes</b> · <b>${bits} bits per number</b> · ${nbk} blocks in this tensor`) + ' ' + ib('weightBlock');
  const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength), f16 = o => GGUF.F16[dv.getUint16(o, true)];
  $('#fields').innerHTML = fields.map((f, i) => {
    let val = '';
    if (!plain && f[2] === 2 && /^(d|dmin|m)$/.test(f[0])) val = ` = <code>${sig(f16(f[1]), 4)}</code>`;
    if (f[0] === 'd' && f[2] === 4) val = ` = <code>${sig(dv.getFloat32(f[1], true), 4)}</code>`;
    if (tn === 'MXFP4' && f[0] === 'e') val = ` = 2^${bytes[0] - 128}`;
    return `<div><i class="f${i % 5}"></i><code>${esc(f[0])}</code><span>${tl(`${f[2]} Byte`, `${f[2]} ${f[2] === 1 ? 'byte' : 'bytes'}`)} · ${esc(tl(...f[3]))}${val}</span></div>`;
  }).join('') + (FORMULA[tn] ? `<p style="margin-top:4px">${esc(tl(...FORMULA[tn]))}</p>` : '');
  const sub = plain ? 0 : subOf(tn), s = percentileAbs([rowVals], 0.99), T = theme();
  $('#valsCap').innerHTML = tl(`<b>Daraus dekodiert: ${fmtInt(bv.length)} Zahlen</b>`, `<b>Decoded: ${fmtInt(bv.length)} numbers</b>`) + (sub ? tl(` · dunkler Strich: Beginn eines Unterblocks (je ${sub} Zahlen eine Skala)`, ` · dark line: start of a sub-block (one scale per ${sub} numbers)`) : '');
  $('#vals').innerHTML = Array.from(bv).map((v, i) => { const k = colorIdx(v, s), r = T.lut[k], g = T.lut[k + 1], b = T.lut[k + 2], ink = (0.299 * r + 0.587 * g + 0.114 * b) > 150 ? '#0f1520' : '#fff'; return `<span class="${sub && i % sub === 0 ? 'sb' : ''}" style="background:rgb(${r},${g},${b});color:${ink}" title="${tl(`Zahl ${fmtInt(v0 + i)}: ${sig(v, 4)}`, `Number ${fmtInt(v0 + i)}: ${sig(v, 4)}`)}">${sig(v, 2)}</span>`; }).join('');
}

// ---------- embedding sub-views ----------
function renderEmbSel() {
  let bar = $('#embSel');
  if (!bar) { bar = document.createElement('div'); bar.id = 'embSel'; bar.className = 'note'; $('#embView').after(bar); }
  const M = S.model, a = S.embA, b = S.embB;
  if (a == null || !M.E) { bar.hidden = true; return; }
  bar.hidden = false;
  const n = fmtInt(a), ta = esc(disp(a).trim() || disp(a)), tb = b != null ? esc(disp(b).trim() || disp(b)) : '';
  bar.innerHTML = `<span${b != null ? '' : ` title="${tl('Eine zweite Zeile anklicken, um zwei zu vergleichen', 'Click a second row to compare the two')}"`}>${tl(`Ausgewählt: Zeile <b>${n}</b>`, `Selected: row <b>${n}</b>`)} <span class="mono">${wsHTML(disp(a))}</span></span>. <button type="button" class="btn" style="padding:3px 10px;font-size:13px" id="selNN">${tl(`Ähnliche Wörter zu <span class="mono">${ta}</span>`, `Similar words to <span class="mono">${ta}</span>`)}</button>` + (b != null ? ` <button type="button" class="btn" style="padding:3px 10px;font-size:13px" id="selCmp">${tl(`Mit <span class="mono">${tb}</span> vergleichen`, `Compare with <span class="mono">${tb}</span>`)}</button>` : '');
  $('#selNN').onclick = () => openNN(a);
  if (b != null) $('#selCmp').onclick = () => setESub('cmp');
}
function setESub(sub) {
  S.eSub = sub;
  document.querySelectorAll('[data-esub]').forEach(b => b.setAttribute('aria-pressed', String(b.dataset.esub === sub)));
  for (const k of ['table', 'nn', 'calc', 'map', 'cmp']) $('#es-' + k).hidden = k !== sub;
  if (!S.model || !S.model.E) return;
  if (sub === 'calc') renderCalcTab();
  if (sub === 'map') renderMapTab();
  if (sub === 'table' && embView) embView.layout();
  if (sub === 'nn') renderNNTab();
  if (sub === 'cmp') { renderPresets(); drawCompare(); }
}
function openNN(id) { S.nnId = id; S.eSub = 'nn'; if (S.tab !== 'meaning') setTab('meaning'); else setESub('nn'); }

// ---------- nearest neighbours (all cores via Web Workers) ----------
const nnWords = () => tl(['König', 'Paris', 'Geld', 'audit', 'Hund', 'Bilanz', 'Wirtschaft', 'Prüfung'], ['King', 'Paris', 'money', 'audit', 'dog', 'Germany', 'auditor', 'financial']);
const NN_WORKER = `
let buf = null;
self.onmessage = async (e) => {
  const { rid, file, t, start, end, q, K } = e.data;
  try {
    const d = t.d, rows = Math.max(32, Math.floor(3e6 / t.rowBytes));
    if (!buf || buf.length < rows * d) buf = new Float32Array(rows * d);
    let top = [], minTop = -Infinity;
    for (let s = start; s < end; s += rows) {
      const n = Math.min(rows, end - s);
      const raw = new Uint8Array(await file.slice(t.absOffset + s * t.rowBytes, t.absOffset + (s + n) * t.rowBytes).arrayBuffer());
      const all = GGUF.dequant(t.type, raw, n * d / t.qk, buf);
      for (let r = 0; r < n; r++) {
        const o = r * d; let dot = 0, nn = 0;
        for (let i = 0; i < d; i++) { const x = all[o + i]; dot += q[i] * x; nn += x * x; }
        const c = nn > 0 ? dot / Math.sqrt(nn) : 0;
        if (top.length < K || c > minTop) { top.push([c, s + r]); if (top.length >= 2 * K) { top.sort((a, b) => b[0] - a[0]); top.length = K; minTop = top[K - 1][0]; } }
      }
      self.postMessage({ rid, progress: n });
    }
    top.sort((a, b) => b[0] - a[0]); if (top.length > K) top.length = K;
    self.postMessage({ rid, top });
  } catch (err) { self.postMessage({ rid, error: String(err && err.message || err) }); }
};`;
let nnPool = null, nnBusy = false, nnSeq = 0;
function getPool() {
  if (nnPool !== null) return nnPool;
  try {
    const src = document.getElementById('core-src').textContent + '\n' + NN_WORKER;
    const url = URL.createObjectURL(new Blob([src], { type: 'text/javascript' }));
    const n = Math.max(2, Math.min(16, navigator.hardwareConcurrency || 4));
    nnPool = Array.from({ length: n }, () => new Worker(url));
  } catch (e) { console.warn('Web Worker nicht verfügbar, rechne im Hauptthread', e); nnPool = false; }
  return nnPool;
}
function killPool() { if (nnPool) nnPool.forEach(w => w.terminate()); nnPool = null; }
async function nnScanMain(M, t, q, K, onP) {
  const V = M.E.dims[1], d = t.d, rows = Math.max(32, Math.floor(3e6 / t.rowBytes)), buf = new Float32Array(rows * d);
  let top = [], minTop = -Infinity;
  for (let s = 0; s < V; s += rows) {
    const n = Math.min(rows, V - s);
    const raw = new Uint8Array(await M.source.file.slice(t.absOffset + s * t.rowBytes, t.absOffset + (s + n) * t.rowBytes).arrayBuffer());
    const all = GGUF.dequant(t.type, raw, n * d / t.qk, buf);
    for (let r = 0; r < n; r++) {
      const o = r * d; let dot = 0, nn = 0;
      for (let i = 0; i < d; i++) { const x = all[o + i]; dot += q[i] * x; nn += x * x; }
      const c = nn > 0 ? dot / Math.sqrt(nn) : 0;
      if (top.length < K || c > minTop) { top.push([c, s + r]); if (top.length >= 2 * K) { top.sort((a, b) => b[0] - a[0]); top.length = K; minTop = top[K - 1][0]; } }
    }
    onP((s + n) / V);
  }
  return top.sort((a, b) => b[0] - a[0]).slice(0, K);
}
function pca2(vs) {
  const n = vs.length, d = vs[0].length;
  const X = vs.map(v => { let s = 0; for (let i = 0; i < d; i++) s += v[i] * v[i]; s = Math.sqrt(s) || 1; const o = new Float64Array(d); for (let i = 0; i < d; i++) o[i] = v[i] / s; return o; });
  const mean = new Float64Array(d); for (const x of X) for (let i = 0; i < d; i++) mean[i] += x[i] / n;
  for (const x of X) for (let i = 0; i < d; i++) x[i] -= mean[i];
  const G = X.map(a => X.map(b => { let s = 0; for (let k = 0; k < d; k++) s += a[k] * b[k]; return s; }));
  const eig = Mx => {
    let v = Array.from({ length: n }, (_, i) => Math.sin(i + 1) + 0.5), lam = 0;
    for (let it = 0; it < 300; it++) { const w = v.map((_, i) => Mx[i].reduce((s, x, j) => s + x * v[j], 0)); lam = Math.hypot(...w) || 1; v = w.map(x => x / lam); }
    return [lam, v];
  };
  const [l1, v1] = eig(G), G2 = G.map((row, i) => row.map((x, j) => x - l1 * v1[i] * v1[j])), [l2, v2] = eig(G2);
  return v1.map((_, i) => [v1[i] * Math.sqrt(l1), v2[i] * Math.sqrt(Math.max(0, l2))]);
}
async function scanTop(q, K, onProgress) {
  const M = S.model, E = M.E, d = E.dims[0], V = E.dims[1], t = { absOffset: E.absOffset, rowBytes: E.rowBytes, type: E.type, d, qk: GGUF.TYPES[E.type][1] };
  const t0 = performance.now();
  if (nnBusy) killPool();
  nnBusy = true;
  let parts, cores = 1, done = 0;
  try {
    const pool = getPool();
    if (!pool) throw new Error('no workers');
    cores = pool.length;
    const per = Math.ceil(V / pool.length), rid = Math.random();
    parts = await Promise.all(pool.map((w, i) => new Promise((res, rej) => {
      const start = i * per, end = Math.min(V, start + per);
      if (start >= end) return res([]);
      w.onmessage = e => { const m = e.data; if (m.rid !== rid) return; if (m.progress) { done += m.progress; onProgress && onProgress(done / V); } else if (m.top) res(m.top); else if (m.error) rej(new Error(m.error)); };
      w.onerror = ev => { if (ev.preventDefault) ev.preventDefault(); rej(new Error(ev.message || 'Worker-Fehler')); };
      w.postMessage({ rid, file: M.source.file, t, start, end, q, K });
    })));
  } catch (e) {
    if (e.message !== 'no workers') { console.warn('Worker-Suche fehlgeschlagen, rechne im Hauptthread', e); killPool(); nnPool = false; }
    cores = 1; parts = [await nnScanMain(M, t, q, K, p => onProgress && onProgress(p))];
  } finally { nnBusy = false; }
  M.source.weightBytes += E.nBytes;
  return { top: parts.flat().sort((a, b) => b[0] - a[0]).slice(0, K), cores, ms: performance.now() - t0 };
}
const unitVec = v => { let n = 0; for (const x of v) n += x * x; n = Math.sqrt(n) || 1; return Float32Array.from(v, x => x / n); };
async function nnCompute(id, onProgress) {
  const M = S.model, E = M.E;
  M._nn = M._nn || new Map();
  if (M._nn.has(id)) return M._nn.get(id);
  if (M.source.kind === 'demo') { const pre = M.info.demo.nn && M.info.demo.nn[id]; return pre ? { top: pre.top, xy: pre.xy, demo: true } : null; }
  const [raw] = await M.source.readRowRange(E, id, 1), qv = decodeRow(E, raw);
  if (isSrv()) {
    await srvEnsure(onProgress ? (p) => onProgress(p) : null);
    const j = await srvJson('/api/nn', { path: M.source.path, id, k: 24 }), vecs = [qv];
    for (const x of j.ids) { const [rw] = await M.source.readRowRange(E, x, 1); vecs.push(decodeRow(E, rw)); }
    const res = { top: j.ids.map((x, k) => [x, j.sims[k]]), xy: pca2(vecs), ms: j.ms, srv: j.device };
    M._nn.set(id, res); return res;
  }
  const r = await scanTop(unitVec(qv), 40, onProgress);
  const best = r.top.filter(x => x[1] !== id).slice(0, 24);
  const vecs = [qv];
  for (const [, j] of best) { const [rw] = await M.source.readRowRange(E, j, 1); vecs.push(decodeRow(E, rw)); }
  const res = { top: best.map(([c, j]) => [j, c]), xy: pca2(vecs), ms: r.ms, cores: r.cores };
  M._nn.set(id, res); updateReadStat();
  return res;
}
function renderNNTab() {
  const M = S.model, ex = nnWords().map(w => [w, lookupWord(w)]).filter(([, id]) => id >= 0 && (M.source.kind !== 'demo' || (M.info.demo.nn && M.info.demo.nn[id])));
  $('#nnExamples').innerHTML = ex.length ? tl('Beispiele: ', 'Examples: ') + ex.map(([w, id]) => `<button type="button" class="ui" data-id="${id}">${esc(w)}</button>`).join('') : '';
  if (S.nnId == null && ex.length) S.nnId = ex[0][1];
  if (S.nnId != null) nnSearch(S.nnId, { list: $('#nnList'), map: $('#nnMap'), status: $('#nnStatus'), prog: $('#nnProg'), input: $('#nnQ') });
}
async function nnSearch(id, H) {
  const M = S.model, seq = ++nnSeq;
  S.nnId = id;
  if (H.input) H.input.value = disp(id).trim() || disp(id);
  if (M.source.kind !== 'demo' && !(M._nn && M._nn.has(id))) {
    H.prog.hidden = false; H.prog.firstChild.style.width = '0%';
    const n = fmtInt(M.E.dims[1]);
    H.status.innerHTML = tl(`Durchsuche ${n} Zeilen …`, `Searching ${n} rows …`);
  }
  let res;
  try { res = await nnCompute(id, p => { if (seq === nnSeq) { H.prog.firstChild.style.width = (p * 100).toFixed(1) + '%'; } }); }
  catch (e) { console.error(e); H.prog.hidden = true; H.status.textContent = tl('Suche fehlgeschlagen: ', 'Search failed: ') + srvMsg(e.message); return; }
  if (seq !== nnSeq) return;
  H.prog.hidden = true;
  if (!res) {
    H.status.innerHTML = tl('Im Beispiel nur für die vorgeschlagenen Wörter. Für jedes Wort: <b>echte .gguf-Datei öffnen</b>.', 'In the example, only for the suggested words. For any word: <b>open a real .gguf file</b>.');
    H.list.innerHTML = ''; H.map.innerHTML = ''; return;
  }
  const E = M.E, n = fmtInt(E.dims[1]);
  if (res.srv) { const ms = nf(res.ms, 0), dev = esc(res.srv); H.status.innerHTML = tl(`${n} Zeilen verglichen · <strong>${ms} ms</strong> · Lupe-Server (${dev})`, `${n} rows compared · <strong>${ms} ms</strong> · Lupe server (${dev})`); }
  else if (res.demo) H.status.innerHTML = tl(`${n} Zeilen verglichen · vorberechnet`, `${n} rows compared · precomputed`);
  else { const by = fmtBytes(E.nBytes), s = nf((res.ms || 0) / 1000, 1), c = fmtInt(res.cores), one = res.cores === 1; H.status.innerHTML = tl(`${n} Zeilen (${by}) verglichen · <strong>${s} s</strong> · <strong>${c} ${one ? 'Kern' : 'Kerne'}</strong>`, `${n} rows (${by}) compared · <strong>${s} s</strong> · <strong>${c} ${one ? 'core' : 'cores'}</strong>`); }
  renderNNList(H.list, id, res); renderNNMap(H.map, id, res);
}
function renderNNList(host, id, res) {
  const max = res.top.length ? Math.max(...res.top.map(x => x[1])) : 1;
  host.innerHTML = `<button type="button" class="nn-row q" data-id="${id}"><span class="rk">★</span><span class="w">${wsHTML(disp(id))}</span><span class="bt"><i style="width:100%"></i></span><span class="v">${nf(1, 2)}</span></button>`
    + res.top.map(([j, c], k) => `<button type="button" class="nn-row" data-id="${j}" title="ID ${fmtInt(j)}"><span class="rk">${k + 1}</span><span class="w">${wsHTML(disp(j))}</span><span class="bt"><i style="width:${Math.max(2, c / max * 92)}%"></i></span><span class="v">${nf(c, 2)}</span></button>`).join('');
}
function renderNNMap(host, id, res) {
  const ids = [id, ...res.top.map(x => x[0])], cos = [1, ...res.top.map(x => x[1])], xy = res.xy.slice(0, ids.length);
  const W = 600, H = 460, pad = 26, padL = 16;
  const xs = xy.map(p => p[0]), ys = xy.map(p => p[1]);
  const x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys);
  const X = v => padL + (x1 > x0 ? (v - x0) / (x1 - x0) : 0.5) * (W - padL - 120), Y = v => pad + (y1 > y0 ? (v - y0) / (y1 - y0) : 0.5) * (H - 2 * pad);
  const placed = [], fits = b => b.x >= 2 && b.x + b.w <= W - 2 && b.y >= 2 && b.y + b.h <= H - 2 && !placed.some(p => b.x < p.x + p.w && b.x + b.w > p.x && b.y < p.y + p.h && b.y + b.h > p.y);
  const order = ids.map((_, i) => i).sort((a, b) => cos[b] - cos[a]);
  const pts = ids.map((j, i) => ({ j, i, x: X(xy[i][0]), y: Y(xy[i][1]), c: cos[i] }));
  for (const p of pts) placed.push({ x: p.x - 6, y: p.y - 6, w: 12, h: 12 });
  let s = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${tl('Bedeutungskarte der Nachbarn', 'Meaning map of the neighbors')}">`;
  const q = pts[0];
  for (const p of pts.slice(1)) s += `<line x1="${q.x}" y1="${q.y}" x2="${p.x}" y2="${p.y}" style="stroke:var(--line);stroke-width:1"/>`;
  const labels = [];
  for (const i of order) {
    const p = pts[i], txt = wsPlain(disp(p.j)).slice(0, 18), w = txt.length * (i === 0 ? 8.6 : 7.4) + 6, h = 16;
    const cand = [[10, -8, 'start', 0], [-10, -8, 'end', -w], [-w / 2, -24, 'middle', -w / 2], [-w / 2, 10, 'middle', -w / 2]];
    for (const [dx, dy, anchor, bx] of cand) {
      const box = { x: p.x + (anchor === 'start' ? dx : anchor === 'end' ? dx - w : bx), y: p.y + (anchor === 'middle' ? dy : dy), w, h };
      if (fits(box)) { placed.push(box); labels.push({ i, x: anchor === 'middle' ? p.x : p.x + dx, y: box.y + 12, anchor, txt }); break; }
    }
  }
  for (const p of pts) {
    const r = p.i === 0 ? 8 : 3 + Math.max(0, p.c) * 10;
    s += `<g class="pt${p.i === 0 ? ' q' : ''}" data-id="${p.j}"><circle cx="${p.x}" cy="${p.y}" r="${r}" style="fill:${p.i === 0 ? 'var(--ink)' : 'var(--c-ffn)'};stroke:var(--surface);stroke-width:2"/></g>`;
  }
  for (const L of labels) s += `<g class="pt${L.i === 0 ? ' q' : ''}" data-id="${pts[L.i].j}"><text x="${L.x}" y="${L.y}" text-anchor="${L.anchor}">${esc(L.txt)}</text></g>`;
  host.innerHTML = s + '</svg>';
  const svg = host.firstChild;
  svg.onmousemove = e => { const g = e.target.closest('.pt'); if (!g) return hideTip(); const j = +g.dataset.id, k = ids.indexOf(j), w = wsHTML(disp(j)), nr = fmtInt(j), c = nf(cos[k], 2); showTip(tl(`<code>${w}</code> · ID ${nr}<br>Ähnlichkeit ${c}${k ? '' : ' (Suchwort)'}<br><span class="m">Klick sucht von hier weiter</span>`, `<code>${w}</code> · ID ${nr}<br>Similarity ${c}${k ? '' : ' (search word)'}<br><span class="m">Click to continue from here</span>`), e); };
  svg.onmouseleave = hideTip;
}

// ---------- data flow (recipe from llama.cpp, tensors from the file) ----------
const KIND_COLOR = { lin: 'var(--c-lin)', attn: 'var(--c-attn)', swa: 'color-mix(in srgb, var(--c-attn) 45%, var(--surface))', mtp: 'var(--c-mtp)', other: 'var(--c-other)' };
function kindLabel(M, k) {
  const win = metaNum(M, 'attention.sliding_window'), w = win ? fmtInt(win) : '';
  return {
    lin: M.arch.startsWith('qwen3') ? tl('Lineare Attention (Gated DeltaNet)', 'Linear attention (Gated DeltaNet)') : tl('Lineare Attention / SSM', 'Linear attention / SSM'),
    attn: M.layers.some(L => L.kind === 'swa') ? tl('Globale Attention (ganzer Text)', 'Global attention (whole text)') : tl('Volle Attention', 'Full attention'),
    swa: win ? tl(`Lokale Attention (Fenster ${w} Tokens)`, `Local attention (${w}-token window)`) : tl('Lokale Attention', 'Local attention'),
    mtp: tl('MTP-Block (rät voraus)', 'MTP block (guesses ahead)'), other: tl('Sonstiger Block', 'Other block'),
  }[k];
}
const VERIFIED = { qwen35: 'llama.cpp b11388, src/models/qwen35.cpp', gemma4: 'llama.cpp b11388, src/models/gemma4.cpp' };
const isMainPass = (M, L) => !(L.kind === 'mtp' && VERIFIED[M.arch] && M.arch === 'qwen35');

function buildRecipe(M, L) {
  const T = {}; for (const t of L.tensors) T[t.name.replace(LAYER_RE, '$3').replace(/\.weight$/, '')] = t;
  const d = showVal(metaNum(M, 'embedding_length')) || '?', heads = showVal(metaNum(M, 'attention.head_count')), kvh = metaNum(M, 'attention.head_count_kv');
  const kvN = kvh == null ? null : fmtInt(typeof kvh === 'object' ? kvh.items[L.idx] : kvh), kvTxt = kvN == null ? '' : tl(`${kvN} Key/Value-Köpfe`, `${kvN} key/value heads`);
  const back = tl(`zurück auf ${d} Zahlen`, `back to ${d} numbers`), nrm = tl('normieren', 'normalize'), nrmRes = tl('Ergebnis normieren', 'normalize the result');
  const N = [], E = [];
  const node = (id, col, row, kind, title, sub, x = {}) => { N.push({ id, col, row, kind, title, sub, ...x }); return id; };
  const w = (id, key, col, row, sub, x) => T[key] ? node(id, col, row, 'w', key, sub, { t: T[key], ...x }) : null;
  const e = (a, b, o = {}) => { if (a && b) E.push({ a, b, ...o }); };
  let r = 0;
  const qwen = M.arch === 'qwen35';
  const inN = node('in', 1.5, r++, 'data', tl('Eingang', 'Input'), tl(`je Token ${d} Zahlen (aus dem vorigen Block)`, `${d} numbers per token (from the previous block)`));
  let normIn = inN;

  if (L.kind === 'mtp' && qwen) {
    // multi-token prediction head (graph_mtp in qwen35.cpp)
    N.length = 0; r = 0;
    const te = node('te', 0.5, r, 'data', 'Embedding', tl('des gerade erzeugten Tokens', 'of the token just generated'));
    const he = node('he', 2.5, r++, 'data', tl('Ergebnis Block 63', 'Result of block 63'), tl('Zahlen aus dem Hauptdurchgang', 'numbers from the main pass'));
    const en = w('enorm', 'nextn.enorm', 0.5, r, nrm), hn = w('hnorm', 'nextn.hnorm', 2.5, r++, nrm);
    e(te, en); e(he, hn);
    const cc = node('cat', 1.5, r++, 'op', tl('Aneinanderhängen', 'Concatenate'), tl(`2 × ${d} Zahlen`, `2 × ${d} numbers`)); e(en, cc); e(hn, cc);
    const ep = w('eh', 'nextn.eh_proj', 1.5, r++, back); e(cc, ep);
    normIn = ep;
  }
  const n1 = w('attn_norm', 'attn_norm', 1.5, r++, tl('Normieren: alle Zahlen auf ähnliche Größe bringen', 'Normalize: scale all numbers to a similar size')); e(normIn, n1);
  let mixOut, joinFrom = normIn;

  if (L.kind === 'lin' && T.ssm_conv1d) {
    const R0 = r;
    const qkv = w('qkv', 'attn_qkv', 0.5, R0, tl('Query, Key und Value in einem Rutsch', 'Query, key and value in one go'));
    const al = w('alpha', 'ssm_alpha', 1.5, R0, tl('Wie stark soll vergessen werden?', 'How much to forget?'));
    const be = w('beta', 'ssm_beta', 2.5, R0, tl('Wie stark soll neu geschrieben werden?', 'How much new to write?'));
    const z = w('z', 'attn_gate', 3.5, R0, tl('Ausgangs-Gate z', 'Output gate z'));
    [qkv, al, be, z].forEach(x => e(n1, x));
    const cv = w('conv', 'ssm_conv1d', 0.5, R0 + 1, tl('kurzer Blick zurück: mischt die letzten Tokens', 'short look back: mixes the last tokens')); e(qkv, cv);
    const dt = w('dt', 'ssm_dt.bias', 1.5, R0 + 1, tl('+ Bias, dann Softplus', '+ bias, then Softplus')); e(al, dt);
    const bs = node('bsig', 2.5, R0 + 1, 'op', 'Sigmoid', tl('→ Schreibstärke β (0 bis 1)', '→ write strength β (0 to 1)')); e(be, bs);
    const sp = node('split', 0.5, R0 + 2, 'op', tl('SiLU, aufteilen', 'SiLU, split'), tl('in q, k, v; q und k auf Länge 1', 'into q, k, v; q, k to length 1')); e(cv, sp);
    const sa = w('A', 'ssm_a', 1.5, R0 + 2, tl('× A → Vergessensrate g', '× A → forget rate g')); e(dt, sa);
    const nh = showVal(metaNum(M, 'ssm.time_step_rank')), st = showVal(metaNum(M, 'ssm.state_size'));
    const dl = node('delta', 1.5, R0 + 3, 'op', tl('Gated-Delta-Regel: der Speicher', 'Gated delta rule: the memory'), tl(`fester Speicher: ${nh} Köpfe × ${st} × ${st} · pro Token: vergessen (g), k → v schreiben (β), mit q lesen`, `fixed memory: ${nh} heads × ${st} × ${st} · per token: forget (g), write k → v (β), read with q`), { span: 3 });
    e(sp, dl); e(sa, dl); e(bs, dl);
    const nm = w('ssm_norm', 'ssm_norm', 1.5, R0 + 4, tl('normieren, dann × SiLU(z)', 'normalize, then × SiLU(z)')); e(dl, nm); e(z, nm, { enter: 'right' });
    mixOut = w('ssm_out', 'ssm_out', 1.5, R0 + 5, back); e(nm, mixOut);
    r = R0 + 6;
  } else {
    const R0 = r, gemma = M.arch.startsWith('gemma');
    let q, k, v;
    if (T.attn_qkv) { q = k = v = w('qkv', 'attn_qkv', 1.5, R0, tl('Query, Key und Value in einem', 'Query, key and value in one')); e(n1, q); }
    else {
      q = w('q', 'attn_q', 0.5, R0, qwen ? tl('Query + Gate: wonach sucht das Token?', 'Query + gate: what does it look for?') : tl('Query: wonach sucht das Token?', 'Query: what does it look for?'));
      k = w('k', 'attn_k', 1.5, R0, tl('Key: wofür steht das Token?', 'Key: what does it stand for?'));
      v = T.attn_v ? w('v', 'attn_v', 2.5, R0, tl('Value: was gibt das Token weiter?', 'Value: what does it pass on?')) : node('v', 2.5, R0 + 1, 'op', tl('Value = Key', 'Value = key'), tl('keine eigene Value-Tabelle', 'no separate value table'));
      e(n1, q); e(n1, k); if (T.attn_v) e(n1, v); else e(k, v);
    }
    const qn = w('qn', 'attn_q_norm', 0.5, R0 + 1, tl('Query je Kopf normieren', 'normalize query per head')), kn = w('kn', 'attn_k_norm', 1.5, R0 + 1, tl('Key je Kopf normieren', 'normalize key per head'));
    e(q, qn); e(k, kn);
    const rp = node('rope', 1, R0 + 2, 'op', 'RoPE', tl('Position einrechnen: wo steht das Token?', 'add position: where is the token?'));
    e(qn || q, rp); e(kn || k, rp);
    const swa = L.kind === 'swa', win = showVal(metaNum(M, 'attention.sliding_window'));
    const over = swa ? tl(`der letzten ${win} Tokens`, `the last ${win} tokens`) : tl('aller vorherigen Tokens', 'all previous tokens'), kv = kvTxt ? ' · ' + kvTxt : '';
    const at = node('attn', 1.5, R0 + 3, 'op', 'Attention', tl(`Query mit den Keys ${over} vergleichen, Values mischen · ${heads} Köpfe${kv} · KV-Cache`, `compare query with the keys of ${over}, mix values · ${heads} heads${kv} · KV cache`), { span: 3 });
    e(rp, at); e(v, at);
    let last = at, rr = R0 + 4;
    if (qwen) { const g = node('gate', 1.5, rr++, 'op', tl('× Sigmoid(Gate)', '× Sigmoid(gate)'), tl('Gate aus attn_q dämpft oder lässt durch', 'gate from attn_q damps or lets through')); e(at, g); e(q, g, { route: 'left' }); last = g; }
    mixOut = w('o', 'attn_output', 1.5, rr++, back); e(last, mixOut);
    if (T.post_attention_norm && T.ffn_norm) { const pn = w('pan', 'post_attention_norm', 1.5, rr++, nrmRes); e(mixOut, pn); mixOut = pn; }
    r = rr;
    void gemma;
  }
  const j1 = node('j1', 1.5, r++, 'join', '+', tl('Eingang wieder aufaddieren', 'add the input back')); e(mixOut, j1); e(joinFrom, j1, { skip: true });
  let pre = j1;
  if (T.ffn_norm) { pre = w('fn', 'ffn_norm', 1.5, r++, tl('Normieren vor dem FFN', 'normalize before the FFN')); e(j1, pre); }
  else if (T.post_attention_norm) { pre = w('pan2', 'post_attention_norm', 1.5, r++, tl('Normieren vor dem FFN', 'normalize before the FFN')); e(j1, pre); }
  let ffOut;
  if (T.ffn_gate_inp) {
    const ro = w('router', 'ffn_gate_inp', 0.5, r, tl('Router: Punktzahl je Experte', 'Router: score per expert')); e(pre, ro);
    const nE = showVal(metaNum(M, 'expert_count')), kE = showVal(metaNum(M, 'expert_used_count'));
    const ex = node('exp', 2, r, 'op', tl(`${kE || 'k'} von ${nE || 'N'} Experten`, `${kE || 'k'} of ${nE || 'N'} experts`), tl('nur die besten Experten rechnen', 'only the best experts compute')); e(ro, ex); e(pre, ex);
    const gx = w('gx', 'ffn_gate_exps', 1, r + 1, tl('Experten: Gate', 'Experts: gate')), ux = w('ux', 'ffn_up_exps', 2, r + 1, tl('Experten: Inhalt', 'Experts: content')), dx = w('dx', 'ffn_down_exps', 3, r + 1, tl('Experten: zurück', 'Experts: back'));
    [gx, ux, dx].forEach(x => e(ex, x));
    ffOut = node('mix', 1.5, r + 2, 'op', tl('Gewichtet mischen', 'Weighted mix'), tl('Ergebnisse der Experten zusammenführen', 'combine the experts’ results')); [gx, ux, dx].forEach(x => e(x, ffOut));
    r += 3;
  } else if (T.ffn_gate) {
    const ff = showVal(metaNum(M, 'feed_forward_length'));
    const g = w('fg', 'ffn_gate', 0.5, r, tl(`auffächern auf ${ff}: entscheidet, was durchkommt`, `expand to ${ff}: decides what gets through`)), u = w('fu', 'ffn_up', 2.5, r, tl(`auffächern auf ${ff}: der Inhalt`, `expand to ${ff}: the content`));
    e(pre, g); e(pre, u);
    const sl = node('silu', 0.5, r + 1, 'op', M.arch.startsWith('gemma') ? 'GELU' : 'SiLU', tl('weiche Schwelle: kleine Werte werden fast 0', 'soft threshold: small values become almost 0')); e(g, sl);
    const mu = node('mul', 1.5, r + 2, 'op', '× (gated FFN)', tl('Schwelle mal Inhalt, Zahl für Zahl', 'threshold × content, number by number')); e(sl, mu); e(u, mu);
    ffOut = w('fd', 'ffn_down', 1.5, r + 3, tl(`zusammenfassen: zurück auf ${d} Zahlen`, `condense: back to ${d} numbers`)); e(mu, ffOut);
    r += 4;
  } else {
    const u = w('fu', 'ffn_up', 1.5, r, tl('auffächern', 'expand')); e(pre, u);
    const ac = node('act', 1.5, r + 1, 'op', tl('Aktivierung', 'Activation'), 'GELU/SiLU'); e(u, ac);
    ffOut = w('fd', 'ffn_down', 1.5, r + 2, tl('zurückfalten', 'fold back')); e(ac, ffOut); r += 3;
  }
  if (T.post_ffw_norm) { const pf = w('pfn', 'post_ffw_norm', 1.5, r++, nrmRes); e(ffOut, pf); ffOut = pf; }
  const j2 = node('j2', 1.5, r++, 'join', '+', tl('wieder aufaddieren', 'add back')); e(ffOut, j2); e(j1, j2, { skip: true });
  let outFrom = j2;
  if (T.layer_output_scale) { const sc = w('los', 'layer_output_scale', 1.5, r++, tl('ganzen Block skalieren', 'scale the whole block')); e(j2, sc); outFrom = sc; }
  if (L.kind === 'mtp' && qwen) {
    const shn = w('shn', 'nextn.shared_head_norm', 1.5, r++, nrm); e(outFrom, shn);
    const hd = node('head', 1.5, r++, 'data', tl('Ausgabe-Kopf (output)', 'Output head (output)'), tl('Vorschlag für das übernächste Token', 'guess for the token after next')); e(shn || outFrom, hd);
  } else { const o = node('out', 1.5, r++, 'data', tl('Ausgang', 'Output'), tl('weiter zum nächsten Block', 'on to the next block')); e(outFrom, o); }
  return { nodes: N, edges: E, verified: VERIFIED[M.arch] || null };
}

let flowUid = 0;
function wrapText(s, max) {
  const words = String(s).split(' '), lines = []; let cur = '';
  for (const w of words) { if ((cur + ' ' + w).trim().length > max && cur) { lines.push(cur); cur = w; } else cur = (cur + ' ' + w).trim(); }
  if (cur) lines.push(cur);
  return lines;
}
function drawFlow(host, R, opts = {}) {
  const COL = 232, PADL = 30, PADT = 12, LANE = 64;
  const geo = n => {
    const span = n.span || 1, wdt = n.kind === 'join' ? 30 : span * COL - 24;
    const sub = n.kind === 'join' ? [] : wrapText(n.sub || '', Math.floor((wdt - 18) / 6.1));
    const lines = (n.kind === 'w' ? 2 : 1) + Math.min(sub.length, n.span ? 3 : 2);
    const h = n.kind === 'join' ? 30 : 14 + lines * 15;
    return { span, w: wdt, h, sub: sub.slice(0, n.span ? 3 : 2) };
  };
  const rows = Math.max(...R.nodes.map(n => n.row)) + 1;
  const rowH = new Array(rows).fill(40);
  for (const n of R.nodes) { n.g = geo(n); rowH[n.row] = Math.max(rowH[n.row], n.g.h + 26); }
  const rowY = []; let acc = PADT; for (let i = 0; i < rows; i++) { rowY.push(acc); acc += rowH[i]; }
  const W = PADL + 4 * COL + LANE, H = acc + 8;
  const byId = new Map(R.nodes.map(n => [n.id, n]));
  for (const n of R.nodes) { n.cx = PADL + n.col * COL; n.y = rowY[n.row] + (rowH[n.row] - n.g.h) / 2; n.x = n.cx - n.g.w / 2; }
  const uid = ++flowUid, laneR = PADL + 4 * COL + 26, laneL = 12;
  let s = `<svg viewBox="0 0 ${W} ${H}" width="${W}" role="img" aria-label="${tl('Datenfluss durch einen Block', 'Data flow through one block')}"><defs><marker id="ah${uid}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" style="fill:var(--ink-2)"/></marker><marker id="as${uid}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" style="fill:var(--muted)"/></marker></defs>`;
  let skipLabel = false;
  for (const ed of R.edges) {
    const a = byId.get(ed.a), b = byId.get(ed.b); if (!a || !b) continue;
    let d;
    if (ed.skip) {
      const ya = a.y + a.g.h / 2, yb = b.y + b.g.h / 2;
      d = `M${a.x + a.g.w},${ya} H${laneR} V${yb} H${b.x + b.g.w + 4}`;
      s += `<path class="edge skip" d="${d}" marker-end="url(#as${uid})"/>`;
      if (!skipLabel) { skipLabel = true; s += `<text class="lane" x="${laneR + 6}" y="${(ya + yb) / 2}" transform="rotate(90 ${laneR + 6} ${(ya + yb) / 2})" text-anchor="middle">${tl('Rest-Verbindung (Residual)', 'Residual connection')}</text>`; }
      continue;
    }
    if (ed.route === 'left') {
      const ya = a.y + a.g.h / 2, yb = b.y + b.g.h / 2;
      d = `M${a.x},${ya} H${laneL} V${yb} H${b.x - 4}`;
    } else if (ed.enter === 'right') {
      const yb = b.y + b.g.h / 2;
      d = `M${a.cx},${a.y + a.g.h} V${yb} H${b.x + b.g.w + 4}`;
    } else {
      const x1 = a.cx, y1 = a.y + a.g.h, y2 = b.y - 4, xt = Math.max(b.x + 14, Math.min(b.x + b.g.w - 14, x1));
      d = Math.abs(xt - x1) < 1 ? `M${x1},${y1} V${y2}` : `M${x1},${y1} V${y2 - 12} H${xt} V${y2}`;
    }
    s += `<path class="edge" d="${d}" marker-end="url(#ah${uid})"/>`;
  }
  for (const n of R.nodes) {
    const { w: nw, h: nh, sub } = n.g;
    if (n.kind === 'join') {
      s += `<g class="node"><circle cx="${n.cx}" cy="${n.y + 15}" r="15" style="fill:var(--surface);stroke:var(--ink);stroke-width:1.6"/><text x="${n.cx}" y="${n.y + 20}" text-anchor="middle" style="font-size:18px;font-weight:700">+</text><text class="t2" x="${n.cx - 24}" y="${n.y + 19}" text-anchor="end">${esc(n.sub)}</text></g>`;
      continue;
    }
    const col = n.kind === 'w' ? `var(${COMPS[n.t.comp].v})` : null;
    const fill = n.kind === 'w' ? `color-mix(in srgb, ${col} 14%, var(--surface))` : n.kind === 'op' ? 'var(--sunken)' : 'var(--surface)';
    const stroke = n.kind === 'w' ? col : n.kind === 'op' ? 'var(--line-2)' : 'var(--ink-2)';
    s += `<g class="node ${n.kind}"${n.kind === 'w' ? ` data-t="${n.t.idx}" tabindex="0" role="button" aria-label="${tl(`${esc(n.t.name)} öffnen`, `Open ${esc(n.t.name)}`)}"` : ''}><rect x="${n.x}" y="${n.y}" width="${nw}" height="${nh}" rx="8" style="fill:${fill};stroke:${stroke};stroke-width:1.5${n.kind === 'data' ? ';stroke-dasharray:4 3' : ''}"/>`;
    let ty = n.y + 19;
    s += `<text class="t1${n.kind === 'w' ? '' : ' op'}" x="${n.x + 10}" y="${ty}">${esc(n.title)}</text>`;
    for (const line of sub) { ty += 15; s += `<text class="t2" x="${n.x + 10}" y="${ty}">${esc(line)}</text>`; }
    if (n.kind === 'w') {
      const t = n.t, shape = t.dims.length === 1 ? `[${fmtInt(t.dims[0])}]` : `${fmtInt(t.dims[0])} → ${fmtInt(t.dims[1])}`;
      ty += 15; s += `<text class="t2" x="${n.x + 10}" y="${ty}" style="fill:var(--muted)">${shape} · ${esc(t.typeName)} · ${fmtBytes(t.nBytes)}</text>`;
    }
    s += '</g>';
  }
  host.innerHTML = s + '</svg>';
  const svg = host.firstChild;
  const tOf = el => { const g = el.closest('.node.w'); return g ? S.model.tensors[+g.dataset.t] : null; };
  svg.onmousemove = e => { const t = tOf(e.target); if (!t) return hideTip(); showTip(tensorTip(t) + `<br><span class="m">${tl('Klick öffnet die Tabelle unter „Gewichte“', 'Click to open it under “Weights”')}</span>`, e); };
  svg.onmouseleave = hideTip;
  svg.onclick = e => { const t = tOf(e.target); if (!t) return; hideTip(); if (opts.onOpen) opts.onOpen(t); else openTensor(t); };
  svg.onkeydown = e => { if (e.key === 'Enter') { const t = tOf(e.target); if (t) { if (opts.onOpen) opts.onOpen(t); else openTensor(t); } } };
  return R;
}
function blockStripHTML(M, cur) {
  return `<div class="bstrip" role="group" aria-label="${tl('Blöcke', 'Blocks')}">${M.layers.map((L, i) => `<button type="button" data-b="${i}" aria-current="${i === cur}" style="background:${KIND_COLOR[L.kind]}${isMainPass(M, L) ? '' : ';opacity:.55'}" title="Block ${esc(L.key)} · ${esc(kindLabel(M, L.kind))} · ${fmtBytes(L.bytes)}" aria-label="Block ${esc(L.key)}"></button>`).join('')}</div>`;
}
function kindsLegend(M) {
  const ks = ['lin', 'swa', 'attn', 'mtp', 'other'].filter(k => M.layers.some(L => L.kind === k));
  return `<div class="legend" style="margin-top:6px">${ks.map(k => `<span><i style="background:${KIND_COLOR[k]}"></i>${fmtInt(M.layers.filter(L => L.kind === k).length)} × ${esc(kindLabel(M, k))}</span>`).join('')}</div>`;
}
function renderFlow() {
  const M = S.model;
  const name = M.meta['general.name'] || M.arch;
  if (!M.layers.length) { const msg = tl('Keine Blöcke (blk.N.*) in dieser Datei, also kein Datenfluss.', 'No blocks (blk.N.*) in this file, so no data flow.'); $('#flowExplain').textContent = msg; $('#flowChain').innerHTML = `<div class="st"><span>${msg}</span></div>`; $('#flowDiagram').innerHTML = ''; return; }
  if (S.flowBlock == null || !M.layers[S.flowBlock]) { const i = M.layers.findIndex(L => L.kind === 'attn'); S.flowBlock = i >= 0 ? i : 0; }
  const main = M.layers.filter(L => isMainPass(M, L)), V = M.vocabSize, d = showVal(metaNum(M, 'embedding_length')) || '?', nb = fmtInt(main.length), arch = esc(M.arch);
  if (!V) {                                              // no vocabulary: not a language model (e.g. the image encoder "clip" of an mmproj file)
    const clip = M.arch === 'clip';
    $('#flowExplain').innerHTML = clip
      ? tl('Kein Sprachmodell, sondern ein <b>Bild-Encoder</b> (Architektur „clip“, die mmproj-Datei zu einem Sprachmodell): Er macht aus Bildausschnitten Vektoren, die das Sprachmodell wie Tokens liest. Tokenizer und Ausgabe-Kopf gibt es hier nicht. Die Blöcke stammen aus der Datei, die Rechenreihenfolge aus dem Programm (llama.cpp, Teil „mtmd“).', 'Not a language model but an <b>image encoder</b> (architecture “clip”, the mmproj file that goes with a language model): it turns image patches into vectors that the language model reads like tokens. There is no tokenizer and no output head here. The blocks come from the file, the compute order from the program (llama.cpp, part “mtmd”).')
      : tl(`Kein Sprachmodell, sondern eine Datei der Architektur „${arch}“ ohne Wörterbuch. Tokenizer und Ausgabe-Kopf gibt es hier nicht. Die Blöcke stammen aus der Datei, die Rechenreihenfolge aus dem Programm (llama.cpp).`, `Not a language model but a file of architecture “${arch}” without a vocabulary. There is no tokenizer and no output head here. The blocks come from the file, the compute order from the program (llama.cpp).`);
    $('#flowChain').innerHTML = `<div class="st blocks"><b>${tl(`${nb} Blöcke nacheinander`, `${nb} blocks in a row`)}</b><span>${clip ? tl('Bild-Encoder, kein Sprachmodell', 'image encoder, not a language model') : tl('kein Sprachmodell (kein Wörterbuch)', 'not a language model (no vocabulary)')}</span>${blockStripHTML(M, S.flowBlock)}${kindsLegend(M)}</div>`;
    renderFlowBlock(); renderBauplan(); return;
  }
  const Vs = fmtInt(V), nm = esc(name);
  $('#flowExplain').innerHTML = tl(`So rechnet <b>${nm}</b>: Text wird zu Tokens, jedes Token zu einer Zeile mit ${d} Zahlen, und ${nb} Blöcke rechnen diese Zahlen nacheinander neu. Am Ende bekommt jedes der ${Vs} Tokens eine Punktzahl; eins wird nach Wahrscheinlichkeit gewählt, dann beginnt alles von vorn mit einem Token mehr. Die Datei liefert nur die Tabellen, <b>die Reihenfolge steht im Programm</b> (llama.cpp), das den Bauplan am Architektur-Namen „${arch}“ erkennt.`, `How <b>${nm}</b> computes: text becomes tokens, each token a row of ${d} numbers, and ${nb} blocks recompute these numbers one after another. At the end each of the ${Vs} tokens gets a score; one is picked by probability, then everything starts again with one more token. The file only supplies the tables; <b>the order is in the program</b> (llama.cpp), which picks the blueprint by the architecture name “${arch}”.`);
  const out = M.byName.get('output.weight') || M.E, on = M.byName.get('output_norm.weight');
  const temp = M.meta['general.sampling.temp'], topp = M.meta['general.sampling.top_p'], topk = M.meta['general.sampling.top_k'], cap = metaNum(M, 'final_logit_softcapping');
  const tS = temp != null ? sig(temp, 2) : '', pS = topp != null ? sig(topp, 2) : '', kS = topk != null ? fmtInt(topk) : '', cS = cap ? fmtInt(cap) : '', gem = M.arch.startsWith('gemma') ? ` · × √${d}` : '';
  const st = (go, b, sp, code) => `<button type="button" class="st" data-go="${go}"><b>${b}</b>${sp ? `<span>${sp}</span>` : ''}${code ? `<code>${code}</code>` : ''}</button>`;
  $('#flowChain').innerHTML = [
    `<div class="st"><b>Text</b></div>`,
    st('tokens', 'Tokenizer', tl(`Wörterbuch: ${Vs}`, `vocabulary: ${Vs}`), 'tokenizer.ggml.*'),
    M.E ? st('embed', tl('Embedding-Tabelle', 'Embedding table'), tl(`ID → Zeile mit ${d} Zahlen${gem}`, `ID → row of ${d} numbers${gem}`), `token_embd ${fmtInt(M.E.dims[1])} × ${fmtInt(M.E.dims[0])}`) : '',
    `<div class="st blocks"><b>${tl(`${nb} Blöcke nacheinander`, `${nb} blocks in a row`)}</b>${blockStripHTML(M, S.flowBlock)}${kindsLegend(M)}${main.length < M.layers.length ? `<span>${tl('Blass: nicht im normalen Durchgang', 'pale: not in the normal pass')}</span>` : ''}</div>`,
    on ? st('onorm', tl('Normieren', 'Normalize'), '', 'output_norm') : '',
    out ? st('head', tl('Ausgabe-Kopf', 'Output head'), tl(`${Vs} Punktzahlen`, `${Vs} scores`), `${esc(out.name.replace('.weight', ''))} ${fmtInt(out.dims[0])} → ${fmtInt(out.dims[1])}`) : '',
    `<div class="st"><b>${tl('Nächstes Token', 'Next token')}</b><span>${tl(`${cap ? `Softcapping ${cS} → ` : ''}Wahrscheinlichkeiten → eins wählen${temp != null ? ` · Temperatur ${tS}${topp != null ? ` · Top-p ${pS}` : ''}${topk != null ? ` · Top-k ${kS}` : ''}` : ''}`, `${cap ? `softcapping ${cS} → ` : ''}probabilities → pick one${temp != null ? ` · temperature ${tS}${topp != null ? ` · top-p ${pS}` : ''}${topk != null ? ` · top-k ${kS}` : ''}` : ''}`)}</span></div>`,
  ].filter(Boolean).join('<span class="ar" aria-hidden="true">→</span>');
  renderFlowBlock();
  renderBauplan();
}
function renderFlowBlock() {
  const M = S.model, L = M.layers[S.flowBlock];
  document.querySelectorAll('#flowChain .bstrip button').forEach(b => b.setAttribute('aria-current', String(+b.dataset.b === S.flowBlock)));
  const R = buildRecipe(M, L);
  $('#flowTitle').textContent = `Block ${L.key}: ${kindLabel(M, L.kind)}`;
  const same = M.layers.filter(x => x.kind === L.kind).length, sm = fmtInt(same), win = showVal(metaNum(M, 'attention.sliding_window'));
  const over = L.kind === 'swa' ? tl(`der letzten ${win} Tokens`, `the last ${win} tokens`) : tl('aller vorherigen Tokens', 'all previous tokens');
  const mech = R.nodes.some(n => n.id === 'delta')
    ? tl('Statt alle früheren Tokens anzusehen, führt der Block einen festen Speicher mit: Pro Token vergisst er etwas (g), schreibt das neue Paar k → v hinein (β) und liest mit q aus.', 'Instead of looking at all earlier tokens, the block carries a fixed memory: per token it forgets a little (g), writes the new pair k → v (β) and reads with q.')
    : tl(`Jedes Token vergleicht seine Query mit den Keys ${over} und mischt deren Values.`, `Each token compares its query with the keys of ${over} and mixes their values.`);
  $('#flowLede').innerHTML = L.kind === 'mtp' && M.arch === 'qwen35'
    ? tl('Dieser Zusatz-Block läuft im normalen Durchgang nicht mit. Er rät aus dem Ergebnis des letzten Blocks das übernächste Token voraus; stimmt der Vorschlag, spart das Zeit (spekulatives Dekodieren).', 'This extra block does not run in the normal pass. From the last block’s result it guesses the token after next; if the guess is right, that saves time (speculative decoding).')
    : (same === 1 ? tl('Der einzige Block dieser Art.', 'The only block of this kind.') : tl(`${sm} Blöcke dieser Art haben denselben Ablauf, nur mit eigenen Tabellen.`, `${sm} blocks of this kind share this flow, each with its own tables.`)) + ' ' + mech;
  drawFlow($('#flowDiagram'), R);
  const nt = fmtInt(L.tensors.length), by = fmtBytes(L.bytes);
  $('#flowNote').textContent = tl(`Farbig: Tabelle aus der Datei · grau: Rechenschritt des Programms · ${nt} Tabellen, ${by} · `, `Colored: table from the file · gray: program step · ${nt} tables, ${by} · `)
    + (R.verified ? tl(`am Quellcode geprüft: ${R.verified}`, `checked against the source: ${R.verified}`) : tl(`Standard-Bauplan, für „${M.arch}“ nicht am Quellcode geprüft`, `standard blueprint, not checked against the source for “${M.arch}”`));
  const fileOrder = L.tensors.slice().sort((a, b) => a.fileOrder - b.fileOrder), short = t => t.name.replace(LAYER_RE, '$3');
  const alpha = fileOrder.map(short).every((x, i, a) => i === 0 || a[i - 1].localeCompare(x) <= 0);
  $('#orderFileCap').innerHTML = tl(`<b>Reihenfolge in der Datei</b> (Block ${esc(L.key)})${alpha ? ': schlicht alphabetisch' : ''}`, `<b>Order in the file</b> (block ${esc(L.key)})${alpha ? ': simply alphabetical' : ''}`);
  $('#orderFile').innerHTML = fileOrder.map(t => `<li>${esc(short(t))} <span class="x">${esc(t.typeName)} · ${fmtBytes(t.nBytes)}</span></li>`).join('');
  const calc = R.nodes.filter(n => n.kind === 'w').sort((a, b) => a.row - b.row || a.col - b.col);
  $('#orderCalc').innerHTML = calc.map(n => `<li>${esc(short(n.t))} <span class="x">${esc(n.sub)}</span></li>`).join('');
}

// ---------- Rundgang (full-screen, step by step; every step shows its content at once) ----------
const TOUR = ['start', 'split', 'lookup', 'rows', 'matrix', 'layers', 'block', 'output', 'nn', 'map', 'end'];
const TOUR_NAMES = {
  get start() { return tl('Start', 'Start'); }, get split() { return tl('Zerlegen', 'Split'); }, get lookup() { return tl('Nachschlagen', 'Look up'); }, get rows() { return tl('Zeilen holen', 'Fetch rows'); },
  get matrix() { return tl('Eingabe', 'Input'); }, get layers() { return tl('Schichten', 'Layers'); }, get block() { return tl('Ein Block', 'One block'); }, get output() { return tl('Ausgabe', 'Output'); },
  get nn() { return tl('Bedeutung', 'Meaning'); }, get map() { return tl('Landkarte', 'Map'); }, get end() { return tl('Zusammenfassung', 'Summary'); },
};
let tourSeq = 0;
function openTour() {
  if (!S.model) return;
  S.tourStep = S.tourStep || 0;
  $('#tour').hidden = false; document.body.style.overflow = 'hidden';
  try { const p = document.documentElement.requestFullscreen && document.documentElement.requestFullscreen(); if (p && p.catch) p.catch(() => {}); } catch {}
  renderTour(); $('#tourNext').focus();
}
function closeTour() {
  $('#tour').hidden = true; document.body.style.overflow = ''; hideTip();
  try { if (document.fullscreenElement) document.exitFullscreen().catch(() => {}); } catch {}
}
function tourGo(i) { S.tourStep = Math.max(0, Math.min(TOUR.length - 1, i)); renderTour(); }
function tourTokens() { const M = S.model; return (M.tok && M.tok.supported && S.tokRes) ? S.tokRes : null; }
function renderTour() {
  const M = S.model, k = TOUR[S.tourStep], seq = ++tourSeq, res = tourTokens();
  $('#tourDots').innerHTML = TOUR.map((x, i) => `<button type="button" data-step="${i}" aria-current="${i === S.tourStep}" aria-label="${i + 1}. ${TOUR_NAMES[x]}" title="${i + 1}. ${TOUR_NAMES[x]}"></button>`).join('');
  $('#tourCount').textContent = `${S.tourStep + 1} / ${TOUR.length} · ${TOUR_NAMES[k]}`;
  $('#tourPrev').disabled = S.tourStep === 0;
  $('#tourNext').textContent = S.tourStep === TOUR.length - 1 ? tl('Fertig ✓', 'Done ✓') : tl('Weiter →', 'Next →');
  const stage = $('#tourStage'), E = M.E, d = E ? fmtInt(E.dims[0]) : showVal(metaNum(M, 'embedding_length')), V = M.vocabSize, name = esc(M.meta['general.name'] || M.arch);
  const ids = res ? res.ids : [], n = ids.length;
  const chips = (big) => res ? res.pieces.map((p, i) => { const tt = tokenText(M.tokens[p.id], M.tokStyle); return `<span class="tk tk-c${i % 6}${p.kind ? ' special' : ''}"><span class="tt">${tt.bytes ? `<span class="ws">${esc(tt.text)}</span>` : wsHTML(tt.text)}</span>${big ? `<small>${fmtInt(p.id)}</small>` : ''}</span>`; }).join('') : '';
  let h = '';
  const head = (eyebrow, title, lede) => `<p class="t-eyebrow">${eyebrow}</p><h2 class="t-title">${title}</h2>${lede ? `<p class="t-lede">${lede}</p>` : ''}`;
  if (k === 'start') {
    const file = esc(M.info.fileName), size = fmtBytes(M.h.fileSize);
    // the tokenizer's reason comes in German from gguf-core.js; English version of its two cases
    const why = !M.tok ? tl('Kein Tokenizer in dieser Datei.', 'No tokenizer in this file.') : tl(M.tok.reason, M.tok.model !== 'gpt2' && M.tok.model !== 'gemma4' ? `Live splitting is built in for BPE tokenizers (Qwen, Llama 3, Mistral Nemo, Gemma 4 …). This model uses “${M.tok.model || 'unknown'}”.` : 'The file has no merge rules.');
    h = head(tl(`Rundgang · ${name}`, `Tour · ${name}`), tl('Was passiert mit deinem Satz?', 'What happens to your sentence?'), tl(`Wir verfolgen einen Satz durch die echte Modelldatei <b>${file}</b> (${size}). Jede Zahl auf den nächsten Seiten stammt aus dieser Datei. Tippe einen eigenen Satz ein oder nimm ein Beispiel.`, `We follow a sentence through the real model file <b>${file}</b> (${size}). Every number on the next pages comes from this file. Type your own sentence or pick an example.`))
      + `<textarea class="t-input" id="tourText" spellcheck="false" aria-label="${tl('Satz für den Rundgang', 'Sentence for the tour')}">${esc(S.text)}</textarea><div class="quick" id="tourEx"></div>`
      + (M.tok && M.tok.supported ? '' : `<p class="t-note">${esc(why)} ${tl('Der Rundgang zeigt trotzdem den Aufbau.', 'The tour still shows the structure.')}</p>`);
  } else if (k === 'split') {
    const chars = Array.from(S.text).length;
    h = head(tl('Schritt 1 · Zerlegen', 'Step 1 · Split'), tl(`Der Satz wird in ${fmtInt(n)} Tokens zerlegt`, `The sentence is split into ${fmtInt(n)} tokens`), tl('Das Modell liest keine Buchstaben, sondern Stücke aus einem festen Wörterbuch. Häufige Wörter sind ein Stück, seltene werden aus mehreren zusammengesetzt.', 'The model doesn’t read letters but pieces from a fixed vocabulary. Common words are one piece; rare words are built from several.'))
      + `<div class="t-flow">${chips(false)}</div><div class="t-big"><div><b>${fmtInt(chars)}</b><span>${tl('Zeichen', 'characters')}</span></div><div><b>${fmtInt(n)}</b><span>${tl('Tokens', 'tokens')}</span></div>${n ? `<div><b>${nf(chars / n, 1)}</b><span>${tl('Zeichen pro Token', 'characters per token')}</span></div>` : ''}</div>`;
  } else if (k === 'lookup') {
    const list = `<span class="mono">[${ids.slice(0, 40).join(', ')}${n > 40 ? ', …' : ''}]</span>`;
    h = head(tl('Schritt 2 · Nachschlagen', 'Step 2 · Look up'), tl('Jedes Token hat eine Nummer im Wörterbuch', 'Every token has a number in the vocabulary'), tl(`Das Wörterbuch hat <b>${fmtInt(V)} Einträge</b>. Die Nummer (Token-ID) ist alles, was vom Wort übrig bleibt. Kleine Nummern sind häufige Stücke, die früh ins Wörterbuch kamen; große Nummern sind seltene, lange Stücke.`, `The vocabulary has <b>${fmtInt(V)} entries</b>. The number (token ID) is all that is left of the word. Small numbers are common pieces that entered the vocabulary early; large numbers are rare, long pieces.`))
      + `<div class="t-flow">${chips(true)}</div><div class="dict" id="tourDict"></div><p class="t-note">${tl(`Das Modell bekommt nur diese Liste: ${list}`, `All the model gets is this list: ${list}`)}</p>`;
  } else if (k === 'rows') {
    h = head(tl('Schritt 3 · Zeilen holen', 'Step 3 · Fetch rows'), tl(`Jede Nummer zeigt auf eine Zeile mit ${d} Zahlen`, `Each number points to a row of ${d} numbers`), tl(`In der Embedding-Tabelle steht für jedes Token eine Zeile mit <b>${d} gelernten Zahlen</b>, eine pro Dimension. Hier die echten Zeilen deiner Tokens; jeder Farbstrich ist eine Zahl (blau negativ, rot positiv).`, `The embedding table holds one row per token with <b>${d} learned numbers</b>, one per dimension. Here are the real rows of your tokens; each colored stripe is one number (blue negative, red positive).`))
      + `<div class="t-rows" id="tourRows">${ids.slice(0, 14).map((id, i) => `<div class="t-row"><div class="lbl">${wsHTML(disp(id))}<small>${tl(`Zeile ${fmtInt(id)}`, `Row ${fmtInt(id)}`)}</small></div><div class="vec"><canvas data-i="${i}"></canvas><div class="nums" data-n="${i}">${tl('lese …', 'reading …')}</div></div></div>`).join('')}</div>${n > 14 ? `<p class="hint">${tl(`… und ${fmtInt(n - 14)} weitere Zeilen`, `… and ${fmtInt(n - 14)} more rows`)}</p>` : ''}`;
  } else if (k === 'matrix') {
    const tot = fmtInt(n * (E ? E.dims[0] : 0));
    h = head(tl('Schritt 4 · Die Eingabe', 'Step 4 · The input'), tl(`Zusammen: eine Tabelle mit ${fmtInt(n)} × ${d} Zahlen`, `Together: a table of ${fmtInt(n)} × ${d} numbers`), tl(`Untereinander gelegt ergeben die Zeilen die Eingabe für das Netz: <b>${tot} Zahlen</b>. Ab hier rechnet das Modell nur noch mit Zahlen; Wörter tauchen erst ganz am Ende wieder auf.`, `Stacked on top of each other, the rows form the input to the network: <b>${tot} numbers</b>. From here on the model only computes with numbers; words come back only at the very end.`))
      + `<div class="t-mx"><div class="labs" id="tourMxLabs"></div><canvas class="t-matrix" id="tourMx"></canvas></div><p class="hint">${tl(`Zeilen: deine ${fmtInt(n)} Tokens · Spalten: ${d} Dimensionen (verkleinert)`, `Rows: your ${fmtInt(n)} tokens · columns: ${d} dimensions (scaled down)`)}</p>`;
  } else if (k === 'layers') {
    const main = M.layers.filter(L => isMainPass(M, L)), P = M.params - (E ? E.nElements : 0);
    h = head(tl('Schritt 5 · Durch die Schichten', 'Step 5 · Through the layers'), tl(`Diese Tabelle läuft durch ${fmtInt(main.length)} Blöcke`, `This table runs through ${fmtInt(main.length)} blocks`), tl('Jeder Block rechnet alle Zahlen jeder Zeile neu. <b>Attention</b> lässt die Tokens einander „ansehen“ und Informationen austauschen; das <b>FFN</b> verarbeitet danach jedes Token für sich. Jeder Block hat eigene Gewichtstabellen.', 'Each block recomputes every number in every row. <b>Attention</b> lets the tokens “look at” each other and exchange information; the <b>FFN</b> then processes each token on its own. Every block has its own weight tables.'))
      + blockStripHTML(M, -1) + kindsLegend(M)
      + `<div class="t-big"><div><b>${fmtInt(main.length)}</b><span>${tl('Blöcke', 'blocks')}</span></div><div><b>${fmtParams(M.params)}</b><span>${tl('Gewichte in der Datei', 'weights in the file')}</span></div><div><b>≈ ${fmtParams(2 * P)}</b><span>${tl('Rechenschritte pro Token (grob 2 je Gewicht)', 'operations per token (about 2 per weight)')}</span></div></div>`
      + `<p class="t-note">${tl('Diesen Teil rechnet das Programm (llama.cpp, auch in LM Studio). Die Lupe zeigt die Tabellen und den Bauplan, rechnet die Schichten aber nicht selbst nach.', 'This part is computed by the program (llama.cpp, also inside LM Studio). GGUF-Lupe shows the tables and the blueprint but doesn’t compute the layers itself.')}</p>`;
  } else if (k === 'block') {
    const L = M.layers.find(x => x.kind === 'attn') || M.layers[0];
    const lin = M.layers.some(x => x.kind === 'lin');
    h = head(tl('Schritt 6 · Ein Block von innen', 'Step 6 · Inside one block'), `Block ${esc(L ? L.key : '–')}: ${esc(L ? kindLabel(M, L.kind) : '')}`, tl(`Von oben nach unten: Normieren, Attention, wieder aufaddieren, Normieren, FFN, wieder aufaddieren. Farbige Kästen sind Tabellen aus der Datei, graue Kästen Rechenschritte des Programms.${lin ? ' Die meisten Blöcke dieses Modells nutzen statt Attention einen festen Speicher (lineare Attention), siehe Tab „Datenfluss“.' : ''}`, `Top to bottom: normalize, attention, add back, normalize, FFN, add back. Colored boxes are tables from the file; gray boxes are steps of the program.${lin ? ' Most blocks of this model use a fixed memory instead of attention (linear attention); see the “Data flow” tab.' : ''}`))
      + `<div class="panel flow-wrap"><div class="flow" id="tourFlow"></div></div>`
      + `<p class="t-lede" style="margin-top:12px">${tl('Die Reihenfolge steht nicht in der Datei, sondern im Programm.', 'The order is not in the file but in the program.')} <button type="button" class="btn" id="tourCode">${tl('Im Original-Quelltext von llama.cpp ansehen →', 'See it in the original llama.cpp source →')}</button></p>`;
  } else if (k === 'output') {
    const out = M.byName.get('output.weight') || E, temp = M.meta['general.sampling.temp'], topp = M.meta['general.sampling.top_p'], topk = M.meta['general.sampling.top_k'];
    const sp = `${topp != null ? `, Top-p ${sig(topp, 2)}` : ''}${topk != null ? `, Top-k ${fmtInt(topk)}` : ''}`;
    const samp = temp != null ? tl(`Temperatur ${sig(temp, 2)}${sp} (aus der Datei)`, `temperature ${sig(temp, 2)}${sp} (from the file)`) : tl('per Zufall nach Wahrscheinlichkeit', 'drawn at random by probability');
    h = head(tl('Schritt 7 · Die Ausgabe', 'Step 7 · The output'), tl(`Am Ende: eine Punktzahl für jedes der ${fmtInt(V)} Tokens`, `At the end: a score for each of the ${fmtInt(V)} tokens`), tl('Nur die letzte Zeile zählt: Sie wird mit dem Ausgabe-Kopf verrechnet und ergibt für jedes Token im Wörterbuch eine Punktzahl. Daraus werden Wahrscheinlichkeiten, und ein Token wird ausgewählt.', 'Only the last row counts: it is multiplied by the output head, which gives a score for every token in the vocabulary. The scores become probabilities, and one token is picked.'))
      + `<div class="t-pipe"><div class="box"><b>${tl('Letzte Zeile', 'Last row')}</b><span>${tl(`${d} Zahlen`, `${d} numbers`)}</span></div><span class="ar">→</span><div class="box"><b>${tl('Ausgabe-Kopf', 'Output head')}</b><span>${out ? `${fmtInt(out.dims[0])} → ${fmtInt(out.dims[1])}` : ''}</span><code>${out ? esc(out.name) + ' · ' + fmtBytes(out.nBytes) : ''}</code></div><span class="ar">→</span><div class="box"><b>${tl(`${fmtInt(V)} Punktzahlen`, `${fmtInt(V)} scores`)}</b><span>${tl('eine je Wörterbuch-Eintrag', 'one per vocabulary entry')}</span></div><span class="ar">→</span><div class="box"><b>${tl('Nächstes Token', 'Next token')}</b><span>${samp}</span></div></div>`
      + `<p class="t-note">${tl('<b>Und dann?</b> Das neue Token wird an den Satz angehängt, und alles beginnt von vorn, mit einem Token mehr. So entsteht eine Antwort Token für Token. Dank KV-Cache muss dabei nur die neue Zeile durch alle Blöcke.', '<b>And then?</b> The new token is appended to the sentence, and everything starts again with one more token. That is how an answer is built, token by token. Thanks to the KV cache, only the new row has to go through all the blocks.')}</p>`;
  } else if (k === 'nn') {
    const words = res ? [...new Set(res.pieces.filter(p => !p.kind).map(p => p.id))].filter(id => /\p{L}{2}/u.test(disp(id))) : [];
    const ex = nnWords().map(w => lookupWord(w)).filter(id => id >= 0);
    const pick = [...words, ...ex].filter((id, i, a) => a.indexOf(id) === i && (M.source.kind !== 'demo' || (M.info.demo.nn && M.info.demo.nn[id]))).slice(0, 12);
    if (S.tourNN == null || !pick.includes(S.tourNN)) { const fromText = pick.filter(id => words.includes(id)).sort((a, b) => disp(b).trim().length - disp(a).trim().length); S.tourNN = fromText[0] ?? pick[0]; }
    h = head(tl('Schritt 8 · Was das Modell gelernt hat', 'Step 8 · What the model has learned'), tl('Welche Wörter liegen in der Tabelle nah beieinander?', 'Which words sit close together in the table?'), tl('Vergleicht man eine Zeile mit allen anderen, landen ähnliche Wörter oben, oft sogar in anderen Sprachen. Niemand hat dem Modell diese Nähe beigebracht; sie ergibt sich aus dem Training.', 'Compare one row with all the others, and similar words come out on top, often even in other languages. Nobody taught the model this closeness; it emerges from training.'))
      + `<div class="quick" id="tourNNPick">${pick.map(id => `<button type="button" class="ui" data-id="${id}" aria-current="${id === S.tourNN}">${esc(disp(id).trim() || disp(id))}</button>`).join('')}</div><div class="progress" id="tourNNProg" hidden><div></div></div><p class="readline" id="tourNNStatus"></p>`
      + `<div class="t-split"><div class="nnmap" id="tourNNMap"></div><div class="nn-list" id="tourNNList"></div></div>`;
  } else if (k === 'map') {
    h = head(tl('Schritt 9 · Die Landkarte', 'Step 9 · The map'), tl('Die Landkarte der Bedeutungen', 'The map of meanings'), tl(`Jedes Token ist ein Punkt, angeordnet nach seiner Zeile in der Embedding-Tabelle. Ähnliche Wörter landen nah beieinander, Sprachen mischen sich, Wochentage, Zahlen und Farben bilden eigene Inseln. Berechnet mit allen ${d} Dimensionen, nur flach gezeichnet.`, `Every token is a dot, placed by its row in the embedding table. Similar words land close together, languages mix, and weekdays, numbers and colors form their own islands. Computed with all ${d} dimensions, only drawn flat.`))
      + `<div class="progress" id="tourMapProg" hidden><div></div></div><p class="readline" id="tourMapStatus"></p><div id="tourMap"></div>`;
  } else if (k === 'end') {
    const main = M.layers.filter(L => isMainPass(M, L)).length;
    const P = fmtParams(M.params), hb = fmtBytes(M.h.headerEnd, 2);
    h = head(tl('Zusammenfassung', 'Summary'), tl('Vom Satz zum nächsten Token', 'From sentence to next token'), '')
      + `<div class="t-big"><div><b>${fmtInt(Array.from(S.text).length)}</b><span>${tl('Zeichen', 'characters')}</span></div><div><b>→ ${fmtInt(n)}</b><span>${tl('Tokens', 'tokens')}</span></div><div><b>→ ${fmtInt(n)} × ${d}</b><span>${tl('Zahlen', 'numbers')}</span></div><div><b>→ ${fmtInt(main)}</b><span>${tl('Blöcke', 'blocks')}</span></div><div><b>→ 1</b><span>${tl('neues Token', 'new token')}</span></div></div>`
      + `<p class="t-lede">${tl(`Alles, was das Modell weiß, steckt in ${P} Zahlen in dieser Datei. Der Header (${hb}) beschreibt nur, wie sie angeordnet sind. Selbst nachsehen: Tabs <b>Tokens</b>, <b>Embedding-Tabelle</b>, <b>Datenfluss</b> und <b>Gewichte</b>.`, `Everything the model knows is in the ${P} numbers in this file. The header (${hb}) only describes how they are arranged. See for yourself in the tabs <b>Tokens</b>, <b>Embedding table</b>, <b>Data flow</b> and <b>Weights</b>.`)}</p>`;
  }
  stage.innerHTML = `<div class="inner">${h}</div>`;
  stage.scrollTop = 0;
  // after-render hooks (real data)
  if (k === 'start') {
    const ex = examples(M);
    $('#tourEx').innerHTML = tl('Beispiele: ', 'Examples: ') + ex.map(([l], i) => `<button type="button" class="ui" data-ex="${i}">${esc(l)}</button>`).join('');
    $('#tourEx').onclick = e => { const b = e.target.closest('[data-ex]'); if (!b) return; S.text = ex[+b.dataset.ex][1]; $('#tourText').value = S.text; $('#tokText').value = S.text; S.tokSel = null; S.wordSel = null; runTokenize(); };
    $('#tourText').oninput = e => { S.text = e.target.value; $('#tokText').value = S.text; clearTimeout(tokTimer); tokTimer = setTimeout(() => { S.tokSel = null; S.wordSel = null; runTokenize(); }, 120); };
  }
  if (k === 'lookup') tourDict($('#tourDict'), ids);
  if ((k === 'rows' || k === 'matrix') && E && n) {
    (async () => {
      const uniq = [...new Set(ids.slice(0, k === 'rows' ? 14 : 200))], vecs = new Map();
      for (const id of uniq) { const cached = embView && embView.cache.get(id); if (cached) { vecs.set(id, cached); continue; } const [raw] = await M.source.readRowRange(E, id, 1); vecs.set(id, decodeRow(E, raw)); }
      if (seq !== tourSeq) return;
      updateReadStat();
      const all = [...vecs.values()].filter(Boolean), sc = all.length ? percentileAbs(all, 0.99) : 1;
      if (k === 'rows') {
        ids.slice(0, 14).forEach((id, i) => {
          const v = vecs.get(id), cv = stage.querySelector(`canvas[data-i="${i}"]`), nn = stage.querySelector(`[data-n="${i}"]`);
          if (!v) { nn.textContent = tl('im eingebauten Beispiel nicht enthalten', 'not in the built-in example'); return; }
          drawVecStrip(cv, v, sc); nn.textContent = Array.from(v.subarray(0, 10)).map(x => sig(x, 3)).join(' · ') + ' · …';
        });
      } else {
        const rowsN = Math.min(ids.length, 200), Wd = Math.min(E.dims[0], 1200), cv = $('#tourMx'), T = theme();
        cv.width = Wd; cv.height = rowsN;
        const ctx = cv.getContext('2d'), img = ctx.createImageData(Wd, rowsN);
        for (let r = 0; r < rowsN; r++) {
          const v = vecs.get(ids[r]);
          for (let x = 0; x < Wd; x++) {
            const o = (r * Wd + x) * 4;
            if (!v) { img.data[o] = T.sunkRGB[0]; img.data[o + 1] = T.sunkRGB[1]; img.data[o + 2] = T.sunkRGB[2]; img.data[o + 3] = 255; continue; }
            const a = Math.floor(x * v.length / Wd), b = Math.max(a + 1, Math.floor((x + 1) * v.length / Wd)); let best = v[a];
            for (let c = a + 1; c < b; c++) if (Math.abs(v[c]) > Math.abs(best)) best = v[c];
            const kk = colorIdx(best, sc * 1.3); img.data[o] = T.lut[kk]; img.data[o + 1] = T.lut[kk + 1]; img.data[o + 2] = T.lut[kk + 2]; img.data[o + 3] = 255;
          }
        }
        ctx.putImageData(img, 0, 0);
        const rowPx = Math.max(10, Math.min(30, Math.floor(420 / rowsN)));
        cv.style.height = rowsN * rowPx + 'px';
        $('#tourMxLabs').style.gridTemplateRows = `repeat(${rowsN}, ${rowPx}px)`;
        $('#tourMxLabs').innerHTML = ids.slice(0, rowsN).map(id => `<span>${wsHTML(disp(id))}</span>`).join('');
      }
    })();
  }
  if (k === 'block') { const L = M.layers.find(x => x.kind === 'attn') || M.layers[0]; if (L) drawFlow($('#tourFlow'), buildRecipe(M, L), { onOpen: t => { closeTour(); openTensor(t); } }); const tc = $('#tourCode'); if (tc) tc.onclick = () => { closeTour(); setTab('code'); }; }
  if (k === 'map' && E && (M._fullMap || (M.source.kind === 'demo' && M.info.demo.fullmap))) {
    const D = M._fullMap || (M._fullMap = fullMapFromDemo(M)); const fm = new FullMap($('#tourMap')); fm.setData(D); fm.find(S.tourNN != null ? disp(S.tourNN).trim() : tl('Montag', 'Monday')); $('#tourMapStatus').textContent = tl(`Alle ${fmtInt(D.n)} Tokens · exakte Nachbarn in allen ${fmtInt(E.dims[0])} Dimensionen`, `All ${fmtInt(D.n)} tokens · exact neighbors in all ${fmtInt(E.dims[0])} dimensions`);
  } else if (k === 'map' && E) {
    const tm = new WordMap($('#tourMap'), { id: 'twm', onNN: id => { S.tourNN = id; tourGo(TOUR.indexOf('nn')); } }), pr = $('#tourMapProg');
    if (!S.model._map) pr.hidden = false;
    getMap((p, what) => { pr.firstChild.style.width = (p * 100).toFixed(1) + '%'; $('#tourMapStatus').textContent = mapProgText(p, what); })
      .then(D => { if (seq !== tourSeq) return; pr.hidden = true; if (!D) { $('#tourMapStatus').textContent = tl('Keine Karte verfügbar.', 'No map available.'); return; } $('#tourMapStatus').textContent = ''; tm.setData(D); const w = S.tourNN != null ? disp(S.tourNN).trim() : tl('Montag', 'Monday'); tm.find(w); });
  }
  if (k === 'nn' && S.tourNN != null && E) {
    const H = { list: $('#tourNNList'), map: $('#tourNNMap'), status: $('#tourNNStatus'), prog: $('#tourNNProg') };
    nnSearch(S.tourNN, H);
    $('#tourNNPick').onclick = e => { const b = e.target.closest('[data-id]'); if (!b) return; S.tourNN = +b.dataset.id; document.querySelectorAll('#tourNNPick button').forEach(x => x.setAttribute('aria-current', String(x === b))); nnSearch(S.tourNN, H); };
    const go = e => { const g = e.target.closest('[data-id]'); if (!g) return; S.tourNN = +g.dataset.id; nnSearch(S.tourNN, H); };
    H.list.onclick = go; H.map.onclick = go;
  }
}
function tourDict(host, ids) {
  const M = S.model, V = M.vocabSize, W = 1000, uniq = [...new Set(ids)].slice(0, 30).sort((a, b) => a - b);
  const lanes = [], items = uniq.map(id => {
    const x = 20 + id / V * (W - 40), txt = wsPlain(disp(id)).slice(0, 14), w = txt.length * 8 + 10;
    let lane = 0; while (lanes[lane] != null && lanes[lane] > x - w / 2) lane++;
    lanes[lane] = x + w / 2; return { id, x, txt, lane };
  });
  const nl = Math.max(1, lanes.length), H = 70 + nl * 22;
  let s = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${tl('Lage der Token-IDs im Wörterbuch', 'Where the token IDs sit in the vocabulary')}"><rect x="20" y="${H - 40}" width="${W - 40}" height="14" rx="7" style="fill:var(--sunken);stroke:var(--line-2)"/>`;
  for (const v of [0, 0.25, 0.5, 0.75, 1]) s += `<text class="ax" x="${20 + v * (W - 40)}" y="${H - 8}" text-anchor="${v === 0 ? 'start' : v === 1 ? 'end' : 'middle'}">${fmtInt(Math.round(v * V))}</text>`;
  for (const it of items) {
    const y = H - 52 - it.lane * 22;
    s += `<line x1="${it.x}" y1="${y + 4}" x2="${it.x}" y2="${H - 33}" style="stroke:var(--ink-2);stroke-width:1"/><circle cx="${it.x}" cy="${H - 33}" r="4" style="fill:var(--ink)"/><text x="${it.x}" y="${y}" text-anchor="middle">${esc(it.txt)}</text>`;
  }
  host.innerHTML = s + '</svg>';
}

// ---------- Wort-Rechnen (A − B + C) ----------
const CALC_PRESETS = [['King', 'man', 'woman'], ['brother', 'man', 'woman'], ['Vater', 'Mann', 'Frau'], ['walking', 'walk', 'swim'], ['Paris', 'France', 'Germany'], ['König', 'Mann', 'Frau']];
const CALC_DE_WORDS = new Set(['Vater', 'Mann', 'Frau', 'König']);   // presets with German words: only on the German page (filtered when rendering, data-p keeps the index)
const normWord = s => s.trim().toLowerCase().normalize('NFD').replace(/\p{M}/gu, '');
function renderCalcTab() {
  const M = S.model;
  const ok = CALC_PRESETS.filter(p => !(isEN() && p.some(w => CALC_DE_WORDS.has(w))) && p.every(w => lookupWord(w) >= 0) && (M.source.kind !== 'demo' || (M.info.demo.calc && M.info.demo.calc[p.map(lookupWord).join(',')])));
  $('#calcEx').innerHTML = ok.length ? tl('Beispiele: ', 'Examples: ') + ok.map((p, i) => `<button type="button" class="ui" data-p="${CALC_PRESETS.indexOf(p)}">${esc(p[0])} − ${esc(p[1])} + ${esc(p[2])}</button>`).join('') : '';
  if (!S.calcDone || S.calcLang !== LANG) { S.calcDone = true; S.calcLang = LANG; calcRun(); }   // texts of the last result follow a language switch
}
let calcSeq = 0;
async function calcRun() {
  const M = S.model, E = M.E; if (!E) return;
  const seq = ++calcSeq, words = ['#calcA', '#calcB', '#calcC'].map(s => $(s).value.trim());
  const ids = words.map(w => /^#\d+$/.test(w) ? +w.slice(1) : lookupWord(w));
  const st = $('#calcStatus'), prog = $('#calcProg');
  const bad = ids.findIndex(i => i < 0 || i >= M.vocabSize);
  if (bad >= 0) { const w = esc(words[bad]); st.innerHTML = tl(`„<b>${w}</b>“ ist kein einzelnes Token im Wörterbuch, hat also keine eigene Zeile.`, `“<b>${w}</b>” is not a single token in the vocabulary, so it has no row of its own.`); $('#calcList').innerHTML = ''; $('#calcEq').innerHTML = ''; return; }
  let top, info;
  const rowsN = fmtInt(E.dims[1]);
  if (M.source.kind === 'demo') {
    const pre = M.info.demo.calc && M.info.demo.calc[ids.join(',')];
    if (!pre) { st.innerHTML = tl('Im eingebauten Beispiel gehen nur die vorgeschlagenen Rechnungen. Für jede andere die echte <b>.gguf-Datei öffnen</b>.', 'The built-in example only has the suggested calculations. <b>Open the real .gguf file</b> for any other.'); $('#calcList').innerHTML = ''; $('#calcEq').innerHTML = ''; return; }
    top = pre.map(([j, c]) => [c, j]); info = tl(`${rowsN} Zeilen verglichen`, `${rowsN} rows compared`);
  }
  const vec = async id => { const [r] = await M.source.readRowRange(E, id, 1); return decodeRow(E, r); };
  const vs = await Promise.all(ids.map(vec));
  let q = null;
  if (vs.every(Boolean)) { const [a, b, c] = vs.map(unitVec); q = unitVec(a.map((x, i) => x - b[i] + c[i])); }
  if (!top && isSrv()) {
    prog.hidden = false; prog.firstChild.style.width = '0%'; st.textContent = tl('Lupe-Server rechnet …', 'Lupe server computing …');
    await srvEnsure((p, m) => { prog.firstChild.style.width = (p * 100).toFixed(1) + '%'; st.textContent = tl(`Server lädt die Tabelle: ${m}`, `Server loading the table: ${srvMsg(m)}`); });
    const j = await srvJson('/api/analogy', { path: M.source.path, ids: ids.join(','), k: 40 });
    if (seq !== calcSeq) return;
    prog.hidden = true; top = j.ids.map((x, k) => [j.sims[k], x]); info = tl(`${rowsN} Zeilen verglichen · <strong>${nf(j.ms, 0)} ms</strong> · Lupe-Server (${esc(j.device)})`, `${rowsN} rows compared · <strong>${nf(j.ms, 0)} ms</strong> · Lupe server (${esc(j.device)})`);
  }
  if (!top) {
    prog.hidden = false; prog.firstChild.style.width = '0%'; st.textContent = tl(`Vergleiche mit allen ${rowsN} Zeilen …`, `Comparing with all ${rowsN} rows …`);
    const r = await scanTop(q, 40, p => { if (seq === calcSeq) prog.firstChild.style.width = (p * 100).toFixed(1) + '%'; });
    if (seq !== calcSeq) return;
    const s = nf(r.ms / 1000, 1), cores = fmtInt(r.cores);
    prog.hidden = true; top = r.top; info = tl(`${rowsN} Zeilen verglichen · <strong>${s} s</strong> · ${cores} Kern${r.cores === 1 ? '' : 'e'}`, `${rowsN} rows compared · <strong>${s} s</strong> · ${cores} core${r.cores === 1 ? '' : 's'}`);
    updateReadStat();
  }
  if (seq !== calcSeq) return;
  const hide = $('#calcHide').checked, inN = new Set(ids.map(i => normWord(disp(i)))), shown = [], hidden = [];
  for (const [c, j] of top) { if (ids.includes(j) || (hide && inN.has(normWord(disp(j))))) { if (!ids.includes(j)) hidden.push(j); continue; } shown.push([j, c]); }
  const hl = hidden.slice(0, 8).map(j => `<span class="mono">${wsHTML(disp(j))}</span>`).join(', ');
  st.innerHTML = info + (hidden.length ? tl(` · ausgeblendet: ${hl}`, ` · hidden: ${hl}`) : '');
  const list = shown.slice(0, 12), max = list.length ? list[0][1] : 1;
  $('#calcList').innerHTML = list.map(([j, c], k) => `<button type="button" class="nn-row${k === 0 ? ' q' : ''}" data-id="${j}"><span class="rk">${k + 1}</span><span class="w">${wsHTML(disp(j))}</span><span class="bt"><i style="width:${Math.max(2, c / max * 92)}%"></i></span><span class="v">${nf(c, 2)}</span></button>`).join('');
  const best = list[0] ? list[0][0] : null, bv = best != null ? await vec(best) : null;
  if (seq !== calcSeq) return;
  const rows = [['', words[0], vs[0]], ['−', words[1], vs[1]], ['+', words[2], vs[2]], ['=', tl('neuer Vektor', 'new vector'), q], ['≈', best != null ? disp(best).trim() : '–', bv]];
  const all = rows.map(r => r[2]).filter(Boolean).map(unitVec), sc = all.length ? percentileAbs(all, 0.99) : 1;
  $('#calcEq').innerHTML = rows.map(([op, w], i) => `<div class="eq-row${i === 3 ? ' sum' : ''}"><span class="op">${op}</span><span class="w">${esc(w)}</span><canvas data-i="${i}"></canvas></div>`).join('');
  rows.forEach(([, , v], i) => { const cv = $(`#calcEq canvas[data-i="${i}"]`); if (v) drawVecStrip(cv, unitVec(v), sc); else cv.hidden = true; });
}

// ---------- Bedeutungs-Landkarte ----------
const REGION_VARS = ['--c-ffn', '--c-attn', '--c-lin', '--c-emb', '--c-head', '--c-moe', '--c-mtp', '--div-pos'];
const MAP_WORKER = `
self.onmessage = e => {
  const { rid, Q, n, d, K, r0, r1 } = e.data; let last = r0;
  try {
    const res = MAPCORE.knnRows(Q, n, d, K, r0, r1, r => { if (r - last >= 16) { self.postMessage({ rid, progress: r - last }); last = r; } });
    self.postMessage({ rid, progress: r1 - last });
    self.postMessage({ rid, done: true, r0, nb: res.nb, sim: res.sim }, [res.nb.buffer, res.sim.buffer]);
  } catch (err) { self.postMessage({ rid, error: String(err && err.message || err) }); }
};`;
function mapFromDemo(M) {
  const m = M.info.demo.map; if (!m) return null;
  return { ids: m.ids, Y: Float32Array.from(m.Y), lab: Uint8Array.from(m.lab), nb: Int32Array.from(m.nb), K: m.K, curated: new Set(m.curated), demo: true };
}
async function computeMap(onP) {
  const M = S.model, E = M.E, d = E.dims[0];
  const sel = MAPCORE.select({ n: 2600, size: M.vocabSize, text: id => disp(id), isNormal: id => tokType(id) === 1, lookup: w => lookupWord(w) });
  const ids = sel.ids, n = ids.length, vecs = [], t0 = performance.now();
  for (let i = 0; i < n; i++) { const [r] = await M.source.readRowRange(E, ids[i], 1); vecs.push(decodeRow(E, r)); if (i % 100 === 0) onP(0.12 * i / n, tl('lese Zeilen', 'reading rows')); }
  const { U } = MAPCORE.prep(vecs); vecs.length = 0;
  const Q = MAPCORE.toInt8(U, n, d), K = 10;
  const nb = new Int32Array(n * K), sim = new Float32Array(n * K);
  let cores = 1;
  try {
    const src = document.getElementById('map-src').textContent + '\n' + MAP_WORKER;
    const url = URL.createObjectURL(new Blob([src], { type: 'text/javascript' }));
    cores = Math.max(2, Math.min(8, navigator.hardwareConcurrency || 4));
    const pool = Array.from({ length: cores }, () => new Worker(url)), per = Math.ceil(n / cores), rid = Math.random(); let done = 0;
    try {
      await Promise.all(pool.map((w, i) => new Promise((res, rej) => {
        const r0 = i * per, r1 = Math.min(n, r0 + per); if (r0 >= r1) return res();
        w.onmessage = e => { const m = e.data; if (m.rid !== rid) return; if (m.progress) { done += m.progress; onP(0.12 + 0.68 * done / n, tl('vergleiche alle Paare', 'comparing all pairs')); } else if (m.done) { nb.set(m.nb, m.r0 * K); sim.set(m.sim, m.r0 * K); res(); } else if (m.error) rej(new Error(m.error)); };
        w.onerror = ev => { if (ev.preventDefault) ev.preventDefault(); rej(new Error(ev.message || 'Worker-Fehler')); };
        w.postMessage({ rid, Q, n, d, K, r0, r1 });
      })));
    } finally { pool.forEach(w => w.terminate()); URL.revokeObjectURL(url); }
  } catch (e) {
    console.warn('Karte: Worker nicht verfügbar, rechne im Hauptthread', e); cores = 1;
    for (let r0 = 0; r0 < n; r0 += 40) { const r1 = Math.min(n, r0 + 40), res = MAPCORE.knnRows(Q, n, d, K, r0, r1); nb.set(res.nb, r0 * K); sim.set(res.sim, r0 * K); onP(0.12 + 0.68 * r1 / n, tl('vergleiche alle Paare', 'comparing all pairs')); await new Promise(r => setTimeout(r)); }
  }
  const Y = await MAPCORE.layout(nb, sim, n, K, MAPCORE.pca2(U, n, d), {}, p => onP(0.8 + 0.2 * p, tl('lege die Karte', 'laying out the map')));
  const lab = MAPCORE.regions(Y, n, 8);
  M.source.weightBytes += n * E.rowBytes; updateReadStat();
  return { ids, Y, lab, nb, K, curated: sel.curated, ms: performance.now() - t0, cores };
}
const mapProgText = (p, what) => tl(`Karte wird berechnet: ${what} … ${Math.round(p * 100)} %`, `Computing the map: ${what} … ${Math.round(p * 100)} %`);
async function getMap(onP) {
  const M = S.model;
  if (M._map) return M._map;
  if (M.source.kind === 'demo') return (M._map = mapFromDemo(M));
  if (!M._mapPromise) M._mapPromise = computeMap(onP).then(m => (M._map = m)).finally(() => { M._mapPromise = null; });
  return M._mapPromise;
}
function regionNames(D) {
  const k = 8, names = [];
  for (let c = 0; c < k; c++) {
    const mem = D.ids.map((id, i) => [id, i]).filter(([, i]) => D.lab[i] === c);
    let cx = 0, cy = 0; for (const [, i] of mem) { cx += D.Y[i * 2]; cy += D.Y[i * 2 + 1]; }
    cx /= mem.length || 1; cy /= mem.length || 1;
    // name a region after the most typical words: curated words closest to its centre
    const dist = i => Math.hypot(D.Y[i * 2] - cx, D.Y[i * 2 + 1] - cy);
    const pool = mem.filter(([id]) => D.curated.has(id)).length >= 3 ? mem.filter(([id]) => D.curated.has(id)) : mem;
    pool.sort((a, b) => dist(a[1]) - dist(b[1]));
    names.push({ c, n: mem.length, x: cx, y: cy, words: pool.slice(0, 3).map(([id]) => disp(id).trim()) });
  }
  return names;
}
class WordMap {
  constructor(root, opt = {}) {
    this.root = root; this.opt = opt; const id = opt.id || 'wm';
    root.innerHTML = `<div class="wm-bar"><div class="field" style="max-width:360px"><label for="${id}-q"></label><input id="${id}-q" type="search" autocomplete="off" spellcheck="false"></div><div class="seg" role="group" aria-label="Zoom"><button type="button" data-z="in">＋</button><button type="button" data-z="out">－</button><button type="button" data-z="fit"></button></div><span class="count wm-pos"></span></div><div class="wm-vp"><canvas></canvas></div><div class="wm-foot"><div class="legend wm-legend"></div><p class="cap wm-sel"></p></div>`;
    this.vp = root.querySelector('.wm-vp'); this.cv = root.querySelector('canvas'); this.q = root.querySelector('input'); this.pos = root.querySelector('.wm-pos'); this.selBox = root.querySelector('.wm-sel');
    this.s = 1; this.cx = 0; this.cy = 0; this.sel = null; this.hover = null; this.relang();
    root.querySelector('.seg').onclick = e => { const b = e.target.closest('[data-z]'); if (!b || !this.D) return; if (b.dataset.z === 'fit') this.fit(); else this.zoomAt(this.W / 2, this.H / 2, b.dataset.z === 'in' ? 1.6 : 1 / 1.6); };
    this.q.addEventListener('keydown', e => { if (e.key === 'Enter') this.find(this.q.value); });
    this.cv.addEventListener('wheel', e => { if (!this.D) return; e.preventDefault(); const b = this.cv.getBoundingClientRect(); this.zoomAt(e.clientX - b.left, e.clientY - b.top, Math.exp(-e.deltaY * 0.0015)); }, { passive: false });
    let drag = null;
    this.cv.addEventListener('pointerdown', e => { drag = { x: e.clientX, y: e.clientY, cx: this.cx, cy: this.cy, moved: false }; this.cv.setPointerCapture(e.pointerId); });
    this.cv.addEventListener('pointermove', e => {
      if (drag) { const dx = e.clientX - drag.x, dy = e.clientY - drag.y; if (Math.abs(dx) + Math.abs(dy) > 4) drag.moved = true; if (drag.moved) { this.cx = drag.cx - dx / this.s; this.cy = drag.cy - dy / this.s; hideTip(); this.draw(); return; } }
      const i = this.pick(e); if (i !== this.hover) { this.hover = i; this.draw(); }
      if (i == null) return hideTip();
      const rg = esc(this.names[this.D.lab[i]].words.join(' · '));
      showTip(`<code>${wsHTML(disp(this.D.ids[i]))}</code> · ID ${fmtInt(this.D.ids[i])}<br><span class="m">${tl(`Region: ${rg} · Klick zeigt Nachbarn`, `Region: ${rg} · click shows neighbors`)}</span>`, e);
    });
    this.cv.addEventListener('pointerup', e => { if (drag && !drag.moved) { const i = this.pick(e); this.select(i); } drag = null; });
    this.cv.addEventListener('pointerleave', () => { this.hover = null; hideTip(); this.draw(); });
    this.cv.addEventListener('dblclick', e => { const b = this.cv.getBoundingClientRect(); this.zoomAt(e.clientX - b.left, e.clientY - b.top, 2); });
    if (window.ResizeObserver) new ResizeObserver(() => this.draw()).observe(this.vp);
  }
  relang() {  // static labels in the current language (the page can switch language at runtime)
    const r = this.root;
    r.querySelector('label').textContent = tl('Wort finden', 'Find word'); this.q.placeholder = tl('z. B. Montag, King, Geld', 'e.g. Monday, king, money');
    r.querySelector('[data-z="in"]').setAttribute('aria-label', tl('Hineinzoomen', 'Zoom in')); r.querySelector('[data-z="out"]').setAttribute('aria-label', tl('Herauszoomen', 'Zoom out'));
    r.querySelector('[data-z="fit"]').textContent = tl('Ganze Karte', 'Whole map');
  }
  setData(D) {
    this.D = D; this.sel = null; this.names = regionNames(D);
    this.idx = new Map(D.ids.map((id, i) => [id, i]));
    this.root.querySelector('.wm-legend').innerHTML = this.names.map(r => `<span><i style="background:var(${REGION_VARS[r.c]})"></i>${esc(r.words.join(' · '))}</span>`).join('');
    this.selBox.innerHTML = '';
    this.fit();
  }
  fit() {
    const D = this.D, n = D.ids.length; let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
    for (let i = 0; i < n; i++) { const x = D.Y[i * 2], y = D.Y[i * 2 + 1]; if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y; }
    this.size(); this.cx = (x0 + x1) / 2; this.cy = (y0 + y1) / 2; this.s = Math.min((this.W - 60) / (x1 - x0 || 1), (this.H - 60) / (y1 - y0 || 1)); this.s0 = this.s; this.draw();
  }
  size() { this.W = this.vp.clientWidth || 800; this.H = this.vp.clientHeight || 560; }
  sx(i) { return (this.D.Y[i * 2] - this.cx) * this.s + this.W / 2; }
  sy(i) { return (this.D.Y[i * 2 + 1] - this.cy) * this.s + this.H / 2; }
  zoomAt(px, py, f) { const wx = (px - this.W / 2) / this.s + this.cx, wy = (py - this.H / 2) / this.s + this.cy; this.s = Math.max(this.s0 * 0.6, Math.min(this.s0 * 40, this.s * f)); this.cx = wx - (px - this.W / 2) / this.s; this.cy = wy - (py - this.H / 2) / this.s; this.draw(); }
  pick(e) { if (!this.D) return null; const b = this.cv.getBoundingClientRect(), x = e.clientX - b.left, y = e.clientY - b.top; let best = null, bd = 100; for (let i = 0; i < this.D.ids.length; i++) { const dx = this.sx(i) - x, dy = this.sy(i) - y, dd = dx * dx + dy * dy; if (dd < bd) { bd = dd; best = i; } } return best; }
  find(word) {
    if (!this.D) return; const w = normWord(word); if (!w) return;
    let i = this.D.ids.findIndex(id => normWord(disp(id)) === w); if (i < 0) i = this.D.ids.findIndex(id => normWord(disp(id)).startsWith(w));
    if (i < 0) { const n = fmtInt(this.D.ids.length); this.q.setCustomValidity(tl(`Nicht auf der Karte (sie zeigt ${n} häufige Wörter)`, `Not on the map (it shows ${n} common words)`)); this.q.reportValidity(); setTimeout(() => this.q.setCustomValidity(''), 2500); return; }
    this.select(i); this.s = Math.max(this.s, this.s0 * 5); this.cx = this.D.Y[i * 2]; this.cy = this.D.Y[i * 2 + 1]; this.draw();
  }
  select(i) {
    this.sel = i; this.draw();
    if (i == null) return;
    const D = this.D, id = D.ids[i], nbs = Array.from(D.nb.slice(i * D.K, i * D.K + D.K)).filter(j => j !== i);
    const wb = `<b class="mono">${wsHTML(disp(id))}</b>`, nl = nbs.map(j => `<button type="button" class="linkbtn mono" data-i="${j}">${wsHTML(disp(D.ids[j]).trim())}</button>`).join(', ');
    this.selBox.innerHTML = tl(`Nächste Nachbarn von ${wb}: ${nl}`, `Nearest neighbors of ${wb}: ${nl}`) + ` <button type="button" class="btn" style="padding:2px 9px;font-size:12px" data-nn="${id}">${tl('Im ganzen Wörterbuch suchen', 'Search the whole vocabulary')}</button>`;
    this.selBox.onclick = e => { const b = e.target.closest('[data-i]'); if (b) { const j = +b.dataset.i; this.select(j); this.cx = D.Y[j * 2]; this.cy = D.Y[j * 2 + 1]; this.draw(); return; } const nn = e.target.closest('[data-nn]'); if (nn) { if (this.opt.onNN) this.opt.onNN(+nn.dataset.nn); else openNN(+nn.dataset.nn); } };
  }
  draw() { if (!this.raf) this.raf = requestAnimationFrame(() => { this.raf = 0; this.paint(); }); }
  paint() {
    if (!this.D || !this.root.offsetParent) return;
    this.size();
    const dpr = window.devicePixelRatio || 1, W = this.W, H = this.H, D = this.D, n = D.ids.length, T = theme();
    if (this.cv.width !== Math.round(W * dpr) || this.cv.height !== Math.round(H * dpr)) { this.cv.width = Math.round(W * dpr); this.cv.height = Math.round(H * dpr); this.cv.style.width = W + 'px'; this.cv.style.height = H + 'px'; }
    const ctx = this.cv.getContext('2d'); ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.fillStyle = T.surface; ctx.fillRect(0, 0, W, H);
    const col = REGION_VARS.map(v => css(v)), zoom = this.s / this.s0;
    const vis = []; for (let i = 0; i < n; i++) { const x = this.sx(i), y = this.sy(i); if (x > -40 && x < W + 40 && y > -40 && y < H + 40) vis.push(i); }
    // land: soft overlapping discs per region
    const R = Math.max(10, Math.min(26, 14 * Math.sqrt(zoom)));
    ctx.globalAlpha = T.dark ? 0.09 : 0.07;
    for (const i of vis) { ctx.fillStyle = col[D.lab[i]]; ctx.beginPath(); ctx.arc(this.sx(i), this.sy(i), R, 0, 6.2832); ctx.fill(); }
    ctx.globalAlpha = 1;
    // neighbour lines of the selection
    const sel = this.sel, nbs = sel != null ? Array.from(D.nb.slice(sel * D.K, sel * D.K + D.K)).filter(j => j !== sel) : [];
    if (sel != null) { ctx.strokeStyle = T.ink2; ctx.lineWidth = 1.2; for (const j of nbs) { ctx.beginPath(); ctx.moveTo(this.sx(sel), this.sy(sel)); ctx.lineTo(this.sx(j), this.sy(j)); ctx.stroke(); } }
    for (const i of vis) { const big = D.curated.has(D.ids[i]); ctx.fillStyle = col[D.lab[i]]; ctx.beginPath(); ctx.arc(this.sx(i), this.sy(i), big ? 3.4 : 2.4, 0, 6.2832); ctx.fill(); }
    // labels, greedy without overlap
    const occ = [], free = (x, y, w, h) => !occ.some(b => x < b[0] + b[2] && x + w > b[0] && y < b[1] + b[3] && y + h > b[1]);
    const order = vis.slice().sort((a, b) => ((b === sel || nbs.includes(b) || b === this.hover) - (a === sel || nbs.includes(a) || a === this.hover)) || (D.curated.has(D.ids[b]) - D.curated.has(D.ids[a])) || D.ids[a] - D.ids[b]);
    const maxLabels = Math.min(600, Math.round(70 * zoom * zoom + 60));
    let placed = 0;
    for (const i of order) {
      if (placed >= maxLabels) break;
      const hot = i === sel || i === this.hover, near = nbs.includes(i), txt = disp(D.ids[i]).trim();
      ctx.font = `${hot ? 700 : near || D.curated.has(D.ids[i]) ? 600 : 400} ${hot ? 14 : 12}px ${T.mono}`;
      const w = ctx.measureText(txt).width, x = this.sx(i) + 5, y = this.sy(i) + 4;
      if (!hot && !free(x - 2, y - 11, w + 4, 14)) continue;
      occ.push([x - 2, y - 11, w + 4, 14]); placed++;
      ctx.lineWidth = 3.5; ctx.strokeStyle = T.surface; ctx.strokeText(txt, x, y);
      ctx.fillStyle = hot || near ? T.ink : T.ink2; ctx.fillText(txt, x, y);
    }
    if (sel != null) { ctx.strokeStyle = T.ink; ctx.lineWidth = 2; ctx.beginPath(); ctx.arc(this.sx(sel), this.sy(sel), 7, 0, 6.2832); ctx.stroke(); }
    // region names when zoomed out
    if (zoom < 1.8) {
      ctx.font = `700 13px ${T.ui}`; ctx.textAlign = 'center';
      for (const r of this.names) { const x = (r.x - this.cx) * this.s + W / 2, y = (r.y - this.cy) * this.s + H / 2 - 18; const t = r.words.join(' · '); ctx.lineWidth = 5; ctx.strokeStyle = T.surface; ctx.globalAlpha = 0.9; ctx.strokeText(t, x, y); ctx.fillStyle = col[r.c]; ctx.globalAlpha = 1; ctx.fillText(t, x, y); }
      ctx.textAlign = 'left';
    }
    const nw = fmtInt(n), zs = nf(zoom, 1);
    this.pos.textContent = tl(`${nw} Wörter · Zoom ${zs}×`, `${nw} words · zoom ${zs}×`);
  }
}
let wordMap = null;
async function renderMapTab() {
  const M = S.model, st = $('#mapStatus'), prog = $('#mapProg');
  if (await renderFullMap()) return;
  $('#mapView').hidden = false;
  if (!wordMap) wordMap = new WordMap($('#mapView'), { id: 'wm' }); else wordMap.relang();
  if (M._map) { if (wordMap.D !== M._map) wordMap.setData(M._map); else if (wordMap.sel != null) wordMap.select(wordMap.sel); else wordMap.draw(); mapInfo(M._map); return; }
  prog.hidden = false; prog.firstChild.style.width = '0%';
  const D = await getMap((p, what) => { prog.firstChild.style.width = (p * 100).toFixed(1) + '%'; st.textContent = mapProgText(p, what); });
  prog.hidden = true;
  if (!D) { st.textContent = tl('Für dieses Beispiel gibt es keine Karte. Öffne eine .gguf-Datei.', 'No map for this example. Open a .gguf file.'); return; }
  if (S.model !== M) return;
  wordMap.setData(D); mapInfo(D);
}
function mapInfo(D) {
  const n = fmtInt(D.ids.length), d = fmtInt(S.model.E.dims[0]), s = nf((D.ms || 0) / 1000, 1), c = fmtInt(D.cores || 1);
  $('#mapStatus').innerHTML = tl('<b>Schnellkarte</b>', '<b>Quick map</b>') + ib('mapQuick') + tl(` · ${n} Wörter · ${d} Dimensionen`, ` · ${n} words · ${d} dimensions`)
    + (D.demo ? '' : tl(` · <strong>${s} s</strong> · ${c} Kern${D.cores === 1 ? '' : 'e'}`, ` · <strong>${s} s</strong> · ${c} core${D.cores === 1 ? '' : 's'}`));
}

// ---------- Bauplan: Datei vs. Programm ----------
function renderBauplan() {
  const M = S.model, a = M.arch, name = M.meta['general.name'] || '', V = VERIFIED[a];
  const hp = k => metaNum(M, k), hv = k => showVal(hp(k));
  const sizes = [hp('block_count') != null && tl(`${hv('block_count')} Blöcke`, `${hv('block_count')} blocks`), hp('embedding_length') != null && tl(`Breite ${hv('embedding_length')}`, `width ${hv('embedding_length')}`), hp('attention.head_count') != null && tl(`${hv('attention.head_count')} Köpfe`, `${hv('attention.head_count')} heads`), hp('feed_forward_length') != null && `FFN ${hv('feed_forward_length')}`, hp('context_length') != null && tl(`Kontext ${hv('context_length')}`, `context ${hv('context_length')}`)].filter(Boolean).join(', ');
  const sets = Object.keys(M.meta).filter(k => k.startsWith(a + '.') && !/block_count|embedding_length|head_count$|feed_forward_length|context_length/.test(k)).slice(0, 6).map(k => `<code>${esc(k.slice(a.length + 1))}</code>`).join(', ');
  const fileFacts = [
    tl(`Name des Bauplans: <code>general.architecture = ${esc(a)}</code>`, `Blueprint name: <code>general.architecture = ${esc(a)}</code>`) + (name ? tl(` (Modell: ${esc(name)})`, ` (model: ${esc(name)})`) : ''),
    tl(`Größen: ${sizes}`, `Sizes: ${sizes}`),
    tl(`Einstellungen: ${sets} …`, `Settings: ${sets} …`),
    tl(`Wörterbuch und Merge-Regeln (${fmtInt(M.vocabSize)} Tokens), Chat-Vorlage`, `Vocabulary and merge rules (${fmtInt(M.vocabSize)} tokens), chat template`) + (M.meta['general.sampling.temp'] != null ? tl(', empfohlene Sampling-Werte', ', recommended sampling values') : ''),
    tl(`${fmtInt(M.h.nTensors)} Tabellen: Name, Form, Speicherformat, Position`, `${fmtInt(M.h.nTensors)} tables: name, shape, storage format, position`),
  ];
  const lin = M.layers.some(L => L.kind === 'lin');
  const progFacts = [
    tl('Reihenfolge der Tabellen (Normieren → Attention → aufaddieren → FFN …)', 'Order of the tables (normalize → attention → add → FFN …)'),
    tl(`Was jeder Rechenschritt tut: RMSNorm, RoPE, Softmax, SiLU${lin ? ', Gated-Delta-Regel' : ''}`, `What each step does: RMSNorm, RoPE, softmax, SiLU${lin ? ', gated delta rule' : ''}`),
    a === 'qwen35' ? tl('Welcher Block welche Variante nutzt: <code>(i + 1) % 4 != 0</code> → lineare Attention (die 4 steht in der Datei)', 'Which block uses which variant: <code>(i + 1) % 4 != 0</code> → linear attention (the 4 is in the file)') : tl('Welcher Block welche Variante nutzt (die Schalter stehen in der Datei)', 'Which block uses which variant (the switches are in the file)'),
    tl('Entpacken der Zahlen (Q4_K, Q6_K …), KV-Cache, Wahl des nächsten Tokens', 'Unpacking the numbers (Q4_K, Q6_K …), KV cache, picking the next token'),
  ];
  $('#bpLede').innerHTML = tl('Die Rechenvorschrift eines Modells steckt <b>nicht</b> in der GGUF-Datei. Der Hersteller veröffentlicht sie als Python-Code; llama.cpp (in LM Studio eingebaut) baut sie für jede Architektur in C++ nach. In der Datei stehen nur die Zahlen, die Größen und der <b>Name</b> des Bauplans.',
    'How a model computes is <b>not</b> stored in the GGUF file. The maker publishes it as Python code; llama.cpp (built into LM Studio) rebuilds it in C++ for each architecture. The file holds only the numbers, the sizes and the <b>name</b> of the blueprint.');
  const qwen = a.startsWith('qwen'), gem = a.startsWith('gemma'), ft = GGUF.FILE_TYPES[M.meta['general.file_type']];
  const ref = qwen ? `transformers: <code>models/${a === 'qwen35' ? 'qwen3_5' : esc(a)}/modeling_….py</code>` : gem ? `transformers: <code>models/${esc(a)}/modeling_….py</code>` : tl('Referenz-Code des Herstellers (meist Python)', 'the maker’s reference code (usually Python)');
  const btn = `<button type="button" class="btn" id="bpCode" style="padding:2px 9px;font-size:12px;margin-top:4px">${tl('Im Original ansehen →', 'View the original →')}</button>`;   // here, not in #bpNote: that one is only shown as a copy in the ⓘ popover
  $('#bpDiagram').innerHTML = `<div class="bp-col">${tl(`<b>1 · Hersteller</b><span>${esc(name || a)}</span><ul><li>Gewichte (z. B. <code>model.safetensors</code>)</li><li>Größen (<code>config.json</code>)</li><li><b>Bauplan als Python-Code</b><br>${ref}</li></ul>`, `<b>1 · Model maker</b><span>${esc(name || a)}</span><ul><li>Weights (e.g. <code>model.safetensors</code>)</li><li>Sizes (<code>config.json</code>)</li><li><b>Blueprint as Python code</b><br>${ref}</li></ul>`)}</div>`
    + `<div class="bp-ar" aria-hidden="true">→</div><div class="bp-col">${tl(`<b>2 · Umwandeln nach GGUF</b><span>Zahlen und Größen → eine Datei</span><ul><li>alle Tabellen${ft ? ` (gepackt, z. B. ${esc(ft)})` : ''}</li><li>Größen und Einstellungen</li><li>nur der <b>Name</b>: <code>${esc(a)}</code></li><li class="x">kein Programmcode</li></ul>`, `<b>2 · Convert to GGUF</b><span>numbers and sizes → one file</span><ul><li>all tables${ft ? ` (packed, e.g. ${esc(ft)})` : ''}</li><li>sizes and settings</li><li>only the <b>name</b>: <code>${esc(a)}</code></li><li class="x">no program code</li></ul>`)}</div>`
    + `<div class="bp-ar" aria-hidden="true">→</div><div class="bp-col">${tl(`<b>3 · llama.cpp / LM Studio</b><span>liest den Namen, wählt den passenden Nachbau</span><ul><li><b>Bauplan als C++-Code</b><br><code>src/models/${esc(a)}.cpp</code>${V ? ' (hier geprüft)' : ''}<br>${btn}</li><li>von Hand aus dem Python-Code nachgebaut</li><li>rund 150 Architekturen, je eine Datei</li></ul>`, `<b>3 · llama.cpp / LM Studio</b><span>reads the name, picks the matching rebuild</span><ul><li><b>Blueprint as C++ code</b><br><code>src/models/${esc(a)}.cpp</code>${V ? ' (verified here)' : ''}<br>${btn}</li><li>rebuilt by hand from the Python code</li><li>about 150 architectures, one file each</li></ul>`)}</div>`;
  $('#bpCode').onclick = () => { setTab('code'); window.scrollTo({ top: 0 }); };
  $('#bpFile').innerHTML = fileFacts.map(x => `<li>${x}</li>`).join('');
  $('#bpProg').innerHTML = progFacts.map(x => `<li>${x}</li>`).join('');
  const q38 = name && /3\.8/.test(name) && a === 'qwen35';
  $('#bpNote').innerHTML = tl(`<b>Darum laufen neue Modelle oft erst nach einem Update von LM Studio:</b> Kennt llama.cpp den Namen nicht, bricht das Laden ab („unknown model architecture“). Ein neues Modell mit bekanntem Namen läuft ohne neuen Code${q38 ? ` (${esc(name)} nutzt den Bauplan „qwen35“)` : ''}. ONNX speichert den Rechenweg mit, GGUF bewusst nicht.`, `<b>That’s why new models often run only after an LM Studio update:</b> if llama.cpp doesn’t know the name, loading stops (“unknown model architecture”). A new model with a known name runs without new code${q38 ? ` (${esc(name)} uses the “qwen35” blueprint)` : ''}. ONNX stores the computation steps too; GGUF deliberately doesn’t.`);
}

// =====================================================================
// Lupe-Server (optional): heavy work on GPU/CPU, model list, raw columns
// =====================================================================
const SRV = { on: false, info: null, device: 'auto' };
try { SRV.device = localStorage.getItem('lupe-device') || 'auto'; } catch {}
const MODEL_HISTORY_KEY = 'lupe-model-history';
SRV.recent = []; SRV.historyPersistent = true;
try {
  const saved = JSON.parse(localStorage.getItem(MODEL_HISTORY_KEY) || '[]');
  if (Array.isArray(saved)) SRV.recent = [...new Set(saved.filter(p => typeof p === 'string' && p))].slice(0, 30);
} catch { SRV.historyPersistent = false; }
const srvUrl = (p, q = {}) => p + '?' + new URLSearchParams({ ...q, device: SRV.device }).toString();
async function srvJson(p, q) { const r = await fetch(srvUrl(p, q)); const j = await r.json(); if (!r.ok) throw new Error(j.error || r.statusText); return j; }
const sleepMs = ms => new Promise(r => setTimeout(r, ms));
async function pollJob(job, onP) {
  for (;;) {
    if (job.state === 'fertig') return job;
    if (job.state === 'fehler') throw new Error(job.message);
    onP && onP(job.progress || 0, job.message || '');
    await sleepMs(350);
    job = await srvJson('/api/job', { id: job.id });
  }
}
class HttpBlob { // Blob-like view of a file on the Lupe-Server (HTTP Range)
  constructor(path, size) { this.path = path; this.size = size; }
  slice(a, b) { const u = srvUrl('/api/file', { path: this.path }), s = a, e = Math.min(b, this.size); return { arrayBuffer: () => fetch(u, { headers: { Range: `bytes=${s}-${e - 1}` } }).then(r => { if (!r.ok) throw new Error('Server: ' + r.status); return r.arrayBuffer(); }) }; }
}
async function detectServer() {
  if (location.protocol === 'file:') {                 // opened as a file while the server runs: offer the server page
    if (await probeLocalServer()) { SRV.fileOnly = true; renderServerBar(); }
    return;
  }
  if (!/^https?:$/.test(location.protocol)) return;
  try { const r = await fetch('/api/info'); if (!r.ok) return; const j = await r.json(); if (j.server !== 'GGUF-Lupe') return; SRV.info = j; SRV.on = true; } catch { return; }
  await renderServerBar();
  updateReadStat();
}
function devLabel(d) { return d.kind === 'GPU' ? `GPU · ${d.name}${d.mem_gb ? ' · ' + nf(d.mem_gb, 0) + ' GB' : ''}` : `CPU · ${d.threads || '?'} ${tl('Threads', 'threads')}${d.mem_gb ? ' · ' + nf(d.mem_gb, 0) + ' GB RAM' : ''}`; }
// safe to call again at any time (language switch): no server = no bar; keeps the loaded model list and the chosen model
async function renderServerBar() {
  const bar = $('#srvBar'), info = SRV.info;
  if (!info) { if (SRV.fileOnly) { bar.hidden = false; bar.innerHTML = `<span class="srv-dot" aria-hidden="true"></span>${tl('<b>Lupe-Server läuft.</b> Als Datei geöffnet: nur das Beispiel.', '<b>Lupe server is running.</b> Opened as a file: example only.')} <a class="btn primary" href="http://127.0.0.1:8765/">${tl('Über den Server öffnen', 'Open via the server')}</a>`; } return; }
  const cur = $('#srvModel'), keep = cur && cur.value !== '' && SRV.models ? (SRV.models[+cur.value] || {}).path : undefined;
  bar.hidden = false;
  const devs = info.devices, gpu = devs.find(d => d.kind === 'GPU'), auto = devLabel(devs.find(d => d.id === info.default_device) || devs[0]).split(' · ')[0];
  const opts = [['auto', tl(`Automatisch (${auto})`, `Automatic (${auto})`)], ...devs.map(d => [d.id, devLabel(d)])], dir = esc(info.cache_dir || '~/.cache/gguf-lupe');
  bar.innerHTML = `<span class="srv-dot" aria-hidden="true"></span><b>${tl('Lupe-Server verbunden', 'Lupe server connected')}</b>
    <label for="srvDev">${tl('Rechnen auf', 'Compute on')}</label><select id="srvDev">${opts.map(([v, l]) => `<option value="${esc(v)}"${v === SRV.device ? ' selected' : ''}>${esc(l)}</option>`).join('')}</select>
    <label for="srvModel">${tl('Modell', 'Model')}</label><select id="srvModel"><option value="">${tl('lade Liste …', 'loading list …')}</option></select><button type="button" class="btn" id="srvOpen">${tl('Öffnen', 'Open')}</button>
    <details><summary>${tl('ⓘ Speicherorte', 'ⓘ Storage locations')}</summary><p class="cap">${tl(`Berechnete Karten: <code>${dir}</code> · eine Datei je Modell, beim nächsten Öffnen wiederverwendet.`, `Computed maps: <code>${dir}</code> · one file per model, reused next time.`)}</p><p class="cap" id="srvHistoryInfo"></p><p class="cap">${tl('GGUF-Dateien bleiben, wo sie sind; entpackte Tabellen liegen nur im Arbeitsspeicher. Im Browser gewählte Dateien musst du neu auswählen.', 'GGUF files stay where they are; unpacked tables live only in memory. Files picked in the browser must be picked again.')}</p></details>`;
  $('#srvDev').onchange = e => { SRV.device = e.target.value; try { localStorage.setItem('lupe-device', SRV.device); } catch {} if (S.model) S.model._srvLoaded = null; };
  const models = SRV.models || await fetch('/api/models').then(r => r.json()).catch(() => []);
  SRV.models = models;
  renderServerModels(keep);
  $('#srvOpen').onclick = () => { const value = $('#srvModel').value, m = value === '' ? null : SRV.models[+value]; if (m) openServerModel(m); };
}
function renderServerModels(preferredPath) {
  const models = SRV.models || [], select = $('#srvModel');
  const option = i => { const m = models[i]; return `<option value="${i}" title="${esc(m.path)}">${esc(m.name)} · ${fmtBytes(m.size)}${m.folder && m.folder !== '.' ? ' · ' + esc(m.folder) : ''}</option>`; };
  const recent = SRV.recent.map(path => models.findIndex(m => m.path === path)).filter(i => i >= 0);
  select.innerHTML = models.length ? (recent.length ? `<optgroup label="${tl('Zuletzt geöffnet', 'Recently opened')}">${recent.map(option).join('')}</optgroup>` : '') + `<optgroup label="${tl('Alle gefundenen Modelle', 'All models found')}">${models.map((_, i) => option(i)).join('')}</optgroup>` : `<option value="">${tl('keine .gguf-Dateien in den Modell-Ordnern', 'no .gguf files in the model folders')}</option>`;
  const selected = models.findIndex(m => m.path === preferredPath);
  if (selected >= 0) select.value = String(selected);
  $('#srvOpen').disabled = !models.length;
  $('#srvHistoryInfo').textContent = SRV.historyPersistent
    ? tl(`Verlauf: die letzten 30 Modellpfade, im lokalen Speicher dieses Browsers für ${location.origin} (lupe-model-history). Beim nächsten Besuch ist das letzte Modell vorausgewählt.`, `History: the last 30 model paths, in this browser’s local storage for ${location.origin} (lupe-model-history). On your next visit the last model is preselected.`)
    : tl('Kein dauerhafter Verlauf: Der Browser sperrt den lokalen Speicher. „Zuletzt geöffnet“ gilt nur bis zum Neuladen.', 'No lasting history: the browser blocks local storage. “Recently opened” lasts only until you reload.');
}
function rememberServerModel(path) {
  SRV.recent = [path, ...SRV.recent.filter(p => p !== path)].slice(0, 30);
  try { localStorage.setItem(MODEL_HISTORY_KEY, JSON.stringify(SRV.recent)); SRV.historyPersistent = true; }
  catch { SRV.historyPersistent = false; }
  renderServerModels(path);
}
async function openServerModel(m) {
  const seq = SRV.openSeq = (SRV.openSeq || 0) + 1;
  $('#modePill').textContent = tl('Lese Header …', 'Reading header …');
  const t0 = performance.now();
  try {
    const blob = new HttpBlob(m.path, m.size), h = await GGUF.parseBlob(blob);
    if (seq !== SRV.openSeq) return;
    const src = new FileSource(blob); src.kind = 'server'; src.path = m.path;
    setModel(buildModel(h, src, { fileName: m.name, parseMs: performance.now() - t0, server: true }));
    rememberServerModel(m.path);
  } catch (e) { if (seq !== SRV.openSeq) return; console.error(e); showError(e); if (S.model) renderFileBar(); }
}
async function srvEnsure(onP, M = S.model) {
  if (M._srvLoaded === SRV.device) return;
  const r = await srvJson('/api/load', { path: M.source.path });
  if (r.state !== 'fertig') await pollJob(r, onP);
  M._srvLoaded = SRV.device;
}
const isSrv = () => S.model && S.model.source.kind === 'server';

// =====================================================================
// Full word map: every token, raw cosine in all dimensions
// =====================================================================
function decodeMapBin(buf, cachePath) {
  const dv = new DataView(buf), hl = dv.getUint32(8, true);
  const hdr = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 12, hl)));
  let o = 12 + hl; o += (4 - o % 4) % 4;
  const n = hdr.n, k = hdr.k, h16 = new Uint16Array(buf, o, n * 2), Y = new Float32Array(n * 2);
  for (let i = 0; i < n * 2; i++) Y[i] = GGUF.F16[h16[i]];
  const nb = new Int32Array(buf.slice(o + n * 4, o + n * 4 + n * k * 4));
  return { hdr, n, Y, nb, k, cachePath };
}
function fullMapFromDemo(M) {
  const fm = M.info.demo.fullmap; if (!fm) return null;
  const n = fm.n, k = fm.k, c = b64(fm.coords), h16 = new Uint16Array(c.buffer, c.byteOffset, n * 2), Y = new Float32Array(n * 2);
  for (let i = 0; i < n * 2; i++) Y[i] = GGUF.F16[h16[i]];
  const p = b64(fm.nb), nb = new Int32Array(n * k);
  for (let i = 0; i < n * k; i++) nb[i] = p[i * 3] | (p[i * 3 + 1] << 8) | (p[i * 3 + 2] << 16);
  return { hdr: fm.hdr, n, Y, nb, k, demo: true };
}
const CAT_COLORS = { lat: '--c-ffn', han: '--c-attn', kana: '--c-head', hang: '--c-lin', cyr: '--c-mtp', arab: '--c-emb', thai: '--c-moe', ind: '--div-pos', grk: '--c-moe', heb: '--c-emb', oth: '--muted', num: '--ink-2', pun: '--ink-2', emo: '--c-head', ws: '--c-norm', byte: '--c-norm', spec: '--ink', res: '--c-other' };
const catName = c => isEN() && c[2] ? c[2] : c[1];   // label of a CATS entry (an English label may sit at index 2)
let FM_N = 0;
class FullMap {
  constructor(root) {
    this.root = root; const u = this.uid = 'fm' + (++FM_N);
    root.innerHTML = `<div class="fm-bar">
        <div class="field" style="max-width:300px"><label for="${u}Q"></label><input id="${u}Q" data-fmq type="search" autocomplete="off" spellcheck="false"></div>
        <div class="seg" role="group" data-g="zoom"><button type="button" data-z="in">＋</button><button type="button" data-z="out">－</button><button type="button" data-z="fit"></button></div>
        <button type="button" class="toggle" data-lens aria-pressed="false"></button>
        <div class="seg" role="group" data-g="color"><button type="button" data-color="cat" aria-pressed="true"></button><button type="button" data-color="dim" aria-pressed="false"></button></div>
        <div class="seg" role="group" data-g="lay"><button type="button" data-lay="map" aria-pressed="true"></button><button type="button" data-lay="xy" aria-pressed="false"></button><button type="button" data-lay="xyz" aria-pressed="false"></button></div>
      </div>
      <div class="fm-dims" hidden></div>
      <div class="fm-filters"></div>
      <div class="fm-vp"><canvas></canvas><canvas class="fm-lens" hidden></canvas></div>
      <p class="count fm-pos"></p>
      <div class="fm-sel note" hidden></div>`;
    this.vp = root.querySelector('.fm-vp'); this.cv = root.querySelector('canvas'); this.lens = root.querySelector('.fm-lens');
    this.pos = root.querySelector('.fm-pos'); this.selBox = root.querySelector('.fm-sel'); this.dimsBox = root.querySelector('.fm-dims'); this.filtBox = root.querySelector('.fm-filters');
    this.s = 1; this.cx = 0; this.cy = 0; this.sel = null; this.hover = null; this.colorMode = 'cat'; this.layMode = 'map'; this.dimA = 0; this.dimB = 1; this.dimC = 2; this.rotY = -0.65; this.rotX = 0.4; this.axisSrc = 'dims'; this.pcs = [0, 1, 2]; this.lensOn = false; this.extra = [];
    this.relang();
    root.querySelector('[data-g="zoom"]').onclick = e => { const b = e.target.closest('[data-z]'); if (!b || !this.D) return; if (b.dataset.z === 'fit') this.fit(); else this.zoomAt(this.W / 2, this.H / 2, b.dataset.z === 'in' ? 1.8 : 1 / 1.8); };
    root.querySelector('[data-lens]').onclick = e => { this.lensOn = !this.lensOn; e.currentTarget.setAttribute('aria-pressed', String(this.lensOn)); this.lens.hidden = !this.lensOn; this.cv.style.cursor = this.lensOn ? 'zoom-in' : ''; };
    root.querySelector('[data-g="color"]').onclick = e => { const b = e.target.closest('[data-color]'); if (b) this.setColor(b.dataset.color); };
    root.querySelector('[data-g="lay"]').onclick = e => { const b = e.target.closest('[data-lay]'); if (b) this.setLayout(b.dataset.lay); };
    root.querySelector('[data-fmq]').addEventListener('keydown', e => { if (e.key === 'Enter') this.find(e.target.value); });
    this.cv.addEventListener('wheel', e => { if (!this.D) return; e.preventDefault(); const b = this.cv.getBoundingClientRect(); this.zoomAt(e.clientX - b.left, e.clientY - b.top, Math.exp(-e.deltaY * 0.0015)); }, { passive: false });
    let drag = null;
    this.cv.addEventListener('pointerdown', e => { drag = { x: e.clientX, y: e.clientY, cx: this.cx, cy: this.cy, ry: this.rotY, rx: this.rotX, moved: false }; this.cv.setPointerCapture(e.pointerId); });
    this.cv.addEventListener('pointermove', e => {
      const b = this.cv.getBoundingClientRect(); this.mx = e.clientX - b.left; this.my = e.clientY - b.top;
      if (drag) { const dx = e.clientX - drag.x, dy = e.clientY - drag.y; if (Math.abs(dx) + Math.abs(dy) > 4) drag.moved = true; if (drag.moved) {
        if (this.layMode === 'xyz' && this.P3) { this.rotY = drag.ry + dx * 0.008; this.rotX = Math.max(-1.5, Math.min(1.5, drag.rx + dy * 0.008)); this.project3(); }   // dragging turns the cloud
        else { this.cx = drag.cx - dx / this.s; this.cy = drag.cy + dy / this.s; }
        hideTip(); this.draw(); return; } }
      if (this.lensOn) { this.drawLens(); hideTip(); return; }
      const i = this.pick(this.mx, this.my); if (i !== this.hover) { this.hover = i; this.draw(); }
      if (i == null) return hideTip();
      showTip(`<code>${wsHTML(disp(i))}</code> · ID ${fmtInt(i)}<br><span class="m">${catName(CATS[S.model._cat[i]])}${this.vals ? ` · Dimension ${fmtInt(this.dimA)}: ${sig(this.vals[i], 3)}` : ''}</span>`, e);
    });
    this.cv.addEventListener('pointerup', e => {
      if (drag && !drag.moved) { if (this.lensOn) { this.zoomAt(this.mx, this.my, 5); this.drawLens(); } else this.select(this.pick(this.mx, this.my)); }
      drag = null;
    });
    this.cv.addEventListener('pointerleave', () => { this.hover = null; hideTip(); if (this.lensOn) { const c = this.lens.getContext('2d'); c.clearRect(0, 0, this.lens.width, this.lens.height); } this.draw(); });
    if (window.ResizeObserver) new ResizeObserver(() => this.draw()).observe(this.vp);
  }
  // language switch: rewrite the static labels, then the generated parts (filters, axis controls, values, selection) and repaint;
  // paint() calls it by itself when the language changed since the last time
  relang() {
    const q = s => this.root.querySelector(s), T = (s, t) => { q(s).textContent = t; }, A = (s, a, v) => q(s).setAttribute(a, v);
    this.en = isEN();
    T(`label[for="${this.uid}Q"]`, tl('Token finden', 'Find token')); A('[data-fmq]', 'placeholder', tl('jedes Token, z. B. König', 'any token, e.g. King'));
    A('[data-g="zoom"]', 'aria-label', 'Zoom'); A('[data-z="in"]', 'aria-label', tl('Hineinzoomen', 'Zoom in')); A('[data-z="out"]', 'aria-label', tl('Herauszoomen', 'Zoom out')); T('[data-z="fit"]', tl('Ganze Karte', 'Whole map'));
    T('[data-lens]', tl('🔍 Lupe', '🔍 Magnifier')); A('[data-lens]', 'title', tl('Klick in die Karte vergrößert die Stelle', 'Click the map to zoom in on that spot'));
    A('[data-g="color"]', 'aria-label', tl('Färbung', 'Coloring')); T('[data-color="cat"]', tl('Farbe: Schrift', 'Color: script')); T('[data-color="dim"]', tl('Farbe: eine Dimension', 'Color: one dimension'));
    A('[data-g="lay"]', 'aria-label', tl('Anordnung', 'Layout')); T('[data-lay="map"]', tl('Anordnung: Landkarte', 'Layout: map')); T('[data-lay="xy"]', tl('Zwei Dimensionen', 'Two dimensions')); T('[data-lay="xyz"]', tl('Drei Dimensionen', 'Three dimensions')); A('[data-lay="xyz"]', 'title', tl('Ziehen dreht die Wolke', 'Drag to rotate the cloud'));
    if (!this.D) return;
    if (this === fullMap && S.model && this.D === S.model._fullMap) fullMapInfo(this.D);   // its status line sits outside the map
    this.renderFilters(); if (!this.dimsBox.hidden) { this.renderDims(); this.renderStats(); }
    if (this.msgF && !this.selBox.hidden) this.msg(this.msgF); else if (this.sel != null) this.select(this.sel);
    this.draw();
  }
  setData(D) {
    const M = S.model; buildVocabCats();
    this.model = M; this.dseq = (this.dseq || 0) + 1; clearTimeout(this.dt); this.dimCache = new Map();
    this.dimA = Math.min(this.dimA, M.E.dims[0] - 1); this.dimB = Math.min(this.dimB, M.E.dims[0] - 1); this.dimC = Math.min(this.dimC, M.E.dims[0] - 1); this.P3 = null;
    this.D = D; this.sel = null; this.extra = []; this.vals = null; this.stats = null; this.msgF = null; this.colorMode = 'cat'; this.layMode = 'map';
    this.root.querySelectorAll('[data-color]').forEach(b => b.setAttribute('aria-pressed', String(b.dataset.color === 'cat')));
    this.root.querySelectorAll('[data-lay]').forEach(b => b.setAttribute('aria-pressed', String(b.dataset.lay === 'map')));
    this.dimsBox.hidden = true; this.P = D.Y;
    this.on = new Uint8Array(CATS.length).fill(1); this.wsOnly = false;
    this.renderFilters(); this.fit(); this.selBox.hidden = true;
  }
  renderFilters() {
    const M = S.model, cnt = M._catCount;
    const order = CATS.map((c, k) => [c, k, cnt[k]]).filter(x => x[2] > 0).sort((a, b) => b[2] - a[2]);
    this.filtBox.innerHTML = `<span class="hint">${tl('Anzeigen:', 'Show:')}</span>` + order.map(([c, k, n]) => `<button type="button" class="fchipT" data-cat="${k}" aria-pressed="${!!this.on[k]}"><i style="background:var(${CAT_COLORS[c[0]]})"></i>${catName(c)}<small>${fmtInt(n)}</small></button>`).join('')
      + `<button type="button" class="toggle" data-ws aria-pressed="${this.wsOnly}">${tl('Nur Wortanfänge', 'Word starts only')}</button><button type="button" class="linkbtn" data-all>${tl('alle', 'all')}</button><button type="button" class="linkbtn" data-none>${tl('keine', 'none')}</button>`;
    this.filtBox.onclick = e => {
      const b = e.target.closest('button'); if (!b) return;
      if (b.dataset.cat != null) this.on[+b.dataset.cat] ^= 1;
      else if (b.hasAttribute('data-ws')) this.wsOnly = !this.wsOnly;
      else if (b.hasAttribute('data-all')) this.on.fill(1);
      else if (b.hasAttribute('data-none')) this.on.fill(0);
      this.renderFilters(); this.draw();
    };
  }
  visible(i) { const M = S.model; return this.on[M._cat[i]] && (!this.wsOnly || M._disp[i].charCodeAt(0) === 32); }
  async setColor(mode) {
    if (mode === 'dim' && !isSrv()) { this.msg(() => tl('Farbe je Dimension braucht den <b>Lupe-Server</b>: Modell darüber öffnen.', 'Color by dimension needs the <b>Lupe server</b>: open the model through it.')); return; }
    this.colorMode = mode;
    this.dseq = (this.dseq || 0) + 1;
    this.root.querySelectorAll('[data-color]').forEach(b => b.setAttribute('aria-pressed', String(b.dataset.color === mode)));
    this.renderDims();
    if (mode === 'dim') await this.loadDims(); else { this.draw(); }
  }
  async setLayout(mode) {
    if ((mode === 'xy' || mode === 'xyz') && !isSrv()) { this.msg(() => tl(`„${mode === 'xy' ? 'Zwei' : 'Drei'} Dimensionen“ braucht den <b>Lupe-Server</b>: Modell darüber öffnen.`, `“${mode === 'xy' ? 'Two' : 'Three'} dimensions” needs the <b>Lupe server</b>: open the model through it.`)); return; }
    this.layMode = mode;
    this.dseq = (this.dseq || 0) + 1;
    this.root.querySelectorAll('[data-lay]').forEach(b => b.setAttribute('aria-pressed', String(b.dataset.lay === mode)));
    this.renderDims();
    this.P3 = null;
    if (mode === 'xy' || mode === 'xyz') await this.loadDims(); else { this.P = this.D.Y; this.fit(); }
  }
  msg(f) { this.msgF = f; this.selBox.hidden = false; this.selBox.innerHTML = f(); }   // f() returns the HTML, so a language switch can show it again
  renderDims() {
    const show = this.colorMode === 'dim' || this.layMode === 'xy' || this.layMode === 'xyz', d = S.model.E.dims[0], ax = this.layMode !== 'map', pca = ax && this.axisSrc === 'pca';
    this.dimsBox.hidden = !show; if (!show) return;
    const u = this.uid, ctl = (id, v, lab) => `<span class="fm-dim"><label for="${u + id}">${lab}</label><button type="button" data-step="${id}:-1" aria-label="${tl('vorige', 'previous')}">◀</button><input type="number" id="${u + id}" data-dim="${id}" min="0" max="${d - 1}" value="${v}"><button type="button" data-step="${id}:1" aria-label="${tl('nächste', 'next')}">▶</button><input type="range" id="${u + id}R" data-dim="${id}" min="0" max="${d - 1}" value="${v}" aria-label="${tl(`${lab} wählen`, `Choose ${lab.toLowerCase()}`)}"></span>`;
    const pcSel = (k, lab) => `<span class="fm-dim"><label for="${u}pc${k}">${lab}</label><select id="${u}pc${k}" data-pc="${k}">${Array.from({ length: 8 }, (_, i) => `<option value="${i}"${this.pcs[k] === i ? ' selected' : ''}>${tl('Hauptrichtung', 'Principal direction')} ${i + 1}</option>`).join('')}</select></span>`;
    const src = ax ? `<div class="fm-axsrc"><span class="count">${tl('Achsen:', 'Axes:')}</span><span class="seg sm" role="group" aria-label="${tl('Achsen', 'Axes')}"><button type="button" data-src="dims" aria-pressed="${!pca}">${tl('einzelne Spalten der Tabelle', 'single table columns')}</button><button type="button" data-src="pca" aria-pressed="${pca}">${tl(`Hauptrichtungen aus allen ${fmtInt(d)} Dimensionen`, `principal directions from all ${fmtInt(d)} dimensions`)}</button></span></div>` : '';
    const axes = pca ? pcSel(0, 'x:') + pcSel(1, 'y:') + (this.layMode === 'xyz' ? pcSel(2, 'z:') : '') + (this.colorMode === 'dim' ? ctl('fmDimA', this.dimA, tl('Farbe: Dimension', 'Color: dimension')) : '')
      : ctl('fmDimA', this.dimA, ax ? 'Dimension x' : 'Dimension') + (ax ? ctl('fmDimB', this.dimB, 'Dimension y') : '') + (this.layMode === 'xyz' ? ctl('fmDimC', this.dimC, 'Dimension z') : '');
    this.dimsBox.innerHTML = src + axes + `<div class="fm-dimstats"></div>`;
    const set = (id, v) => { v = Math.max(0, Math.min(d - 1, v | 0)); if (id === 'fmDimA') this.dimA = v; else if (id === 'fmDimC') this.dimC = v; else this.dimB = v; this.dimsBox.querySelectorAll(`[data-dim="${id}"]`).forEach(x => { x.value = v; }); clearTimeout(this.dt); this.dt = setTimeout(() => this.loadDims(), 60); };
    this.dimsBox.onclick = e => {
      const sb = e.target.closest('[data-src]');
      if (sb) { if (this.axisSrc !== sb.dataset.src) { this.axisSrc = sb.dataset.src; this.P3 = null; this.renderDims(); this.loadDims(); } return; }
      const b = e.target.closest('[data-step]'); if (!b) return; const [id, st] = b.dataset.step.split(':'); set(id, (id === 'fmDimA' ? this.dimA : id === 'fmDimC' ? this.dimC : this.dimB) + +st);
    };
    this.dimsBox.oninput = e => { const id = e.target.dataset.dim; if (id) set(id, +e.target.value); };
    this.dimsBox.onchange = e => { const k = e.target.dataset.pc; if (k != null) { this.pcs[+k] = +e.target.value; this.P3 = null; this.loadDims(); } };
  }
  async fetchPca() {
    // principal directions of the whole table, computed on the server from all dimensions
    const M = this.model;
    if (this.pcaData && this.pcaData.path === M.source.path) return this.pcaData;
    await srvEnsure((p, m) => { const sm = srvMsg(m), pc = Math.round(p * 100); if (S.model === M) this.pos.textContent = tl(`Server lädt die Tabelle: ${sm} ${pc} %`, `Server is loading the table: ${sm} ${pc} %`); }, M);
    this.pos.textContent = tl('Hauptrichtungen werden berechnet …', 'Computing principal directions …');
    const r = await fetch(srvUrl('/api/pca', { path: M.source.path, k: 8 }));
    if (!r.ok) throw new Error((await r.json()).error || r.statusText);
    const buf = await r.arrayBuffer(), dv = new DataView(buf), ml = dv.getUint32(0, true), meta = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 4, ml)));
    let o = 4 + ml; o += o % 2;
    const h = new Uint16Array(buf.slice(o)), proj = new Float32Array(h.length);
    for (let k = 0; k < h.length; k++) proj[k] = GGUF.F16[h[k]];
    return (this.pcaData = { meta, proj, path: M.source.path });
  }
  axLabel(k) {
    if (this.axisSrc === 'pca' && this.pcaData && this.layMode !== 'map') { const c = this.pcs[k], r = nf(this.pcaData.meta.ratio[c] * 100, 2); return tl(`Hauptrichtung ${c + 1} (${r} %)`, `Principal direction ${c + 1} (${r} %)`); }
    return `Dimension ${fmtInt([this.dimA, this.dimB, this.dimC][k])}`;
  }
  pcaHTML(PC, toks) {
    const m = PC.meta, used = this.layMode === 'xyz' ? this.pcs.slice(0, 3) : this.pcs.slice(0, 2), sum = used.reduce((a, c) => a + m.ratio[c], 0), avg = 1 / m.d;
    const pct = x => nf(x * 100, 2) + ' %';
    const W = 320, H = 78, mx = m.ratio[0], bw = (W - 16) / 9.6, hb = r => Math.max(1, r / mx * (H - 28));
    const bars = m.ratio.slice(0, 8).map((r, i) => `<rect x="${(4 + i * bw).toFixed(1)}" y="${(H - 16 - hb(r)).toFixed(1)}" width="${(bw - 4).toFixed(1)}" height="${hb(r).toFixed(1)}" fill="var(${used.includes(i) ? '--c-attn' : '--c-other'})"><title>${tl('Hauptrichtung', 'Principal direction')} ${i + 1}: ${pct(r)}</title></rect><text x="${(4 + i * bw + bw / 2 - 2).toFixed(1)}" y="${H - 4}" text-anchor="middle" class="ax">${i + 1}</text>`).join('')
      + `<rect x="${(4 + 8 * bw + 10).toFixed(1)}" y="${(H - 16 - hb(avg)).toFixed(1)}" width="${(bw - 4).toFixed(1)}" height="${hb(avg).toFixed(1)}" fill="var(--c-norm)"><title>${tl('eine einzelne Spalte im Schnitt', 'one single column on average')}: ${pct(avg)}</title></rect><text x="${(4 + 8 * bw + 10 + bw / 2).toFixed(1)}" y="${H - 4}" text-anchor="middle" class="ax">${tl('1 Spalte', '1 column')}</text>`
      + `<text x="${W - 4}" y="11" text-anchor="end" class="ax">${pct(mx)}</text>`;
    const one = c => { const p = pct(m.ratio[c]), dims = m.dims[c].slice(0, 6).map(([j, w]) => `${fmtInt(j)} (${w > 0 ? '+' : '−'})`).join(', '), pos = toks(m.pos[c].slice(0, 8)), neg = toks(m.neg[c].slice(0, 8));
      return tl(`<p class="cap"><b>Hauptrichtung ${c + 1}</b>: ${p} der Unterschiede · am stärksten beteiligt: Dimension ${dims}</p><p class="cap"><b>Plus-Ende:</b> ${pos}</p><p class="cap"><b>Minus-Ende:</b> ${neg}</p>`,
        `<p class="cap"><b>Principal direction ${c + 1}</b>: ${p} of the differences · strongest dimensions: ${dims}</p><p class="cap"><b>Plus end:</b> ${pos}</p><p class="cap"><b>Minus end:</b> ${neg}</p>`); };
    const three = used.length === 3, d = fmtInt(m.d), rest = fmtInt(m.d - used.length), n = fmtInt(m.n);
    return `<p class="fm-pca-sum">${tl(`${three ? 'Diese drei Richtungen' : 'Diese zwei Richtungen'} erfassen <b>${pct(sum)}</b> der Unterschiede`, `${three ? 'These three directions' : 'These two directions'} capture <b>${pct(sum)}</b> of the differences`)}</p>
      <svg class="fm-pca-bars" viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" role="img" aria-label="${tl('Anteil der Unterschiede je Richtung', 'Share of the differences per direction')}">${bars}</svg>
      <details><summary>${tl('ⓘ Erklärung · Enden der Richtungen', 'ⓘ Explanation · ends of the directions')}</summary><p class="hint">${tl(`Jede Richtung mischt alle ${d} Dimensionen, so gewählt, dass sich die Tokens darauf am stärksten unterscheiden. Eine einzelne Spalte erfasst im Schnitt nur ${pct(avg)}, der Rest verteilt sich auf die übrigen ${rest} Richtungen. Darum kann kein Bild alle Dimensionen zeigen.`, `Each direction mixes all ${d} dimensions, chosen so that the tokens differ most along it. A single column captures only ${pct(avg)} on average; the rest is spread over the other ${rest} directions. That’s why no picture can show all dimensions.`)}</p>${used.map(one).join('')}<p class="hint">${tl(`Hauptkomponenten über alle ${n} Zeilen der Embedding-Tabelle, jede Zeile auf Länge 1 gebracht (wie bei der Landkarte).`, `Principal components over all ${n} rows of the embedding table, each row scaled to length 1 (as for the map).`)}</p></details>`;
  }
  async fetchDim(i) {
    const M = this.model, cache = this.dimCache;
    this.dimCache = this.dimCache || new Map();
    if (this.dimCache.has(i)) return this.dimCache.get(i);
    await srvEnsure((p, m) => { const sm = srvMsg(m), pc = Math.round(p * 100); if (S.model === M) this.pos.textContent = tl(`Server lädt die Tabelle: ${sm} ${pc} %`, `Server is loading the table: ${sm} ${pc} %`); }, M);
    const response = await fetch(srvUrl('/api/dim', { path: M.source.path, i }));
    if (!response.ok) throw new Error((await response.json()).error || response.statusText);
    const buf = await response.arrayBuffer();
    const dv = new DataView(buf), ml = dv.getUint32(0, true), meta = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 4, ml)));
    let o = 4 + ml; o += o % 2;
    const h = new Uint16Array(buf.slice(o)), v = new Float32Array(h.length);
    for (let k = 0; k < h.length; k++) v[k] = GGUF.F16[h[k]];
    const r = { meta, v };
    if (cache.size > 40) cache.delete(cache.keys().next().value);
    cache.set(i, r); return r;
  }
  async loadDims() {
    const seq = (this.dseq = (this.dseq || 0) + 1);
    const M = this.model, pca = this.layMode !== 'map' && this.axisSrc === 'pca';
    this.stats = null;
    try {
      const A = !pca || this.colorMode === 'dim' ? await this.fetchDim(this.dimA) : null;
      const B = !pca && this.layMode !== 'map' ? await this.fetchDim(this.dimB) : null, Cz = !pca && this.layMode === 'xyz' ? await this.fetchDim(this.dimC) : null;
      const PC = pca ? await this.fetchPca() : null;
      if (seq !== this.dseq || S.model !== M) return;
      this.vals = this.colorMode === 'dim' && A ? A.v : null;
      if (this.colorMode === 'dim' && A) { const s = Math.max(Math.abs(A.meta.min), Math.abs(A.meta.max)); this.valScale = Math.min(s, 4 * A.meta.std + Math.abs(A.meta.mean)) || 1; }
      if (PC) {
        const k = PC.meta.k, n = PC.meta.n, pr = PC.proj, [a, b, c] = this.pcs;
        if (this.layMode === 'xyz') { const P3 = new Float32Array(n * 3); for (let i = 0; i < n; i++) { P3[i * 3] = pr[i * k + a]; P3[i * 3 + 1] = pr[i * k + b]; P3[i * 3 + 2] = pr[i * k + c]; } this.P3 = P3; this.axStd = [PC.meta.std[a], PC.meta.std[b], PC.meta.std[c]]; this.project3(); this.fit(); }
        else { const P = new Float32Array(n * 2); for (let i = 0; i < n; i++) { P[i * 2] = pr[i * k + a]; P[i * 2 + 1] = pr[i * k + b]; } this.P = P; this.fit(); }
      } else if (B && Cz) { const n = A.v.length, P3 = new Float32Array(n * 3); for (let i = 0; i < n; i++) { P3[i * 3] = A.v[i] - A.meta.mean; P3[i * 3 + 1] = B.v[i] - B.meta.mean; P3[i * 3 + 2] = Cz.v[i] - Cz.meta.mean; } this.P3 = P3; this.axStd = [A.meta.std, B.meta.std, Cz.meta.std]; this.project3(); this.fit(); }
      else if (B) { const P = new Float32Array(A.v.length * 2); for (let i = 0; i < A.v.length; i++) { P[i * 2] = A.v[i]; P[i * 2 + 1] = B.v[i]; } this.P = P; this.fit(); }
      this.stats = { A, B, Cz, PC, a: this.dimA, b: this.dimB, c: this.dimC }; this.renderStats();
      this.draw();
    } catch (e) { if (seq !== this.dseq || S.model !== M) return; console.error(e); this.msg(() => tl('Server-Fehler: ', 'Server error: ') + esc(srvMsg(e.message))); }
  }
  renderStats() {   // legend, values and explanation under the axis controls (again after a language switch)
    const st = this.dimsBox.querySelector('.fm-dimstats'), X = this.stats; if (!st || !X) return;
    const { A, B, Cz, PC } = X, d = fmtInt(S.model.E.dims[0]), a = fmtInt(X.a), b = fmtInt(X.b), toks = l => l.map(i => `<button type="button" class="linkbtn mono" data-tok="${i}">${wsHTML(disp(i).trim() || disp(i))}</button>`).join(' · ');
    const one = (Y, i, axis) => { const lo = sig(Y.meta.min, 3), hi = sig(Y.meta.max, 3), mu = sig(Y.meta.mean, 3), sd = sig(Y.meta.std, 3), top = toks(Y.meta.top.slice(0, 10)), bot = toks(Y.meta.bottom.slice(0, 10));
      return tl(`<p class="cap">${axis}Dimension <b>${fmtInt(i)}</b>: rohe Werte von ${lo} bis ${hi} · Mittel ${mu} · Streuung ${sd}</p><p class="cap"><b>Größte Werte:</b> ${top}</p><p class="cap"><b>Kleinste Werte:</b> ${bot}</p>`,
        `<p class="cap">${axis}Dimension <b>${fmtInt(i)}</b>: raw values from ${lo} to ${hi} · mean ${mu} · spread ${sd}</p><p class="cap"><b>Largest values:</b> ${top}</p><p class="cap"><b>Smallest values:</b> ${bot}</p>`); };
    const v2 = sig(this.valScale || 1, 2), leg = this.colorMode === 'dim' ? `<p class="fm-leg" title="${tl(`Werte über ±${v2} voll gefärbt`, `values beyond ±${v2} fully colored`)}"><span class="mono">−${v2}</span><i aria-hidden="true"></i><span class="mono">+${v2}</span><span class="hint">${tl(`Dimension ${a}: blau negativ · hell ≈ 0 · rot positiv`, `Dimension ${a}: blue negative · light ≈ 0 · red positive`)}</span></p>` : '';
    if (PC) st.innerHTML = leg + this.pcaHTML(PC, toks);
    else {
      const help = Cz ? tl(`Drei Spalten der Tabelle als Achsen, jeder Punkt ein Token; Ziehen dreht die Wolke. Drei von ${d} Spalten sind nur ein Schnitt durch den Raum, darum bleibt es eine Wolke. Mit „Hauptrichtungen“ fließen alle Dimensionen in die Achsen ein.`, `Three table columns as axes, each dot a token; drag to rotate the cloud. Three of ${d} columns are only a slice through the space, so it stays a cloud. With “principal directions”, all dimensions feed into the axes.`)
        : B ? tl(`Jeder Punkt ist ein Token: weiter rechts = größerer Wert in Dimension ${a}, weiter oben = größerer Wert in Dimension ${b}. Meist eine Wolke um 0: Eine Dimension allein trägt kaum Bedeutung, erst alle ${d} zusammen.`, `Each dot is a token: further right = larger value in dimension ${a}, higher up = larger value in dimension ${b}. Usually a cloud around 0: one dimension alone carries little meaning, only all ${d} together.`)
        : tl(`Eine Dimension ist eine Spalte der Tabelle. Mit ◀ ▶ oder dem Regler blätterst du durch alle ${d}; die Karte bleibt, nur die Farbe zeigt die neue Spalte.`, `A dimension is one column of the table. ◀ ▶ or the slider step through all ${d}; the map stays, only the color shows the new column.`);
      st.innerHTML = leg + `<details><summary>${tl('ⓘ Dimensionswerte und Erklärung', 'ⓘ Dimension values and explanation')}</summary><p class="hint">${help}</p>${one(A, X.a, B ? 'x: ' : '')}${B ? one(B, X.b, 'y: ') : ''}${Cz ? one(Cz, X.c, 'z: ') : ''}</details>`;
    }
    st.onclick = e => { const btn = e.target.closest('[data-tok]'); if (btn) this.focus(+btn.dataset.tok); };
  }
  fit() {
    const P = this.P, n = this.D.n; let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
    for (let i = 0; i < n; i++) { const x = P[i * 2], y = P[i * 2 + 1]; if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y; }
    this.size(); this.cx = (x0 + x1) / 2; this.cy = (y0 + y1) / 2; this.s = Math.min((this.W - 40) / (x1 - x0 || 1), (this.H - 40) / (y1 - y0 || 1)); this.s0 = this.s; this.draw();
  }
  size() { this.W = this.vp.clientWidth || 800; this.H = this.vp.clientHeight || 600; }
  sx(i) { return (this.P[i * 2] - this.cx) * this.s + this.W / 2; }
  sy(i) { return -(this.P[i * 2 + 1] - this.cy) * this.s + this.H / 2; }
  zoomAt(px, py, f) { const wx = (px - this.W / 2) / this.s + this.cx, wy = -(py - this.H / 2) / this.s + this.cy; this.s = Math.max(this.s0 * 0.5, Math.min(this.s0 * 4000, this.s * f)); this.cx = wx - (px - this.W / 2) / this.s; this.cy = wy + (py - this.H / 2) / this.s; this.draw(); }
  pick(x, y) {
    if (!this.D) return null; let best = null, bd = 64;
    const n = this.D.n;
    for (let i = 0; i < n; i++) { const dx = this.sx(i) - x; if (dx > 8 || dx < -8) continue; const dy = this.sy(i) - y; if (dy > 8 || dy < -8) continue; const dd = dx * dx + dy * dy; if (dd < bd && this.visible(i)) { bd = dd; best = i; } }
    return best;
  }
  find(q) {
    const M = S.model, w = normWord(q), raw = q.trim(); if (!w || !this.D) return;
    // exact spelling first (word-start variant before the bare piece), then ignoring case/accents, then prefix
    let i = M._disp.indexOf(' ' + raw); if (i < 0) i = M._disp.indexOf(raw);
    if (i < 0) i = M._disp.findIndex(t => normWord(t) === w); if (i < 0) i = M._disp.findIndex(t => normWord(t).startsWith(w));
    const inp = this.root.querySelector('[data-fmq]');
    if (i < 0) { inp.setCustomValidity(tl('Kein Token mit diesem Text', 'No token with this text')); inp.reportValidity(); setTimeout(() => inp.setCustomValidity(''), 2000); return; }
    this.focus(i);
  }
  focus(i) { this.on[S.model._cat[i]] = 1; this.renderFilters(); this.select(i); this.s = Math.max(this.s, this.s0 * 30); this.cx = this.P[i * 2]; this.cy = this.P[i * 2 + 1]; this.draw(); }
  async select(i) {
    this.sel = i; this.extra = []; this.msgF = null; this.draw();
    if (i == null) { this.selBox.hidden = true; return; }
    const D = this.D, nbs = Array.from(D.nb.slice(i * D.k, i * D.k + D.k)), tok = `<b class="mono">${wsHTML(disp(i))}</b> (ID ${fmtInt(i)})`, dd = fmtInt(D.hdr.dims || S.model.E.dims[0]);
    this.selBox.hidden = false;
    this.selBox.innerHTML = tl(`${tok} · ${nbs.length} nächste Nachbarn in allen ${dd} Dimensionen`, `${tok} · ${nbs.length} nearest neighbors in all ${dd} dimensions`) + `${ib('fmNeighbors')}<br>${nbs.map(j => `<button type="button" class="linkbtn mono" data-tok="${j}">${wsHTML(disp(j))}</button>`).join(' · ')}`
      + (isSrv() || S.model.source.kind === 'file' ? ` <button type="button" class="btn" style="padding:2px 9px;font-size:12px" data-more>${tl('24 Nachbarn im ganzen Wörterbuch', '24 neighbors in the whole vocabulary')}</button>` : '');
    this.selBox.onclick = async e => {
      const b = e.target.closest('[data-tok]'); if (b) { this.focus(+b.dataset.tok); return; }
      if (e.target.closest('[data-more]')) { openNN(i); }
    };
  }
  draw() { if (!this.raf) this.raf = requestAnimationFrame(() => { this.raf = 0; this.paint(); }); }
  paint() {
    if (!this.D || !this.root.offsetParent) return;
    if (this.en !== isEN()) this.relang();   // language switched since the labels were written
    this.size();
    const dpr = window.devicePixelRatio || 1, W = this.W, H = this.H, n = this.D.n, M = S.model, T = theme();
    const PW = Math.round(W * dpr), PH = Math.round(H * dpr);
    if (this.cv.width !== PW || this.cv.height !== PH) { this.cv.width = PW; this.cv.height = PH; this.cv.style.width = W + 'px'; this.cv.style.height = H + 'px'; this.lens.width = PW; this.lens.height = PH; this.lens.style.width = W + 'px'; this.lens.style.height = H + 'px'; }
    const ctx = this.cv.getContext('2d');
    const img = ctx.createImageData(PW, PH), px = img.data, bg = T.surfRGB;
    for (let q = 0; q < px.length; q += 4) { px[q] = bg[0]; px[q + 1] = bg[1]; px[q + 2] = bg[2]; px[q + 3] = 255; }
    const catRGB = CATS.map(c => hex2rgb(css(CAT_COLORS[c[0]]) || '#888888'));
    const zoom = this.s / this.s0, dot = zoom > 60 ? 3 : zoom > 8 ? 2 : 1, rad = Math.max(1, Math.round(dot * dpr)), vals = this.colorMode === 'dim' ? this.vals : null, lut = T.lut, vs = this.valScale || 1;
    let shown = 0;
    for (let i = 0; i < n; i++) {
      if (!this.visible(i)) continue;
      const x = Math.round(this.sx(i) * dpr), y = Math.round(this.sy(i) * dpr);
      if (x < 0 || y < 0 || x >= PW || y >= PH) continue;
      shown++;
      let r, g, b;
      if (vals) { const k = colorIdx(vals[i], vs); r = lut[k]; g = lut[k + 1]; b = lut[k + 2]; } else { const c = catRGB[M._cat[i]]; r = c[0]; g = c[1]; b = c[2]; }
      for (let yy = y; yy < Math.min(PH, y + rad); yy++) for (let xx = x; xx < Math.min(PW, x + rad); xx++) { const q = (yy * PW + xx) * 4; px[q] = r; px[q + 1] = g; px[q + 2] = b; }
    }
    ctx.putImageData(img, 0, 0);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    if (this.layMode === 'xy') this.axes(ctx, T); else if (this.layMode === 'xyz' && this.P3) this.axes3(ctx, T);
    // neighbour lines
    const sel = this.sel, nbs = sel != null ? Array.from(this.D.nb.slice(sel * this.D.k, sel * this.D.k + this.D.k)) : [];
    if (sel != null) { ctx.strokeStyle = T.ink; ctx.lineWidth = 1.2; ctx.globalAlpha = 0.75; for (const j of nbs) { ctx.beginPath(); ctx.moveTo(this.sx(sel), this.sy(sel)); ctx.lineTo(this.sx(j), this.sy(j)); ctx.stroke(); } ctx.globalAlpha = 1; }
    // labels: token id == point index, low ids are the frequent tokens -> natural priority
    // occupancy grid (8 px cells) instead of pairwise box tests: labels stay cheap with 248k points
    const GS = 8, gw = Math.ceil(W / GS) + 2, gh = Math.ceil(H / GS) + 2, grid = new Uint8Array(gw * gh);
    const cells = (x, y, w, h, set) => { const x0 = Math.max(0, Math.floor(x / GS)), x1 = Math.min(gw - 1, Math.floor((x + w) / GS)), y0 = Math.max(0, Math.floor(y / GS)), y1 = Math.min(gh - 1, Math.floor((y + h) / GS)); for (let yy = y0; yy <= y1; yy++) for (let xx = x0; xx <= x1; xx++) { const q = yy * gw + xx; if (set) grid[q] = 1; else if (grid[q]) return false; } return true; };
    const approxW = (t, px) => { let w = 0; for (const ch of t) w += ch.codePointAt(0) > 0x2e80 ? px * 1.05 : px * 0.6; return w; };
    const label = (i, weight) => {
      const x = this.sx(i) + 4, y = this.sy(i) + 4; if (x < -60 || x > W || y < 0 || y > H + 10) return false;
      const txt = (disp(i).trim() || disp(i)).slice(0, 24), fs = weight === 700 ? 13.5 : 12, w = approxW(txt, fs);
      if (weight !== 700 && !cells(x - 2, y - 11, w + 4, 14, false)) return false;
      cells(x - 2, y - 11, w + 4, 14, true);
      ctx.font = `${weight} ${fs}px ${T.mono}`; ctx.lineWidth = 3.5; ctx.strokeStyle = T.surface; ctx.strokeText(txt, x, y); ctx.fillStyle = weight >= 600 ? T.ink : T.ink2; ctx.fillText(txt, x, y); return true;
    };
    if (sel != null) { label(sel, 700); nbs.forEach(j => label(j, 600)); }
    if (this.hover != null && this.hover !== sel) label(this.hover, 700);
    const maxL = Math.min(450, Math.round(25 * zoom + 25));
    let placed = 0, checked = 0;
    for (let i = 0; i < n && placed < maxL && checked < 60000; i++) {
      if (!this.visible(i) || M._partial[i]) continue;
      const x = this.sx(i), y = this.sy(i); if (x < 0 || x > W || y < 0 || y > H) continue;
      checked++; if (label(i, 400)) placed++;
    }
    if (sel != null) { ctx.strokeStyle = T.ink; ctx.lineWidth = 2; ctx.beginPath(); ctx.arc(this.sx(sel), this.sy(sel), 7, 0, 6.2832); ctx.stroke(); }
    const zt = nf(zoom, zoom < 10 ? 1 : 0), pt = this.lensOn ? tl('Lupe: Klick vergrößert die Stelle', 'Magnifier: click to zoom in on that spot') : this.layMode === 'xyz' ? tl('Ziehen dreht, Mausrad zoomt, Klick zeigt die echten Nachbarn', 'Drag to rotate, scroll to zoom, click shows the true neighbors') : tl('Mausrad zoomt, Ziehen verschiebt, Klick zeigt die echten Nachbarn', 'Scroll to zoom, drag to pan, click shows the true neighbors');
    this.pos.textContent = tl(`${fmtInt(shown)} von ${fmtInt(n)} Tokens sichtbar · Zoom ${zt}×`, `${fmtInt(shown)} of ${fmtInt(n)} tokens visible · zoom ${zt}×`);
    if (this.pos.title !== pt) this.pos.title = pt;
    if (this.lensOn) this.drawLens();
  }
  axes(ctx, T) {
    // one scale for both axes (same this.s), zero lines, ticks on the bottom and left edge
    const W = this.W, H = this.H, s = this.s, wx0 = this.cx - W / 2 / s, wx1 = this.cx + W / 2 / s, wy0 = this.cy - H / 2 / s, wy1 = this.cy + H / 2 / s;
    const nice = r => { const e = Math.pow(10, Math.floor(Math.log10(r))), f = r / e; return (f < 1.5 ? 1 : f < 3.5 ? 2 : f < 7.5 ? 5 : 10) * e; };
    const step = nice((wx1 - wx0) / 7), X = v => (v - this.cx) * s + W / 2, Y = v => -(v - this.cy) * s + H / 2, lab = v => Math.abs(v) < step / 1e6 ? '0' : sig(v, 3);
    ctx.save(); ctx.lineWidth = 1; ctx.strokeStyle = T.line2;
    const x0 = X(0), y0 = Y(0);
    ctx.beginPath(); if (x0 > 0 && x0 < W) { ctx.moveTo(x0 + 0.5, 0); ctx.lineTo(x0 + 0.5, H); } if (y0 > 0 && y0 < H) { ctx.moveTo(0, y0 + 0.5); ctx.lineTo(W, y0 + 0.5); } ctx.stroke();
    ctx.font = `11px ${T.mono}`; ctx.fillStyle = T.muted; ctx.textBaseline = 'alphabetic';
    for (let k = Math.ceil(wx0 / step); k * step <= wx1; k++) { const v = k * step, x = X(v); if (x < 30 || x > W - 30) continue; ctx.fillRect(Math.round(x), H - 30, 1, 5); ctx.fillText(lab(v), x + 3, H - 32); }
    for (let k = Math.ceil(wy0 / step); k * step <= wy1; k++) { const v = k * step, y = Y(v); if (y < 30 || y > H - 44) continue; ctx.fillRect(0, Math.round(y), 5, 1); ctx.fillText(lab(v), 8, y + 4); }
    ctx.font = `600 12px ${T.ui}`; ctx.fillStyle = T.ink2;
    const tx = `${this.axLabel(0)} →`, tw = ctx.measureText(tx).width;
    ctx.fillText(tx, W - tw - 10, H - 10); ctx.fillText(`↑ ${this.axLabel(1)}`, 10, 18);
    ctx.font = `11px ${T.ui}`; ctx.fillStyle = T.muted; ctx.fillText(this.axisSrc === 'pca' ? tl('Achsen mischen alle Dimensionen · gleicher Maßstab für x und y', 'axes mix all dimensions · same scale for x and y') : tl('rohe Werte aus der Tabelle · gleicher Maßstab für x und y', 'raw table values · same scale for x and y'), 10, H - 10);
    ctx.restore();
  }
  project3() {
    // turn the three raw columns (centred) and look at them along z: screen x/y = the rotated x/y
    const P3 = this.P3, n = P3.length / 3, P = this.P && this.P.length === n * 2 && this.P !== this.D.Y ? this.P : new Float32Array(n * 2);
    const cy = Math.cos(this.rotY), sy = Math.sin(this.rotY), cx = Math.cos(this.rotX), sx = Math.sin(this.rotX);
    for (let i = 0; i < n; i++) { const x = P3[i * 3], y = P3[i * 3 + 1], z = P3[i * 3 + 2], x1 = cy * x + sy * z, z1 = -sy * x + cy * z; P[i * 2] = x1; P[i * 2 + 1] = cx * y - sx * z1; }
    this.P = P;
  }
  axes3(ctx, T) {
    const cy = Math.cos(this.rotY), sy = Math.sin(this.rotY), cx = Math.cos(this.rotX), sx = Math.sin(this.rotX), L = 4 * Math.max(...this.axStd);
    const proj = (x, y, z) => { const x1 = cy * x + sy * z, z1 = -sy * x + cy * z, px = x1, py = cx * y - sx * z1; return [(px - this.cx) * this.s + this.W / 2, -(py - this.cy) * this.s + this.H / 2]; };
    const o = proj(0, 0, 0);
    ctx.save(); ctx.lineWidth = 1.5; ctx.font = `600 12px ${T.ui}`;
    [[L, 0, 0, 'x', 0, '--c-ffn'], [0, L, 0, 'y', 1, '--c-attn'], [0, 0, L, 'z', 2, '--c-lin']].forEach(([x, y, z, nm, k, col]) => {
      const e = proj(x, y, z), m = proj(-x, -y, -z); ctx.lineWidth = 1.5; ctx.strokeStyle = css(col); ctx.beginPath(); ctx.moveTo(m[0], m[1]); ctx.lineTo(e[0], e[1]); ctx.stroke();
      const lab = `${nm}: ${this.axLabel(k)}`, lw = ctx.measureText(lab).width, lx = Math.max(6, Math.min(this.W - lw - 6, e[0] + 4)), ly = Math.max(14, Math.min(this.H - 24, e[1] - 4));
      ctx.lineWidth = 4; ctx.strokeStyle = T.surface; ctx.strokeText(lab, lx, ly); ctx.lineWidth = 1.5; ctx.fillStyle = css(col); ctx.fillText(lab, lx, ly);
    });
    ctx.fillStyle = T.muted; ctx.font = `11px ${T.ui}`; ctx.fillText(this.axisSrc === 'pca' ? tl('Achsen aus allen Dimensionen', 'axes from all dimensions') : tl('rohe Werte aus der Tabelle', 'raw table values'), 10, this.H - 10);
    ctx.restore();
  }
  drawLens() {
    if (!this.lensOn || this.mx == null) return;
    const dpr = window.devicePixelRatio || 1, c = this.lens.getContext('2d'), T = theme(), R = 120, F = 6, M = S.model;
    c.setTransform(dpr, 0, 0, dpr, 0, 0); c.clearRect(0, 0, this.W, this.H);
    const mx = this.mx, my = this.my, wx = (mx - this.W / 2) / this.s + this.cx, wy = -(my - this.H / 2) / this.s + this.cy, s2 = this.s * F;
    c.save(); c.beginPath(); c.arc(mx, my, R, 0, 6.2832); c.fillStyle = T.surface; c.fill(); c.clip();
    const X = i => (this.P[i * 2] - wx) * s2 + mx, Y = i => -(this.P[i * 2 + 1] - wy) * s2 + my, n = this.D.n, inR = [];
    const catHex = CATS.map(cc => css(CAT_COLORS[cc[0]]));
    for (let i = 0; i < n; i++) {
      if (!this.visible(i)) continue;
      const dx = this.P[i * 2] - wx, dy = this.P[i * 2 + 1] - wy; if (Math.abs(dx) * s2 > R || Math.abs(dy) * s2 > R) continue;
      const x = X(i), y = Y(i); c.fillStyle = this.colorMode === 'dim' && this.vals ? (() => { const k = colorIdx(this.vals[i], this.valScale || 1); return `rgb(${T.lut[k]},${T.lut[k + 1]},${T.lut[k + 2]})`; })() : catHex[M._cat[i]];
      c.fillRect(x - 2, y - 2, 4, 4); inR.push(i);
    }
    const occ = [];
    c.font = `600 12px ${T.mono}`;
    for (const i of inR.slice(0, 400)) {
      if (M._partial[i]) continue;
      const txt = (disp(i).trim() || disp(i)).slice(0, 20), x = X(i) + 4, y = Y(i) + 4, w = c.measureText(txt).width;
      if (occ.some(b => x < b[0] + b[2] && x + w > b[0] && y - 11 < b[1] + b[3] && y + 3 > b[1])) continue;
      occ.push([x, y - 11, w, 14]); c.lineWidth = 3; c.strokeStyle = T.surface; c.strokeText(txt, x, y); c.fillStyle = T.ink; c.fillText(txt, x, y);
      if (occ.length > 60) break;
    }
    c.restore(); c.strokeStyle = T.ink; c.lineWidth = 2; c.beginPath(); c.arc(mx, my, R, 0, 6.2832); c.stroke();
    const cap = tl(`Lupe ${F}× · ${fmtInt(inR.length)} Tokens`, `Magnifier ${F}× · ${fmtInt(inR.length)} tokens`); c.font = `600 11.5px ${T.ui}`; const cw = c.measureText(cap).width, cy = my + R + 18 > this.H - 4 ? my - R - 10 : my + R + 18; c.lineWidth = 4; c.strokeStyle = T.surface; c.strokeText(cap, mx - cw / 2, cy); c.fillStyle = T.ink; c.fillText(cap, mx - cw / 2, cy);
  }
}
let fullMap = null;
async function renderFullMap() {
  const M = S.model, st = $('#mapStatus'), prog = $('#mapProg');
  $('#mapView').hidden = true; $('#fullMapView').hidden = false;
  if (!fullMap) fullMap = new FullMap($('#fullMapView'));
  if (M._fullMap) { if (fullMap.D !== M._fullMap) fullMap.setData(M._fullMap); else fullMap.draw(); fullMapInfo(M._fullMap); return true; }
  let D = null;
  if (isSrv()) {
    const c = await srvJson('/api/map/cached', { path: M.source.path, n: M.vocabSize, d: M.E.dims[0] }).catch(() => ({}));
    if (c.cache) D = decodeMapBin(await fetch(srvUrl('/api/map', { path: M.source.path, cache: c.cache })).then(r => r.arrayBuffer()), c.cache);
    else {
      const dev = SRV.info.devices.find(d => d.id === (SRV.device === 'auto' ? SRV.info.default_device : SRV.device)) || SRV.info.devices[0];
      const n = fmtInt(M.vocabSize), dd = fmtInt(M.E.dims[0]), dur = c.estimate_s ? fmtDur(c.estimate_s) : '', dl = esc(devLabel(dev)), dir = esc(SRV.info.cache_dir || '~/.cache/gguf-lupe');
      st.innerHTML = tl(`<b>${n} Tokens · ${dd} Dimensionen</b> · Karte noch nicht berechnet${dur ? ` · Dauer <b>${dur}</b>` : ''} <button type="button" class="btn primary" id="mapGo">Komplette Karte berechnen</button><details><summary>ⓘ Berechnung und Speicherort</summary>Alle Tokenpaare werden in allen Dimensionen verglichen, auf ${dl}. Die Karte wird unter <code>${dir}</code> gespeichert und beim nächsten Öffnen wiederverwendet.</details>`,
        `<b>${n} tokens · ${dd} dimensions</b> · map not computed yet${dur ? ` · takes <b>${dur}</b>` : ''} <button type="button" class="btn primary" id="mapGo">Compute full map</button><details><summary>ⓘ Computation and storage</summary>All token pairs are compared in all dimensions, on ${dl}. The map is saved in <code>${dir}</code> and reused next time.</details>`);
      await new Promise(res => { $('#mapGo').onclick = res; });
      prog.hidden = false;
      const job = await pollJob(await srvJson('/api/map/start', { path: M.source.path }), (p, m) => { prog.firstChild.style.width = (p * 100).toFixed(1) + '%'; const sm = srvMsg(m); st.textContent = tl(`Server rechnet: ${sm}`, `Server is computing: ${sm}`); });
      prog.hidden = true;
      D = decodeMapBin(await fetch(srvUrl('/api/map', { path: M.source.path, cache: job.result.cache })).then(r => r.arrayBuffer()), job.result.cache);
    }
  } else if (M.source.kind === 'demo') D = fullMapFromDemo(M);
  if (!D || D.n !== M.vocabSize) { $('#fullMapView').hidden = true; return false; }
  if (S.model !== M) return true;
  M._fullMap = D; fullMap.setData(D); fullMapInfo(D);
  return true;
}
function fmtDur(s) { const sec = fmtInt(Math.max(5, Math.round(s / 5) * 5)), min = fmtInt(Math.round(s / 60)), h = nf(s / 3600, 1); return s < 90 ? tl(`ca. ${sec} Sekunden`, `about ${sec} seconds`) : s < 5400 ? tl(`ca. ${min} Minuten`, `about ${min} minutes`) : tl(`ca. ${h} Stunden`, `about ${h} hours`); }
function fullMapInfo(D) {
  const h = D.hdr || {}, n = fmtInt(D.n), d = fmtInt(h.dims || S.model.E.dims[0]), dev = esc(h.device || ''), eng = esc(h.engine || ''), sec = h.total_s ? nf(h.total_s, 0) : '', cr = esc(h.created || ''), file = D.cachePath ? `<br>${tl('Datei', 'File')}: <code>${esc(D.cachePath)}</code>` : '';
  $('#mapStatus').innerHTML = tl(`<b>Komplette Karte · ${n} Tokens · ${d} Dimensionen</b><details><summary>ⓘ Kartendetails${D.cachePath ? ' und Speicherort' : ''}</summary>Ähnlichkeit = Kosinus der rohen Zeilen, exakt über alle Dimensionen; jedes Token kennt seine ${D.k} nächsten Nachbarn. Flach ist nur die Zeichnung. ${D.demo ? 'Vorberechnet beim Erstellen des Beispiels.' : `Berechnet auf ${dev} (${eng})${sec ? ` in ${sec} s` : ''}, gespeichert am ${cr}.`}${file}</details>`,
    `<b>Full map · ${n} tokens · ${d} dimensions</b><details><summary>ⓘ Map details${D.cachePath ? ' and storage' : ''}</summary>Similarity = cosine of the raw rows, exact over all dimensions; each token knows its ${D.k} nearest neighbors. Only the drawing is flat. ${D.demo ? 'Precomputed when the example was made.' : `Computed on ${dev} (${eng})${sec ? ` in ${sec} s` : ''}, saved on ${cr}.`}${file}</details>`);
}

// =====================================================================
// Rechenweg: how llama.cpp computes this file, straight from its source (embedded, gzip)
// =====================================================================
let LL = null;
async function llLoad() {
  if (LL) return LL;
  const bytes = b64($('#ll-data').textContent.trim());
  LL = JSON.parse(await new Response(new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'))).text());
  LL.defs = new Map(); LL.byName = new Map(); LL.lines = {}; LL.starts = {};
  for (const [f, src] of Object.entries(LL.files)) {
    const lines = src.split('\n'), starts = new Array(lines.length); let o = 0;
    lines.forEach((l, i) => { starts[i] = o; o += l.length + 1; });
    LL.lines[f] = lines; LL.starts[f] = starts;
    lines.forEach((l, i) => {
      const m = /^(?:[A-Za-z_][\w:<>,\s*&]*?\s)?((?:llama_model_\w+|llm_\w+)(?:::\w+(?:<[^>]*>)?)*)::(~?\w+)\s*\(/.exec(l);
      if (!m || /;\s*$/.test(l)) return;
      const cls = m[1].replace(/<[^>]*>/g, ''), d = { file: f, line: i, cls, name: m[2], key: cls + '::' + m[2] };
      if (!LL.defs.has(d.key)) LL.defs.set(d.key, d);
      if (!LL.byName.has(d.name)) LL.byName.set(d.name, []);
      LL.byName.get(d.name).push(d);
    });
  }
  return LL;
}
// strip comments, strings and char literals (same length, so offsets and braces stay valid)
function llStrip(s) {
  let out = '', i = 0, st = 0; // 0 code, 1 line comment, 2 block comment, 3 string, 4 char
  while (i < s.length) {
    const c = s[i], n = s[i + 1];
    if (st === 0) {
      if (c === '/' && n === '/') { st = 1; out += '  '; i += 2; continue; }
      if (c === '/' && n === '*') { st = 2; out += '  '; i += 2; continue; }
      if (c === '"') { st = 3; out += '"'; i++; continue; }
      if (c === "'") { st = 4; out += "'"; i++; continue; }
      out += c; i++; continue;
    }
    if (st === 1) { if (c === '\n') { st = 0; out += c; } else out += ' '; i++; continue; }
    if (st === 2) { if (c === '*' && n === '/') { st = 0; out += '  '; i += 2; continue; } out += c === '\n' ? c : ' '; i++; continue; }
    if (c === '\\') { out += '  '; i += 2; continue; }
    if ((st === 3 && c === '"') || (st === 4 && c === "'")) { st = 0; out += c; i++; continue; }
    out += c === '\n' ? c : ' '; i++;
  }
  return out;
}
function llBody(d) { // line range of a function body: [first line after '{', line of matching '}']
  if (d.body) return d.body;
  const f = d.file, src = LL.files[f], cs = LL.stripped?.[f] || ((LL.stripped = LL.stripped || {})[f] = llStrip(src));
  let i = LL.starts[f][d.line], par = 0;
  for (; i < cs.length; i++) { const c = cs[i]; if (c === '(') par++; else if (c === ')') par--; else if (c === '{' && par === 0) break; else if (c === ';' && par === 0) return null; }
  let depth = 0, j = i;
  for (; j < cs.length; j++) { if (cs[j] === '{') depth++; else if (cs[j] === '}' && --depth === 0) break; }
  const lineOf = off => { const st = LL.starts[f]; let lo = 0, hi = st.length - 1; while (lo < hi) { const m = (lo + hi + 1) >> 1; if (st[m] <= off) lo = m; else hi = m - 1; } return lo; };
  return (d.body = { open: lineOf(i), close: lineOf(j), o0: i, o1: j });
}
// which class, file and functions llama.cpp uses for a GGUF architecture name
function llResolve(arch) {
  const e = LL.archs[arch], cls = e && LL.factory[e];
  if (!cls) return { arch, enum: e || null, missing: true };
  const chain = []; for (let c = cls; c && c !== 'llama_model_base' && chain.length < 6; c = LL.parents[c]) chain.push(c);
  const def = what => { for (const c of chain) { const d = LL.defs.get(c + '::' + what); if (d) return d; } return null; };
  const R = { arch, enum: e, cls, chain, hparams: def('load_arch_hparams'), tensors: def('load_arch_tensors'), builder: def('build_arch_graph'), graph: null };
  if (R.builder) { // the last make_unique<...> in build_arch_graph is the main pass (an earlier one is often the MTP draft graph)
    const b = llBody(R.builder), txt = LL.files[R.builder.file].slice(b.o0, b.o1), all = [...txt.matchAll(/make_unique<\s*([\w:]+)/g)].map(m => m[1]);
    const T = all.filter(t => !/mtp/i.test(t)).pop() || all.pop();
    if (T) {
      const parts = T.split('::'), last = parts.pop();
      if (parts.length) R.graph = LL.defs.get(T + '::' + last) || null;
      else for (const c of chain) { R.graph = LL.defs.get(`${c}::${last}::${last}`) || LL.defs.get(`${T}::${T}`) || (LL.alias[c] && LL.defs.get(`${LL.alias[c]}::${last}::${last}`)); if (R.graph) break; }
    }
  }
  R.file = (R.graph || R.tensors || R.hparams || {}).file || null;
  R.members = llMembers(R);
  return R;
}
// member -> tensor name, parsed from load_arch_tensors (this class and its parents)
function llMembers(R) {
  const mem = new Map(), add = (k, e, s) => { if (!mem.has(k) && LL.tensors[e]) mem.set(k, { e, pat: LL.tensors[e] + (s ? '.' + s : '') }); };
  for (const c of R.chain) {
    const d = LL.defs.get(c + '::load_arch_tensors'); if (!d) continue;
    const b = llBody(d), txt = LL.files[d.file].slice(b.o0, b.o1);
    for (const m of txt.matchAll(/(\blayer\.)?([\w.]+)\s*=\s*create_tensor\(\s*tn\(\s*(LLM_TENSOR_\w+)\s*(?:,\s*"(\w+)")?/g)) add((m[1] ? 'L.' : 'M.') + m[2], m[3], m[4]);
    if (/create_tensor_qkv\s*\(/.test(txt)) for (const [k, e, s] of [['wq', 'Q', 'weight'], ['wk', 'K', 'weight'], ['wv', 'V', 'weight'], ['wqkv', 'QKV', 'weight'], ['wq_b', 'Q', 'bias'], ['wk_b', 'K', 'bias'], ['wv_b', 'V', 'bias'], ['wqkv_b', 'QKV', 'bias']]) add('L.' + k, 'LLM_TENSOR_ATTN_' + e, s);
  }
  return mem;
}
// all tensors of the open file that match a pattern like "blk.%d.attn_norm.weight"
function llFound(pat) {
  const M = S.model, key = 'll:' + pat; M._llf = M._llf || new Map();
  if (M._llf.has(key)) return M._llf.get(key);
  const re = new RegExp('^' + pat.replace(/[.*+?^${}()|[\]\\]/g, '\\$&').replace('%d', '(\\d+)') + '$'), r = M.tensors.filter(t => re.test(t.name));
  M._llf.set(key, r); return r;
}
function llRef(ref, R) { // "model.layers[il].attn_norm" | "model.output" | "layer.wq" -> {k, pat}
  const m = /^(?:model\.layers\[\w+\]\.|layer\.)([\w.]+)$/.exec(ref);
  const k = m ? 'L.' + m[1] : 'M.' + ref.replace(/^model\./, '');
  const x = R.members.get(k) || R.members.get(k.replace(/^L\./, 'M.'));
  return x ? { k, ...x } : { k, pat: null };
}
// ---------- steps: the function body read statement by statement
const LL_VARS = {   // getters: the label follows the language (part F reads LL_VARS[name])
  get Qcur() { return tl('Q (Frage)', 'Q (query)'); }, get Kcur() { return tl('K (Schlüssel)', 'K (key)'); }, get Vcur() { return tl('V (Inhalt)', 'V (value)'); }, get cur() { return tl('cur (laufender Vektor)', 'cur (running vector)'); },
  get inpL() { return tl('Eingang des Blocks', 'block input'); }, get inpSA() { return tl('Kopie für das Residual', 'copy for the residual'); }, get ffn_inp() { return tl('Eingang des FFN', 'FFN input'); },
};
function llCond(c, M) {
  const t = c.replace(/\s+/g, ' ').trim();
  const known = [[/hparams\.is_recr\(il\)/, tl('der Block ein linearer Block ist (laufender Zustand statt Attention)', 'the block is a linear block (running state instead of attention)')], [/hparams\.is_swa\(il\)/, tl('der Block nur ein Fenster der letzten Tokens sieht (lokale Attention)', 'the block only sees a window of recent tokens (local attention)')],
    [/ffn_gate_inp/, tl('der Block Experten hat (MoE)', 'the block has experts (MoE)')], [/il == n_layer - 1/, tl('es der letzte Block ist', 'it is the last block')], [/^inp_out_ids$/, tl('nur bestimmte Tokens eine Vorhersage brauchen', 'only certain tokens need a prediction')], [/hparams\.has_kv\(il\)/, tl('der Block eigene K- und V-Werte rechnet', 'the block computes its own K and V values')],
    [/params\.gtype == LLM_GRAPH_TYPE_DECODER_MTP/, tl('der Entwurfs-Durchlauf (MTP) gerechnet wird', 'the draft pass (MTP) is computed')]];
  for (const [re, s] of known) if (re.test(t)) return s;
  const mm = /^model\.(?:layers\[il\]\.)?([\w.]+)$/.exec(t);
  if (mm) return tl(`die Datei die Tabelle <code>${esc(mm[1])}</code> hat`, `the file has the table <code>${esc(mm[1])}</code>`);
  return `<code>${esc(t)}</code>`;
}
function llStep(st, refs) {
  const R = [
    [/\bres->t_logits\s*=/, 'head', tl('Ergebnis: eine Punktzahl (Logit) für jedes Token im Wörterbuch', 'Result: one score (logit) for every token in the vocabulary')],
    [/\bres->t_embd\s*=/, 'other', tl('Zwischenergebnis: der fertige Vektor vor dem Ausgabe-Kopf', 'Intermediate result: the finished vector before the output head')],
    [/build_inp_embd\s*\(/, 'emb', tl('Token-IDs nachschlagen: jede ID holt ihre Zeile aus der Embedding-Tabelle', 'Look up token IDs: each ID fetches its row from the embedding table')],
    [/build_inp_pos\s*\(/, 'other', tl('Positionen der Tokens bereitlegen (0, 1, 2, …)', 'Prepare the token positions (0, 1, 2, …)')],
    [/build_inp_(?:mem_hybrid|attn\w*|rs|kv\w*)\s*\(|build_attn_inp\w*\s*\(|build_rs_inp\w*\s*\(/, 'other', tl('Gedächtnis anlegen: KV-Cache bzw. laufender Zustand der früheren Tokens', 'Set up memory: KV cache or running state of the earlier tokens')],
    [/build_inp_out_ids\s*\(/, 'other', tl('Merken, welche Tokens am Ende eine Vorhersage brauchen', 'Note which tokens need a prediction at the end')],
    [/build_norm\s*\([^;]*LLM_NORM_RMS/, 'norm', tl('Normieren (RMSNorm): alle Zahlen auf vergleichbare Größe bringen', 'Normalize (RMSNorm): bring all numbers to a comparable size')],
    [/build_norm\s*\(/, 'norm', tl('Normieren (LayerNorm)', 'Normalize (LayerNorm)')],
    [/ggml_rms_norm\s*\(/, 'norm', tl('Normieren (RMSNorm)', 'Normalize (RMSNorm)')],
    [/ggml_l2_norm\s*\(/, 'norm', tl('Auf Länge 1 bringen (L2-Normierung)', 'Scale to length 1 (L2 normalization)')],
    [/ggml_norm\s*\(/, 'norm', tl('Normieren', 'Normalize')],
    [/ggml_rope\w*\s*\(/, 'attn', tl('Position einrechnen (RoPE): den Vektor je nach Position im Text drehen', 'Add the position (RoPE): rotate the vector by its position in the text')],
    [/build_attn\w*\s*\(/, 'attn', tl('Attention: jedes Token vergleicht sich mit den früheren (Q·K → Softmax → mal V)', 'Attention: each token compares itself with the earlier ones (Q·K → softmax → times V)')],
    [/build_moe_ffn\s*\(/, 'moe', tl('Experten (MoE): ein Router wählt wenige Experten, nur die rechnen', 'Experts (MoE): a router picks a few experts, only those compute')],
    [/build_ffn\s*\(/, 'ffn', tl('Feed-Forward: auf viele Zahlen aufblähen, Aktivierung, zurück auf die Breite', 'Feed-forward: expand to many numbers, activation, back to the width')],
    [/ggml_ssm_conv\s*\(/, 'lin', tl('Kurze Faltung über die letzten Tokens (Conv1D)', 'Short convolution over the last few tokens (Conv1D)')],
    [/ggml_(?:ssm_scan|gated_delta_net|rwkv_wkv\w*|gated_linear_attn|delta_net\w*)\s*\(|build_delta_net\w*\s*\(/, 'lin', tl('Laufenden Zustand aktualisieren: lineare Attention (DeltaNet / SSM)', 'Update the running state: linear attention (DeltaNet / SSM)')],
    [/build_rs\s*\(/, 'lin', tl('Gespeicherten Zustand der früheren Tokens holen', 'Fetch the stored state of the earlier tokens')],
    [/build_lora_mm(?:_id)?\s*\(|ggml_mul_mat(?:_id)?\s*\(/, 'mm', ''],
    [/ggml_soft_max\w*\s*\(/, 'attn', tl('Softmax: Punktzahlen in Wahrscheinlichkeiten umrechnen', 'Softmax: turn scores into probabilities')],
    [/ggml_sigmoid\s*\(/, 'other', tl('Sigmoid: jede Zahl auf 0 bis 1 drücken (wie ein Gate)', 'Sigmoid: squeeze each number into 0 to 1 (like a gate)')],
    [/ggml_(silu|gelu\w*|relu|swiglu\w*|geglu\w*|tanh|softplus|exp|elu)\s*\(/, 'other', tl('Aktivierung: $1', 'Activation: $1')],
    [/ggml_add\s*\(/, 'add', ''],
    [/ggml_mul\s*\(/, 'other', tl('Elementweise multiplizieren (z. B. mit einem Gate oder Gewicht)', 'Multiply element by element (e.g. by a gate or a weight)')],
    [/ggml_scale\w*\s*\(/, 'other', tl('Mit einem festen Faktor skalieren', 'Scale by a fixed factor')],
    [/ggml_get_rows\s*\(/, 'other', tl('Nur bestimmte Zeilen behalten', 'Keep only certain rows')],
    [/ggml_(?:concat|reshape|view|permute|transpose|cont|repeat)\w*\s*\(/, 'shape', tl('Umformen: gleiche Zahlen, andere Anordnung', 'Reshape: same numbers, different arrangement')],
    [/build_cvec\s*\(/, 'other', tl('Steuervektor addieren (nur falls einer gesetzt ist)', 'Add a control vector (only if one is set)')],
  ];
  for (const [re, comp, lab] of R) {
    const m = re.exec(st); if (!m) continue;
    if (comp === 'mm') {
      const r = refs[0], nm = r && (r.pat || r.k).replace(/^blk\.%d\./, '').replace(/^[LM]\./, '').replace(/\.weight$/, '');
      if (r && /^(output|M\.output)$/.test(nm)) return { comp: 'head', label: tl('Ausgabe-Kopf: Vektor mal Tabelle <code>output</code>, eine Punktzahl je Token', 'Output head: vector times the table <code>output</code>, one score per token') };
      return { comp: r ? 'mm' : 'other', label: r ? tl(`Matrix mal Vektor mit der Tabelle <code>${esc(nm)}</code>`, `Matrix times vector with the table <code>${esc(nm)}</code>`) : tl('Matrix-Multiplikation', 'Matrix multiplication') };
    }
    if (comp === 'add') return { comp: 'other', label: /\b(inpSA|inpL|residual|ffn_inp|sa_out|ffn_residual|attn_out)\b/.test(st) ? tl('Residual: den Eingang wieder dazuaddieren (Abkürzung um den Teilschritt)', 'Residual: add the input back (the shortcut around the sub-step)') : tl('Addieren', 'Add') };
    return { comp, label: lab.replace('$1', m[1] || '') };
  }
  return null;
}
function llSteps(d, R, depth, seen) {
  const f = d.file, b = llBody(d); if (!b) return [];
  const src = LL.lines[f], cs = (LL.sLines = LL.sLines || {})[f] || (LL.sLines[f] = LL.stripped[f].split('\n')), out = [];
  let lvl = 0, buf = '', bufLine = -1, par = 0, cm = [], pendIf = false;
  const refsOf = s => [...s.matchAll(/\b(model\.layers\[\w+\]\.[\w.]+|model\.(?!layers\b)[\w.]+|layer\.[\w.]+)/g)].map(m => ({ ref: m[1], ...llRef(m[1], R) })).filter((r, i, a) => a.findIndex(x => x.ref === r.ref) === i);
  for (let i = b.open; i <= b.close; i++) {
    const s0 = i === b.open ? b.o0 - LL.starts[f][i] + 1 : 0, e0 = i === b.close ? b.o1 - LL.starts[f][i] : cs[i].length, code = cs[i].slice(s0, e0);
    const raw = src[i], t = code.trim();
    if (!t) { const c = raw.trim(); if (c.startsWith('//')) cm.push(c.replace(/^\/\/+\s?/, '')); else if (!c) cm = []; continue; }
    const lead = /^\}+/.exec(t); if (lead && !buf) lvl -= lead[0].length;
    if (!buf) {
      const ctl = /^(?:\}\s*)?(else\s+if|if|for|while|else)\b\s*(?:\((.*)\))?/.exec(t);
      if (ctl) {
        const kw = ctl[1], cond = ctl[2] || '';
        let text;
        if (kw === 'for') { const fm = /(\w+)\s*<\s*(n_layer\w*)/.exec(cond); const n = S.model.layers.filter(L => isMainPass(S.model, L)).length; text = fm && /^n_layer$/.test(fm[2]) ? tl(`Für jeden Block (${fm[1]} = 0 … ${fmtInt(n - 1)}, also ${fmtInt(n)}-mal):`, `For each block (${fm[1]} = 0 … ${fmtInt(n - 1)}, so ${fmtInt(n)} times):`) : tl(`Wiederholen: <code>for (${esc(cond)})</code>`, `Repeat: <code>for (${esc(cond)})</code>`); }
        else if (kw === 'while') text = tl(`Solange <code>${esc(cond)}</code>:`, `While <code>${esc(cond)}</code>:`);
        else if (kw === 'else') text = tl('Sonst:', 'Otherwise:');
        else text = (kw === 'if' ? tl('Wenn ', 'If ') : tl('Sonst, wenn ', 'Otherwise, if ')) + llCond(cond, S.model) + ':';
        out.push({ type: 'ctl', line: i, lvl, text, file: f, cm: cm.join(' ') }); cm = [];
        const opens = (t.match(/\{/g) || []).length - (t.match(/\}/g) || []).length + (lead ? lead[0].length : 0);
        lvl += Math.max(0, opens); pendIf = !/\{\s*$/.test(t) && !/;\s*$/.test(t);
        if (!/;\s*$/.test(t) || /\{\s*$/.test(t)) continue;
      }
    }
    if (!buf) bufLine = i;
    buf += ' ' + code;
    for (const ch of code) { if (ch === '(') par++; else if (ch === ')') par--; }
    if (par > 0 || !/[;{}]\s*$/.test(t)) continue;
    const stmt = buf.trim(), ln = bufLine, eff = pendIf ? lvl + 1 : lvl; buf = ''; par = 0; pendIf = false;
    const opens = (stmt.match(/\{/g) || []).length - (stmt.match(/\}/g) || []).length + (lead && stmt.startsWith('}') ? lead[0].length : 0);
    const cbm = /^cb\(/.test(stmt) && /\bcb\(\s*(\w+)\s*,\s*"([^"]*)"/.exec(src.slice(ln, i + 1).join(' '));
    if (cbm) { const last = out[out.length - 1]; if (last && last.type === 'step' && !last.named) last.named = cbm[2]; lvl += Math.max(0, opens); cm = []; continue; }
    const refs = refsOf(stmt), s = llStep(stmt, refs), tgt = /^(?:[\w:<>]+\s*\*?\s+|auto\s*\*?\s*)?(\w+)\s*=[^=]/.exec(stmt);
    // helper defined in a model file (same file, a *-base file or the class's own graph): show its inside as sub-steps
    const hc = !s && /\b(build_\w+|\w+_impl)\s*\(/.exec(stmt), hd = hc && llHelper(hc[1], f);
    if (s) out.push({ type: 'step', line: ln, lvl: eff, file: f, ...s, refs, tgt: tgt && tgt[1], cm: cm.join(' ') });
    else if (hd) {
      const item = { type: 'call', line: ln, lvl: eff, file: f, name: hc[1], def: hd, tgt: tgt && tgt[1], cm: cm.join(' '), refs };
      out.push(item);
      if (depth < 2 && !seen.has(hd.key)) { seen.add(hd.key); for (const x of llSteps(hd, R, depth + 1, seen)) out.push({ ...x, lvl: x.lvl + eff + 1, sub: true }); }
    }
    cm = []; lvl += Math.max(0, opens);
  }
  return out;
}
function llHelper(name, file) {
  const c = LL.byName.get(name); if (!c) return null;
  return c.find(d => d.file === file) || c.find(d => /-base\.cpp$/.test(d.file)) || null;
}
// ---------- code view
const LL_KW = /^(?:const|auto|if|else|for|while|return|int|int64_t|int32_t|uint32_t|float|bool|void|struct|template|nullptr|true|false|static|switch|case|break|continue|default|new|std|size_t|double|char|using|typename|this)$/;
function llHi(line, f, st) { // syntax colour one line; st.block carries /* */ across lines
  let h = '', i = 0;
  const re = /(\/\/.*$)|(\/\*)|("(?:\\.|[^"\\])*")|('(?:\\.|[^'\\])+')|(model\.layers\[\w+\]\.[\w.]+|model\.(?!layers\b)[\w.]+|layer\.[\w.]+)|\b(\d+(?:\.\d+)?f?)\b|\b(LLM_\w+|GGML_\w+)\b|\b([A-Za-z_]\w*)(?=\s*\()|\b([A-Za-z_]\w*)\b/g;
  if (st.block) { const e = line.indexOf('*/'); if (e < 0) return `<span class="c-cm">${esc(line)}</span>`; h += `<span class="c-cm">${esc(line.slice(0, e + 2))}</span>`; i = e + 2; st.block = false; }
  re.lastIndex = i; let m;
  while ((m = re.exec(line))) {
    h += esc(line.slice(i, m.index)); i = re.lastIndex;
    if (m[1]) h += `<span class="c-cm">${esc(m[1])}</span>`;
    else if (m[2]) { const e = line.indexOf('*/', m.index + 2); if (e < 0) { h += `<span class="c-cm">${esc(line.slice(m.index))}</span>`; st.block = true; return h; } h += `<span class="c-cm">${esc(line.slice(m.index, e + 2))}</span>`; i = re.lastIndex = e + 2; }
    else if (m[3] || m[4]) h += `<span class="c-str">${esc(m[0])}</span>`;
    else if (m[5]) h += `<span class="c-ref" data-ref="${esc(m[5])}">${esc(m[5])}</span>`;
    else if (m[6]) h += `<span class="c-num">${m[6]}</span>`;
    else if (m[7]) h += `<span class="c-enum">${m[7]}</span>`;
    else if (m[8]) { const d = !LL_KW.test(m[8]) && llHelper(m[8], f) || (/^(build_|ggml_)/.test(m[8]) && (LL.byName.get(m[8]) || []).find(x => x.file === 'llama-graph.cpp')); h += d ? `<span class="c-fn" data-def="${esc(d.key)}">${m[8]}</span>` : `<span class="${LL_KW.test(m[8]) ? 'c-kw' : /^ggml_/.test(m[8]) ? 'c-op' : 'c-call'}">${m[8]}</span>`; }
    else h += LL_KW.test(m[9]) ? `<span class="c-kw">${m[9]}</span>` : m[9];
  }
  return h + esc(line.slice(i));
}
function llComp(c) { return c === 'mm' ? 'other' : c; }
const llLine = i => tl(`Z. ${i + 1}`, `line ${i + 1}`);
async function renderCode() {
  const M = S.model, host = $('#codeSteps');
  if (!LL) { host.innerHTML = `<p class="count">${tl('Lade den llama.cpp-Quelltext …', 'Loading the llama.cpp source …')}</p>`; await llLoad(); }
  const arch = S.codeArch || M.arch, R = llResolve(arch), own = arch === M.arch, me = tl(' (diese Datei)', ' (this file)');
  $('#codeFile').innerHTML = Object.keys(LL.archs).filter(a => LL.factory[LL.archs[a]]).sort().map(a => `<option value="${esc(a)}"${a === arch ? ' selected' : ''}>${esc(a)}${a === M.arch ? me : ''}</option>`).join('');
  $('#codeFile').onchange = e => { S.codeArch = e.target.value; S.codePart = 'graph'; renderCode(); };
  const nm = M.meta['general.name'] || M.arch, ac = `<code>${esc(arch)}</code>`;
  if (R.missing) {
    $('#codeExplain').innerHTML = tl(`Im Header steht die Architektur ${ac}. llama.cpp ${LL.version} kennt diesen Namen nicht und bricht beim Laden mit „unknown model architecture: '${esc(arch)}'“ ab. Ohne passenden Bauplan im Programm sind die Tabellen nur Zahlen.`,
      `The header names the architecture ${ac}. llama.cpp ${LL.version} doesn’t know this name and stops loading with “unknown model architecture: '${esc(arch)}'”. Without a matching blueprint in the program, the tables are just numbers.`);
    $('#codeChain').innerHTML = `<p class="count">${tl(`llama.cpp ${LL.version} kennt die Architektur ${ac} nicht.`, `llama.cpp ${LL.version} doesn’t know the architecture ${ac}.`)}</p>`;
    host.innerHTML = ''; $('#codeView').innerHTML = ''; $('#codeHead').innerHTML = ''; $('#codeParts').innerHTML = ''; return;
  }
  const link = (f, l) => `${LL.repo}${f === 'llama-graph.cpp' ? '' : 'models/'}${f}${l != null ? '#L' + (l + 1) : ''}`, nl = fmtInt(LL.lines[R.file].length);
  $('#codeChain').innerHTML = [
    `<div class="st"><b>${tl('Header der Datei', 'File header')}</b><span>general.architecture</span><code>"${esc(arch)}"</code></div>`,
    `<div class="st"><b>llama-arch.cpp</b><span>${tl('Name → Kennung', 'Name → ID')}</span><code>${esc(R.enum)}</code></div>`,
    `<div class="st"><b>llama-model.cpp</b><span>${tl('Kennung → Klasse', 'ID → class')}</span><code>new ${esc(R.cls)}</code></div>`,
    `<a class="st" href="${link(R.file)}" target="_blank" rel="noopener"><b>src/models/${esc(R.file)}</b><span>${tl(`${nl} Zeilen C++`, `${nl} lines of C++`)}</span><code>${R.chain.length > 1 ? tl('erbt von ', 'inherits from ') + esc(R.chain.slice(1).join(' → ')) : tl('Einstellungen, Tabellen, Rechenweg', 'settings, tables, code path')}</code></a>`,
  ].join('<span class="ar" aria-hidden="true">→</span>');
  $('#codeLicense').innerHTML = `<a href="https://github.com/ggml-org/llama.cpp" target="_blank" rel="noopener">llama.cpp</a> ${LL.version} · MIT · <details style="display:inline"><summary>${tl('Lizenztext', 'License text')}</summary><pre style="white-space:pre-wrap;font-size:12px">${esc(LL.license)}</pre></details>`;
  const parts = [['hparams', tl('1 · Einstellungen lesen', '1 · Read settings'), R.hparams], ['tensors', tl('2 · Tabellen laden', '2 · Load tables'), R.tensors], ['graph', tl('3 · Rechenweg', '3 · Code path'), R.graph]].filter(p => p[2]);
  if (!parts.some(p => p[0] === S.codePart)) S.codePart = parts.some(p => p[0] === 'graph') ? 'graph' : parts[0][0];
  const how = {
    hparams: tl('Hier liest llama.cpp die Einstellungen aus dem Header; was hier fehlt, ist im Programm fest eingebaut.', 'Here llama.cpp reads the settings from the header; anything not listed here is built into the program.'),
    tensors: tl('Hier holt llama.cpp die Tabellen aus der Datei, jede über ihren Namen in der GGUF-Datei.', 'Here llama.cpp loads the tables from the file, each by its name in the GGUF file.'),
    graph: tl('Links die Rechenschritte in Alltagssprache, automatisch aus dem Quelltext gelesen; eingerückt steht das Innere eines Teilschritts.', 'Left: the steps in plain words, read automatically from the source; indented lines show the inside of a sub-step.'),
  }[S.codePart];
  $('#codeExplain').innerHTML = (own
    ? tl(`Die Datei nennt nur ihren Architektur-Namen ${ac}. llama.cpp (das Programm in LM Studio) findet dazu den Programmteil, der genau diese Architektur rechnet, hier im Original (Version ${LL.version}).`,
      `The file only states its architecture name, ${ac}. llama.cpp (the program inside LM Studio) looks it up and finds the part of the program that computes exactly this architecture, shown here in the original (version ${LL.version}).`)
    : tl(`Zum Vergleich: so rechnet llama.cpp die Architektur ${ac}. Deine Datei (${esc(nm)}) ist <code>${esc(M.arch)}</code>, deshalb lassen sich die Tabellen hier nicht in ihr öffnen.`,
      `For comparison: this is how llama.cpp computes the architecture ${ac}. Your file (${esc(nm)}) is <code>${esc(M.arch)}</code>, so the tables can’t be opened in it here.`)) + ' ' + how;
  $('#codeParts').innerHTML = parts.map(([k, l]) => `<button type="button" data-part="${k}" aria-pressed="${k === S.codePart}">${l}</button>`).join('');
  $('#codeParts').onclick = e => { const b = e.target.closest('[data-part]'); if (b) { S.codePart = b.dataset.part; renderCode(); } };
  const d = parts.find(p => p[0] === S.codePart)[2];
  S.codeR = R; S.codeDef = d;
  if (S.codePart === 'graph') renderCodeSteps(R, d); else if (S.codePart === 'tensors') renderCodeTensors(R, d, own); else renderCodeHparams(R, d, own);
  showCode(d.file, d, d.line);
}
function stepHTML(x, own) {
  const pad = `padding-left:${10 + Math.min(x.lvl, 8) * 14}px`;
  if (x.type === 'ctl') return `<button type="button" class="ll-ctl" data-file="${x.file}" data-line="${x.line}" style="${pad}">${x.text}${x.cm ? `<small>// ${esc(x.cm)}</small>` : ''}</button>`;
  const comp = x.type === 'call' ? 'other' : llComp(x.comp), refs = (x.refs || []).filter(r => r.pat);
  const col = x.comp === 'mm' && refs[0] && own ? (llFound(refs[0].pat)[0] || {}).comp || 'other' : comp;
  const vr = LL_VARS[x.tgt], vt = Array.isArray(vr) ? vr[isEN() ? 1 : 0] : vr;   // LL_VARS (part E): string or [de, en]
  const meta = [llLine(x.line), x.tgt ? `→ <code>${esc(x.tgt)}</code>${vt ? ' ' + vt : ''}` : '', x.named ? `${tl('heißt', 'named')} <code>${esc(x.named)}</code>` : '',
    ...refs.slice(0, 3).map(r => { const n = own ? llFound(r.pat).length : 0; return `<span class="ll-tref" data-pat="${esc(r.pat)}">${esc(r.pat.replace('%d', 'N'))}</span>${own ? (n ? tl(` ${fmtInt(n)}× in der Datei`, ` ${fmtInt(n)}× in the file`) : tl(' nicht in dieser Datei', ' not in this file')) : ''}`; })].filter(Boolean).join(' · ');
  const lab = x.type === 'call' ? tl(`Teilschritt <code>${esc(x.name)}()</code>`, `Sub-step <code>${esc(x.name)}()</code>`) : x.label;
  return `<button type="button" class="ll-st${x.comp === 'shape' ? ' shape' : ''}${x.type === 'call' ? ' call' : ''}" data-file="${x.file}" data-line="${x.line}" style="${pad};--c:var(${COMPS[col] ? COMPS[col].v : '--c-other'})"><span class="ll-dot" aria-hidden="true"></span><span class="ll-lab">${lab}</span><span class="ll-meta">${meta}</span>${x.cm ? `<small>// ${esc(x.cm)}</small>` : ''}</button>`;
}
function renderCodeSteps(R, d) {
  const host = $('#codeSteps'), own = R.arch === S.model.arch, steps = llSteps(d, R, 0, new Set([d.key]));
  for (let k = steps.length - 1; k >= 0; k--) { const x = steps[k]; if (x.type !== 'ctl') continue; let j = k + 1; while (j < steps.length && steps[j].comp === 'shape') j++; if (j >= steps.length || steps[j].lvl <= x.lvl) steps.splice(k, 1); }
  const shapes = steps.filter(x => x.comp === 'shape').length;
  host.classList.toggle('show-shape', !!S.codeShape);
  const ns = fmtInt(steps.filter(x => x.type === 'step' && x.comp !== 'shape').length), cb0 = `<label class="ll-shape"><input type="checkbox" id="codeShape"${S.codeShape ? ' checked' : ''}> ${tl('zeigen', 'show')}</label>`;
  host.innerHTML = `<p class="ll-sum">${tl(`${ns} Rechenschritte${shapes ? ` · ${fmtInt(shapes)} reine Umform-Schritte ${cb0}` : ''}`, `${ns} steps${shapes ? ` · ${fmtInt(shapes)} reshape-only steps ${cb0}` : ''}`)}</p>` + steps.map(x => stepHTML(x, own)).join('');
  const cb = $('#codeShape'); if (cb) cb.onchange = e => { S.codeShape = e.target.checked; host.classList.toggle('show-shape', S.codeShape); };
  wireSteps(host);
}
function renderCodeTensors(R, d, own) {
  const b = llBody(d), f = d.file, src = LL.lines[f], rows = [];
  for (let i = b.open; i <= b.close; i++) {
    const m = /(\blayer\.)?([\w.]+)\s*=\s*create_tensor\(\s*tn\(\s*(LLM_TENSOR_\w+)\s*(?:,\s*"(\w+)")?/.exec(src[i]);
    if (m && LL.tensors[m[3]]) rows.push({ line: i, mem: (m[1] ? 'layer.' : '') + m[2], pat: LL.tensors[m[3]] + (m[4] ? '.' + m[4] : ''), opt: /TENSOR_NOT_REQUIRED/.test(src[i]) });
    else if (/create_tensor_qkv\s*\(/.test(src[i])) rows.push({ line: i, mem: tl('layer.wq / wk / wv (oder wqkv)', 'layer.wq / wk / wv (or wqkv)'), pat: LL.tensors.LLM_TENSOR_ATTN_Q + '.weight', qkv: true });
  }
  $('#codeSteps').innerHTML = `<p class="ll-sum">${tl('Name im Programm ← Name in der GGUF-Datei', 'Name in the program ← name in the GGUF file')}</p>` + rows.map(r => {
    const found = own ? llFound(r.pat) : [], comp = found[0] ? found[0].comp : 'other';
    const hit = !own ? '' : ' · ' + (found.length ? tl(`${fmtInt(found.length)}× in der Datei`, `${fmtInt(found.length)}× in the file`) : r.opt ? tl('optional, fehlt in dieser Datei', 'optional, missing in this file') : tl('fehlt in dieser Datei', 'missing in this file'));
    return `<button type="button" class="ll-st" data-file="${f}" data-line="${r.line}" style="padding-left:10px;--c:var(${COMPS[comp].v})"><span class="ll-dot" aria-hidden="true"></span><span class="ll-lab"><code>${esc(r.mem)}</code> ← <span class="ll-tref" data-pat="${esc(r.pat)}">${esc(r.pat.replace('%d', 'N'))}</span>${r.qkv ? ' (+ K, V)' : ''}</span><span class="ll-meta">${llLine(r.line)}${hit}</span></button>`;
  }).join('');
  wireSteps($('#codeSteps'));
}
function renderCodeHparams(R, d, own) {
  const b = llBody(d), f = d.file, src = LL.lines[f], M = S.model, rows = [];
  for (let i = b.open; i <= b.close; i++) {
    const m = /ml\.(get_key_or_arr|get_key|get_arr|get_arr_n)\(\s*(LLM_KV_\w+)\s*,\s*([\w.\[\]]+)/.exec(src[i]);
    if (!m || !LL.kv[m[2]]) continue;
    const key = LL.kv[m[2]].replace('%s', R.arch), v = own ? M.meta[key] : undefined;
    rows.push({ line: i, key, target: m[3], v, opt: /,\s*false\s*\)/.test(src[i]) });
  }
  const show = v => v === undefined ? null : v && v.items ? show(Array.from(v.items.slice(0, 7)).concat(arrLen(v) > 7 ? new Array(arrLen(v) - 7).fill(0) : [])) : ArrayBuffer.isView(v) ? show(Array.from(v.slice(0, 7))) : Array.isArray(v) ? `[${v.slice(0, 6).map(x => typeof x === 'number' ? sig(x, 4) : esc(String(x))).join(', ')}${v.length > 6 ? tl(`, … ${fmtInt(v.length)} Werte`, `, … ${fmtInt(v.length)} values`) : ''}]` : typeof v === 'number' ? sig(v, 6) : esc(String(v));
  $('#codeSteps').innerHTML = `<p class="ll-sum">${tl(`Schlüssel im Header${own ? ' = Wert in deiner Datei' : ''} · → Variable im Programm`, `Key in the header${own ? ' = value in your file' : ''} · → variable in the program`)}</p>` + rows.map(r => {
    const val = show(r.v);
    return `<button type="button" class="ll-st" data-file="${f}" data-line="${r.line}" style="padding-left:10px;--c:var(--c-norm)"><span class="ll-dot" aria-hidden="true"></span><span class="ll-lab"><code>${esc(r.key)}</code>${own ? ` = <b>${val ?? `<i>${tl('nicht in der Datei', 'not in the file')}</i>`}</b>` : ''}</span><span class="ll-meta">${llLine(r.line)} · → <code>${esc(r.target)}</code>${r.opt ? ' · optional' : ''}${own && val == null && r.opt ? tl(' · Standardwert aus dem Programm', ' · default from the program') : ''}</span></button>`;
  }).join('');
  wireSteps($('#codeSteps'));
}
function wireSteps(host) {
  host.onclick = e => {
    const t = e.target.closest('.ll-tref'); if (t && S.codeR.arch === S.model.arch) { const f = llFound(t.dataset.pat); if (f.length) { openTensor(f[0]); return; } }
    const b = e.target.closest('[data-line]'); if (!b) return;
    host.querySelectorAll('[aria-current="true"]').forEach(x => x.removeAttribute('aria-current')); b.setAttribute('aria-current', 'true');
    showCode(b.dataset.file, null, +b.dataset.line);
  };
}
function showCode(f, d, line) {
  const view = $('#codeView'), lines = LL.lines[f];
  if (S.codeShown !== f) {
    const st = { block: false };
    view.innerHTML = lines.map((l, i) => `<span class="ln"><i>${i + 1}</i>${llHi(l, f, st) || ' '}</span>`).join('');
    S.codeShown = f;
    view.onmouseover = e => {
      const r = e.target.closest('.c-ref'); if (!r) return hideTip();
      const x = llRef(r.dataset.ref, S.codeR), own = S.codeR.arch === S.model.arch;
      if (!x.pat) return showTip(`<code>${esc(r.dataset.ref)}</code><br><span class="m">${tl('keine Tabelle (Hilfswert im Programm)', 'not a table (helper value in the program)')}</span>`, e);
      const fd = own ? llFound(x.pat) : [], t = fd[0], ti = t ? `${t.dims.map(fmtInt).join(' × ')} · ${GGUF.TYPES[t.type] ? GGUF.TYPES[t.type][0] : t.type}` : '';
      showTip(`<code>${esc(r.dataset.ref)}</code> = ${tl('Tabelle', 'table')} <code>${esc(x.pat.replace('%d', 'N'))}</code><br><span class="m">${own ? (t ? tl(`${fmtInt(fd.length)}× in deiner Datei · ${ti} · Klick öffnet sie`, `${fmtInt(fd.length)}× in your file · ${ti} · click to open`) : tl('nicht in dieser Datei', 'not in this file')) : tl('andere Architektur als deine Datei', 'different architecture from your file')}</span>`, e);
    };
    view.onmouseout = e => { if (e.target.closest('.c-ref')) hideTip(); };
    view.onclick = e => {
      const r = e.target.closest('.c-ref'); if (r && S.codeR.arch === S.model.arch) { const x = llRef(r.dataset.ref, S.codeR); const fd = x.pat ? llFound(x.pat) : []; if (fd.length) openTensor(fd[0]); return; }
      const fn = e.target.closest('.c-fn'); if (fn) { const dd = LL.defs.get(fn.dataset.def); if (dd) showCode(dd.file, dd, dd.line); }
    };
  }
  const D = d || S.codeDef;
  view.querySelectorAll('.ln.in, .ln.hit').forEach(x => x.classList.remove('in', 'hit'));
  if (D && D.file === f) { const b = llBody(D); if (b) for (let i = D.line; i <= b.close; i++) view.children[i] && view.children[i].classList.add('in'); }
  const el = view.children[line]; if (el) { el.classList.add('hit'); view.scrollTop = Math.max(0, el.offsetTop - view.clientHeight / 3); }
  const link = `${LL.repo}${f === 'llama-graph.cpp' ? '' : 'models/'}${f}#L${line + 1}`;
  $('#codeHead').innerHTML = `<b class="mono">src/${f === 'llama-graph.cpp' ? '' : 'models/'}${esc(f)}</b> <span class="count">${tl('Zeile', 'line')} ${fmtInt(line + 1)} · llama.cpp ${LL.version}</span>${f !== (S.codeDef && S.codeDef.file) ? ` <button type="button" class="linkbtn" data-back>${tl(`zurück zu ${esc(S.codeDef.file)}`, `back to ${esc(S.codeDef.file)}`)}</button>` : ''} <a class="linkbtn" href="${link}" target="_blank" rel="noopener">${tl('auf GitHub', 'on GitHub')}</a>`;
  const bk = $('#codeHead [data-back]'); if (bk) bk.onclick = () => showCode(S.codeDef.file, S.codeDef, S.codeDef.line);
}

// =====================================================================
// Durchlauf: the model really computes (llama.cpp in the Lupe-Server), every step recorded
// =====================================================================
const RUN = { llm: null, info: null, pos: null, layer: null, dim: null, stack: null, lens: null, blocks: {}, step: null, view: '3d', same: false, head: -1, seq: 0, busy: false, ex: null, exList: null };
const STEP_DE = {
  'model.input_embed': 'Embedding-Zeile holen', inp_embd: 'Embedding-Zeile holen', inp_scaled: 'Embedding skalieren',
  norm: 'Normieren', attn_norm: '× Norm-Gewicht', attn_post_norm: '× Norm-Gewicht', ffn_norm: '× Norm-Gewicht', result_norm: 'Letztes Normieren',
  Qcur_full: 'Q und Gate berechnen', Qcur_normed: 'Q × Norm-Gewicht', Qcur: 'Q berechnen / drehen', Kcur_normed: 'K × Norm-Gewicht', Kcur: 'K berechnen / drehen', Vcur: 'V berechnen',
  kq: 'Punktzahlen Q·K', kq_soft_max: 'Aufmerksamkeit (Softmax)', kqv: 'V nach Aufmerksamkeit mischen', kqv_out: 'Köpfe zusammenfügen',
  gate_sigmoid: 'Gate (Sigmoid)', attn_gated: 'mit Gate multiplizieren', attn_output: 'Attention-Ausgabe', attn_residual: '+ Eingang (Residual)', ffn_inp: '+ Eingang (Residual)',
  ffn_gate: 'FFN: Gate', ffn_up: 'FFN: hoch', ffn_swiglu: 'FFN: Aktivierung', ffn_gelu: 'FFN: Aktivierung', ffn_geglu: 'FFN: Aktivierung', ffn_out: 'FFN: runter', ffn_down: 'FFN: runter',
  post_ffn: '+ Eingang (Residual)', l_out: 'Ergebnis des Blocks',
  linear_attn_qkv_mixed: 'q, k, v berechnen', conv_input: 'letzte Tokens dazunehmen', conv_output_raw: 'Faltung über die letzten Tokens', conv_output_silu: 'Aktivierung (SiLU)',
  q_conv_predelta: 'q normieren', k_conv_predelta: 'k normieren', v_conv_predelta: 'v', alpha: 'Zerfall berechnen', a_softplus: 'Zerfall (Softplus)', gate: 'Vergessen',
  beta: 'Schreibstärke berechnen', beta_sigmoid: 'Schreibstärke β', final_output: 'aus dem Speicher lesen', z: 'Gate z', linear_attn_out: 'Ausgabe des Speichers',
  result_output: 'Ausgabe-Kopf', h_nextn: 'Letztes Normieren',
};
const OP_DE = { MUL_MAT: 'Matrix × Vektor', MUL: 'malnehmen', ADD: 'addieren', RMS_NORM: 'normieren', SOFT_MAX: 'Softmax', ROPE: 'drehen (Position)', GATED_DELTA_NET: 'Speicher aktualisieren (DeltaNet)',
  SSM_CONV: 'Faltung', SILU: 'SiLU', SIGMOID: 'Sigmoid', SOFTPLUS: 'Softplus', SWIGLU: 'SwiGLU', GEGLU: 'GeGLU', GET_ROWS: 'Zeilen holen', SCALE: 'skalieren', CONCAT: 'aneinanderhängen', L2_NORM: 'L2-normieren', GELU: 'GELU', EXP: 'e hoch', SUB: 'subtrahieren', DIV: 'teilen', NEG: 'Vorzeichen' };
const STEP_EN = {
  'model.input_embed': 'Fetch embedding row', inp_embd: 'Fetch embedding row', inp_scaled: 'Scale embedding',
  norm: 'Normalize', attn_norm: '× norm weight', attn_post_norm: '× norm weight', ffn_norm: '× norm weight', result_norm: 'Final normalization',
  Qcur_full: 'Compute Q and gate', Qcur_normed: 'Q × norm weight', Qcur: 'Compute / rotate Q', Kcur_normed: 'K × norm weight', Kcur: 'Compute / rotate K', Vcur: 'Compute V',
  kq: 'Scores Q·K', kq_soft_max: 'Attention (softmax)', kqv: 'Mix V by attention', kqv_out: 'Merge heads',
  gate_sigmoid: 'Gate (sigmoid)', attn_gated: 'Multiply by gate', attn_output: 'Attention output', attn_residual: '+ input (residual)', ffn_inp: '+ input (residual)',
  ffn_gate: 'FFN: gate', ffn_up: 'FFN: up', ffn_swiglu: 'FFN: activation', ffn_gelu: 'FFN: activation', ffn_geglu: 'FFN: activation', ffn_out: 'FFN: down', ffn_down: 'FFN: down',
  post_ffn: '+ input (residual)', l_out: 'Block result',
  linear_attn_qkv_mixed: 'Compute q, k, v', conv_input: 'Include recent tokens', conv_output_raw: 'Convolution over recent tokens', conv_output_silu: 'Activation (SiLU)',
  q_conv_predelta: 'Normalize q', k_conv_predelta: 'Normalize k', v_conv_predelta: 'v', alpha: 'Compute decay', a_softplus: 'Decay (softplus)', gate: 'Forget',
  beta: 'Compute write strength', beta_sigmoid: 'Write strength β', final_output: 'Read from memory', z: 'Gate z', linear_attn_out: 'Memory output',
  result_output: 'Output head', h_nextn: 'Final normalization',
};
const OP_EN = { MUL_MAT: 'Matrix × vector', MUL: 'multiply', ADD: 'add', RMS_NORM: 'normalize', SOFT_MAX: 'softmax', ROPE: 'rotate (position)', GATED_DELTA_NET: 'update memory (DeltaNet)',
  SSM_CONV: 'convolution', SILU: 'SiLU', SIGMOID: 'sigmoid', SOFTPLUS: 'softplus', SWIGLU: 'SwiGLU', GEGLU: 'GeGLU', GET_ROWS: 'fetch rows', SCALE: 'scale', CONCAT: 'concatenate', L2_NORM: 'L2-normalize', GELU: 'GELU', EXP: 'e to the power', SUB: 'subtract', DIV: 'divide', NEG: 'flip sign' };
// label of a graph node / ggml op in the current language; undefined for unknown keys
const pickLabel = (de, en, k) => Object.hasOwn(de, k) ? (isEN() && en[k]) || de[k] : undefined;
const stepLabel = k => pickLabel(STEP_DE, STEP_EN, k), opLabel = k => pickLabel(OP_DE, OP_EN, k);
const INFO = {   // ⓘ popover texts: [de, en] (or a function using tl)
  run: ['Das Modell rechnet hier wirklich: <b>llama.cpp</b>, dasselbe Programm wie in LM Studio, beantwortet deine Frage mit den Gewichten aus der Datei. Danach wird der ganze Text in einem Durchgang noch einmal gerechnet und dabei <b>jedes Zwischenergebnis</b> mitgeschrieben (höchstens 100 Tokens). Gewählt wird immer das wahrscheinlichste Token, deshalb gibt dieselbe Frage dieselbe Antwort.',
    'The model really computes here: <b>llama.cpp</b>, the same program as in LM Studio, answers your question with the weights from the file. Then the whole text is computed again in a single pass, recording <b>every intermediate result</b> (up to 100 tokens). It always picks the most likely token, so the same question gives the same answer.'],
  ex: ['Ein mit llama.cpp aufgezeichneter Durchlauf, mit beliebigem Modell; zum Ansehen braucht es weder Server noch Grafikkarte. Gespeichert sind die Rechenschritte jedes Blocks, Aufmerksamkeit und Vorhersage für jedes Token, der Weg durch alle Schichten für die wichtigsten Tokens und alle Zahlen von drei Blöcken. Eigene Beispiele: mit dem Lupe-Server eine Frage durchrechnen, dann „Als Beispiel speichern“. Sie landen im Ordner <code>examples</code> neben der Seite; wer den Ordner weitergibt, gibt sie mit.',
    'A forward pass recorded with llama.cpp, with any model; viewing it needs neither a server nor a GPU. Stored are the steps of every block, attention and prediction for every token, the path through all layers for the key tokens, and all numbers of three blocks. To add your own, run a question with the Lupe server and click “Save as example”. It is saved in the <code>examples</code> folder next to the page; share the folder to pass it on.'],
  toks: ['Jedes Kästchen ist ein Token. Grau: deine Eingabe samt Chat-Vorlage; farbig: die Antwort des Modells, darunter die Wahrscheinlichkeit, mit der es gewählt wurde. Ein Klick wählt das Token, dessen Weg du verfolgst.',
    'Each cell is a token. Gray: your input including the chat template; colored: the model’s answer, with the probability it was picked shown below. Click a token to follow its path.'],
  stack: ['Ein Token ist eine Zeile aus vielen Zahlen, eine pro Dimension. Jede Platte ist <b>diese Zeile nach einer Schicht</b>: unten die Embedding-Zeile, oben das Ergebnis des letzten Blocks; jedes Kästchen eine Dimension (blau negativ, rot positiv). Mausrad, Ziehen oder der Regler wechseln die Schicht. Ein Klick wählt eine Dimension, sie erscheint als Säule durch alle Platten.',
    'A token is a row of many numbers, one per dimension. Each plate is <b>that row after one layer</b>: the embedding row at the bottom, the last block’s result at the top; each cell is one dimension (blue negative, red positive). Scroll, drag or use the slider to change the layer. Click a cell to pick a dimension; it shows as a column through all plates.'],
  scale: ['Die Zahlen werden von Schicht zu Schicht deutlich größer (Länge rechts). Damit in jeder Schicht das Muster sichtbar bleibt, ist jede Platte für sich skaliert. „Gleiche Skala“ zeigt die echten Größen.',
    'The numbers grow a lot from layer to layer (length on the right). To keep the pattern visible in every layer, each plate is scaled on its own. “Same scale” shows the true sizes.'],
  plane: ['Die gewählte Schicht flach ausgebreitet, jedes Kästchen eine Dimension. Ein Klick wählt die Dimension für den Verlauf darunter.',
    'The chosen layer laid out flat, each cell one dimension. Click a cell to pick the dimension for the curve below.'],
  dim: ['Der Wert dieser einen Dimension durch alle Schichten. Das graue Band zeigt, wie groß die Zahlen einer Schicht typischerweise sind. Einzelne Dimensionen sagen selten etwas; die Bedeutung steckt im Zusammenspiel aller.',
    'The value of this one dimension through all layers. The gray band shows how large the numbers of a layer typically are. A single dimension rarely means much; the meaning lies in all of them together.'],
  lens: ['<b>Logit-Linse:</b> Der Vektor nach dieser Schicht wird probehalber schon hier durch den Ausgabe-Kopf geschickt. So sieht man, ab welcher Schicht das Modell „weiß“, was als Nächstes kommt; frühe Schichten liefern meist Unsinn, die Antwort entsteht oft erst am Ende. Die Kurve zeigt die Wahrscheinlichkeit des Tokens, das tatsächlich folgt.',
    '<b>Logit lens:</b> the vector after this layer is sent through the output head early, as a test. This shows from which layer on the model “knows” what comes next; early layers mostly give nonsense, and the answer often forms only at the end. The curve shows the probability of the token that actually follows.'],
  block: ['Die echten Rechenschritte dieses Blocks, in der Reihenfolge, in der llama.cpp sie ausgeführt hat. „×“ nennt die Tabelle aus der GGUF-Datei, die der Schritt benutzt. Ein Klick auf einen Schritt zeigt seine Zahlen für dein Token, ein Klick auf eine Zahl, wie sie entstanden ist.',
    'The real steps of this block, in the order llama.cpp ran them. “×” names the table from the GGUF file that the step uses. Click a step to see its numbers for your token, and a number to see how it came about.'],
  attn: ['<b>Aufmerksamkeit:</b> Auf welche früheren Tokens schaut dein Token in dieser Schicht? Jeder Kopf schaut für sich; die Balken eines Kopfes ergeben zusammen 100 %. Ein Token sieht nur sich selbst und was davor steht.',
    '<b>Attention:</b> which earlier tokens does your token look at in this layer? Each head looks on its own; the bars of one head add up to 100 %. A token sees only itself and what comes before it.'],
  lin: ['Diese Schicht hat keine Aufmerksamkeits-Tabelle. Sie trägt einen <b>festen Speicher</b> von Token zu Token weiter (Gated DeltaNet): Jedes Token schreibt etwas hinein (Schreibstärke β) und lässt Altes verblassen (Behalten). Darum kostet sie bei langen Texten kaum mehr Rechenzeit.',
    'This layer has no attention table. It carries a <b>fixed-size memory</b> from token to token (Gated DeltaNet): each token writes something into it (write strength β) and lets old content fade (keep). That is why long texts cost it barely any extra compute.'],
  out: ['Was das Modell an dieser Stelle als nächstes Token vorschlägt. Bei der Antwort wurde jeweils das oberste genommen.',
    'What the model proposes as the next token at this position. For the answer, the top one was taken each time.'],
  zoom: ['Eine einzelne Zahl, nachgerechnet aus den echten Eingaben dieses Schritts und, falls vorhanden, der Gewichtszeile aus der GGUF-Datei. Bei Matrix × Vektor rundet llama.cpp die Eingabe vorher auf 8 Bit, um schneller zu rechnen; die Lupe rechnet exakt und mit derselben Rundung nach. Die gerundete Rechnung trifft llama.cpp fast genau, der Rest sind Rundungen beim Aufsummieren.',
    'A single number, recomputed from this step’s real inputs and, if there is one, the weight row from the GGUF file. For matrix × vector, llama.cpp first rounds the input to 8 bits to compute faster; GGUF-Lupe recomputes it both exactly and with the same rounding. The rounded result matches llama.cpp almost exactly; the rest is rounding during summation.'],
};
const pctf = p => p >= 0.9995 ? '100 %' : p >= 0.1 ? nf(Math.floor(p * 1000) / 10, 1) + ' %' : p >= 0.001 ? nf(p * 100, 2) + ' %' : `< ${nf(0.1, 1)} %`;
const tokTxt = id => { const ex = RUN.ex; if (ex && !ex.sameVocab) { const t = ex.h.tok && ex.h.tok[String(id)]; return t == null ? '#' + id : t; } return disp(id); };
const tokHTML = id => { const t = tokTxt(id); return t === '' ? '<span class="ws">∅</span>' : wsHTML(t.length > 24 ? t.slice(0, 22) + '…' : t); };
const tokQ = id => { const t = tokTxt(id), s = t.trim() ? esc(t.trim()) : wsHTML(t) || '∅'; return tl(`„${s}“`, `“${s}”`); };
function resetRun() { Object.assign(RUN, { info: null, pos: null, layer: null, dim: null, stack: null, lens: null, blocks: {}, step: null, seq: RUN.seq + 1, ex: null, detail: null }); }

// ---------- one data layer: live run on the server, or a recorded example (examples/*.js)
class ExMissing extends Error {}
function exArr(key) {
  const ex = RUN.ex, a = ex.h.arrays[key]; if (!a) return null;
  const h = new Uint16Array(ex.buf, ex.o + a[0] * 2, a[1]), v = new Float32Array(a[1]);
  for (let k = 0; k < a[1]; k++) v[k] = GGUF.F16[h[k]];
  return v;
}
const RD = {
  stack: p => { if (!RUN.ex) return runBin('/api/run/stack', { run: RUN.info.id, p }); const m = RUN.ex.h.stacks[String(p)], v = m && exArr('stack:' + p); return v ? Promise.resolve({ meta: m, v }) : Promise.reject(new ExMissing('stack')); },
  lens: p => RUN.ex ? (RUN.ex.h.lens[String(p)] ? Promise.resolve(RUN.ex.h.lens[String(p)]) : Promise.reject(new ExMissing('lens'))) : srvJson('/api/run/lens', { run: RUN.info.id, p }),
  block: b => RUN.ex ? Promise.resolve(RUN.ex.h.blocks[String(b)]) : srvJson('/api/run/block', { run: RUN.info.id, b }),
  vec: (n, p) => { if (!RUN.ex) return runBin('/api/run/vec', { run: RUN.info.id, n, p }); const m = RUN.ex.h.vecs[`${n}:${p}`], v = m && exArr(`vec:${n}:${p}`); return v ? Promise.resolve({ meta: m, v }) : Promise.reject(new ExMissing('vec')); },
  explain: (n, p, j) => {
    if (!RUN.ex) return srvJson('/api/run/explain', { run: RUN.info.id, n, p, j });
    const E = RUN.ex.h.explains, pre = `${n}:${p}:`;
    if (E[pre + j]) return Promise.resolve(E[pre + j]);
    const js = Object.keys(E).filter(k => k.startsWith(pre)).map(k => +k.slice(pre.length));
    if (!js.length) return Promise.reject(new ExMissing('explain'));
    const near = js.reduce((a, b) => Math.abs(b - j) < Math.abs(a - j) ? b : a);
    return Promise.resolve({ ...E[pre + near], snapped: true });
  },
};
function loadScriptData(src, cb) {
  // a <script> works everywhere: opened as a file, through the server, and online
  return new Promise((res, rej) => {
    const s = document.createElement('script');
    window[cb] = d => { delete window[cb]; res(d); };
    s.src = src; s.onload = () => { s.remove(); if (window[cb]) { delete window[cb]; rej(new Error(tl(`Datei ohne Inhalt: ${src}`, `File has no content: ${src}`))); } };
    s.onerror = () => { s.remove(); delete window[cb]; rej(new Error(tl(`nicht gefunden: ${src}`, `not found: ${src}`))); };
    document.head.appendChild(s);
  });
}
async function loadExampleList(force) {
  if (RUN.exList && !force) return RUN.exList;
  try { RUN.exList = await loadScriptData('examples/index.js' + (/^https?:$/.test(location.protocol) ? '?t=' + Date.now() : ''), 'lupeBeispielListe'); }
  catch { RUN.exList = []; }
  return RUN.exList;
}
const dateLoc = d => { const m = /^(\d{4})-(\d{2})-(\d{2})$/.exec(d || ''); return m ? new Date(+m[1], m[2] - 1, +m[3]).toLocaleDateString(LOC(), { day: 'numeric', month: 'short', year: 'numeric' }) : (d || ''); };
function exampleCards(list) {
  const cur = RUN.ex && RUN.ex.h.id;
  return list.map(x => `<button type="button" class="run-ex-card" data-ex="${esc(x.id)}" aria-current="${x.id === cur}"><b>${esc(x.title)}</b><small>${esc(x.model)} · ${tl(`${fmtInt(x.tokens)} Tokens`, `${fmtInt(x.tokens)} tokens`)} · ${dateLoc(x.recorded)} · ${nf(x.mb, 1)} MB</small></button>`).join('');
}
async function renderExamples(auto) {
  const box = $('#runEx'), all = await loadExampleList();
  if (!all.length) { box.hidden = true; return; }
  const mine = x => (x.lang || 'de') === LANG, list = all.filter(mine).concat(all.filter(x => !mine(x)));   // examples in the page language first
  box.hidden = false;
  box.innerHTML = `<div class="run-ex-h"><b>${tl('Beispiele', 'Examples')}</b>${ib('ex')}<span class="count">${tl('aufgezeichnet mit llama.cpp', 'recorded with llama.cpp')}</span></div><div class="run-ex-list">${exampleCards(list)}</div>`;
  box.querySelector('.run-ex-list').onclick = e => { const b = e.target.closest('[data-ex]'); if (b) loadExample(list.find(x => x.id === b.dataset.ex)); };
  if (auto && !RUN.info) loadExample(list[0]);
  else if (RUN.ex && $('#runStatus').innerHTML === RUN.exSt) exStatus();   // status line untouched since loading: redo it in the current language
}
function exStatus() {
  const st = $('#runStatus'), h = RUN.ex.h, I = h.info, na = I.tokens.length - I.n_prompt, end = `${dateLoc(h.recorded)} · llama.cpp ${esc(h.llama)} · ${esc(h.device)}`;
  st.innerHTML = tl(`<b>Beispiel</b> · ${esc(h.model)} · ${fmtInt(I.n_prompt)} + ${fmtInt(na)} Tokens · ${fmtInt(I.n_layer)} Blöcke · ${fmtInt(I.n_graph)} Rechenschritte · ${end}`,
    `<b>Example</b> · ${esc(h.model)} · ${fmtInt(I.n_prompt)} + ${fmtInt(na)} tokens · ${fmtInt(I.n_layer)} blocks · ${fmtInt(I.n_graph)} steps · ${end}`);
  RUN.exSt = st.innerHTML;
}
async function loadExample(item) {
  const st = $('#runStatus'), prog = $('#runProg'), seq = ++RUN.seq, mb = nf(item.mb, 1);
  st.textContent = tl(`Lade Beispiel „${item.title}“ (${mb} MB) …`, `Loading example “${item.title}” (${mb} MB) …`); prog.hidden = false; prog.firstChild.style.width = '30%';
  try {
    const bytes = b64(await loadScriptData(item.file, 'lupeBeispiel'));
    const buf = await new Response(new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'))).arrayBuffer();
    if (seq !== RUN.seq) return;
    const dv = new DataView(buf), hl = dv.getUint32(0, true), h = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 4, hl)));
    let o = 4 + hl; o += o % 2;
    resetRun();
    const M = S.model, sameVocab = !!M && M.vocabSize === h.n_vocab && h.info.tokens.slice(0, 40).every(id => disp(id) === (h.tok || {})[String(id)]);
    RUN.ex = { h, buf, o, sameVocab };
    RUN.info = h.info; RUN.pos = h.mainPos; RUN.layer = h.info.layer_nodes.length - 1;
    exStatus();
    $('#runBody').hidden = false;
    $('#runEx').querySelectorAll('[data-ex]').forEach(b => b.setAttribute('aria-current', String(b.dataset.ex === h.id)));
    await selectPos(RUN.pos, true);
  } catch (e) { st.innerHTML = `<b>${tl('Beispiel ließ sich nicht laden:', 'Could not load the example:')}</b> ${esc(e.message)}`; }
  finally { prog.hidden = true; }
}
async function saveExample() {
  const I = RUN.info, btn = $('#runSave'), st = $('#runSaveSt'); if (!I || RUN.ex || !btn) return;
  btn.disabled = true; st.textContent = tl('speichere …', 'saving …');
  try {
    const job = await srvJson('/api/run/export', { run: I.id, lang: LANG });
    const done = await pollJob(job, (p, m) => { st.textContent = `${srvMsg(m)} (${Math.round(p * 100)} %)`; });
    const fn = `<code>${esc(done.result.file)}</code> · ${nf(done.result.mb, 1)} MB`;
    st.innerHTML = tl(`gespeichert als ${fn}`, `saved as ${fn}`);
    await loadExampleList(true); renderExamples();
  } catch (e) { st.textContent = tl('Speichern fehlgeschlagen: ', 'Saving failed: ') + srvMsg(e.message); btn.disabled = false; }
}
function showNoStack() {
  const I = RUN.info, have = RUN.ex ? RUN.ex.h.stackPos : [], box = $('#runMiss'), q = tokQ(I.tokens[RUN.pos]);
  $('.run-grid').hidden = true; box.hidden = false;
  box.innerHTML = `<p style="margin:0">${tl(`${q}: Weg durch die Schichten nicht gespeichert. Gespeichert für:`, `${q}: path through the layers not stored. Stored for:`)}</p><div class="rt-list">${have.map(k => `<button type="button" class="rt has" data-k="${k}"><span>${tokHTML(I.tokens[k])}</span></button>`).join('')}</div>`;
  box.onclick = e => { const b = e.target.closest('[data-k]'); if (b) selectPos(+b.dataset.k); };
}

// ---------- info popover
function infoHTML(btn) {
  if (btn.dataset.infoEl) return btn.dataset.infoEl.split(' ').map(id => { const el = document.getElementById(id); return el ? el.innerHTML : ''; }).filter(Boolean).join('<br><br>');
  const v = INFO[btn.dataset.info];
  return typeof v === 'function' ? v() : Array.isArray(v) ? v[isEN() ? 1 : 0] : (v || '');
}
function infoPop(btn) {
  const pop = $('#infoPop'), key = btn.dataset.info || '#' + btn.dataset.infoEl;
  if (!pop.hidden && pop.dataset.key === key) { pop.hidden = true; return; }
  pop.innerHTML = infoHTML(btn); pop.dataset.key = key; pop.hidden = false;
  const r = btn.getBoundingClientRect(), w = Math.min(380, innerWidth - 24);
  pop.style.width = w + 'px';
  pop.style.left = Math.max(12, Math.min(innerWidth - w - 12, r.left + r.width / 2 - w / 2)) + 'px';
  const h = pop.offsetHeight; pop.style.top = (r.bottom + 8 + h > innerHeight - 8 ? Math.max(8, r.top - h - 8) : r.bottom + 8) + 'px';
}
document.addEventListener('click', e => { const b = e.target.closest('.ib'); if (b) { e.preventDefault(); infoPop(b); return; } const pop = $('#infoPop'); if (pop && !pop.hidden && !e.target.closest('#infoPop')) pop.hidden = true; });
document.addEventListener('keydown', e => { if (e.key === 'Escape') { const pop = $('#infoPop'); if (pop) pop.hidden = true; } });
document.addEventListener('scroll', () => { const pop = $('#infoPop'); if (pop) pop.hidden = true; }, true);
const ib = key => `<button type="button" class="ib" data-info="${key}" aria-label="${tl('Erklärung', 'Explanation')}">i</button>`;

// ---------- data
async function runBin(p, q) {
  const r = await fetch(srvUrl(p, q));
  if (!r.ok) throw new Error(srvMsg((await r.json().catch(() => ({}))).error || r.statusText));
  const buf = await r.arrayBuffer(), dv = new DataView(buf), ml = dv.getUint32(0, true), meta = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 4, ml)));
  let o = 4 + ml; o += o % 2;
  const h = new Uint16Array(buf.slice(o)), v = new Float32Array(h.length);
  for (let k = 0; k < h.length; k++) v[k] = GGUF.F16[h[k]];
  return { meta, v };
}

// ---------- tab
async function renderRunTab() {
  const M = S.model, note = $('#runNote'), form = $('#runForm');
  const live = SRV.on && isSrv() && !!M.vocabSize;       // a live run needs a language model (an image encoder has no vocabulary)
  renderExamples(!live);                                 // without server: open the first example right away
  if (SRV.on && isSrv() && !M.vocabSize) {
    form.hidden = true; note.hidden = false;
    note.innerHTML = tl(`<b>Kein Sprachmodell</b> · Architektur „${esc(M.arch)}“ ohne Wörterbuch · oben ein Sprachmodell öffnen`, `<b>Not a language model</b> · architecture “${esc(M.arch)}” has no vocabulary · open a language model above`);
    if (RUN.info) { $('#runBody').hidden = false; drawRunAll(); } else $('#runBody').hidden = true;
    return;
  }
  if (!live) {
    form.hidden = true; note.hidden = false;
    note.innerHTML = (!SRV.on
      ? tl(`<b>Eigene Fragen:</b> Lupe-Server starten (<code>start-lupe-server.cmd</code>)`, `<b>Your own questions:</b> start the Lupe server (<code>start-lupe-server.cmd</code>)`)
      : tl(`<b>Eigene Fragen:</b> oben dein Modell über die Server-Liste öffnen`, `<b>Your own questions:</b> open your model from the server list above`)) + ' ' + ib('runServer');
    if (!SRV.on && location.protocol === 'file:' && await probeLocalServer()) note.innerHTML = tl(`<b>Lupe-Server läuft</b> · Seite ist als Datei geöffnet <a class="btn primary" href="http://127.0.0.1:8765/#durchlauf">Über den Server öffnen</a>`, `<b>Lupe server is running</b> · page opened as a file <a class="btn primary" href="http://127.0.0.1:8765/#durchlauf">Open via the server</a>`);
    if (RUN.info) { $('#runBody').hidden = false; drawRunAll(); } else $('#runBody').hidden = true;
    return;
  }
  if (!RUN.llm) {
    note.hidden = false; note.textContent = tl('Frage llama.cpp nach Rechengeräten …', 'Asking llama.cpp for devices…');
    try { RUN.llm = await srvJson('/api/llm/info'); } catch (e) { note.innerHTML = tl('Fehler: ', 'Error: ') + esc(srvMsg(e.message)); return; }
  }
  const L = RUN.llm;
  if (!L.devices) {
    note.hidden = false;
    const v = esc(L.version), err = L.error ? esc(srvMsg(L.error)) + ' ' : '';
    note.innerHTML = tl(`<b>llama.cpp fehlt</b> (offizielle Version ${v}) ${err}<span class="run-get">Einmal herunterladen: <button type="button" class="btn" data-get="vulkan">Grafikkarte (Vulkan, 33 MB)</button> <button type="button" class="btn" data-get="cuda">NVIDIA (CUDA, 580 MB)</button> <button type="button" class="btn" data-get="cpu">nur CPU (19 MB)</button></span>`,
      `<b>llama.cpp is missing</b> (official version ${v}) ${err}<span class="run-get">One-time download: <button type="button" class="btn" data-get="vulkan">GPU (Vulkan, 33 MB)</button> <button type="button" class="btn" data-get="cuda">NVIDIA (CUDA, 580 MB)</button> <button type="button" class="btn" data-get="cpu">CPU only (19 MB)</button></span>`);
    note.querySelectorAll('[data-get]').forEach(b => b.onclick = () => getLlama(b.dataset.get));
    form.hidden = true; return;
  }
  note.hidden = true; form.hidden = false;
  const sel = $('#runDev');
  if (!sel.options.length) {
    sel.innerHTML = L.devices.map(d => `<option value="${esc(d.id)}">${esc(d.kind === 'CPU' ? 'CPU' : d.name)}${d.kind !== 'CPU' ? ` (${esc(d.backend)}, ${nf(d.total_gb, 0)} GB)` : ''}</option>`).join('');
    const gpu = L.devices.filter(d => d.kind === 'GPU').sort((a, b) => b.total_gb - a.total_gb)[0];
    sel.value = gpu ? gpu.id : 'cpu';
  }
  const dq = ['Was ist die Hauptstadt von Frankreich?', 'What is the capital of France?']; if (!$('#runQ').value || dq.includes($('#runQ').value)) $('#runQ').value = tl(dq[0], dq[1]);   // default question follows the language until edited
  if (RUN.info && (RUN.ex || RUN.info.path === M.source.path)) { $('#runBody').hidden = false; drawRunAll(); }
}
async function probeLocalServer() {
  try { await fetch('http://127.0.0.1:8765/api/info', { mode: 'no-cors', cache: 'no-store' }); return true; } catch { return false; }
}
async function getLlama(kind) {
  const note = $('#runNote');
  note.innerHTML = tl(`Lade llama.cpp (${esc(kind)}) von GitHub …`, `Downloading llama.cpp (${esc(kind)}) from GitHub…`) + '<div class="progress"><div></div></div>';
  try {
    const job = await srvJson('/api/llm/get', { kind });
    await pollJob(job, (p, m) => { note.querySelector('.progress div').style.width = (p * 100).toFixed(0) + '%'; });
    RUN.llm = null; renderRunTab();
  } catch (e) { note.innerHTML = tl('Herunterladen fehlgeschlagen: ', 'Download failed: ') + esc(srvMsg(e.message)); }
}
async function startRun(e) {
  e.preventDefault();
  if (RUN.busy) return;
  const M = S.model, q = $('#runQ').value.trim(); if (!q) return;
  const prog = $('#runProg'), st = $('#runStatus'), seq = ++RUN.seq;
  RUN.busy = true; $('#runGo').disabled = true; prog.hidden = false; prog.firstChild.style.width = '0%'; st.textContent = tl('Starte …', 'Starting…');
  try {
    const job = await srvJson('/api/run/start', { path: M.source.path, q, chat: $('#runChat').checked ? 1 : 0, max: $('#runMax').value, dev: $('#runDev').value });
    const done = await pollJob(job, (p, m) => { prog.firstChild.style.width = (p * 100).toFixed(1) + '%'; st.textContent = srvMsg(m); });
    if (seq !== RUN.seq || S.model !== M) return;
    const info = await srvJson('/api/run/info', { run: done.result.run });
    resetRun(); RUN.info = info;
    RUN.pos = info.n_prompt - 1; RUN.layer = info.layer_nodes.length - 1;
    const np = fmtInt(info.n_prompt), na = fmtInt(info.tokens.length - info.n_prompt), nb = fmtInt(info.n_layer), ns = fmtInt(info.n_graph), tg = nf(info.timing.gen_s, 1), tr = nf(info.timing.rec_s, 2);
    st.innerHTML = tl(`<b>${np} + ${na} Tokens</b> · ${nb} Blöcke · ${ns} Rechenschritte · Antwort ${tg} s · Aufzeichnung ${tr} s`, `<b>${np} + ${na} tokens</b> · ${nb} blocks · ${ns} steps · answer ${tg} s · recording ${tr} s`) +
      ` · ${esc(info.device)} <button type="button" class="btn" id="runSave" style="padding:3px 10px;font-size:13px">${tl('Als Beispiel speichern', 'Save as example')}</button> <span class="count" id="runSaveSt"></span>`;
    $('#runSave').onclick = saveExample;
    $('#runEx').querySelectorAll('[data-ex]').forEach(b => b.setAttribute('aria-current', 'false'));
    $('#runBody').hidden = false;
    await selectPos(RUN.pos, true);
  } catch (err) { st.innerHTML = tl('<b>Fehler:</b> ', '<b>Error:</b> ') + esc(srvMsg(err.message)); }
  finally { RUN.busy = false; $('#runGo').disabled = false; prog.hidden = true; }
}
function drawRunAll() { renderRunToks(); drawStack(); drawPlane(); drawDim(); renderLens(); renderBlock(); renderAttn(); renderOut(); }

// ---------- tokens
function renderRunToks() {
  const I = RUN.info, box = $('#runToks');
  const has = RUN.ex ? new Set(RUN.ex.h.stackPos) : null;
  const ni = fmtInt(I.n_prompt), na = fmtInt(I.tokens.length - I.n_prompt), end = I.why === 'Ende';   // 'Ende' = server value
  box.innerHTML = `<div class="rt-head"><b>Text</b> ${ib('toks')}<span class="count">${tl(`Eingabe ${ni} Tokens · Antwort ${na}${end ? ' · Modell hat das Ende gewählt' : ''}`, `input ${ni} tokens · answer ${na}${end ? ' · model chose to stop' : ''}`)}</span></div><div class="rt-list">` +
    I.tokens.map((t, k) => { const ans = k >= I.n_prompt, p = ans ? I.p_next[k - 1] : null, hk = has && has.has(k);
      return `<button type="button" class="rt${ans ? ' ans' : ''}${k === RUN.pos ? ' sel' : ''}${hk ? ' has' : ''}${tokType(t) === 3 || tokType(t) === 4 ? ' sp' : ''}" data-k="${k}" title="${tl(`Position ${k} · Token ${fmtInt(t)}${hk ? ' · Weg gespeichert' : ''}`, `Position ${k} · token ${fmtInt(t)}${hk ? ' · path stored' : ''}`)}"><span>${tokHTML(t)}</span>${ans ? `<small>${pctf(p)}</small>` : ''}</button>`; }).join('') +
    (end ? `<span class="rt end" title="${tl('Das Modell hat das Ende-Token gewählt', 'The model chose the end token')}">⏹</span>` : '') + '</div>';
  box.onclick = e => { const b = e.target.closest('.rt[data-k]'); if (b) selectPos(+b.dataset.k); };
  $('#runStackTitle').innerHTML = tl(`Der Weg von ${tokQ(I.tokens[RUN.pos])} durch alle Schichten`, `The path of ${tokQ(I.tokens[RUN.pos])} through all layers`);
}
async function selectPos(k, first) {
  const I = RUN.info; RUN.pos = k; const seq = ++RUN.seq;
  renderRunToks();                                       // also sets the stack title (so a language switch redraws it)
  try {
    const [st, lens] = await Promise.all([RD.stack(k).catch(e => { if (e instanceof ExMissing) return null; throw e; }), RD.lens(k).catch(e => ({ error: e instanceof ExMissing ? null : e.message }))]);
    if (seq !== RUN.seq) return;
    if (!st) { RUN.stack = null; RUN.lens = null; showNoStack(); renderOut(); renderAttn(); if (first) renderBlock(); return; }
    $('.run-grid').hidden = false; $('#runMiss').hidden = true;
    const d = st.meta.cols, C = Math.round(Math.sqrt(d * 1.25)), R = Math.ceil(d / C);
    RUN.stack = { v: st.v, rows: st.meta.rows, cols: d, C, R, rms: st.meta.rms, planes: null };
    RUN.lens = lens.error ? null : lens; RUN.lensErr = lens.error || null;
    if (RUN.dim == null || first) { const L = RUN.layer, o = L * d; let best = 0; for (let j = 1; j < d; j++) if (Math.abs(st.v[o + j]) > Math.abs(st.v[o + best])) best = j; RUN.dim = best; }
    drawStack(); drawPlane(); drawDim(); renderLens(); renderOut(); renderAttn();
    if (RUN.step != null) openStep(RUN.step, true);
    if (first) renderBlock();
  } catch (e) { $('#runStatus').innerHTML = tl('<b>Fehler:</b> ', '<b>Error:</b> ') + esc(srvMsg(e.message)); }
}
function selectLayer(L) {
  const rows = RUN.stack ? RUN.stack.rows : RUN.info ? RUN.info.layer_nodes.length : 0; if (!rows) return;
  L = Math.max(0, Math.min(rows - 1, L)); if (L === RUN.layer) return;
  RUN.layer = L; RUN.step = null;
  $('#runLayer').value = L;
  drawStack(); drawPlane(); drawDim(); renderLens(); renderBlock(); renderAttn();
}
const layerName = L => L === 0 ? tl('Embedding (vor Block 1)', 'Embedding (before block 1)') : tl(`nach Block ${L}`, `after block ${L}`);

// ---------- the stack of layers
function ensurePlanes() {
  const st = RUN.stack; if (st.planes && st.planesSame === RUN.same) return st.planes;
  const T = theme(), lut = T.lut, { rows, cols: d, C, R, v } = st;
  let glob = null;
  if (RUN.same) { const a = []; for (let k = 0; k < v.length; k += 7) a.push(Math.abs(v[k])); a.sort((x, y) => x - y); glob = a[Math.floor(a.length * 0.995)] || 1; }
  st.planes = [];
  for (let l = 0; l < rows; l++) {
    const c = document.createElement('canvas'); c.width = C; c.height = R;
    const g = c.getContext('2d'), img = g.createImageData(C, R), px = img.data, s = glob || Math.max(1e-9, 2.5 * st.rms[l]);
    for (let q = 0; q < C * R; q++) {
      const o = q * 4;
      if (q < d) { const kk = colorIdx(v[l * d + q], s); px[o] = lut[kk]; px[o + 1] = lut[kk + 1]; px[o + 2] = lut[kk + 2]; } else { px[o] = T.surfRGB[0]; px[o + 1] = T.surfRGB[1]; px[o + 2] = T.surfRGB[2]; }
      px[o + 3] = 255;
    }
    g.putImageData(img, 0, 0); st.planes.push(c);
  }
  st.planesSame = RUN.same;
  return st.planes;
}
function drawStack() {
  const cv = $('#runStackCv'), st = RUN.stack; if (!st || !cv.offsetParent) return;
  const dpr = devicePixelRatio || 1, W = cv.parentElement.clientWidth - 28, H = Math.round(Math.max(360, Math.min(640, W * 0.9)));
  if (cv.width !== Math.round(W * dpr) || cv.height !== Math.round(H * dpr)) { cv.width = Math.round(W * dpr); cv.height = Math.round(H * dpr); cv.style.width = W + 'px'; cv.style.height = H + 'px'; }
  const ctx = cv.getContext('2d'), T = theme(), L = RUN.layer;
  ctx.setTransform(1, 0, 0, 1, 0, 0); ctx.clearRect(0, 0, cv.width, cv.height);
  $('#runLayer').max = st.rows - 1; $('#runLayer').value = L; $('#runLayerOut').textContent = layerName(L);
  if (RUN.view === '2d') return drawFlat(ctx, W, H, dpr);
  const planes = ensurePlanes(), { rows, C, R } = st;
  const k = (W - 122) / (C + R), dh = (C + R) * k * 0.5, top = 14, dz = Math.max(1.5, (H - dh - top - 18) / Math.max(1, rows - 1)), x0 = 110 + R * k;
  RUN.geo = { k, dz, x0, top, rows, dh, C, R };
  ctx.imageSmoothingEnabled = false;
  const yOf = l => top + (rows - 1 - l) * dz;
  for (let l = 0; l < rows; l++) {
    const y0 = yOf(l);
    ctx.setTransform(dpr * k, dpr * k * 0.5, -dpr * k, dpr * k * 0.5, dpr * x0, dpr * y0);
    ctx.globalAlpha = l <= L ? 1 : 0.06;
    ctx.drawImage(planes[l], 0, 0);
  }
  ctx.globalAlpha = 1; ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const corner = (l, i, j) => [x0 + (i - j) * k, yOf(l) + (i + j) * k * 0.5];
  const diamond = l => { const a = corner(l, 0, 0), b = corner(l, C, 0), c = corner(l, C, R), d = corner(l, 0, R); ctx.beginPath(); ctx.moveTo(...a); ctx.lineTo(...b); ctx.lineTo(...c); ctx.lineTo(...d); ctx.closePath(); };
  ctx.strokeStyle = T.line2; ctx.lineWidth = 1; diamond(rows - 1); ctx.stroke();
  ctx.strokeStyle = T.ink; ctx.lineWidth = 1.6; diamond(L); ctx.stroke();
  // the chosen dimension as a column through the layers
  if (RUN.dim != null) {
    const i = RUN.dim % C, j = Math.floor(RUN.dim / C), [xb, yb] = corner(0, i + 0.5, j + 0.5), [, yt] = corner(L, i + 0.5, j + 0.5);
    ctx.strokeStyle = T.ink; ctx.lineWidth = 2; ctx.beginPath(); ctx.moveTo(xb, yb); ctx.lineTo(xb, yt); ctx.stroke();
    ctx.beginPath(); ctx.arc(xb, yt, 4, 0, 6.2832); ctx.stroke();
  }
  // layer ticks on the left
  ctx.font = `11px ${T.ui}`; ctx.fillStyle = T.muted; ctx.textAlign = 'right';
  const step = rows > 40 ? 16 : 8;
  for (let l = 0; l < rows; l += step) { if (Math.abs(l - L) * dz < 13) continue; const [xl, yl] = corner(l, 0, R); ctx.fillText(l === 0 ? 'Embedding' : `Block ${l}`, xl - 6, yl + 4); }
  const [xl, yl] = corner(L, 0, R); ctx.fillStyle = T.ink; ctx.font = `600 12px ${T.ui}`; ctx.fillText(layerName(L), xl - 6, yl + 4);
  ctx.textAlign = 'left';
}
function drawFlat(ctx, W, H, dpr) {
  const st = RUN.stack, T = theme(), lut = T.lut, { rows, cols: d, v } = st, padL = 74, padB = 8, w = Math.max(10, Math.floor(W - padL - 8)), h = H - padB - 10;
  const bins = Math.min(w, d), per = d / bins, img = new ImageData(bins, rows);
  for (let l = 0; l < rows; l++) {
    const s = Math.max(1e-9, 2.5 * st.rms[l]), r = rows - 1 - l;
    for (let b = 0; b < bins; b++) {
      let best = 0; const a0 = Math.floor(b * per), a1 = Math.max(a0 + 1, Math.floor((b + 1) * per));
      for (let j = a0; j < a1; j++) { const x = v[l * d + j]; if (Math.abs(x) > Math.abs(best)) best = x; }
      const kk = colorIdx(best, s), o = (r * bins + b) * 4; img.data[o] = lut[kk]; img.data[o + 1] = lut[kk + 1]; img.data[o + 2] = lut[kk + 2]; img.data[o + 3] = l <= RUN.layer ? 255 : 70;
    }
  }
  const off = document.createElement('canvas'); off.width = bins; off.height = rows; off.getContext('2d').putImageData(img, 0, 0);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.imageSmoothingEnabled = false;
  ctx.drawImage(off, padL, 10, w, h);
  RUN.geo = { flat: true, padL, w, h, rows, top: 10, per, bins };
  ctx.font = `11px ${T.ui}`; ctx.fillStyle = T.muted; ctx.textAlign = 'right';
  const rh = h / rows, step = rows > 40 ? 16 : 8;
  for (let l = 0; l < rows; l += step) ctx.fillText(l === 0 ? 'Embedding' : `Block ${l}`, padL - 6, 10 + (rows - 1 - l + 0.5) * rh + 4);
  ctx.strokeStyle = T.ink; ctx.lineWidth = 1.5; ctx.strokeRect(padL - 0.5, 10 + (rows - 1 - RUN.layer) * rh, w + 1, rh);
  if (RUN.dim != null) { const x = padL + (RUN.dim / d) * w; ctx.beginPath(); ctx.moveTo(x, 10); ctx.lineTo(x, 10 + h); ctx.stroke(); }
  ctx.textAlign = 'left';
}
function stackHit(e) {
  const cv = $('#runStackCv'), g = RUN.geo, st = RUN.stack; if (!g || !st) return null;
  const b = cv.getBoundingClientRect(), mx = e.clientX - b.left, my = e.clientY - b.top;
  if (g.flat) {
    if (mx < g.padL || mx > g.padL + g.w || my < g.top || my > g.top + g.h) return null;
    const l = st.rows - 1 - Math.floor((my - g.top) / (g.h / st.rows)), dd = Math.min(st.cols - 1, Math.floor((mx - g.padL) / g.w * st.cols));
    return { l, d: dd };
  }
  const y0 = g.top + (g.rows - 1 - RUN.layer) * g.dz, xr = (mx - g.x0) / g.k, yr = (my - y0) / (g.k * 0.5);
  const i = Math.floor((yr + xr) / 2), j = Math.floor((yr - xr) / 2);
  if (i < 0 || j < 0 || i >= g.C || j >= g.R || j * g.C + i >= st.cols) return { l: null, y: my };
  return { l: RUN.layer, d: j * g.C + i };
}

// ---------- the chosen layer, flat
function drawPlane() {
  const cv = $('#runPlaneCv'), st = RUN.stack; if (!st || !cv.offsetParent) return;
  const L = RUN.layer, { C, R, cols: d, v } = st, T = theme(), lut = T.lut;
  const cell = Math.max(2, Math.floor(Math.min(cv.parentElement.clientWidth - 28, 520) / C)), W = C * cell, H = R * cell, dpr = devicePixelRatio || 1;
  cv.width = W * dpr; cv.height = H * dpr; cv.style.width = W + 'px'; cv.style.height = H + 'px';
  const ctx = cv.getContext('2d'); ctx.setTransform(1, 0, 0, 1, 0, 0); ctx.imageSmoothingEnabled = false;
  ctx.drawImage(ensurePlanes()[L], 0, 0, C, R, 0, 0, W * dpr, H * dpr);
  RUN.planeCell = cell;
  if (RUN.dim != null) { ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.strokeStyle = T.ink; ctx.lineWidth = 2; ctx.strokeRect((RUN.dim % C) * cell - 1, Math.floor(RUN.dim / C) * cell - 1, cell + 2, cell + 2); }
  let lo = 0, hi = 0; for (let j = 0; j < d; j++) { const x = v[L * d + j]; if (x < v[L * d + lo]) lo = j; if (x > v[L * d + hi]) hi = j; }
  $('#runPlaneTitle').textContent = tl(`${layerName(L)} · ${fmtInt(d)} Dimensionen`, `${layerName(L)} · ${fmtInt(d)} dimensions`);
  const len = sig(Math.sqrt(st.rms[L] ** 2 * d), 3), vh = sig(v[L * d + hi], 3), vl = sig(v[L * d + lo], 3);
  $('#runPlaneInfo').innerHTML = tl(`Länge <b>${len}</b> · max ${vh} (Dimension ${fmtInt(hi)}) · min ${vl} (Dimension ${fmtInt(lo)})`, `Length <b>${len}</b> · max ${vh} (dimension ${fmtInt(hi)}) · min ${vl} (dimension ${fmtInt(lo)})`);
}

// ---------- one dimension through all layers
function drawDim() {
  const svg = $('#runDimSvg'), st = RUN.stack; if (!st || RUN.dim == null) return;
  const d = RUN.dim, rows = st.rows, D = st.cols, W = Math.max(260, svg.parentElement.clientWidth - 28), H = 150, pl = 44, pr = 8, pt = 10, pb = 22, T = theme();
  const ys = Array.from({ length: rows }, (_, l) => st.v[l * D + d]);
  const band = st.rms.map(r => r);
  let mx = 1e-9; for (let l = 0; l < rows; l++) mx = Math.max(mx, Math.abs(ys[l]), band[l]);
  const X = l => pl + (W - pl - pr) * l / Math.max(1, rows - 1), Y = y => pt + (H - pt - pb) * (0.5 - y / (2 * mx));
  const bandPath = 'M' + band.map((b, l) => `${X(l).toFixed(1)},${Y(b).toFixed(1)}`).join('L') + 'L' + band.map((b, l) => `${X(rows - 1 - l).toFixed(1)},${Y(-band[rows - 1 - l]).toFixed(1)}`).join('L') + 'Z';
  const line = 'M' + ys.map((y, l) => `${X(l).toFixed(1)},${Y(y).toFixed(1)}`).join('L');
  const ticks = [-mx, 0, mx].map(t => `<text x="${pl - 6}" y="${Y(t) + 4}" text-anchor="end" class="ax">${sig(t, 2)}</text>`).join('');
  const xt = [0, Math.round((rows - 1) / 2), rows - 1].map(l => `<text x="${X(l)}" y="${H - 6}" text-anchor="middle" class="ax">${l === 0 ? 'Emb.' : 'Block ' + l}</text>`).join('');
  svg.setAttribute('viewBox', `0 0 ${W} ${H}`); svg.setAttribute('width', W); svg.setAttribute('height', H);
  svg.innerHTML = `<path d="${bandPath}" fill="${T.line}" opacity=".7"/><line x1="${pl}" x2="${W - pr}" y1="${Y(0)}" y2="${Y(0)}" stroke="${T.line2}"/>${ticks}${xt}
    <line x1="${X(RUN.layer)}" x2="${X(RUN.layer)}" y1="${pt}" y2="${H - pb}" stroke="${T.ink}" stroke-dasharray="3 3"/>
    <path d="${line}" fill="none" stroke="var(--c-attn)" stroke-width="2"/><circle cx="${X(RUN.layer)}" cy="${Y(ys[RUN.layer])}" r="4" fill="var(--c-attn)"/>`;
  $('#runDimTitle').textContent = tl(`Dimension ${fmtInt(d)} durch alle Schichten · hier ${sig(ys[RUN.layer], 3)}`, `Dimension ${fmtInt(d)} through all layers · here ${sig(ys[RUN.layer], 3)}`);
  svg.onclick = e => { const b = svg.getBoundingClientRect(), x = (e.clientX - b.left) * W / b.width; selectLayer(Math.round((x - pl) / (W - pl - pr) * (rows - 1))); };
}
function selectDim(d) { RUN.dim = d; drawStack(); drawPlane(); drawDim(); }

// ---------- logit lens
const lensNext = (nxt, row, fol) => { const t = `<b class="mono">${tokHTML(nxt)}</b>`, rk = fmtInt(row.rank_next), p = pctf(row.p_next);
  return tl(`Tatsächlich ${fol ? 'folgt' : 'gewählt'}: ${t} · hier Rang ${rk} mit ${p}`, `${fol ? 'Actual next token' : 'Chosen'}: ${t} · rank ${rk} here (${p})`); };
function renderLens() {
  const box = $('#runLens'), I = RUN.info, Lr = RUN.lens;
  if (!Lr) { box.innerHTML = `<p class="count">${RUN.lensErr ? tl('Nicht verfügbar: ', 'Not available: ') + esc(srvMsg(RUN.lensErr)) : tl('lädt …', 'loading…')}</p>`; return; }
  const row = Lr.rows[RUN.layer], nxt = Lr.next, rows = Lr.rows.length, mxp = Math.max(...row.top.map(t => t[1]), 1e-9);
  const W = Math.max(240, box.clientWidth || 300), H = 70, X = l => 6 + (W - 12) * l / (rows - 1), Y = p => H - 14 - (H - 24) * p;
  const area = nxt != null ? `M${X(0)},${Y(0)}` + Lr.rows.map((r, l) => `L${X(l).toFixed(1)},${Y(r.p_next || 0).toFixed(1)}`).join('') + `L${X(rows - 1)},${Y(0)}Z` : '';
  box.innerHTML = `<p class="lens-l">${layerName(RUN.layer)}:</p><ol class="lens-top">${row.top.map(([id, p]) => `<li class="${id === nxt ? 'hit' : ''}"><span class="mono">${tokHTML(id)}</span><i style="width:${(p / mxp * 100).toFixed(1)}%"></i><b>${pctf(p)}</b></li>`).join('')}</ol>` +
    (nxt != null ? `<p class="count">${lensNext(nxt, row, RUN.pos + 1 < I.tokens.length)}</p>
      <svg class="lens-curve" viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" role="img" aria-label="${tl('Wahrscheinlichkeit des nächsten Tokens je Schicht', 'Probability of the next token per layer')}"><path d="${area}" fill="var(--c-head)" opacity=".35"/><path d="${area.replace(/Z$/, '')}" fill="none" stroke="var(--c-head)" stroke-width="1.5"/>
      <line x1="${X(RUN.layer)}" x2="${X(RUN.layer)}" y1="4" y2="${H - 12}" stroke="var(--ink)" stroke-dasharray="3 3"/><text x="6" y="${H - 2}" class="ax">Embedding</text><text x="${W - 6}" y="${H - 2}" text-anchor="end" class="ax">${tl('letzter Block', 'last block')}</text><text x="${W - 6}" y="12" text-anchor="end" class="ax">100 %</text></svg>` : '') +
    (Lr.check ? '' : `<p class="count">${tl('Hinweis: Linse bei dieser Architektur nur ungefähr', 'Note: lens only approximate for this architecture')}</p>`);
  const sv = box.querySelector('.lens-curve'); if (sv) sv.onclick = e => { const b = sv.getBoundingClientRect(); selectLayer(Math.round(((e.clientX - b.left) * W / b.width - 6) / (W - 12) * (rows - 1))); };
}

// ---------- the steps inside one block
async function renderBlock() {
  const I = RUN.info, box = $('#runFlow'), L = RUN.layer, seq = RUN.seq;
  $('#runDetail').hidden = true;
  if (L === 0) {
    const tok = I.tokens[RUN.pos];
    $('#runBlockTitle').innerHTML = tl('Schicht 0: die Embedding-Zeile', 'Layer 0: the embedding row') + ` ${ib('block')}`;
    box.innerHTML = `<p class="count">${tl(`Token ${tokQ(tok)} (ID ${fmtInt(tok)}) → Zeile ${fmtInt(tok)} der Embedding-Tabelle`, `Token ${tokQ(tok)} (ID ${fmtInt(tok)}) → row ${fmtInt(tok)} of the embedding table`)} <button type="button" class="linkbtn" id="runToEmb">${tl('ansehen', 'view')}</button></p>`;
    $('#runToEmb').onclick = () => { setTab('embed'); jumpToRow(tok); };
    return;
  }
  const b = L - 1, kind = I.layer_kinds[b];
  $('#runBlockTitle').innerHTML = (kind === 'attn' ? tl(`Block ${L}: Attention und FFN`, `Block ${L}: Attention and FFN`) : kind === 'lin' ? tl(`Block ${L}: Lineare Attention (Speicher) und FFN`, `Block ${L}: Linear attention (memory) and FFN`) : tl(`Block ${L}: Rechenschritte und FFN`, `Block ${L}: Steps and FFN`)) +
    ` ${ib('block')} <span class="seg sm rf-nav" role="group" aria-label="${tl('Schritt für Schritt', 'Step by step')}"><button type="button" data-st="-1">${tl('◀ Schritt', '◀ Step')}</button><button type="button" data-st="1">${tl('Schritt ▶', 'Step ▶')}</button></span>`;
  $('#runBlockTitle').querySelector('.rf-nav').onclick = e => { const x = e.target.closest('[data-st]'); if (x) stepNav(+x.dataset.st); };
  let data = RUN.blocks[b];
  if (!data) { box.innerHTML = `<p class="count">${tl('lädt …', 'loading…')}</p>`; try { data = RUN.blocks[b] = await RD.block(b); } catch (e) { box.innerHTML = tl('Fehler: ', 'Error: ') + esc(srvMsg(e.message)); return; } if (seq !== RUN.seq && RUN.layer !== L) return; }
  if (RUN.layer !== L) return;
  const wshort = w => w.replace(/^blk\.\d+\./, '').replace(/\.weight$/, '');
  box.innerHTML = data.steps.map((s, k) => {
    if (s.tail && !(data.steps[k - 1] || {}).tail) return `<span class="rf-tail">${tl('nach dem letzten Block:', 'after the last block:')}</span>` + stepChip(s, k, I, wshort);
    return stepChip(s, k, I, wshort);
  }).join('');
  box.onclick = e => { const b2 = e.target.closest('.rf'); if (b2) openStep(+b2.dataset.n); };
  if (RUN.pendingStep) { const st = data.steps, want = RUN.pendingStep; RUN.pendingStep = null; if (st.length) openStep(want === 'first' ? st[0].i : st[st.length - 1].i); }
}
function stepNav(dir) {
  const L = RUN.layer, data = RUN.blocks[L - 1];
  if (L === 0) { if (dir > 0) { RUN.pendingStep = 'first'; selectLayer(1); } return; }
  if (!data) return;
  const st = data.steps, k = st.findIndex(s => s.i === RUN.step), nk = k < 0 ? (dir > 0 ? 0 : st.length - 1) : k + dir;
  if (nk >= 0 && nk < st.length) { openStep(st[nk].i); $('#runFlow').querySelector(`.rf[data-n="${st[nk].i}"]`)?.scrollIntoView({ block: 'nearest' }); return; }
  const L2 = L + dir; if (L2 < 1 || L2 >= RUN.info.layer_nodes.length) return;
  RUN.pendingStep = dir > 0 ? 'first' : 'last'; selectLayer(L2);
}
function stepChip(s, k, I, wshort) {
  {
    const lab = stepLabel(s.base) || opLabel(s.op) || s.op, ws = s.ins.filter(x => x.w), size = s.ne[0] * (s.tok ? (s.ne[2] > 1 && s.ne[1] !== I.tokens.length ? s.ne[1] : 1) : 1);
    return `${k ? '<span class="rf-ar" aria-hidden="true">→</span>' : ''}<button type="button" class="rf${s.i === RUN.step ? ' sel' : ''}" data-n="${s.i}" style="--c:var(${COMPS[s.comp] ? COMPS[s.comp].v : '--c-other'})"><b>${esc(lab)}</b><small class="mono">${esc(s.name)}</small>${ws.map(x => `<em class="mono">× ${esc(wshort(x.w))}</em>`).join('')}</button>`;
  }
}
async function openStep(n, keep) {
  const I = RUN.info, box = $('#runDetail'), seq = RUN.seq;
  RUN.step = n;
  document.querySelectorAll('#runFlow .rf').forEach(b => b.classList.toggle('sel', +b.dataset.n === n));
  box.hidden = false; if (!keep) box.innerHTML = `<p class="count">${tl('lädt …', 'loading…')}</p>`;
  try {
    const { meta, v } = await RD.vec(n, RUN.pos);
    if (RUN.step !== n) return;
    const s = (RUN.blocks[RUN.layer - 1] || { steps: [] }).steps.find(x => x.i === n) || { base: meta.name, op: meta.op, ins: [] };
    const lab = stepLabel(s.base) || opLabel(meta.op) || meta.op;
    const shape = meta.shape.filter((x, k) => k === 0 || x > 1), heads = shape.length > 1 ? shape[1] : 1;
    const ins = (s.ins || []).map(x => x.w ? `<button type="button" class="linkbtn mono" data-w="${esc(x.w)}">${esc(x.w)}</button>` : `<span class="mono">${esc(x.name)}</span>`).join(' , ');
    const all = (RUN.blocks[RUN.layer - 1] || { steps: [] }).steps, no = all.findIndex(x => x.i === n);
    const op = esc(opLabel(meta.op) || meta.op), nz = fmtInt(meta.n), hd = heads > 1 ? ` (${fmtInt(shape[1])} × ${fmtInt(shape[0])})` : '', tq = tokQ(I.tokens[RUN.pos]);
    box.innerHTML = `<div class="rd-head">${no >= 0 ? `<span class="count">${tl(`Schritt ${no + 1} von ${all.length}`, `Step ${no + 1} of ${all.length}`)}</span>` : ''}<b>${esc(lab)}</b> <code>${esc(meta.name)}</code> <span class="count">${tl(`${op} aus ${ins || '–'} · ${nz} Zahlen${hd} für ${tq}`, `${op} from ${ins || '–'} · ${nz} numbers${hd} for ${tq}`)} · min ${sig(meta.min, 3)} · max ${sig(meta.max, 3)}</span> ${ib('zoom')}</div>
      <canvas class="rd-stripe"></canvas><p class="count rd-hint" hidden></p><div class="rd-zoom"></div>`;
    box.querySelectorAll('[data-w]').forEach(b => b.onclick = () => { const t = S.model.byName.get(b.dataset.w); if (t) openTensor(t); });
    RUN.detail = { n, v, meta, heads, ne0: shape[0] };
    drawStripe();
    const cv = box.querySelector('.rd-stripe');
    cv.onmousemove = e => { const j = stripeIdx(e); if (j == null) return hideTip(); const hh = heads > 1 ? Math.floor(j / shape[0]) + 1 : 0, nr = j % shape[0], val = sig(v[j], 4);
      showTip(tl(`Stelle ${fmtInt(j)}${hh ? ` (Kopf ${hh}, Nr. ${nr})` : ''}: <b>${val}</b><br><span class="m">Klick: so entstand diese Zahl</span>`, `Entry ${fmtInt(j)}${hh ? ` (head ${hh}, no. ${nr})` : ''}: <b>${val}</b><br><span class="m">Click: how this number was computed</span>`), e); };
    cv.onmouseleave = hideTip;
    cv.onclick = e => { const j = stripeIdx(e); if (j != null) explain(n, j); };
    let best = 0; for (let j = 1; j < v.length; j++) if (Math.abs(v[j]) > Math.abs(v[best])) best = j;
    explain(n, RUN.detail.j != null && RUN.detail.j < v.length && keep ? RUN.detail.j : best);
  } catch (e) {
    if (e instanceof ExMissing && RUN.ex) {
      const h = RUN.ex.h, bl = h.detailBlocks.map(b => b + 1).join(', '), many = h.detailBlocks.length > 1, mt = tokQ(I.tokens[h.mainPos]);
      box.innerHTML = `<p class="count">${tl(`Im Beispiel gespeichert: alle Rechenschritte für ${many ? 'Blöcke' : 'Block'} ${bl} beim Token ${mt}`, `Stored in this example: all steps for ${many ? 'blocks' : 'block'} ${bl} at token ${mt}`)}</p><p>${h.detailBlocks.map(b => `<button type="button" class="btn" data-xb="${b}" style="padding:3px 10px;font-size:13px">${tl(`Block ${b + 1} ansehen`, `View block ${b + 1}`)}</button>`).join(' ')}</p>`;
      box.querySelectorAll('[data-xb]').forEach(x => x.onclick = async () => { const b = +x.dataset.xb; if (RUN.pos !== h.mainPos) await selectPos(h.mainPos); RUN.pendingStep = 'first'; if (RUN.layer === b + 1) renderBlock(); else selectLayer(b + 1); });
      return;
    }
    if (seq === RUN.seq || keep) box.innerHTML = tl('Fehler: ', 'Error: ') + esc(srvMsg(e.message));
  }
}
function drawStripe() {
  const D = RUN.detail, box = $('#runDetail'), cv = box && box.querySelector('.rd-stripe'); if (!D || !cv) return;
  const { v, heads, ne0 } = D, n = v.length, W = Math.max(200, box.clientWidth - 24), H = 96, dpr = devicePixelRatio || 1, T = theme();
  cv.width = W * dpr; cv.height = H * dpr; cv.style.width = W + 'px'; cv.style.height = H + 'px';
  const ctx = cv.getContext('2d'); ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const sorted = Array.from(v, Math.abs).sort((a, b) => a - b), p99 = sorted[Math.floor(sorted.length * 0.995)] || 1e-9, vmax = sorted[sorted.length - 1] || 1e-9;
  const mx = Math.max(1e-9, Math.min(vmax, p99 * 1.3)), mid = H / 2, per = n / W;
  ctx.fillStyle = T.line; ctx.fillRect(0, mid, W, 1);
  for (let x = 0; x < W; x++) {
    const a0 = Math.floor(x * per), a1 = Math.max(a0 + 1, Math.floor((x + 1) * per)); let b = 0;
    for (let j = a0; j < a1 && j < n; j++) if (Math.abs(v[j]) > Math.abs(b)) b = v[j];
    const clip = Math.abs(b) > mx, h = Math.min(1, Math.abs(b) / mx) * (mid - 6);
    ctx.fillStyle = b < 0 ? css('--div-neg') : css('--div-pos'); ctx.fillRect(x, b < 0 ? mid : mid - h, Math.max(1, 1 / per), Math.max(0.5, h));
    if (clip) { ctx.fillStyle = T.ink; ctx.fillRect(x - 1, b < 0 ? mid + h + 1 : mid - h - 4, Math.max(3, 1 / per + 2), 3); }
  }
  if (heads > 1) { ctx.fillStyle = T.line2; for (let h = 1; h < heads; h++) ctx.fillRect(Math.round(h * ne0 / per), 0, 1, H); }
  if (D.j != null) { const x = (D.j + 0.5) / per; ctx.strokeStyle = T.ink; ctx.lineWidth = 1.5; ctx.beginPath(); ctx.moveTo(x, 2); ctx.lineTo(x, H - 2); ctx.stroke(); }
  if (RUN.ex) { const pre = `${D.n}:${RUN.pos}:`; ctx.fillStyle = T.ink; for (const k of Object.keys(RUN.ex.h.explains)) if (k.startsWith(pre)) { const x = (+k.slice(pre.length) + 0.5) / per; ctx.fillRect(x - 3, 0, 6, 3); } }
  D.per = per;
  const hint = box.querySelector('.rd-hint'), cut = vmax > mx * 1.01, r = vmax / mx, rf = nf(r, r < 10 ? 1 : 0);   // legend only when bars are clipped
  if (hint) { hint.hidden = !cut; hint.innerHTML = cut ? tl(`Balken mit Strich abgeschnitten · bis ${rf}× größer`, `Bars with a dash are clipped · up to ${rf}× larger`) : ''; }
}
function stripeIdx(e) { const D = RUN.detail; if (!D) return null; const b = e.target.getBoundingClientRect(); const j = Math.floor((e.clientX - b.left) * D.per); return j >= 0 && j < D.v.length ? j : null; }

// ---------- zooming into one number
async function explain(n, j) {
  const I = RUN.info, box = $('#runDetail .rd-zoom'); if (!box) return;
  RUN.detail.j = j; drawStripe();
  box.innerHTML = `<p class="count">${tl('rechne nach …', 'recomputing…')}</p>`;
  let x; try { x = await RD.explain(n, RUN.pos, j); } catch (e) { box.innerHTML = e instanceof ExMissing ? `<p class="count">${tl('Für diesen Schritt ist im Beispiel keine Einzel-Rechnung gespeichert.', 'No breakdown stored in this example for this step.')}</p>` : tl('Fehler: ', 'Error: ') + esc(srvMsg(e.message)); return; }
  if (RUN.detail.n !== n || RUN.detail.j !== j) return;
  if (x.snapped) { RUN.detail.j = j = x.j; drawStripe(); }
  box.innerHTML = (x.snapped ? `<p class="count">${tl(`Gespeichert nur für markierte Stellen (Striche oben) · gezeigt: Stelle ${fmtInt(x.j)}`, `Stored only for marked entries (ticks on top) · showing entry ${fmtInt(x.j)}`)}</p>` : '') + zoomHTML(x);
  box.querySelectorAll('[data-w]').forEach(b => b.onclick = () => { const t = S.model.byName.get(b.dataset.w); if (t) openTensor(t); });
  box.querySelectorAll('[data-k]').forEach(b => b.onclick = () => selectPos(+b.dataset.k));
}
function zoomHTML(x) {
  const I = RUN.info, f = v => v == null ? '–' : sig(v, 5), name = esc(x.name), at = `${name}[${fmtInt(x.j)}]`;
  const check = (tot, val, extra) => `<p class="zz-sum">${tl('Nachgerechnet', 'Recomputed')}: <b>${f(tot)}</b> · llama.cpp: <b>${f(val)}</b>${extra || ''}</p>`;
  const bars = (top, la, lb, key) => { const mx = Math.max(...top.map(t => Math.abs(t.p)), 1e-12);   // key: 'Token' or 'Nr.' (logic value; shown translated)
    return `<table class="zz-terms"><thead><tr><th>${key === 'Token' ? 'Token' : tl('Nr.', '#')}</th><th>${la}</th><th>${lb}</th><th>${tl('Produkt', 'Product')}</th><th></th></tr></thead><tbody>${top.map(t => `<tr><td class="mono">${key === 'Token' ? `<button type="button" class="linkbtn mono" data-k="${t.i}">${tokHTML(I.tokens[t.i])}</button>` : fmtInt(t.i)}</td><td>${f(t.a)}</td><td>${t.b == null ? '' : f(t.b)}</td><td><b>${f(t.p)}</b></td><td class="zz-bar"><i style="width:${(Math.abs(t.p) / mx * 100).toFixed(1)}%;background:var(${t.p < 0 ? '--div-neg' : '--div-pos'})"></i></td></tr>`).join('')}</tbody></table>`; };
  const curve = c => { if (!c || c.length < 2) return ''; const W = 300, H = 90, n = c[c.length - 1][0], lx = v => 8 + (W - 16) * Math.log(v) / Math.log(n), ys = c.map(p => p[1]), lo = Math.min(0, ...ys), hi = Math.max(0, ...ys), Y = v => 8 + (H - 22) * (1 - (v - lo) / ((hi - lo) || 1));
    return `<svg class="zz-curve" viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" role="img" aria-label="${tl('Zwischensumme', 'Running sum')}"><line x1="8" x2="${W - 8}" y1="${Y(0)}" y2="${Y(0)}" stroke="var(--line-2)"/><path d="M${c.map(p => `${lx(p[0]).toFixed(1)},${Y(p[1]).toFixed(1)}`).join('L')}" fill="none" stroke="var(--c-ffn)" stroke-width="2"/><text x="8" y="${H - 2}" class="ax">${tl('1 Produkt', '1 product')}</text><text x="${W - 8}" y="${H - 2}" text-anchor="end" class="ax">${tl(`alle ${fmtInt(n)}`, `all ${fmtInt(n)}`)}</text></svg><p class="count">${tl('Zwischensumme, größte Produkte zuerst', 'Running sum, largest products first')} ${ib('runSum')}</p>`; };
  const k = x.kind;
  if (k === 'matmul') {
    const wb = `<button type="button" class="linkbtn mono" data-w="${esc(x.weight)}">${esc(x.weight)}</button> (${esc(x.wtype || '')})`, nt = fmtInt(x.n_terms), row = fmtInt(x.row);
    const q8 = x.total_q8 != null ? tl(` · wie llama.cpp gerundet (8 Bit, Blöcke zu ${x.q8_block}): <b>${f(x.total_q8)}</b>`, ` · rounded like llama.cpp (8 bit, blocks of ${x.q8_block}): <b>${f(x.total_q8)}</b>`) : '';
    return `<p class="zz-f">${tl(`<b>${at}</b> = Σ Gewicht × Eingabe über ${nt} Paare · Gewichtszeile ${row} von ${wb}`, `<b>${at}</b> = Σ weight × input over ${nt} pairs · weight row ${row} of ${wb}`)}</p><p class="zz-sum">${tl('Exakt', 'Exact')}: <b>${f(x.total)}</b>${q8} · llama.cpp: <b>${f(x.value)}</b></p>
      <p class="count">${tl(`Σ positive Produkte ${f(x.pos_sum)} · Σ negative ${f(x.neg_sum)}`, `Σ positive products ${f(x.pos_sum)} · Σ negative ${f(x.neg_sum)}`)}</p><div class="zz-two">${bars(x.top, tl('Gewicht', 'Weight'), tl('Eingabe', 'Input'), 'Nr.')}<div>${curve(x.curve)}</div></div>`;
  }
  if (k === 'score') { const a = tokQ(I.tokens[x.pos]), b = tokQ(I.tokens[x.key]), nt = fmtInt(x.n_terms);
    return `<p class="zz-f">${tl(`<b>Punktzahl</b> von Kopf ${x.head + 1}: Q von ${a} · K von ${b} = Σ über ${nt} Zahlen`, `<b>Score</b> of head ${x.head + 1}: Q of ${a} · K of ${b} = Σ over ${nt} numbers`)}</p>${check(x.total, x.value)}<p class="count">${tl(`Danach × 1/√${nt}, dann Softmax`, `Then × 1/√${nt}, then softmax`)}</p><div class="zz-two">${bars(x.top, 'Q', 'K', 'Nr.')}<div>${curve(x.curve)}</div></div>`; }
  if (k === 'softmax') { const a = tokQ(I.tokens[x.pos]), b = tokQ(I.tokens[x.key]), nt = fmtInt(x.n_terms);
    return `<p class="zz-f">${tl(`<b>Aufmerksamkeit</b> von Kopf ${x.head + 1}: ${a} schaut auf ${b} = e<sup>Punktzahl</sup> / Σ e<sup>Punktzahl</sup> über ${nt} Tokens`, `<b>Attention</b> of head ${x.head + 1}: ${a} looks at ${b} = e<sup>score</sup> / Σ e<sup>score</sup> over ${nt} tokens`)}</p>${check(x.total, x.value)}${bars(x.top.map(t => ({ i: t.i, a: t.a, b: null, p: t.p })), tl('Punktzahl × Skala', 'Score × scale'), '', 'Token')}`; }
  if (k === 'mix') return `<p class="zz-f">${tl(`<b>${at}</b> = Σ Aufmerksamkeit × V-Wert über ${fmtInt(x.n_terms)} Tokens (Kopf ${x.head + 1}, Zahl ${x.dim})`, `<b>${at}</b> = Σ attention × V value over ${fmtInt(x.n_terms)} tokens (head ${x.head + 1}, number ${x.dim})`)}</p>${check(x.total, x.value)}${bars(x.top, tl('Anteil', 'Share'), tl('V-Wert', 'V value'), 'Token')}`;
  if (k === 'binary') { const nm = (x.inputs || []).map(i => i.w ? `<button type="button" class="linkbtn mono" data-w="${esc(i.w)}">${esc(i.w.replace(/^blk\.\d+\./, ''))}</button>` : `<span class="mono">${esc(i.name)}</span>`);
    return `<p class="zz-f"><b>${at}</b> = ${nm[0] || 'a'} ${x.sym} ${nm[1] || 'b'}</p><p class="zz-big">${f(x.a)} ${x.sym} ${f(x.b)} = <b>${f(x.total)}</b></p>${check(x.total, x.value)}`; }
  if (k === 'rmsnorm') return `<p class="zz-f">${tl(`<b>${at}</b> = x ÷ √(Mittelwert von x² + ε) über ${fmtInt(x.n_terms)} Zahlen`, `<b>${at}</b> = x ÷ √(mean of x² + ε) over ${fmtInt(x.n_terms)} numbers`)} ${ib('runNorm')}</p><p class="zz-big">${f(x.x)} ÷ √(${f(x.mean_sq)} + ${sig(x.eps, 2)}) = ${f(x.x)} ÷ ${f(x.rms)} = <b>${f(x.total)}</b></p>${check(x.total, x.value)}`;
  if (k === 'glu') return `<p class="zz-f"><b>${at}</b> = ${tl('SiLU(Gate) × Hoch', 'SiLU(gate) × up')} ${ib('runGlu')}</p><p class="zz-big">SiLU(${f(x.gate)}) × ${f(x.up)} = ${f(x.act)} × ${f(x.up)} = <b>${f(x.total)}</b></p>${check(x.total, x.value)}`;
  if (k === 'unary') return `<p class="zz-f"><b>${at}</b> = ${esc(x.fn)}(${tl('Eingabe', 'input')})</p><p class="zz-big">${esc(x.fn)}(${f(x.x)}) = <b>${f(x.total)}</b></p>${check(x.total, x.value)}`;
  if (k === 'scale') return `<p class="zz-f"><b>${at}</b> = ${tl('Eingabe', 'input')} × ${f(x.s)}${x.b ? ' + ' + f(x.b) : ''}</p>${check(x.total, x.value)}`;
  if (k === 'lookup') { const wb = `<button type="button" class="linkbtn mono" data-w="${esc(x.weight)}">${esc(x.weight)}</button>`, tq = tokQ(x.token);
    return `<p class="zz-f">${tl(`<b>${at}</b> = Zahl ${fmtInt(x.j)} aus Zeile ${fmtInt(x.token)} (Token ${tq}) von ${wb}`, `<b>${at}</b> = number ${fmtInt(x.j)} from row ${fmtInt(x.token)} (token ${tq}) of ${wb}`)}</p>${check(x.total, x.value)}`; }
  if (k === 'rope') return x.rotated
    ? `<p class="zz-f">${tl(`<b>${at}</b>: Paar aus Zahl ${fmtInt(x.j % 1e9)} und Zahl ${fmtInt(x.partner)}, gedreht um einen Winkel je nach Position`, `<b>${at}</b>: pair of number ${fmtInt(x.j % 1e9)} and number ${fmtInt(x.partner)}, rotated by an angle that depends on the position`)} ${ib('runRope')}</p><p class="zz-big">${tl(`Position ${x.pos} · Winkel ${sig(x.angle, 4)} rad · vorher (${f(x.x)}, ${f(x.x_partner)}) → nachher <b>${f(x.value)}</b>`, `Position ${x.pos} · angle ${sig(x.angle, 4)} rad · before (${f(x.x)}, ${f(x.x_partner)}) → after <b>${f(x.value)}</b>`)}</p><p class="count">${tl(`Gedreht werden nur die ersten ${fmtInt(x.n_dims)} Zahlen jedes Kopfes`, `Only the first ${fmtInt(x.n_dims)} numbers of each head are rotated`)}</p>`
    : `<p class="zz-f">${tl(`<b>${at}</b>: außerhalb der gedrehten ersten Zahlen, bleibt ${f(x.x)}`, `<b>${at}</b>: outside the rotated first numbers, stays ${f(x.x)}`)} ${ib('runRope')}</p>`;
  const op = esc(opLabel(x.op) || x.op), ins = (x.inputs || []).map(i => `<span class="mono">${esc(i.w || i.name)}</span>`).join(', ') || '–';
  return `<p class="zz-f">${tl(`<b>${at}</b> = <b>${f(x.value)}</b> · Ergebnis von ${op} aus ${ins}`, `<b>${at}</b> = <b>${f(x.value)}</b> · result of ${op} from ${ins}`)}</p><p class="count">${x.note ? esc(srvMsg(x.note)) : tl('Für diesen Schritt gibt es keine Einzel-Nachrechnung.', 'No breakdown available for this step.')}</p>`;
}

// ---------- attention / linear memory of the chosen block
async function renderAttn() {
  const I = RUN.info, box = $('#runAttn'), head = $('#runAttnTitle'), L = RUN.layer, pos = RUN.pos, seq = RUN.seq;
  if (!I) return;
  if (L === 0) { head.innerHTML = tl('Aufmerksamkeit', 'Attention'); box.innerHTML = `<p class="count">${tl('Embedding: kein Blick auf andere Tokens · Schicht ab Block 1 wählen', 'Embedding: no look at other tokens · pick a layer from block 1 on')}</p>`; return; }
  const b = L - 1, kind = I.layer_kinds[b];
  if (kind === 'attn' && I.attn_nodes[String(b)] != null) {
    head.innerHTML = tl(`Block ${L}: Worauf schaut ${tokQ(I.tokens[pos])}?`, `Block ${L}: What does ${tokQ(I.tokens[pos])} look at?`) + ` ${ib('attn')}`;
    let d; try { d = await RD.vec(I.attn_nodes[String(b)], pos); } catch (e) { box.innerHTML = e instanceof ExMissing ? `<p class="count">${tl('Im Beispiel nicht gespeichert.', 'Not stored in this example.')}</p>` : tl('Fehler: ', 'Error: ') + esc(srvMsg(e.message)); return; }
    if (RUN.layer !== L || RUN.pos !== pos) return;
    const n = d.meta.shape[0], nh = d.meta.shape[1], v = d.v, hsel = RUN.head;
    const w = Array.from({ length: pos + 1 }, (_, k) => { if (hsel >= 0) return v[hsel * n + k]; let s = 0; for (let h = 0; h < nh; h++) s += v[h * n + k]; return s / nh; });
    const mx = Math.max(...w, 1e-9);
    const top = tokQ(I.tokens[w.indexOf(mx)]);
    box.innerHTML = `<div class="at-heads"><span class="count">${tl('Kopf:', 'Head:')}</span><button type="button" class="toggle" data-h="-1" aria-pressed="${hsel < 0}">${tl(`Ø alle ${nh}`, `mean of ${nh}`)}</button>${Array.from({ length: nh }, (_, h) => `<button type="button" class="toggle sm" data-h="${h}" aria-pressed="${hsel === h}">${h + 1}</button>`).join('')}</div>
      <div class="at-row">${w.map((x, k) => `<button type="button" class="at${k === pos ? ' self' : ''}" data-k="${k}" title="${pctf(x)}"><i style="height:${(x / mx * 100).toFixed(1)}%"></i><span class="mono">${tokHTML(I.tokens[k])}</span></button>`).join('')}</div>
      <p class="count">${tl(`Stärkster Blick: ${top} mit ${pctf(mx)}${hsel < 0 ? ' (Durchschnitt aller Köpfe)' : ''}`, `Most attention: ${top} with ${pctf(mx)}${hsel < 0 ? ' (mean of all heads)' : ''}`)}</p>`;
    box.onclick = e => { const hb = e.target.closest('[data-h]'); if (hb) { RUN.head = +hb.dataset.h; renderAttn(); return; } const tb = e.target.closest('.at[data-k]'); if (tb) selectPos(+tb.dataset.k); };
    return;
  }
  if (kind === 'lin') {
    head.innerHTML = tl(`Block ${L}: Speicher statt Rückblick`, `Block ${L}: Memory instead of look-back`) + ` ${ib('lin')}`;
    let data = RUN.blocks[b]; if (!data) { try { data = RUN.blocks[b] = await RD.block(b); } catch (e) { box.innerHTML = tl('Fehler: ', 'Error: ') + esc(srvMsg(e.message)); return; } }
    const nb = data.steps.find(s => s.base === 'beta_sigmoid'), ng = data.steps.find(s => s.base === 'gate');
    if (!nb || !ng) { box.innerHTML = `<p class="count">${tl('Speicher wird von Token zu Token weitergetragen.', 'Memory is carried from token to token.')}</p>`; return; }
    let B, G; try { [B, G] = await Promise.all([RD.vec(nb.i, pos), RD.vec(ng.i, pos)]); } catch (e) { box.innerHTML = e instanceof ExMissing ? `<p class="count">${tl('Im Beispiel nicht gespeichert.', 'Not stored in this example.')}</p>` : tl('Fehler: ', 'Error: ') + esc(srvMsg(e.message)); return; }
    if (RUN.layer !== L || RUN.pos !== pos) return;
    const keep = Array.from(G.v, g => Math.exp(g)), beta = Array.from(B.v);
    const strip = (vals, col, lab) => `<div class="lm"><span class="count">${lab}</span><div class="lm-bars">${vals.map((x, h) => `<i title="${tl(`Kopf ${h + 1}`, `Head ${h + 1}`)}: ${nf(x * 100, 1)} %" style="height:${(Math.max(0, Math.min(1, x)) * 100).toFixed(1)}%;background:var(${col})"></i>`).join('')}</div></div>`;
    const mb = nf(beta.reduce((a, c) => a + c, 0) / beta.length * 100, 0), mk = nf(keep.reduce((a, c) => a + c, 0) / keep.length * 100, 0), nh = fmtInt(beta.length), src = `<span class="mono">${esc(nb.name)}</span>`, srg = `<span class="mono">${esc(ng.name)}</span>`;
    box.innerHTML = strip(beta, '--c-lin', tl(`Schreibstärke β je Kopf · Ø ${mb} %`, `Write strength β per head · mean ${mb} %`)) +
      strip(keep, '--c-norm', tl(`Behalten je Kopf · Ø ${mk} %`, `Keep per head · mean ${mk} %`)) +
      `<p class="count">${tl(`${nh} Köpfe, je eigener Speicher · Werte aus ${src} und ${srg} · Behalten = e<sup>gate</sup>`, `${nh} heads, each with its own memory · values from ${src} and ${srg} · keep = e<sup>gate</sup>`)}</p>`;
    return;
  }
  head.innerHTML = `Block ${L}`; box.innerHTML = `<p class="count">${tl('Für diese Schicht gibt es keine Aufmerksamkeits-Tabelle.', 'No attention table for this layer.')}</p>`;
}

// ---------- next token at this position
function renderOut() {
  const I = RUN.info, box = $('#runOut'), pos = RUN.pos; if (!I) return;
  const top = I.top[pos].slice(0, 10), nxt = pos + 1 < I.tokens.length ? I.tokens[pos + 1] : null, mx = top[0][1] || 1;
  $('#runOutTitle').innerHTML = tl(`Nach ${tokQ(I.tokens[pos])}: das nächste Token`, `After ${tokQ(I.tokens[pos])}: the next token`) + ` ${ib('out')}`;
  const nq = nxt == null ? '' : tokQ(nxt), rk = nxt == null ? '' : fmtInt(I.rank_next[pos]);
  box.innerHTML = `<ol class="out-top">${top.map(([id, p]) => `<li class="${id === nxt ? 'hit' : ''}"><span class="mono">${tokHTML(id)}</span><i style="width:${(p / mx * 100).toFixed(1)}%"></i><b>${pctf(p)}</b></li>`).join('')}</ol>` +
    `<p class="count">${nxt == null ? (I.why === 'Ende' ? tl('Danach hat das Modell das Ende-Token gewählt.', 'Then the model chose the end token.') : tl('Hier endet der aufgezeichnete Text.', 'The recorded text ends here.')) : pos + 1 >= I.n_prompt ? tl(`Gewählt: ${nq} (Rang ${rk})`, `Chosen: ${nq} (rank ${rk})`) : tl(`Im Text folgt ${nq} · beim Modell Rang ${rk}`, `Next in the text: ${nq} · model rank ${rk}`)}</p>`;
}

// ---------- wiring
function initRunTab() {
  $('#runForm').addEventListener('submit', startRun);
  $('#runLayer').addEventListener('input', e => selectLayer(+e.target.value));
  $('#runView').onclick = e => { const b = e.target.closest('[data-v]'); if (!b) return; RUN.view = b.dataset.v; $('#runView').querySelectorAll('[data-v]').forEach(x => x.setAttribute('aria-pressed', String(x.dataset.v === RUN.view))); drawStack(); };
  $('#runSame').onchange = e => { RUN.same = e.target.checked; drawStack(); drawPlane(); };
  const cv = $('#runStackCv');
  let drag = null;
  cv.addEventListener('wheel', e => { if (!RUN.stack) return; e.preventDefault(); selectLayer(RUN.layer + (e.deltaY < 0 ? 1 : -1)); }, { passive: false });
  cv.addEventListener('pointerdown', e => { drag = { y: e.clientY, l: RUN.layer, moved: false }; cv.setPointerCapture(e.pointerId); });
  cv.addEventListener('pointermove', e => {
    if (drag && Math.abs(e.clientY - drag.y) > 4) { drag.moved = true; hideTip(); const g = RUN.geo; const step = g && !g.flat ? g.dz : (g ? g.h / g.rows : 8); selectLayer(drag.l + Math.round((drag.y - e.clientY) / Math.max(2, step))); return; }
    const h = stackHit(e); if (!h || h.d == null) return hideTip();
    const st = RUN.stack, val = sig(st.v[h.l * st.cols + h.d], 4);
    showTip(tl(`${layerName(h.l)} · Dimension ${fmtInt(h.d)}: <b>${val}</b><br><span class="m">Klick wählt die Dimension · Mausrad wechselt die Schicht</span>`, `${layerName(h.l)} · dimension ${fmtInt(h.d)}: <b>${val}</b><br><span class="m">Click selects the dimension · mouse wheel changes the layer</span>`), e);
  });
  cv.addEventListener('pointerup', e => { if (drag && !drag.moved) { const h = stackHit(e); if (h && h.d != null) { if (h.l !== RUN.layer && h.l != null) selectLayer(h.l); selectDim(h.d); } } drag = null; });
  cv.addEventListener('pointerleave', hideTip);
  const pc = $('#runPlaneCv');
  const planeIdx = e => { const st = RUN.stack, b = pc.getBoundingClientRect(), c = RUN.planeCell || 5, i = Math.floor((e.clientX - b.left) / c), j = Math.floor((e.clientY - b.top) / c), d = j * st.C + i; return i >= 0 && i < st.C && d >= 0 && d < st.cols ? d : null; };
  pc.addEventListener('mousemove', e => { if (!RUN.stack) return; const d = planeIdx(e); if (d == null) return hideTip(); showTip(`Dimension ${fmtInt(d)}: <b>${sig(RUN.stack.v[RUN.layer * RUN.stack.cols + d], 4)}</b>`, e); });
  pc.addEventListener('mouseleave', hideTip);
  pc.addEventListener('click', e => { if (!RUN.stack) return; const d = planeIdx(e); if (d != null) selectDim(d); });
  if (window.ResizeObserver) new ResizeObserver(() => { if (RUN.stack && S.tab === 'run') { drawStack(); drawPlane(); drawDim(); drawStripe(); } }).observe($('#runBody'));
}

// ---------- tables ----------
function renderTensorTable() {
  const M = S.model, tb = $('#tensorTable tbody');
  $('#nTensors').textContent = fmtInt(M.tensors.length);
  tb.innerHTML = M.tensors.map(t => `<tr data-i="${t.idx}"><td class="r">${t.fileOrder}</td><td class="mono">${esc(t.name)}</td><td><span class="sw" style="background:var(${COMPS[t.comp].v})"></span>${COMPS[t.comp].label}</td><td class="mono">[${t.dims.map(fmtInt).join(', ')}]</td><td class="mono">${esc(t.typeName)}</td><td class="r">${fmtInt(t.nElements)}</td><td class="r">${fmtBytes(t.nBytes)}</td><td class="r">${fmtInt(t.absOffset)}</td></tr>`).join('');
  M.rowEls = new Map([...tb.children].map(tr => [+tr.dataset.i, tr]));
  for (const t of M.tensors) t._q = (t.name + ' ' + t.typeName + ' ' + COMPS[t.comp].label).toLowerCase();
  sortTensors(); filterTensors();
}
function sortTensors() {
  const M = S.model, { key, dir } = S.sort, tb = $('#tensorTable tbody');
  const val = t => key === 'comp' ? COMPS[t.comp].label : t[key];
  const arr = M.tensors.slice().sort((a, b) => { const x = val(a), y = val(b); return (typeof x === 'string' ? x.localeCompare(y, 'de', { numeric: true }) : x - y) * dir; });
  const frag = document.createDocumentFragment(); for (const t of arr) frag.appendChild(M.rowEls.get(t.idx)); tb.appendChild(frag);
  document.querySelectorAll('#tensorTable th[aria-sort]').forEach(th => th.setAttribute('aria-sort', th.querySelector('button').dataset.sort === key ? (dir > 0 ? 'ascending' : 'descending') : 'none'));
}
function filterTensors() {
  const M = S.model, q = $('#tensorFilter').value.trim().toLowerCase().split(/\s+/).filter(Boolean), tf = S.tf || {};
  let n = 0, b = 0;
  for (const t of M.tensors) {
    const show = q.every(w => t._q.includes(w)) && (!tf.comp || t.comp === tf.comp) && (!tf.type || t.typeName === tf.type);
    M.rowEls.get(t.idx).hidden = !show; if (show) { n++; b += t.nBytes; }
  }
  $('#tensorCount').textContent = tl(`${fmtInt(n)} von ${fmtInt(M.tensors.length)} Tensoren · ${fmtBytes(b)}`, `${fmtInt(n)} of ${fmtInt(M.tensors.length)} tensors · ${fmtBytes(b)}`);
  renderFilterChips($('#tensorFilters'), [
    tf.comp && [tl('Komponente: ', 'Component: ') + COMPS[tf.comp].label, () => { S.tf.comp = null; filterTensors(); }],
    tf.type && [tl('Typ: ', 'Type: ') + tf.type, () => { S.tf.type = null; filterTensors(); }],
  ]);
}
function showTensorsWhere(f) { S.tf = { comp: null, type: null, ...f }; $('#tensorFilter').value = ''; setTab('tensors'); filterTensors(); }
function metaValueHTML(M, k, v) {
  if (v && typeof v === 'object' && v.items) {
    const n = arrLen(v), items = Array.from(v.items.slice(0, 12));
    const show = x => v.arrayType === 'string' ? `<span class="mono">${wsHTML(JSON.stringify(x).slice(1, -1))}</span>` : esc(typeof x === 'number' ? sig(x, 6) : String(x));
    let s = `<span class="arr-items">[${items.map(show).join(', ')}${n > 12 ? ', …' : ''}]</span>`;
    if (n > 12) { const more = Array.from(v.items.slice(0, 2000)); s += `<details><summary>${tl(`${fmtInt(n)} Einträge, erste ${fmtInt(more.length)} anzeigen`, `${fmtInt(n)} entries, show first ${fmtInt(more.length)}`)}</summary><pre>${esc(more.map(x => v.arrayType === 'string' ? JSON.stringify(x) : String(x)).join(', '))}</pre></details>`; }
    return s;
  }
  if (typeof v === 'string') {
    if (v.length > 140 || v.includes('\n')) return `<details><summary>${esc(v.slice(0, 90).replace(/\s+/g, ' '))}… <span class="count">(${tl(`${fmtInt(v.length)} Zeichen`, `${fmtInt(v.length)} characters`)})</span></summary><pre>${esc(v)}</pre></details>`;
    return `<span class="mono">${esc(v)}</span>`;
  }
  let s = typeof v === 'number' ? (Number.isInteger(v) ? fmtInt(v) : sig(v, 7)) : esc(String(v));
  if (k === 'general.file_type' && GGUF.FILE_TYPES[v]) s += ` <span class="count">= ${GGUF.FILE_TYPES[v]}</span>`;
  if (/^tokenizer\.ggml\..*_token_id$/.test(k) && M.tokens[v] != null) s += ` <span class="count">= <span class="mono">${wsHTML(disp(v))}</span></span>`;
  return s;
}
function renderMeta() {
  const M = S.model, tb = $('#metaTable tbody'), so = S.metaSort || { key: 'file', dir: 1 };
  let keys = Object.keys(M.meta);
  $('#nMeta').textContent = fmtInt(keys.length);
  const ty = k => { const v = M.meta[k]; return v && v.items ? `${v.arrayType}[${fmtInt(arrLen(v))}]` : (M.h.metaTypes[k] || typeof v); };
  if (so.key === 'key') keys.sort((a, b) => a.localeCompare(b) * so.dir);
  else if (so.key === 'type') keys.sort((a, b) => (ty(a).localeCompare(ty(b)) || a.localeCompare(b)) * so.dir);
  tb.innerHTML = keys.map(k => `<tr data-k="${esc(k.toLowerCase())}"><td class="mono">${esc(k)}</td><td class="mono" style="color:var(--muted)">${esc(ty(k))}</td><td class="meta-val">${metaValueHTML(M, k, M.meta[k])}</td></tr>`).join('');
  document.querySelectorAll('#metaHead th[aria-sort]').forEach(th => th.setAttribute('aria-sort', th.querySelector('button').dataset.sort === so.key ? (so.dir > 0 ? 'ascending' : 'descending') : 'none'));
  filterMeta();
}
function filterMeta() {
  const q = $('#metaFilter').value.trim().toLowerCase(); let n = 0;
  for (const tr of $('#metaTable tbody').children) { const show = !q || tr.dataset.k.includes(q) || tr.textContent.toLowerCase().includes(q); tr.hidden = !show; if (show) n++; }
  $('#metaCount').textContent = tl(`${fmtInt(n)} von ${fmtInt(Object.keys(S.model.meta).length)} Einträgen`, `${fmtInt(n)} of ${fmtInt(Object.keys(S.model.meta).length)} entries`);
}

// =====================================================================
// tabs, model switching, events
// =====================================================================
const TABS = ['overview', 'tokens', 'embed', 'weights', 'tensors', 'meta', 'flow', 'run', 'meaning', 'code'];
const AREA = { overview: 'reader', tokens: 'reader', embed: 'reader', weights: 'reader', tensors: 'reader', meta: 'reader', flow: 'arch', run: 'arch', meaning: 'arch', code: 'arch' };
const PANE = { meaning: 'embed' };                       // "Bedeutung" shows the embedding pane with its meaning views
const LAST = { reader: 'overview', arch: 'flow' };
const HASH = { overview: 'ueberblick', tokens: 'tokens', embed: 'embedding', flow: 'datenfluss', run: 'durchlauf', meaning: 'bedeutung', code: 'rechenweg', weights: 'gewichte', tensors: 'tensoren', meta: 'metadaten' };
function setTab(name, fromHash) {
  if (!TABS.includes(name)) name = 'overview';
  S.tab = name;
  const area = AREA[name], pane = PANE[name] || name; LAST[area] = name;
  document.querySelectorAll('.areas [data-area]').forEach(b => b.setAttribute('aria-pressed', String(b.dataset.area === area)));
  document.querySelectorAll('.tabrow [data-area]').forEach(b => { b.hidden = b.dataset.area !== area; });
  for (const n of TABS) { $('#t-' + n).setAttribute('aria-selected', String(n === name)); const pn = PANE[n] || n; $('#p-' + pn).hidden = pn !== pane; }
  if (!fromHash) { try { history.replaceState(null, '', '#' + HASH[name]); } catch {} }
  hideTip();
  if (!S.model) return;
  if (name === 'overview') { drawStrip(); renderBars(); renderLayers(); }
  if (name === 'tokens') setTokSub(S.tokSub);
  if (name === 'embed' || name === 'meaning') {
    const allowed = name === 'embed' ? ['table', 'cmp'] : ['nn', 'calc', 'map'];
    $('#p-embed').dataset.mode = name; if (!allowed.includes(S.eSub)) S.eSub = allowed[0];
    renderEmbed();
  }
  if (name === 'flow') renderFlow();
  if (name === 'run') renderRunTab();
  if (name === 'code') renderCode().catch(e => { console.error(e); $('#codeSteps').textContent = tl('Fehler: ', 'Error: ') + srvMsg(e.message); });
  if (name === 'weights') { if (!S.sel) selectTensor(defaultTensor(S.model)); else setWSub(S.wSub); }
}
function defaultTensor(M) {
  if (M.source.kind === 'demo') return M.byName.get('blk.3.attn_q.weight') || M.byName.get(Object.keys(M.source.sets).find(n => n !== 'token_embd.weight'));
  const fa = M.layers.find(L => L.kind === 'attn');
  return (fa && fa.tensors.find(t => /attn_q\.weight$/.test(t.name))) || M.tensors.find(t => /blk\.0\..*\.weight$/.test(t.name) && t.dims.length > 1) || M.tensors.slice().sort((a, b) => b.nBytes - a.nBytes)[0];
}
function setModel(M) {
  if (fullMap) { fullMap.dseq = (fullMap.dseq || 0) + 1; clearTimeout(fullMap.dt); }
  S.model = M; S.sel = null; S.sample = null; S.tokSel = null; S.wordSel = null; S.embA = null; S.embB = null; S.embFocus = null; S.embJumpTo = null; THEME = null; S.tf = { comp: null, type: null }; S.nnId = null; S.flowBlock = null; S.tourNN = null; S.tourStep = 0; S.calcDone = false; S.codeArch = null; S.codePart = 'graph'; S.codeShown = null; resetRun(); if (fullMap) fullMap.D = null;
  $('#errBox').hidden = true;
  $('#tensorNames').innerHTML = M.tensors.map(t => `<option value="${esc(t.name)}"></option>`).join('');
  $('#tensorFilter').value = ''; $('#metaFilter').value = ''; $('#vocabQ').value = ''; $('#mergeQ').value = '';
  ['#vocabSel', '#mergeSel'].forEach(s => { const e = $(s); if (e) e.remove(); });
  vocabList = null; mergeList = null; $('#vocabList').innerHTML = ''; $('#mergeList').innerHTML = ''; $('#specBody').innerHTML = '';
  if (embView) embView.d = null;
  if (wView) wView.d = null;
  renderFileBar(); renderOverview(); wExplain(); quickPicks(); renderTensorTable(); renderMeta(); renderTokensTab();
  setTab(S.tab, true);
}
function showError(e) {
  const box = $('#errBox');
  const msg = esc(srvMsg(String(e.message || e)));
  box.innerHTML = tl(`<b>Datei konnte nicht gelesen werden.</b> ${msg} · Die vorige Ansicht bleibt.`, `<b>Could not read the file.</b> ${msg} · The previous view stays.`);
  box.hidden = false;
}
async function openFile(file) {
  if (!file) return;
  $('#modePill').textContent = tl('Lese Header …', 'Reading header…');
  const t0 = performance.now();
  try {
    const h = await GGUF.parseBlob(file);
    setModel(buildModel(h, new FileSource(file), { fileName: file.name, parseMs: performance.now() - t0 }));
  } catch (e) { console.error(e); showError(e); if (S.model) renderFileBar(); }
}
async function gunzip(bytes) {
  const ds = new DecompressionStream('gzip');
  return new Response(new Blob([bytes]).stream().pipeThrough(ds)).arrayBuffer();
}
async function bootDemo() {
  try {
    const d = JSON.parse(document.getElementById('demo-data').textContent);
    const buf = await gunzip(b64(d.headerGz));
    const h = GGUF.parse(buf, d.fileSize);
    setModel(buildModel(h, new DemoSource(d), { fileName: d.fileName, demo: d }));
  } catch (e) {
    console.error(e);
    $('#fileName').textContent = tl('Keine Datei geladen', 'No file loaded');
    $('#readStat').textContent = tl('Eingebautes Beispiel lässt sich in diesem Browser nicht entpacken · eine .gguf-Datei öffnen', 'This browser cannot unpack the built-in example · open a .gguf file');
  }
}

const fi = $('#fileInput');
$('#openBtn').onclick = () => fi.click();
fi.onchange = () => { openFile(fi.files[0]); fi.value = ''; };
let dragDepth = 0;
addEventListener('dragenter', e => { if ([...(e.dataTransfer?.types || [])].includes('Files')) { dragDepth++; $('#dropzone').hidden = false; e.preventDefault(); } });
addEventListener('dragover', e => { if ([...(e.dataTransfer?.types || [])].includes('Files')) e.preventDefault(); });
addEventListener('dragleave', () => { if (--dragDepth <= 0) { dragDepth = 0; $('#dropzone').hidden = true; } });
addEventListener('drop', e => { e.preventDefault(); dragDepth = 0; $('#dropzone').hidden = true; const f = e.dataTransfer?.files?.[0]; if (f) openFile(f); });

const strip = $('#strip');
strip.onmousemove = e => {
  const M = S.model; if (!M) return;
  const b = strip.getBoundingClientRect(), x = e.clientX - b.left, off = x / b.width * M.h.fileSize;
  S.stripHover = x; drawStrip();
  const hb = fmtBytes(M.h.headerEnd, 2), nv = fmtInt(M.vocabSize), nt = fmtInt(M.h.nTensors);
  if (off < M.h.dataStart || x <= 3) return showTip(tl(`<b>Header</b> · ${hb}<br>Bauplan, Wörterbuch (${nv} Tokens) und das Verzeichnis aller ${nt} Tensoren<br><span class="m">Nur dieser Teil wird beim Öffnen gelesen</span>`, `<b>Header</b> · ${hb}<br>Blueprint, vocabulary (${nv} tokens) and the directory of all ${nt} tensors<br><span class="m">Only this part is read when opening</span>`), e);
  showTip(tensorTip(tensorAt(off)) + `<br><span class="m">${tl('Klick öffnet die Tabelle', 'Click opens the table')}</span>`, e);
};
strip.onmouseleave = () => { S.stripHover = null; if (S.model) drawStrip(); hideTip(); };
strip.onclick = e => { const M = S.model, b = strip.getBoundingClientRect(), off = (e.clientX - b.left) / b.width * M.h.fileSize; if (off >= M.h.dataStart) openTensor(tensorAt(off)); };

$('#lmComp').onclick = () => { S.layerMode = 'comp'; $('#lmComp').setAttribute('aria-pressed', 'true'); $('#lmBits').setAttribute('aria-pressed', 'false'); renderLayers(); };
$('#lmBits').onclick = () => { S.layerMode = 'bits'; $('#lmBits').setAttribute('aria-pressed', 'true'); $('#lmComp').setAttribute('aria-pressed', 'false'); renderLayers(); };
document.querySelectorAll('[data-sub]').forEach(b => { b.onclick = () => setTokSub(b.dataset.sub); });
document.querySelectorAll('[data-wsub]').forEach(b => { b.onclick = () => setWSub(b.dataset.wsub); });
let tokTimer = 0;
$('#tokText').addEventListener('input', e => { S.text = e.target.value; clearTimeout(tokTimer); tokTimer = setTimeout(() => { S.tokSel = null; S.wordSel = null; runTokenize(); }, 120); });
$('#tokFlow').onclick = e => { const b = e.target.closest('.tk'); if (!b) return; S.tokSel = +b.dataset.i; markTokSel(); renderTokDetail(); const p = S.tokRes.pieces[S.tokSel]; if (p.word != null) { S.wordSel = p.word; renderWords(); } };
$('#tokFlow').onmousemove = e => { const b = e.target.closest('.tk'); if (!b) return hideTip(); const p = S.tokRes.pieces[+b.dataset.i], raw = S.model.tokens[p.id];
  showTip(tl(`Token-ID <b>${fmtInt(p.id)}</b><br>Rohform: <code>${esc(raw)}</code>${p.fallback ? '<br><span class="m">Byte-Fallback</span>' : ''}<br><span class="m">Klick zeigt die Zeile in der Embedding-Tabelle</span>`, `Token ID <b>${fmtInt(p.id)}</b><br>Raw form: <code>${esc(raw)}</code>${p.fallback ? '<br><span class="m">Byte fallback</span>' : ''}<br><span class="m">Click shows its row in the embedding table</span>`), e); };
$('#tokFlow').onmouseleave = hideTip;
$('#wordPick').onclick = e => { const b = e.target.closest('[data-w]'); if (b) { S.wordSel = +b.dataset.w; renderWords(); } };
$('#vocabQ').oninput = () => { clearTimeout(vocabTimer); vocabTimer = setTimeout(() => S.model._vocabReady && filterVocab(), 150); };
$('#vocabCats').onclick = e => { const b = e.target.closest('[data-cat]'); if (!b || !S.model._vocabReady) return; setVocabFilter({ cat: b.dataset.cat === '' ? null : +b.dataset.cat }); };
let mergeTimer = 0;
$('#mergeQ').oninput = () => { clearTimeout(mergeTimer); mergeTimer = setTimeout(() => S.model._mergesReady && filterMerges(), 150); };
$('#embModeText').onclick = () => { S.embMode = 'text'; renderEmbed(); };
$('#embModeAll').onclick = () => { S.embMode = 'all'; if (S.embA != null) S.embJumpTo = S.embA; renderEmbed(); };
$('#embJump').addEventListener('keydown', e => {
  if (e.key !== 'Enter') return;
  const q = e.target.value.trim(); if (!q) return;
  const id = /^#?\d+$/.test(q) ? +q.replace('#', '') : lookupWord(q);
  if (id >= 0 && id < S.model.vocabSize) { e.target.setCustomValidity(''); jumpToRow(id); }
  else { e.target.setCustomValidity(tl('Nicht im Wörterbuch', 'Not in the vocabulary')); e.target.reportValidity(); }
});
$('#embJump').addEventListener('input', e => e.target.setCustomValidity(''));
$('#tokPresets').onclick = e => { const b = e.target.closest('[data-a]'); if (!b) return; S.embA = +b.dataset.a; S.embB = +b.dataset.b; if (embView) { embView.selRow = S.embA; embView.draw(); } drawCompare(); };
$('#rowsHead').onclick = () => { S.rowMode = 'head'; $('#rowsHead').setAttribute('aria-pressed', 'true'); $('#rowsSpread').setAttribute('aria-pressed', 'false'); loadSample(); };
$('#rowsSpread').onclick = () => { S.rowMode = 'spread'; $('#rowsSpread').setAttribute('aria-pressed', 'true'); $('#rowsHead').setAttribute('aria-pressed', 'false'); loadSample(); };
$('#histLin').onclick = () => { S.histLog = false; $('#histLin').setAttribute('aria-pressed', 'true'); $('#histLog').setAttribute('aria-pressed', 'false'); drawDist(); };
$('#histLog').onclick = () => { S.histLog = true; $('#histLog').setAttribute('aria-pressed', 'true'); $('#histLin').setAttribute('aria-pressed', 'false'); drawDist(); };
$('#blkPrev').onclick = () => { S.blk--; drawStore(); };
$('#blkNext').onclick = () => { S.blk++; drawStore(); };
$('#tensorPick').addEventListener('change', e => { const t = S.model.byName.get(e.target.value.trim()); if (t) openTensor(t); });
$('#quickPicks').onclick = e => { const b = e.target.closest('button[data-name]'); if (b) openTensor(S.model.byName.get(b.dataset.name)); };
$('#lupeMsg').onclick = e => { const b = e.target.closest('[data-pick]'); if (b) openTensor(S.model.byName.get(b.dataset.pick)); };
$('#tensorTable tbody').onclick = e => { const tr = e.target.closest('tr'); if (tr) openTensor(S.model.tensors[+tr.dataset.i]); };
$('#tensorTable thead').onclick = e => { const b = e.target.closest('button[data-sort]'); if (!b) return; const k = b.dataset.sort; S.sort = { key: k, dir: S.sort.key === k ? -S.sort.dir : (k === 'nBytes' || k === 'nElements' ? -1 : 1) }; sortTensors(); };
$('#tensorFilter').oninput = filterTensors;
$('#metaFilter').oninput = filterMeta;
TABS.forEach(n => {
  const b = $('#t-' + n); b.onclick = () => setTab(n);
  b.onkeydown = e => { if (e.key === 'ArrowRight' || e.key === 'ArrowLeft') { const vis = TABS.filter(t => AREA[t] === AREA[n]), i = vis.indexOf(n), j = (i + (e.key === 'ArrowRight' ? 1 : vis.length - 1)) % vis.length; setTab(vis[j]); $('#t-' + vis[j]).focus(); } };
});
document.querySelectorAll('.areas [data-area]').forEach(b => { b.onclick = () => setTab(LAST[b.dataset.area]); });
$('#tourTab').onclick = () => openTour();
const fromHash = () => { const h = location.hash.slice(1); return Object.keys(HASH).find(k => HASH[k] === h) || 'overview'; };
addEventListener('hashchange', () => setTab(fromHash(), true));
setTab(fromHash(), true);

let rz = 0;
const redraw = () => {
  if (!S.model) return; THEME = null;
  if (S.tab === 'overview') { drawStrip(); renderBars(); renderLayers(); }
  if (S.tab === 'tokens' && S.tokSub === 'split') renderTokDetail();
  if (S.tab === 'tokens' && S.tokSub === 'vocab' && vocabList) vocabList.draw(true);
  if (S.tab === 'embed' || S.tab === 'meaning') { embView && embView.layout(); drawCompare(); }
  if (S.tab === 'weights') { wView && wView.layout(); renderWSub(); }
};
addEventListener('resize', () => { clearTimeout(rz); rz = setTimeout(redraw, 150); });
matchMedia('(prefers-color-scheme: dark)').addEventListener('change', redraw);
new MutationObserver(redraw).observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });
if (document.fonts && document.fonts.ready) document.fonts.ready.then(() => { embView && embView.draw(); wView && wView.draw(); });

// new controls
$('#tourBtn').onclick = openTour;
$('#tourClose').onclick = closeTour;
$('#tourPrev').onclick = () => tourGo(S.tourStep - 1);
$('#tourNext').onclick = () => { if (S.tourStep === TOUR.length - 1) closeTour(); else tourGo(S.tourStep + 1); };
$('#tourDots').onclick = e => { const b = e.target.closest('[data-step]'); if (b) tourGo(+b.dataset.step); };
addEventListener('keydown', e => {
  if ($('#tour').hidden) return;
  if (e.key === 'Escape') { closeTour(); return; }
  if (/^(INPUT|TEXTAREA|SELECT)$/.test(e.target.tagName)) return;
  if (e.key === 'ArrowRight' || e.key === 'PageDown' || e.key === ' ') { e.preventDefault(); if (S.tourStep < TOUR.length - 1) tourGo(S.tourStep + 1); }
  else if (e.key === 'ArrowLeft' || e.key === 'PageUp') { e.preventDefault(); tourGo(S.tourStep - 1); }
});
document.querySelectorAll('[data-esub]').forEach(b => { b.onclick = () => setESub(b.dataset.esub); });
const nnHosts = () => ({ list: $('#nnList'), map: $('#nnMap'), status: $('#nnStatus'), prog: $('#nnProg'), input: $('#nnQ') });
$('#nnGo').onclick = () => {
  const inp = $('#nnQ'), q = inp.value.trim(); if (!q || !S.model) return;
  const id = /^#?\d+$/.test(q) ? +q.replace('#', '') : lookupWord(q);
  if (id >= 0 && id < S.model.vocabSize) { inp.setCustomValidity(''); nnSearch(id, nnHosts()); }
  else { inp.setCustomValidity(tl('Nicht im Wörterbuch', 'Not in the vocabulary')); inp.reportValidity(); }
};
$('#nnQ').addEventListener('keydown', e => { if (e.key === 'Enter') $('#nnGo').click(); });
$('#nnQ').addEventListener('input', e => e.target.setCustomValidity(''));
$('#nnExamples').onclick = e => { const b = e.target.closest('[data-id]'); if (b) nnSearch(+b.dataset.id, nnHosts()); };
$('#nnList').onclick = e => { const b = e.target.closest('[data-id]'); if (b) nnSearch(+b.dataset.id, nnHosts()); };
$('#nnMap').onclick = e => { const g = e.target.closest('[data-id]'); if (g) nnSearch(+g.dataset.id, nnHosts()); };
$('#vocabWS').onclick = () => { if (S.model && S.model._vocabReady) setVocabFilter({ ws: !S.vf.ws }); };
$('#specHead').onclick = e => { const b = e.target.closest('[data-sort]'); if (!b) return; const k = b.dataset.sort; S.spSort = { key: k, dir: S.spSort.key === k ? -S.spSort.dir : 1 }; renderSpecial(); };
$('#metaHead').onclick = e => { const b = e.target.closest('[data-sort]'); if (!b) return; const k = b.dataset.sort; S.metaSort = { key: k, dir: S.metaSort.key === k && k !== 'file' ? -S.metaSort.dir : 1 }; renderMeta(); };
$('#flowChain').onclick = e => {
  const bb = e.target.closest('.bstrip [data-b]'); if (bb) { S.flowBlock = +bb.dataset.b; renderFlowBlock(); return; }
  const st = e.target.closest('[data-go]'); if (!st) return;
  const M = S.model, g = st.dataset.go;
  if (g === 'tokens') setTab('tokens');
  else if (g === 'embed') { S.eSub = 'table'; setTab('embed'); }
  else if (g === 'onorm') openTensor(M.byName.get('output_norm.weight'));
  else if (g === 'head') openTensor(M.byName.get('output.weight') || M.E);
};

$('#calcGo').onclick = () => calcRun();
['#calcA', '#calcB', '#calcC'].forEach(sel => $(sel).addEventListener('keydown', e => { if (e.key === 'Enter') calcRun(); }));
$('#calcHide').onchange = () => calcRun();
$('#calcEx').onclick = e => { const b = e.target.closest('[data-p]'); if (!b) return; const p = CALC_PRESETS[+b.dataset.p]; $('#calcA').value = p[0]; $('#calcB').value = p[1]; $('#calcC').value = p[2]; calcRun(); };
$('#calcList').onclick = e => { const b = e.target.closest('[data-id]'); if (b) openNN(+b.dataset.id); };

// ⓘ texts for part A (tokens tab: BPE result line)
Object.assign(INFO, {
  bpeResult: ['Der Tokenizer hört auf, sobald keine Regel mehr auf ein Nachbarpaar passt; was übrig bleibt, sind die Tokens. <b>Grau</b> sind einzelne Bytes: Ein Umlaut wie „ü“ besteht aus zwei Bytes, die erst zusammen ein Zeichen ergeben.',
    'The tokenizer stops as soon as no rule fits any neighboring pair; what is left are the tokens. <b>Gray</b> pieces are single bytes: a letter like “ü” is made of two bytes that only form a character together.'],
});
// ---------- ⓘ texts of part B (vocabulary, special tokens, embedding table, weights): [de, en]
Object.assign(INFO, {
  vocabWordStart: ['Viele Tokens beginnen mit einem Leerzeichen: Es gehört zum Token und markiert den Anfang eines Wortes.',
    'Many tokens start with a space: it is part of the token and marks the start of a word.'],
  specCounts: ['<b>Reserviert:</b> freie Plätze im Wörterbuch für spätere Erweiterungen. <b>Byte-Tokens:</b> Rückfallebene für Zeichen, die kein eigenes Token haben.',
    '<b>Reserved:</b> free slots in the vocabulary for later extensions. <b>Byte tokens:</b> a fallback for characters that have no token of their own.'],
  embCosine: ['Misst, wie ähnlich die Richtungen zweier Zeilen sind: 1 heißt gleiche Richtung, 0 unabhängig, −1 entgegengesetzt. Verwandte Wörter liegen messbar näher beieinander als unverwandte, obwohl die Eingabe-Embeddings nur der Startpunkt sind.',
    'Measures how similar the directions of two rows are: 1 means same direction, 0 unrelated, −1 opposite. Related words are measurably closer than unrelated ones, even though the input embeddings are only the starting point.'],
  weightHist: ['Fast alle Gewichte liegen nahe 0, nur wenige sind groß. Die logarithmische Achse macht die seltenen großen Werte sichtbar.',
    'Almost all weights are close to 0; only a few are large. The logarithmic axis makes the rare large values visible.'],
  weightNorm: ['Norm-Gewichte skalieren jede Dimension; Ausreißer verstärken einzelne Kanäle.',
    'Norm weights scale each dimension; outliers amplify individual channels.'],
  weightProfile: ['Für jede Spalte (Eingangs-Dimension) der mittlere Betrag |w| über die Stichprobenzeilen. Spitzen zeigen Dimensionen, auf die diese Tabelle besonders stark reagiert.',
    'For each column (input dimension), the mean magnitude |w| over the sampled rows. Peaks show dimensions this table reacts to especially strongly.'],
  weightRowLen: ['Länge einer Zeile = Wurzel aus der Summe ihrer quadrierten Zahlen. Lange Zeilen erzeugen große Ausgaben.',
    'Length of a row = square root of the sum of its squared numbers. Long rows produce large outputs.'],
  weightBlock: ['Statt jede Zahl mit 16 Bit zu speichern, teilt sich ein Block gemeinsame Skalen und speichert pro Zahl nur einen kurzen Code. Bytes je Block × 8 ÷ Zahlen je Block = Bit pro Zahl.',
    'Instead of storing every number with 16 bits, a block shares common scales and stores only a short code per number. Bytes per block × 8 ÷ numbers per block = bits per number.'],
});
// ⓘ texts for part D (tour, word arithmetic, quick meaning map)
Object.assign(INFO, {
  mapQuick: () => {   // ⓘ after "Schnellkarte" in #mapStatus (mapInfo)
    const M = S.model, D = M && M._map, n = fmtInt(D ? D.ids.length : 2600), V = fmtInt(M ? M.vocabSize : 0), web = !(D && D.demo);
    return tl(`Die Schnellkarte zeigt ${n} häufige Wörter${web ? ' und wird im Browser berechnet' : ''}. Die komplette Karte aller ${V} Tokens gibt es mit dem Lupe-Server.`,
      `The quick map shows ${n} common words${web ? ' and is computed in your browser' : ''}. For the complete map of all ${V} tokens, start the Lupe server.`);
  },
});
// ⓘ texts for part E (full map)
Object.assign(INFO, {
  fmNeighbors: ['Die Nachbarn sind exakt berechnet: Kosinus-Ähnlichkeit über alle Dimensionen. Die Linien zeigen, wo sie auf der flachen Karte liegen. Liegt ein Nachbar weit weg, zeigt das, was beim Plattdrücken verloren geht.',
    'The neighbors are computed exactly: cosine similarity over all dimensions. The lines show where they sit on the flat map. A neighbor that lands far away shows what gets lost when the space is flattened.'],
});
// ---------- part G: ⓘ texts for the forward pass (Durchlauf)
Object.assign(INFO, {
  runServer: ['Der <b>Lupe-Server</b> (<code>start-lupe-server.cmd</code>) lässt das Modell mit llama.cpp auf deinem Rechner rechnen, auf der Grafikkarte oder einfach auf der CPU. Läuft er, öffnest du dein Modell über die Server-Liste; dann rechnet llama.cpp mit genau dieser Datei.',
    'The <b>Lupe server</b> (<code>start-lupe-server.cmd</code>) runs the model with llama.cpp on your own computer, on the GPU or simply on the CPU. Once it runs, open your model from the server list; llama.cpp then computes with exactly that file.'],
  runSum: ['Die Kurve zählt die Produkte zusammen, das größte zuerst (waagerecht logarithmisch: links 1 Produkt, rechts alle). Wenige Produkte bestimmen meist schon fast das Ergebnis.',
    'The curve adds up the products, largest first (log scale across: 1 product on the left, all on the right). A few products usually get close to the result already.'],
  runNorm: ['<b>Normieren</b> bringt alle Zahlen des Tokens auf eine vergleichbare Größe. Danach wird mit dem Norm-Gewicht aus der Datei malgenommen.',
    '<b>Normalizing</b> brings all numbers of the token to a comparable size. After that they are multiplied by the norm weight from the file.'],
  runGlu: ['Das <b>Gate</b> entscheidet, wie viel von „Hoch“ durchkommt: Ist das Gate stark negativ, ist SiLU(Gate) fast 0, und fast nichts kommt durch.',
    'The <b>gate</b> decides how much of “up” gets through: if the gate is strongly negative, SiLU(gate) is almost 0 and almost nothing gets through.'],
  runRope: ['Die Zahlen werden paarweise um einen Winkel gedreht, der von der Position im Text abhängt. So kann die Attention erkennen, wie weit zwei Tokens auseinanderliegen. Gedreht werden nur die ersten Zahlen jedes Kopfes.',
    'The numbers are rotated in pairs by an angle that depends on the position in the text. This lets attention tell how far apart two tokens are. Only the first numbers of each head are rotated.'],
});
// points moved out of the visible forward-pass UI, appended to part F's existing texts (whatever form they have: [de, en], function or string)
{
  const txt = v => typeof v === 'function' ? v() : Array.isArray(v) ? tl(v[0], v[1]) : (v || '');
  const add = (k, more) => { const base = INFO[k]; INFO[k] = () => txt(base) + more(); };
  add('toks', () => RUN.ex ? tl(' <b>Grün unterstrichen:</b> Für diese Tokens ist im Beispiel der Weg durch die Schichten gespeichert.', ' <b>Green underline:</b> for these tokens the example stores the path through the layers.') : '');
  add('out', () => tl(' Bei deiner Eingabe steht das nächste Token schon fest; das Modell sagt es trotzdem an jeder Stelle voraus.', ' In your input the next token is already given; the model still predicts it at every position.'));
}
// ⓘ texts for the static parts of the page (src.html)
Object.assign(INFO, {
  bpe: ['Der Tokenizer startet mit einzelnen Zeichen und klebt dann Schritt für Schritt Paare zusammen, immer nach einer festen Regelliste aus dem Training. Je kleiner die Regelnummer, desto häufiger war das Paar und desto früher wird es geklebt.',
    'The tokenizer starts with single characters and glues pairs together step by step, always following a fixed list of rules from training. The lower the rule number, the more frequent the pair was, and the earlier it gets glued.'],
  tokFlow: ['Jedes farbige Kästchen ist ein Token, darunter seine Nummer im Wörterbuch (Token-ID). Klick auf ein Token zeigt seine Zeile in der Embedding-Tabelle.',
    'Each colored box is one token, with its number in the vocabulary (token ID) below. Click a token to see its row in the embedding table.'],
  merges: ['Diese Liste <b>ist</b> der gelernte Tokenizer: Regel 0 wurde im Training zuerst gefunden (das häufigste Paar), danach die nächsthäufigen. Jede Regel klebt zwei Stücke zu einem neuen Token zusammen. Ein Klick auf einen Balken filtert die Liste.',
    'This list <b>is</b> the learned tokenizer: rule 0 was found first in training (the most frequent pair), then the next most frequent ones. Each rule glues two pieces into a new token. Click a bar to filter the list.'],
  special: ['Spezial-Tokens sind keine Wörter, sondern <b>Steuerzeichen</b>: Sie markieren, wo eine Chat-Nachricht beginnt, wann das Modell nachdenkt oder wo ein Bild eingefügt wird. Im Chat stehen sie unsichtbar zwischen den Nachrichten.',
    'Special tokens are not words but <b>control signals</b>: they mark where a chat message starts, when the model is thinking, or where an image goes. In a chat they sit invisibly between the messages.'],
  meaning: ['Jedes Token ist eine Zeile der Embedding-Tabelle. Ähnliche Wörter haben ähnliche Zeilen; diese Nähe hat das Modell im Training selbst gelernt. Hier kannst du Nachbarn suchen, mit Zeilen rechnen und alle Tokens als Landkarte sehen.',
    'Every token is one row of the embedding table. Similar words have similar rows; the model learned this closeness on its own in training. Here you can find neighbors, do arithmetic with rows and see all tokens as a map.'],
  nnCap: ['Die Lupe vergleicht die Zeile eines Wortes mit <b>allen Zeilen</b> der Embedding-Tabelle (Kosinus-Ähnlichkeit) und zeigt die nächsten Nachbarn. Diese Nähe hat das Modell im Training selbst gelernt; niemand hat ihm ein Wörterbuch der Bedeutungen gegeben. Ein Klick auf einen Nachbarn sucht dort weiter.',
    'The Lupe compares a word’s row with <b>every row</b> of the embedding table (cosine similarity) and shows the nearest neighbors. The model learned this closeness on its own in training; nobody gave it a dictionary of meanings. Click a neighbor to continue from there.'],
  nnMap: ['Die Nachbarn auf zwei Achsen verkleinert (Hauptrichtungen). Nah auf der Karte heißt ähnlich, aber nur ungefähr: Die echte Ähnlichkeit steckt in allen Dimensionen.',
    'The neighbors squeezed onto two axes (principal directions). Close on the map means similar, but only roughly: the real similarity lives in all dimensions.'],
  calcCap: ['Jede Zeile ist eine <b>Richtung</b> in einem Raum mit allen Dimensionen des Modells. A − B + C ergibt eine neue Richtung, die zu keinem Token gehört; die Lupe sucht die Zeilen, die ihr am nächsten liegen. Die Eingabewörter und ihre Schreibvarianten werden ausgeblendet, wie in der Forschung üblich. „König − Mann + Frau = Königin“ klappt in großen Modellen nur teilweise: Die Embeddings sind der Startpunkt, die Bedeutung entsteht erst in den Schichten.',
    'Every row is a <b>direction</b> in a space with all of the model’s dimensions. A − B + C gives a new direction that belongs to no token; the Lupe looks for the rows closest to it. The input words and their spelling variants are hidden, as is usual in research. “King − man + woman = queen” only partly works in large models: the embeddings are the starting point, and meaning builds up in the layers.'],
  mapCap: ['Jeder Punkt ist ein Token. Die Ähnlichkeit wird aus allen Dimensionen berechnet; die Zeichnung ist eine Projektion und kann Abstände verzerren. Ein Klick zeigt die echten Nachbarn. Mit dem Lupe-Server: jede Dimension als Farbe, zwei oder drei Dimensionen als Achsen. Mausrad zoomt, Ziehen verschiebt.',
    'Each dot is a token. Similarity is computed from all dimensions; the drawing is a projection and can distort distances. Click a dot to see its real neighbors. With the Lupe server: any dimension as color, two or three dimensions as axes. Scroll to zoom, drag to pan.'],
  cmp: ['In der Tabelle eine Zeile anklicken wählt sie als A, die vorherige Auswahl wird zu B.',
    'Click a row in the table to make it A; the previous choice becomes B.'],
  orderProof: ['Die Datei liefert nur die Tabellen mit ihren Namen und den Namen der Architektur. In welcher Reihenfolge gerechnet wird, steht im Programm (llama.cpp, auch in LM Studio). Links die Tabellen dieses Blocks so, wie sie in der Datei liegen, rechts so, wie das Programm sie benutzt.',
    'The file only provides the tables with their names, plus the name of the architecture. The order of the computation lives in the program (llama.cpp, also inside LM Studio). Left: this block’s tables as they lie in the file. Right: as the program uses them.'],
  tensors: ['Das Inhaltsverzeichnis der Datei: jede Zahlentabelle mit Name, Form, Speicherformat und Position. Ein Klick auf eine Zeile öffnet sie unter „Gewichte“.',
    'The file’s table of contents: every table of numbers with name, shape, storage format and position. Click a row to open it under “Weights”.'],
  meta: ['Alle Einträge aus dem Header: der Bauplan des Modells (Schichten, Breite, Kontextlänge), die Tokenizer-Daten und die Chat-Vorlage.',
    'Every entry from the header: the model’s blueprint (layers, width, context length), the tokenizer data and the chat template.'],
  footer: ['Alles läuft lokal im Browser, die Datei wird nicht hochgeladen. Gelesen werden nur der Header (Wörterbuch, Bauplan, Inhaltsverzeichnis) und die Zeilen, die du dir ansiehst. Die Seite lädt nichts aus dem Internet, auch die Schriften sind eingebaut.<br><br>Tokenizer: Nachbau des BPE-Tokenizers aus llama.cpp, für Qwen3.8 und Gemma 4 in 88 Testfällen identisch mit llama.cpp b11388. Gewichte: Dequantisierung für F32, F16, BF16, Q4_0 bis Q8_0, Q2_K bis Q6_K, IQ4_NL, IQ4_XS, TQ2_0, MXFP4 und NVFP4, bitgenau geprüft gegen gguf-py 0.19.',
    'Everything runs locally in your browser; the file is never uploaded. Only the header (vocabulary, blueprint, table of contents) and the rows you look at are read. The page loads nothing from the internet; even the fonts are built in.<br><br>Tokenizer: a re-implementation of the llama.cpp BPE tokenizer, identical to llama.cpp b11388 in 88 test cases for Qwen3.8 and Gemma 4. Weights: dequantization for F32, F16, BF16, Q4_0 to Q8_0, Q2_K to Q6_K, IQ4_NL, IQ4_XS, TQ2_0, MXFP4 and NVFP4, checked bit-exact against gguf-py 0.19.'],
});
// ---------- texts from the Lupe server (German) -> English for srvMsg() in p0: first match wins, so specific before general
const SRV_EN = [
  // jobs: waiting, status words (lupe_server.py; job states are compared in logic, not shown)
  [/^wartet auf freie Rechenzeit$/, 'Waiting for free compute time'],
  [/^wartet$/, 'waiting'],
  [/^läuft$/, 'running'],
  [/^fertig$/, 'done'],
  [/^fehler$/, 'error'],
  [/^bereit$/, 'ready'],
  [/^abgebrochen$/, 'canceled'],
  // embedding table and word map (lupe_server.py)
  [/^Verzeichnis der Datei lesen$/, 'Reading the tensor directory'],
  [/^Tabelle entpacken$/, 'Unpacking the table'],
  [/^Diese Datei hat keine token_embd-Tabelle\.$/, 'This file has no token_embd table.'],
  [/^alle Paare vergleichen \((\d+) %, noch ca\. (\d+) s\)$/, 'Comparing all pairs ($1 %, about $2 s left)'],
  [/^Karte auslegen \((\d+)\/(\d+)\)$/, 'Laying out the map ($1/$2)'],
  [/^Kosinus der rohen Zeilen, exakt über alle Dimensionen$/, 'Cosine of the raw rows, exact over all dimensions'],
  // HTTP errors (lupe_server.py)
  [/^unbekannter Job$/, 'unknown job'],
  [/^unbekannt$/, 'unknown'],
  [/^Nur \.gguf-Dateien aus den freigegebenen Modell-Ordnern\.$/, 'Only .gguf files from the allowed model folders.'],
  [/^Tabelle noch nicht geladen \(\/api\/load\)\.$/, 'Table not loaded yet (/api/load).'],
  [/^Karte nicht gefunden$/, 'Map not found'],
  [/^Bitte eine Frage oder einen Text eingeben\.$/, 'Please enter a question or some text.'],
  // llama.cpp: download, worker process, loading (lupe_run.py, lupe_worker.py, lupe_infer.py)
  [/^unbekannte Variante$/, 'unknown variant'],
  [/^(\S+\.(?:zip|tar\.gz)) entpacken$/, 'Unpacking $1'],
  [/^Der llama\.cpp-Prozess ist nicht gestartet \(siehe Server-Fenster\)\.$/, 'The llama.cpp process did not start (see the server window).'],
  [/^Der llama\.cpp-Prozess ist beendet worden \(Code ([^)]*)\)\. Häufigste Ursache: zu wenig Grafikspeicher\. Beim nächsten Versuch wird er neu gestartet; sonst „Rechnen auf: CPU“ wählen\.$/, 'The llama.cpp process stopped (code $1). Most common cause: not enough GPU memory. It restarts on the next try; otherwise choose CPU under “llama.cpp runs on”.'],
  [/^llama\.cpp ließ sich nicht laden: ggml_tensor-Layout passt nicht zu dieser llama\.cpp-Version$/, 'Could not load llama.cpp: the ggml_tensor layout does not match this llama.cpp version'],
  [/^llama\.cpp ließ sich nicht laden: Es ist schon eine andere llama\.cpp-Version geladen \((.*)\)\. Server neu starten, um zu wechseln\.$/s, 'Could not load llama.cpp: a different llama.cpp version is already loaded ($1). Restart the server to switch.'],
  [/^llama\.cpp ließ sich nicht laden: (.*)$/s, 'Could not load llama.cpp: $1'],
  [/^ggml_tensor-Layout passt nicht zu dieser llama\.cpp-Version$/, 'The ggml_tensor layout does not match this llama.cpp version'],
  [/^Es ist schon eine andere llama\.cpp-Version geladen \((.*)\)\. Server neu starten, um zu wechseln\.$/s, 'A different llama.cpp version is already loaded ($1). Restart the server to switch.'],
  [/^Kein llama\.cpp gefunden\. Siehe README: einmal die passende Version herunterladen\.$/, 'No llama.cpp found. See the README: download the matching version once.'],
  [/^llama\.cpp konnte das Modell nicht laden: (.*)$/s, 'llama.cpp could not load the model: $1'],
  [/^llama\.cpp konnte keinen Rechen-Kontext anlegen: (.*)$/s, 'llama.cpp could not create a compute context: $1'],
  // forward pass: run, record, layer preview (lupe_worker.py, lupe_infer.py, lupe_run.py)
  [/^Modell in llama\.cpp laden \(([\d.]+) GB\)$/, 'Loading the model into llama.cpp ($1 GB)'],
  [/^Die Eingabe ist (\d+) Tokens lang\. Zusammen mit der Antwort sind höchstens (\d+) erlaubt\.$/, 'The input is $1 tokens long. Input and answer together can be at most $2 tokens.'],
  [/^Antwort erzeugen$/, 'Generating the answer'],
  [/^Antwort erzeugen: (\d+) Tokens$/, 'Generating the answer: $1 tokens'],
  [/^Durchlauf aufzeichnen$/, 'Recording the forward pass'],
  [/^Ausgabe-Kopf für die Schicht-Vorschau entpacken \(nur beim ersten Mal\)$/, 'Unpacking the output head for the layer preview (first time only)'],
  [/^Ausgabe-Kopf entpacken$/, 'Unpacking the output head'],
  [/^llama_decode meldet Fehler (-?\d+): (.*)$/s, 'llama_decode reports error $1: $2'],
  [/^Aufzeichnung fehlgeschlagen: (.*)$/s, 'Recording failed: $1'],
  [/^'?Dieser Durchlauf ist nicht mehr im Speicher\. Bitte neu rechnen\.'?$/, 'This forward pass is no longer in memory. Please run it again.'],
  [/^Länge$/, 'length'],
  [/^Ende$/, 'end'],
  [/^roh$/, 'raw'],
  // one number recomputed (lupe_worker.py: note of /api/run/explain)
  [/^Nachrechnen nicht möglich: Diese Zwischenstufe lässt sich nicht auslesen\.$/, 'Cannot recompute: this intermediate result cannot be read out.'],
  [/^Nachrechnen nicht möglich: llama_decode meldet Fehler (-?\d+): (.*)$/s, 'Cannot recompute: llama_decode reports error $1: $2'],
  [/^Nachrechnen nicht möglich: Aufzeichnung fehlgeschlagen: (.*)$/s, 'Cannot recompute: recording failed: $1'],
  [/^Nachrechnen nicht möglich: (.*)$/s, 'Cannot recompute: $1'],
  [/^Diese Zwischenstufe lässt sich nicht auslesen\.$/, 'This intermediate result cannot be read out.'],
  // saving an example (lupe_export.py)
  [/^Weg durch die Schichten: Token (\d+) von (\d+)$/, 'Path through the layers: token $1 of $2'],
  [/^Rechenschritte der Blöcke$/, 'Steps of all blocks'],
  [/^Aufmerksamkeit und Speicher aller Tokens$/, 'Attention and memory of all tokens'],
  [/^Schritte im Detail: (\d+) von (\d+)$/, 'Steps in detail: $1 of $2'],
];
// errors from the GGUF parser (gguf-core.js, German) -> English, same mechanism as the server texts
SRV_EN.push(
  [/^Keine GGUF-Datei \(Magic "GGUF" fehlt am Dateianfang\)\.$/, 'Not a GGUF file (the magic "GGUF" is missing at the start).'],
  [/^Nicht unterstützte GGUF-Version (\d+)$/, 'Unsupported GGUF version $1'],
  [/^Unbekannter Metadaten-Typ (\d+) bei Byte (\d+)$/, 'Unknown metadata type $1 at byte $2'],
  [/^Unbekannter Array-Typ (\d+)$/, 'Unknown array type $1'],
  [/^Unbekannter Metadaten-Typ (\d+) für Schlüssel "(.*)"$/, 'Unknown metadata type $1 for key "$2"'],
);
// ---------- language switch: static texts carry data-en, everything else is rendered again
const DE_ORIG = new WeakMap(), I18N_ATTRS = ['placeholder', 'aria-label', 'title'];
function i18nStatic() {
  document.querySelectorAll('[data-en],[data-en-placeholder],[data-en-aria-label],[data-en-title]').forEach(el => {
    let o = DE_ORIG.get(el);
    if (!o) { o = { html: el.innerHTML }; for (const a of I18N_ATTRS) o[a] = el.getAttribute(a); DE_ORIG.set(el, o); }
    if (el.hasAttribute('data-en')) el.innerHTML = isEN() ? el.getAttribute('data-en') : o.html;
    for (const a of I18N_ATTRS) if (el.hasAttribute('data-en-' + a)) el.setAttribute(a, isEN() ? el.getAttribute('data-en-' + a) : o[a]);
  });
  document.documentElement.lang = LANG;
  document.querySelectorAll('.lang [data-lang]').forEach(b => b.setAttribute('aria-pressed', String(b.dataset.lang === LANG)));
}
function setLang(l) {
  if (l === LANG || (l !== 'de' && l !== 'en')) return;
  const old = LANG; LANG = l;
  try { localStorage.setItem('lupe-lang', l); } catch {}
  hideTip(); $('#infoPop').hidden = true;
  i18nStatic();
  if (S.text === DEFAULT_TEXTS[old]) S.text = DEFAULT_TEXTS[l];
  if (!S.model) return;
  relangAll();
}
// everything that setModel() renders once, then the open tab (setTab renders it), the tour and the server bar
function relangAll() {
  renderFileBar(); renderOverview(); wExplain(); quickPicks(); renderTensorTable(); renderMeta(); renderTokensTab();
  if (embView) embView.relang();
  if (wView) wView.relang();
  setTab(S.tab, true);
  if (!$('#tour').hidden) renderTour();
  renderServerBar();
}
document.querySelectorAll('.lang [data-lang]').forEach(b => { b.onclick = () => setLang(b.dataset.lang); });
i18nStatic();
initRunTab();
bootDemo().then(detectServer);
})();
