/* Word map: exact cosine kNN (full dimensions) -> UMAP-style layout -> k-means regions.
   Pure JS, shared by the page (main thread + Web Workers) and build-demo (Node). */
const MAPCORE = (() => {
  function rng(seed) { let a = seed >>> 0; return () => { a = (a + 0x6D2B79F5) >>> 0; let t = a; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }
  const tick = () => new Promise(r => setTimeout(r, 0));

  /** unit-normalise, subtract the common mean direction, normalise again -> one Float32Array n*d */
  function prep(vecs) {
    const n = vecs.length, d = vecs[0].length, U = new Float32Array(n * d), mean = new Float64Array(d);
    for (let r = 0; r < n; r++) { const v = vecs[r]; let s = 0; for (let i = 0; i < d; i++) s += v[i] * v[i]; s = Math.sqrt(s) || 1; for (let i = 0; i < d; i++) { const x = v[i] / s; U[r * d + i] = x; mean[i] += x / n; } }
    for (let r = 0; r < n; r++) { let s = 0; for (let i = 0; i < d; i++) { const x = U[r * d + i] - mean[i]; U[r * d + i] = x; s += x * x; } s = Math.sqrt(s) || 1; for (let i = 0; i < d; i++) U[r * d + i] /= s; }
    return { U, n, d };
  }
  /** int8 copy (for shipping to workers) */
  function toInt8(U, n, d) { const Q = new Int8Array(n * d); for (let r = 0; r < n; r++) { let mx = 0; for (let i = 0; i < d; i++) mx = Math.max(mx, Math.abs(U[r * d + i])); const s = 127 / (mx || 1); for (let i = 0; i < d; i++) Q[r * d + i] = Math.round(U[r * d + i] * s); } return Q; }
  /** exact kNN for rows [r0, r1) against all rows; X is Float32Array or Int8Array (cosine via norms) */
  function knnRows(X, n, d, K, r0, r1, onRow) {
    const norms = new Float64Array(n);
    for (let j = 0; j < n; j++) { let s = 0; for (let i = 0; i < d; i++) { const x = X[j * d + i]; s += x * x; } norms[j] = Math.sqrt(s) || 1; }
    const nb = new Int32Array((r1 - r0) * K), sim = new Float32Array((r1 - r0) * K);
    for (let r = r0; r < r1; r++) {
      const best = []; let worst = -Infinity; const a = r * d;
      for (let j = 0; j < n; j++) {
        if (j === r) continue;
        const b = j * d; let s = 0; for (let i = 0; i < d; i++) s += X[a + i] * X[b + i];
        s /= norms[r] * norms[j];
        if (best.length < K) { best.push([s, j]); if (best.length === K) { best.sort((x, y) => y[0] - x[0]); worst = best[K - 1][0]; } }
        else if (s > worst) { let p = K - 1; while (p > 0 && best[p - 1][0] < s) { best[p] = best[p - 1]; p--; } best[p] = [s, j]; worst = best[K - 1][0]; }
      }
      if (best.length < K) best.sort((x, y) => y[0] - x[0]);
      for (let k = 0; k < K; k++) { nb[(r - r0) * K + k] = best[k] ? best[k][1] : r; sim[(r - r0) * K + k] = best[k] ? best[k][0] : 0; }
      onRow && onRow(r);
    }
    return { nb, sim };
  }
  /** first two principal components (power iteration) as start positions */
  function pca2(U, n, d) {
    const R = rng(5), Y = new Float32Array(n * 2), comps = [];
    for (let c = 0; c < 2; c++) {
      let v = Float64Array.from({ length: d }, () => R() - 0.5);
      for (let it = 0; it < 12; it++) {
        const t = new Float64Array(n); for (let r = 0; r < n; r++) { let s = 0; for (let i = 0; i < d; i++) s += U[r * d + i] * v[i]; t[r] = s; }
        const nv = new Float64Array(d); for (let r = 0; r < n; r++) { const tr = t[r]; for (let i = 0; i < d; i++) nv[i] += U[r * d + i] * tr; }
        for (const u of comps) { let dot = 0; for (let i = 0; i < d; i++) dot += nv[i] * u[i]; for (let i = 0; i < d; i++) nv[i] -= dot * u[i]; }
        let nr = 0; for (let i = 0; i < d; i++) nr += nv[i] * nv[i]; nr = Math.sqrt(nr) || 1; for (let i = 0; i < d; i++) nv[i] /= nr; v = nv;
      }
      comps.push(v);
      let mx = 0; for (let r = 0; r < n; r++) { let s = 0; for (let i = 0; i < d; i++) s += U[r * d + i] * v[i]; Y[r * 2 + c] = s; mx = Math.max(mx, Math.abs(s)); }
      for (let r = 0; r < n; r++) Y[r * 2 + c] = Y[r * 2 + c] / (mx || 1) * 10;
    }
    return Y;
  }
  /** fuzzy kNN graph + UMAP-style SGD (a, b for min_dist ≈ 0.1) */
  async function layout(nb, sim, n, K, Y, opts = {}, onP) {
    const R = rng(11), a = 1.577, b = 0.895, epochs = opts.epochs || 400, neg = 5;
    const w = new Float32Array(n * K);
    for (let i = 0; i < n; i++) {
      const dist = k => 1 - sim[i * K + k], rho = dist(0), target = Math.log2(K); let lo = 0, hi = Infinity, sig = 1;
      for (let it = 0; it < 64; it++) { let s = 0; for (let k = 0; k < K; k++) s += Math.exp(-Math.max(0, dist(k) - rho) / sig); if (Math.abs(s - target) < 1e-5) break; if (s > target) { hi = sig; sig = (lo + hi) / 2; } else { lo = sig; sig = hi === Infinity ? sig * 2 : (lo + hi) / 2; } }
      for (let k = 0; k < K; k++) w[i * K + k] = Math.exp(-Math.max(0, dist(k) - rho) / sig);
    }
    const edge = new Map();
    for (let i = 0; i < n; i++) for (let k = 0; k < K; k++) { const j = nb[i * K + k]; if (j === i) continue; const key = i < j ? i * n + j : j * n + i, v = w[i * K + k], o = edge.get(key); edge.set(key, o === undefined ? v : o + v - o * v); }
    const E = [...edge.entries()], ei = new Int32Array(E.length), ej = new Int32Array(E.length), ew = new Float32Array(E.length);
    E.forEach(([key, v], t) => { ei[t] = Math.floor(key / n); ej[t] = key % n; ew[t] = v; });
    let wmax = 0; for (const x of ew) wmax = Math.max(wmax, x);
    const every = Float64Array.from(ew, x => wmax / x), next = Float64Array.from(every);
    const clip = x => x > 4 ? 4 : x < -4 ? -4 : x;
    for (let ep = 0; ep < epochs; ep++) {
      const alpha = 1 - ep / epochs;
      for (let t = 0; t < E.length; t++) {
        if (next[t] > ep + 1) continue;
        next[t] += every[t];
        const i = ei[t], j = ej[t];
        let dx = Y[i * 2] - Y[j * 2], dy = Y[i * 2 + 1] - Y[j * 2 + 1], d2 = dx * dx + dy * dy;
        if (d2 > 0) { const g = (-2 * a * b * Math.pow(d2, b - 1)) / (1 + a * Math.pow(d2, b)); const gx = clip(g * dx) * alpha, gy = clip(g * dy) * alpha; Y[i * 2] += gx; Y[i * 2 + 1] += gy; Y[j * 2] -= gx; Y[j * 2 + 1] -= gy; }
        for (let q = 0; q < neg; q++) {
          const k = Math.floor(R() * n); if (k === i) continue;
          dx = Y[i * 2] - Y[k * 2]; dy = Y[i * 2 + 1] - Y[k * 2 + 1]; d2 = dx * dx + dy * dy;
          const g = (2 * b) / ((0.001 + d2) * (1 + a * Math.pow(d2, b)));
          Y[i * 2] += clip(g * dx) * alpha; Y[i * 2 + 1] += clip(g * dy) * alpha;
        }
      }
      if (ep % 25 === 0) { onP && onP(ep / epochs); await tick(); }
    }
    return Y;
  }
  /** k-means on the 2D layout -> region label per word */
  function regions(Y, n, k) {
    const R = rng(3), C = [];
    for (let c = 0; c < k; c++) { const i = Math.floor(R() * n); C.push([Y[i * 2], Y[i * 2 + 1]]); }
    const lab = new Uint8Array(n);
    for (let it = 0; it < 40; it++) {
      for (let i = 0; i < n; i++) { let bi = 0, bd = Infinity; for (let c = 0; c < k; c++) { const dx = Y[i * 2] - C[c][0], dy = Y[i * 2 + 1] - C[c][1], dd = dx * dx + dy * dy; if (dd < bd) { bd = dd; bi = c; } } lab[i] = bi; }
      const S = C.map(() => [0, 0, 0]); for (let i = 0; i < n; i++) { const s = S[lab[i]]; s[0] += Y[i * 2]; s[1] += Y[i * 2 + 1]; s[2]++; }
      S.forEach((s, c) => { if (s[2]) C[c] = [s[0] / s[2], s[1] / s[2]]; });
    }
    return lab;
  }
  // words for the map: curated German/English everyday words first, then the most frequent whole words (low IDs)
  const CUR_DE = 'Mensch Leben Welt Familie Freund Liebe Herz Kopf Hand Auge Wasser Feuer Erde Luft Sonne Mond Stern Himmel Meer Berg Fluss Wald Garten Haus Wohnung Zimmer Küche Tisch Stuhl Bett Tür Fenster Auto Fahrrad Zug Bus Flugzeug Schiff Straße Weg Brücke Stadt Dorf Land Staat Regierung Partei Wahl Gesetz Recht Gericht Polizei Krieg Frieden Armee Geld Bank Kredit Zinsen Steuer Bilanz Prüfung Prüfer Bericht Gewinn Verlust Umsatz Kosten Preis Markt Handel Firma Unternehmen Konzern Aktie Börse Kunde Vertrag Rechnung Zahlung Schule Universität Lehrer Schüler Student Arzt Krankenhaus Krankheit Medizin Essen Brot Fleisch Käse Milch Wein Bier Kaffee Apfel Hund Katze Pferd Kuh Schwein Vogel Fisch Baum Blume Musik Lied Film Buch Zeitung Bild Farbe rot blau grün gelb schwarz weiß groß klein schnell langsam gut schlecht schön alt neu jung eins zwei drei vier fünf sechs sieben acht neun zehn hundert tausend Montag Dienstag Mittwoch Donnerstag Freitag Samstag Sonntag Januar Februar März April Juni Juli August September Oktober November Dezember Frühling Sommer Herbst Winter Morgen Abend Nacht Woche Monat Jahr Zeit Deutschland Frankreich Italien Spanien Österreich Schweiz England Russland China Japan Amerika Europa Berlin München Hamburg Wien Zürich Paris London Rom Madrid Mann Frau Kind Junge Mädchen Vater Mutter Bruder Schwester Sohn Tochter Onkel Tante König Kaiser Prinz Gott Kirche Computer Daten Software Internet Telefon';
  const CUR_EN = 'king queen prince princess man woman boy girl father mother brother sister son daughter uncle aunt family friend doctor lawyer teacher student nurse engineer police army war peace money bank tax price market business company profit loss audit report law court government president minister election country city village house home school university hospital church car train plane ship road bridge river mountain ocean forest tree flower dog cat horse cow bird fish water fire earth sun moon star sky food bread meat cheese milk wine beer coffee tea apple red blue green yellow black white big small fast slow good bad happy sad one two three four five six seven eight nine ten hundred thousand Monday Tuesday Wednesday Thursday Friday Saturday Sunday January February March April June July August September October November December spring summer autumn winter morning evening night Germany France Italy Spain Austria Switzerland England Russia China Japan America Europe Tokyo computer data software internet phone music song movie book newspaper picture color walk walking swim swimming run running';
  /** opts: { n, size, text(id) -> display text or null, isNormal(id), lookup(word) -> id or -1 } -> { ids, curated:Set } */
  function select(o) {
    const seen = new Set(), ids = [], curated = new Set();
    const add = (id, t) => { const k = t.trim().toLowerCase(); if (seen.has(k)) return false; seen.add(k); ids.push(id); return true; };
    for (const w of (CUR_DE + ' ' + CUR_EN).split(' ')) { const id = o.lookup(w); if (id >= 0 && o.isNormal(id)) { const t = o.text(id); if (t && t.trim().toLowerCase() === w.toLowerCase() && add(id, t)) curated.add(id); } }
    for (let i = 0; i < o.size && ids.length < o.n; i++) { if (!o.isNormal(i)) continue; const t = o.text(i); if (t && /^ \p{L}{4,}$/u.test(t)) add(i, t); }
    return { ids, curated };
  }
  return { rng, prep, toInt8, knnRows, pca2, layout, regions, select };
})();
if (typeof module !== 'undefined') module.exports = MAPCORE;
