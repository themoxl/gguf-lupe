// i18n check: English page (browser language en-US) on every tab and sub-view, list German words still visible;
// then switch DE <-> EN on several tabs (with a recorded example loaded) and look for errors.   node e2e14.cjs [--shots]
const { chromium } = require('playwright-core');
const path = require('path'), fs = require('fs');
const exe = process.env.LOCALAPPDATA + '/ms-playwright/chromium-1243/chrome-win64/chrome.exe';
const FILE = 'file:///' + path.resolve(require('fs').existsSync('gguf-lupe.html') ? 'gguf-lupe.html' : '../gguf-lupe.html').split(path.sep).join('/');
const SHOTS = process.argv.includes('--shots'), OUT = 'shots15'; if (SHOTS) fs.mkdirSync(OUT, { recursive: true });
const sleep = ms => new Promise(r => setTimeout(r, ms));
const GER = /^(der|die|das|den|dem|des|und|oder|nicht|ist|sind|ein|eine|einen|einer|für|auf|aus|nach|zeigt|zeile|zeilen|spalte|spalten|schicht|schichten|wörter|datei|tabelle|klick|alle|noch|wird|werden|kein|keine|fehler|beispiel|beispiele|rechnen|dimensionen|gewicht|gewichte|vorhersage|wie|hier|dieser|diese|dieses|von|bis|zum|zur|im|am|als|auch|nur|sich|wenn|dann|weil|beim|vom|über|unter|jede|jeder|jedes|seine|ihre|deine|dein|du|dir|ganze|nächste|nächstes|vorherige|größte|kleinste|länge|wert|werte|zahl|zahlen|gelesen|geladen|öffnen|laden|lade|speichern|gespeichert|aufgezeichnet|berechnet|vergleichen|verglichen|suchen|ähnliche|bedeutung|landkarte|rundgang|durchlauf|datenfluss|rechenweg|überblick|metadaten|tensoren|wörterbuch|schritt|schritte|kopf|köpfe|aufmerksamkeit|ausgabe|eingabe|antwort|frage|modell|modells|bauplan|reihenfolge|programm)$/i;
const SKIP = '#tip,#infoPop,canvas,svg text.tok,.tok-flow,.ids,.nn-list,.nnmap,#tokText,.rt-list,.vl,#metaTable tbody,.ll-code,.steps,.wordpick,.eq,#calcList,.tok-detail,.quick,#specBody,.lm,.at-row,.calc-row,#runToks .rt,#tourStage .tk,.tour-tokens,#fileName,.mv-sc,input,textarea,select option';
(async () => {
  const browser = await chromium.launch({ executablePath: exe });
  const ctx = await browser.newContext({ viewport: { width: 1400, height: 1000 }, locale: 'en-US' });
  const page = await ctx.newPage();
  const errs = []; page.on('pageerror', e => errs.push('pageerror: ' + e.message + ' ' + (e.stack || '').split('\n')[1])); page.on('console', m => { if (m.type() === 'error' && !/ERR_CONNECTION_REFUSED|ERR_FILE_NOT_FOUND/.test(m.text())) errs.push('console: ' + m.text()); });
  await page.goto(FILE);
  await page.waitForFunction(() => window.getComputedStyle && document.querySelector('#facts') && document.querySelector('#facts').children.length > 0, null, { timeout: 60000 });
  console.log('lang:', await page.evaluate(() => document.documentElement.lang), '| pill:', await page.$eval('#modePill', e => e.textContent));
  const german = async (label) => {
    const r = await page.evaluate(([GERsrc, SKIP]) => {
      const GER = new RegExp(GERsrc, 'i'), hits = new Map(); let words = 0;
      const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
      for (let n; (n = walker.nextNode());) {
        const el = n.parentElement; if (!el || !el.offsetParent && getComputedStyle(el).position !== 'fixed') continue;
        if (el.closest(SKIP)) continue;
        const cs = getComputedStyle(el); if (cs.visibility === 'hidden') continue;
        for (const w of n.textContent.split(/[^A-Za-zÄÖÜäöüß]+/)) {
          if (w.length < 2) continue; words++;
          if (GER.test(w) || /[äöüÄÖÜß]/.test(w)) { const k = (el.id ? '#' + el.id : el.tagName.toLowerCase() + (typeof el.className === 'string' && el.className ? '.' + el.className.split(' ')[0] : '')); const s = hits.get(k) || new Set(); s.add(w); hits.set(k, s); }
        }
      }
      for (const a of document.querySelectorAll('[aria-label],[placeholder],[title]')) {
        if (!a.offsetParent || a.closest(SKIP.replace(',input,textarea,select option', ''))) continue;
        for (const at of ['aria-label', 'placeholder', 'title']) { const v = a.getAttribute(at); if (!v) continue; const bad = v.split(/[^A-Za-zÄÖÜäöüß]+/).filter(w => w.length > 1 && (GER.test(w) || /[äöüÄÖÜß]/.test(w))); if (bad.length) { const k = '@' + at + ' ' + (a.id ? '#' + a.id : a.tagName.toLowerCase()); hits.set(k, new Set(bad)); } }
      }
      return { words, hits: [...hits].map(([k, s]) => [k, [...s].slice(0, 8).join(' ')]) };
    }, [GER.source, SKIP]);
    console.log(`\n== ${label}: ${r.words} visible words` + (r.hits.length ? `, German suspects:` : ', no German'));
    for (const [k, w] of r.hits) console.log(`   ${k.padEnd(28)} ${w}`);
  };
  const go = async (area, tab, wait = 900) => { await page.click(`.areas [data-area="${area}"]`); await sleep(150); await page.click('#t-' + tab); await sleep(wait); };
  const shot = async n => { if (SHOTS) await page.screenshot({ path: `${OUT}/${n}.png` }); };
  for (const [area, tab, subs] of [['reader', 'overview'], ['reader', 'tokens', ['split', 'vocab', 'merges', 'special']], ['reader', 'embed', ['table', 'cmp']], ['reader', 'weights', ['table', 'dist', 'prof', 'store']], ['reader', 'tensors'], ['reader', 'meta'], ['arch', 'flow'], ['arch', 'run'], ['arch', 'meaning', ['nn', 'calc', 'map']], ['arch', 'code']]) {
    await go(area, tab, tab === 'run' ? 3000 : 1000);
    if (!subs) { await german(tab); await shot('en-' + tab); continue; }
    for (const s of subs) {
      const sel = tab === 'tokens' ? `[data-sub="${s}"]` : tab === 'weights' ? `[data-wsub="${s}"]` : `[data-esub="${s}"]`;
      await page.click(sel); await sleep(s === 'map' ? 2500 : s === 'nn' || s === 'calc' ? 1500 : 800);
      await german(`${tab}/${s}`); await shot(`en-${tab}-${s}`);
    }
  }
  // every info button on the overview, tokens, run tab: popover not empty, English
  for (const [area, tab] of [['reader', 'overview'], ['reader', 'tokens'], ['reader', 'tensors'], ['arch', 'flow'], ['arch', 'run'], ['arch', 'code']]) {
    await go(area, tab, tab === 'run' ? 1500 : 600);
    const n = await page.$$eval(`#p-${tab} .ib`, b => b.filter(x => x.offsetParent).length);
    const res = [];
    for (let i = 0; i < n; i++) {
      await page.evaluate(([t, i]) => [...document.querySelectorAll(`#p-${t} .ib`)].filter(x => x.offsetParent)[i].click(), [tab, i]); await sleep(120);
      const t = await page.$eval('#infoPop', e => e.hidden ? '' : e.textContent.trim());
      res.push(t ? (/[äöüß]|\b(der|die|das|und|ist)\b/.test(t) ? 'DE?' : 'ok') + ':' + t.slice(0, 40) : 'EMPTY');
      await page.keyboard.press('Escape');
    }
    console.log(`ⓘ ${tab}: ${res.join(' | ')}`);
  }
  // tour, first three steps
  await page.click('#tourBtn'); await sleep(700);
  for (let i = 0; i < 3; i++) { await german('tour step ' + (i + 1)); await page.click('#tourNext'); await sleep(500); }
  await page.click('#tourClose'); await sleep(300); console.log('tour closed');
  // switching back and forth on several tabs
  for (const [area, tab] of [['arch', 'run'], ['reader', 'overview'], ['reader', 'tokens'], ['arch', 'meaning'], ['reader', 'weights'], ['arch', 'flow'], ['arch', 'code']]) {
    await go(area, tab, tab === 'run' ? 1500 : 700);
    await page.click('.lang [data-lang="de"]'); await sleep(700);
    const de = await page.evaluate(() => document.querySelector('main').innerText.slice(0, 60).replace(/\s+/g, ' '));
    await shot('de-' + tab);
    await page.click('.lang [data-lang="en"]'); await sleep(700);
    const en = await page.evaluate(() => document.querySelector('main').innerText.slice(0, 60).replace(/\s+/g, ' '));
    console.log(`switch on ${tab}: DE «${de}» → EN «${en}»`);
  }
  console.log('stored lang:', await page.evaluate(() => { try { return localStorage.getItem('lupe-lang'); } catch { return '?'; } }));
  console.log('\nerrors:', errs.length ? '\n  ' + errs.join('\n  ') : 'none');
  await browser.close();
})().catch(e => { console.error('FAIL', e); process.exit(1); });
