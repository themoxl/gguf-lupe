const { chromium } = require('playwright-core'); const path = require('path');
const exe = process.env.LOCALAPPDATA + '/ms-playwright/chromium-1243/chrome-win64/chrome.exe';
const FILE = 'file:///' + path.resolve(require('fs').existsSync('gguf-lupe.html') ? 'gguf-lupe.html' : '../gguf-lupe.html').split(path.sep).join('/');
const D = require('os').homedir().replace(/\\/g, '/') + '/.lmstudio/models/lmstudio-community/';
const files = [D + 'Qwen3.8-27B-GGUF/mmproj-Qwen3.8-27B-BF16.gguf', D + 'gemma-4-12B-it-QAT-GGUF/mmproj-gemma-4-12B-it-QAT-BF16.gguf', D + 'gemma-4-12B-it-QAT-GGUF/gemma-4-12B-it-QAT-Q4_0.gguf'];
const sleep = ms => new Promise(r => setTimeout(r, ms));
(async () => {
  const browser = await chromium.launch({ executablePath: exe });
  const page = await (await browser.newContext({ viewport: { width: 1300, height: 900 }, locale: 'de-DE' })).newPage();
  const errs = []; page.on('pageerror', e => errs.push('pageerror: ' + e.message)); page.on('console', m => { if (m.type() === 'error' && !/ERR_CONNECTION_REFUSED/.test(m.text())) errs.push(m.text()); });
  const txt = s => page.$eval(s, e => e.textContent.trim().replace(/\s+/g, ' ').slice(0, 220)).catch(() => '(fehlt)');
  await page.goto(FILE); await page.waitForFunction(() => document.querySelector('#modePill').textContent.includes('Beispiel'), null, { timeout: 60000 });
  for (const f of files) {
    const t0 = Date.now(); await page.setInputFiles('#fileInput', f);
    await page.waitForFunction(n => document.querySelector('#fileName').textContent.includes(n), path.basename(f), { timeout: 60000 }); await sleep(600);
    console.log('\n' + path.basename(f), Date.now() - t0, 'ms');
    for (const [tab, sel] of [['overview', '#ovExplain'], ['tokens', '#p-tokens'], ['embed', '#p-embed'], ['weights', '#lupeInfo'], ['tensors', '#p-tensors'], ['meta', '#p-meta'], ['flow', '#flowExplain'], ['run', '#runNote'], ['code', '#codeExplain']]) {
      if (tab === 'flow') await page.click('.areas [data-area="arch"]');
      if (tab === 'overview') await page.click('.areas [data-area="reader"]');
      await page.click('#t-' + tab); await sleep(700);
      console.log(`  ${tab.padEnd(8)} ${(await txt(sel)).slice(0, 150)}`);
    }
  }
  console.log('\nerrors:', errs.length ? errs.join(' | ') : 'none');
  await browser.close();
})().catch(e => { console.error('FAIL', e); process.exit(1); });
