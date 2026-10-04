// two areas + recorded examples, as a plain file (no server) and through the server
const { chromium } = require('playwright-core');
const path = require('path'), fs = require('fs');
const exe = process.env.LOCALAPPDATA + '/ms-playwright/chromium-1243/chrome-win64/chrome.exe';
const FILE = 'file:///' + path.resolve(require('fs').existsSync('gguf-lupe.html') ? 'gguf-lupe.html' : '../gguf-lupe.html').split(path.sep).join('/');
const SRV = process.env.LUPE || 'http://127.0.0.1:8766/';
const OUT = 'shots16'; fs.mkdirSync(OUT, { recursive: true });
const sleep = ms => new Promise(r => setTimeout(r, ms));
(async () => {
  const browser = await chromium.launch({ executablePath: exe });
  const ctx = await browser.newContext({ viewport: { width: 1400, height: 980 }, locale: 'de-DE' });
  const page = await ctx.newPage();
  const errs = []; page.on('pageerror', e => errs.push('pageerror: ' + e.message + ' ' + (e.stack || '').split('\n')[1])); page.on('console', m => { if (m.type() === 'error' && !/ERR_CONNECTION_REFUSED/.test(m.text())) errs.push('console: ' + m.text()); });
  const txt = s => page.$eval(s, e => e.textContent.trim().replace(/\s+/g, ' ').slice(0, 400)).catch(() => '(fehlt)');
  const visibleTabs = () => page.$$eval('.tabrow [role=tab]:not([hidden]), .tabrow .tab-tour:not([hidden])', b => b.map(x => x.textContent.trim().replace(/\s+/g, ' ')).join(' | '));
  // ---------------- as a file
  await page.goto(FILE);
  await page.waitForFunction(() => document.querySelector('#modePill').textContent.includes('Beispiel'), null, { timeout: 60000 });
  console.log('FILE reader tabs:', await visibleTabs());
  await page.click('.areas [data-area="arch"]'); await sleep(400);
  console.log('FILE arch tabs:', await visibleTabs(), '| active', await page.$eval('[role=tab][aria-selected=true]', e => e.id), '| hash', await page.evaluate(() => location.hash));
  await page.screenshot({ path: `${OUT}/01-arch-flow.png` });
  await page.click('#t-meaning'); await sleep(800);
  console.log('Bedeutung subtabs visible:', await page.$$eval('#p-embed [data-esub]', b => b.filter(x => getComputedStyle(x).display !== 'none').map(x => x.textContent).join(' | ')), '| active sub', await page.$eval('#p-embed [data-esub][aria-pressed=true]', e => e.textContent));
  await page.click('.areas [data-area="reader"]'); await sleep(300);
  console.log('back to reader active:', await page.$eval('[role=tab][aria-selected=true]', e => e.id));
  await page.click('#t-embed'); await sleep(600);
  console.log('Embedding subtabs visible:', await page.$$eval('#p-embed [data-esub]', b => b.filter(x => getComputedStyle(x).display !== 'none').map(x => x.textContent).join(' | ')));
  await page.click('.areas [data-area="arch"]'); await page.click('#t-run');
  let t0 = Date.now();
  await page.waitForSelector('#runBody:not([hidden]) #runToks .rt', { timeout: 60000 });
  await sleep(1200);
  console.log('FILE example auto-loaded in', Date.now() - t0, 'ms |', await txt('#runStatus'));
  console.log('  examples:', await txt('#runEx'));
  console.log('  note:', await txt('#runNote'));
  console.log('  tokens:', (await txt('#runToks')).slice(0, 200));
  console.log('  lens:', (await txt('#runLens')).slice(0, 120));
  console.log('  block:', await txt('#runBlockTitle'), '| steps', await page.$$eval('#runFlow .rf', b => b.length));
  await page.screenshot({ path: `${OUT}/02-file-example.png` });
  // a block with stored numbers: last block (64), a step + zoom
  await page.locator('#runFlow .rf', { hasText: 'attn_output-63' }).click();
  await page.waitForSelector('#runDetail .rd-zoom .zz-f', { timeout: 30000 }); await sleep(400);
  console.log('  step 64 attn_output:', (await txt('#runDetail .rd-zoom')).slice(0, 220));
  // click somewhere on the stripe -> snaps to a stored explanation
  const sb = await page.locator('#runDetail .rd-stripe').boundingBox();
  await page.mouse.click(sb.x + sb.width * 0.3, sb.y + sb.height / 2); await sleep(700);
  console.log('  stripe click:', (await txt('#runDetail .rd-zoom')).slice(0, 200));
  // a block without stored numbers
  await page.locator('#runLayer').fill('10'); await page.dispatchEvent('#runLayer', 'input'); await sleep(800);
  await page.locator('#runFlow .rf').first().click(); await sleep(600);
  console.log('  block 10 step:', (await txt('#runDetail')).slice(0, 200));
  await page.locator('#runDetail [data-xb]').first().click(); await sleep(1500);
  console.log('  jump ->', await txt('#runBlockTitle'), '| detail:', (await txt('#runDetail .rd-head')).slice(0, 100));
  // a token without stored layer path
  const noStack = await page.$$eval('#runToks .rt[data-k]:not(.has)', b => b.map(x => x.dataset.k));
  await page.click(`#runToks .rt[data-k="${noStack[3]}"]`); await sleep(800);
  console.log('  token without stack:', (await txt('#runMiss')).slice(0, 160), '| grid hidden', await page.$eval('.run-grid', e => e.hidden), '| attn', (await txt('#runAttnTitle')).slice(0, 60));
  await page.click('#runMiss [data-k]'); await sleep(900);
  console.log('  back to stored token: grid hidden', await page.$eval('.run-grid', e => e.hidden));
  // the Gemma example: other vocabulary
  t0 = Date.now(); await page.click('#runEx [data-ex*="sky"]');
  await page.waitForFunction(() => /gemma/i.test(document.querySelector('#runStatus').textContent), null, { timeout: 60000 }); await sleep(1500);
  console.log('\nGEMMA example', Date.now() - t0, 'ms |', await txt('#runStatus'));
  console.log('  tokens:', (await txt('#runToks')).slice(0, 260));
  console.log('  lens:', (await txt('#runLens')).slice(0, 160));
  console.log('  out:', (await txt('#runOut')).slice(0, 160));
  console.log('  block:', await txt('#runBlockTitle'), '| steps', await page.$$eval('#runFlow .rf', b => b.map(x => x.querySelector('small').textContent).slice(0, 14).join(' ')));
  console.log('  attn:', (await txt('#runAttn')).slice(0, 200));
  await page.screenshot({ path: `${OUT}/03-gemma-example.png` });
  await page.locator('#runFlow').scrollIntoViewIfNeeded(); await page.screenshot({ path: `${OUT}/04-gemma-lower.png` });
  console.log('FILE errors:', errs.length ? errs.join(' | ') : 'none');
  // ---------------- through the server
  errs.length = 0;
  await page.goto(SRV);
  for (let i = 0; i < 60; i++) { if (await page.$('#srvBar:not([hidden]) #srvModel option')) break; await sleep(500); }
  const idx = await page.$$eval('#srvModel option', o => o.find(x => x.textContent.includes('Qwen3.8')).value);
  await page.selectOption('#srvModel', idx); await page.click('#srvOpen');
  await page.waitForFunction(() => document.querySelector('#modePill').textContent === 'Über Lupe-Server', null, { timeout: 60000 });
  await page.click('.areas [data-area="arch"]'); await page.click('#t-run');
  await page.waitForSelector('#runForm:not([hidden])', { timeout: 60000 }); await sleep(800);
  console.log('\nSRV form + examples:', (await txt('#runEx')).slice(0, 160), '| body hidden', await page.$eval('#runBody', e => e.hidden));
  await page.fill('#runQ', 'Wie viele Beine hat eine Spinne?'); t0 = Date.now(); await page.click('#runGo');
  await page.waitForSelector('#runSave', { timeout: 300000 }); await sleep(800);
  console.log('SRV live run', Date.now() - t0, 'ms |', (await txt('#runStatus')).slice(0, 200));
  console.log('  answer tokens:', (await txt('#runToks')).slice(-160));
  await page.click('#runEx [data-ex*="frankreich"]');
  await page.waitForFunction(() => /^Beispiel ·/.test(document.querySelector('#runStatus').textContent.trim()), null, { timeout: 60000 }); await sleep(1000);
  console.log('SRV example from list:', (await txt('#runStatus')).slice(0, 120));
  await page.setViewportSize({ width: 400, height: 900 }); await sleep(800);
  console.log('mobile overflow:', await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth));
  await page.screenshot({ path: `${OUT}/05-mobile.png` });
  console.log('SRV errors:', errs.length ? errs.join(' | ') : 'none');
  await browser.close();
})().catch(e => { console.error('FAIL', e); process.exit(1); });
