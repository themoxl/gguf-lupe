// Assembles the single-file page: node build.cjs <standalone-out.html>   (also writes out/gguf-lupe.html for the artifact)
const fs = require('fs');
const src = fs.readFileSync('src.html', 'utf8');
const pre = 'const GGUF_PRE = ' + JSON.stringify(JSON.parse(fs.readFileSync('pre_table.json', 'utf8'))) + ';';
const core = fs.readFileSync('gguf-core.js', 'utf8').replace("if (typeof module !== 'undefined') module.exports = GGUF;", '');
const demo = fs.readFileSync('demo.json', 'utf8');
const mapc = fs.readFileSync('map-core.js', 'utf8').replace("if (typeof module !== 'undefined') module.exports = MAPCORE;", '');
const app = fs.readFileSync('app.js', 'utf8');
const ll = fs.readFileSync('ll.b64', 'utf8').trim();
const fonts = fs.readFileSync('fonts/fonts.css', 'utf8').replace(/url\(([a-z0-9-]+\.woff2)\)/g, (m, f) => `url(data:font/woff2;base64,${fs.readFileSync('fonts/' + f).toString('base64')})`);
for (const [n, x] of [['pre', pre], ['core', core], ['map', mapc], ['app', app]]) if (/<\/script/i.test(x)) throw new Error(n + ' contains </script');
const page = src.replace('/*PRE*/', () => pre.replace(/</g, '\u003c')).replace('/*CORE*/', () => core).replace('/*MAP*/', () => mapc).replace('/*DEMO*/', () => demo).replace('/*APP*/', () => app).replace('/*LL*/', () => ll).replace('/*FONTS*/', () => fonts);
fs.mkdirSync('out', { recursive: true });
fs.writeFileSync('out/gguf-lupe.html', page);
fs.writeFileSync(process.argv[2], '<!doctype html>\n<html lang="de">\n<meta charset="utf-8">\n<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n' + page + '\n</html>\n');
console.log('page', (page.length / 1e6).toFixed(2), 'MB ->', process.argv[2]);
