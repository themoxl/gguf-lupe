"""Record a question with the running Lupe server and save it as an example (examples/<id>.js + examples/index.js).
Same as asking in the Forward pass tab and clicking "Save as example", for any model the server can open.

usage: python make_example.py "<question>" [max_new] --model PATH [--server http://127.0.0.1:8765] [--title TEXT] [--lang de|en]
"""
import argparse, json, time, urllib.parse, urllib.request
ap = argparse.ArgumentParser(description='Record a question with the running Lupe server and save it as an example in examples/.')
ap.add_argument('question', help='question or text for the model')
ap.add_argument('max_new', nargs='?', type=int, default=40, help='answer at most this many tokens (default: 40)')
ap.add_argument('--model', required=True, help='path of the .gguf file (inside one of the server\'s model folders)')
ap.add_argument('--server', default='http://127.0.0.1:8765', help='address of the running Lupe server (default: %(default)s)')
ap.add_argument('--title', default='', help='title in the example list (default: the question)')
ap.add_argument('--lang', choices=('de', 'en'), help='language of the example (default: guessed from the question)')
a = ap.parse_args()
def get(path_, **q):
    with urllib.request.urlopen(a.server + path_ + '?' + urllib.parse.urlencode(q), timeout=900) as r:
        return json.loads(r.read())
def wait(job):
    while job.get('state') not in ('fertig', 'fehler'):          # the server's job states (done / error)
        time.sleep(0.5); job = get('/api/job', id=job['id'])
    if job['state'] != 'fertig':
        raise SystemExit('failed: ' + str(job.get('message')))
    return job
t0 = time.time()
run = wait(get('/api/run/start', path=a.model, q=a.question, chat='1', max=str(a.max_new), dev='auto'))['result']['run']
res = wait(get('/api/run/export', run=run, title=a.title, **({'lang': a.lang} if a.lang else {})))['result']
print(json.dumps(res, ensure_ascii=False), f'{time.time() - t0:.0f} s')
