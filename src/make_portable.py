"""Build the portable Windows package: unzip, double-click start-lupe-server.cmd, no install, no admin rights.

    python src/make_portable.py [--llama DIR] [--out DIR] [--python 3.12.10]

What goes in:
  python/      the official embeddable Python from python.org, with numpy and gguf installed into it
  llama.cpp/   the DLLs of one llama.cpp release (b11388, Vulkan build: runs on most graphics cards and on the CPU)
  src/         the server files
  the page, the examples, the start script, the licenses
PyTorch stays out (about 3 GB); without it the server computes with numpy, only a bit slower.
Needs internet while building (python.org, PyPI), not when the package is used.
"""
import argparse, glob, os, shutil, subprocess, sys, urllib.request, zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SERVER = ['lupe_server.py', 'lupe_run.py', 'lupe_worker.py', 'lupe_infer.py', 'lupe_export.py', 'make_example.py', 'requirements.txt', 'LICENSE-llama.cpp']
TOP = ['gguf-lupe.html', 'start-lupe-server.cmd', 'README.md', 'README.de.md', 'LICENSE', 'NOTICE.md']
LLAMA_KEEP = ['llama.dll', 'ggml.dll', 'ggml-base.dll', 'ggml-vulkan.dll', 'ggml-cpu-*.dll', 'libomp.dll', 'LICENSE*']


def fetch(url, dest):
    if not os.path.isfile(dest):
        print('  lade', url)
        tmp = dest + '.part'
        with urllib.request.urlopen(url) as r, open(tmp, 'wb') as f:
            shutil.copyfileobj(r, f)
        os.replace(tmp, dest)
    return dest


def main():
    ap = argparse.ArgumentParser(description='Portables Windows-Paket der GGUF-Lupe bauen')
    ap.add_argument('--python', default='3.12.10', help='Version der Embeddable-Python von python.org')
    ap.add_argument('--llama', default=os.path.join(os.path.expanduser('~'), '.cache', 'gguf-lupe', 'llama.cpp-b11388', 'vulkan'),
                    help='Ordner mit den DLLs einer llama.cpp-Version (b11388, Vulkan-Build)')
    ap.add_argument('--out', default=os.path.join(ROOT, 'dist'))
    a = ap.parse_args()
    name = 'gguf-lupe-portable-win64'
    out, cache = os.path.join(a.out, name), os.path.join(a.out, '_cache')
    shutil.rmtree(out, ignore_errors=True)
    os.makedirs(cache, exist_ok=True)

    print('1. Python', a.python, '(embeddable)')
    v = a.python
    z = fetch(f'https://www.python.org/ftp/python/{v}/python-{v}-embed-amd64.zip', os.path.join(cache, f'python-{v}-embed-amd64.zip'))
    py = os.path.join(out, 'python')
    zipfile.ZipFile(z).extractall(py)
    pth = glob.glob(os.path.join(py, 'python*._pth'))[0]
    tag = os.path.basename(pth)[len('python'):-len('._pth')]                 # '312'
    with open(pth, 'w', encoding='ascii') as f:                              # the embeddable Python only sees what is listed here
        f.write(f'python{tag}.zip\n.\nLib\\site-packages\n..\\src\nimport site\n')

    print('2. numpy und gguf in diese Python')
    site = os.path.join(py, 'Lib', 'site-packages')
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--quiet', '--disable-pip-version-check', '--no-compile', '--target', site,
                           '--only-binary=:all:', '--platform', 'win_amd64', '--python-version', v.rsplit('.', 1)[0], '--implementation', 'cp',
                           'numpy', 'gguf>=0.19'])
    for d in glob.glob(os.path.join(site, 'bin')) + glob.glob(os.path.join(site, '**', '__pycache__'), recursive=True):
        shutil.rmtree(d, ignore_errors=True)

    print('3. Seite, Beispiele, Server')
    for f in TOP:
        shutil.copy2(os.path.join(ROOT, f), out)
    for d in ('examples', 'LICENSES'):
        shutil.copytree(os.path.join(ROOT, d), os.path.join(out, d))
    os.makedirs(os.path.join(out, 'src'))
    for f in SERVER:
        shutil.copy2(os.path.join(HERE, f), os.path.join(out, 'src'))

    print('4. llama.cpp aus', a.llama)
    if not os.path.isfile(os.path.join(a.llama, 'llama.dll')):
        sys.exit('   keine llama.dll gefunden. Einmal über die Seite herunterladen (Durchlauf → „Grafikkarte (Vulkan)“) oder --llama angeben.')
    dl = os.path.join(out, 'llama.cpp', 'vulkan')
    os.makedirs(dl)
    for pat in LLAMA_KEEP:
        for f in glob.glob(os.path.join(a.llama, pat)):
            shutil.copy2(f, dl)

    print('5. ZIP')
    zp = shutil.make_archive(os.path.join(a.out, name), 'zip', a.out, name)
    mb = os.path.getsize(zp) / 1e6
    print(f'fertig: {zp} ({mb:.0f} MB)')


if __name__ == '__main__':
    main()
