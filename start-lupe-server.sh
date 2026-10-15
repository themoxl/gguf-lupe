#!/bin/sh
# Start the GGUF-Lupe server (macOS / Linux).
# Python: $LUPE_PYTHON, else the first line of lupe-python.txt, else python3.
# A configured Python that no longer exists (e.g. a deleted conda environment) is skipped with a note.
# Further options are passed on, e.g.:  ./start-lupe-server.sh --device cpu
cd "$(dirname "$0")" || exit 1
case "${LC_ALL:-${LC_MESSAGES:-${LANG:-}}}" in de*) DE=1 ;; *) DE= ;; esac   # system language (the server does the same)
PY="${LUPE_PYTHON:-}"
SRC=LUPE_PYTHON
if [ -z "$PY" ] && [ -f lupe-python.txt ]; then
  PY="$(head -n 1 lupe-python.txt | tr -d '\r')"
  SRC=lupe-python.txt
fi
if [ -n "$PY" ] && ! command -v "$PY" >/dev/null 2>&1; then
  if [ -n "$DE" ]; then echo "Hinweis: Das Python aus $SRC gibt es nicht mehr: $PY - nehme python3. $SRC korrigieren oder loeschen."
  else echo "Note: the Python from $SRC no longer exists: $PY - using python3. Fix or delete $SRC."; fi
  PY=
fi
[ -z "$PY" ] && PY=python3
if ! "$PY" -c "import sys" >/dev/null 2>&1; then
  if [ -n "$DE" ]; then echo "Python laesst sich nicht starten: $PY - Python 3.9 oder neuer installieren oder den Pfad in lupe-python.txt eintragen."
  else echo "Python could not be started: $PY - install Python 3.9 or newer, or put its path into lupe-python.txt."; fi
  exit 1
fi
if ! "$PY" -c "import numpy, gguf" 2>/dev/null; then
  if [ -n "$DE" ]; then
    echo "Es fehlen Python-Pakete. Einmal installieren mit:"
    echo "   $PY -m pip install numpy gguf"
    echo "Optional fuer Grafikkarte (NVIDIA/Apple) oder schnellere CPU: PyTorch, siehe https://pytorch.org"
  else
    echo "Python packages are missing. Install them once with:"
    echo "   $PY -m pip install numpy gguf"
    echo "Optional, for a GPU (NVIDIA/Apple) or a faster CPU: PyTorch, see https://pytorch.org"
  fi
  exit 1
fi
exec "$PY" src/lupe_server.py "$@"
