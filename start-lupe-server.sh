#!/bin/sh
# Start the GGUF-Lupe server (macOS / Linux).
# Python: $LUPE_PYTHON, else the first line of lupe-python.txt, else python3.
# Further options are passed on, e.g.:  ./start-lupe-server.sh --device cpu
cd "$(dirname "$0")" || exit 1
PY="${LUPE_PYTHON:-}"
[ -z "$PY" ] && [ -f lupe-python.txt ] && PY="$(head -n 1 lupe-python.txt)"
[ -z "$PY" ] && PY=python3
if ! "$PY" -c "import numpy, gguf" 2>/dev/null; then
  case "${LC_ALL:-${LC_MESSAGES:-${LANG:-}}}" in          # system language, German or English (the server does the same)
    de*) echo "Es fehlen Python-Pakete. Einmal installieren mit:"
         echo "   $PY -m pip install numpy gguf"
         echo "Optional fuer Grafikkarte (NVIDIA/Apple) oder schnellere CPU: PyTorch, siehe https://pytorch.org" ;;
    *)   echo "Python packages are missing. Install them once with:"
         echo "   $PY -m pip install numpy gguf"
         echo "Optional, for a GPU (NVIDIA/Apple) or a faster CPU: PyTorch, see https://pytorch.org" ;;
  esac
  exit 1
fi
exec "$PY" lupe_server.py "$@"
