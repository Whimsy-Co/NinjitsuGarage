#!/usr/bin/env bash
# Works on Linux, macOS, and Windows (Git Bash / WSL).
set -euo pipefail
cd "$(dirname "$0")"

if command -v python3 >/dev/null 2>&1; then PY=python3; else PY=python; fi

"$PY" -m venv .venv

if [ -f .venv/Scripts/activate ]; then
    VENV_PY=.venv/Scripts/python
    ACTIVATE=".venv\\Scripts\\activate"
else
    VENV_PY=.venv/bin/python
    ACTIVATE="source .venv/bin/activate"
fi

"$VENV_PY" -m pip install --upgrade pip
"$VENV_PY" -m pip install -r requirements.txt

echo "Done. Activate with: $ACTIVATE"
