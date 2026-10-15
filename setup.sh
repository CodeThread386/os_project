#!/usr/bin/env bash
# One-step setup: virtual environment, install, self-check.
set -euo pipefail
cd "$(dirname "$0")"

python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else "Python 3.11+ is required")'
[ -d .venv ] || python3 -m venv .venv
.venv/bin/pip install --quiet --upgrade pip
.venv/bin/pip install --quiet -e '.[dev]'

if ! command -v bwrap >/dev/null; then
    echo "note: bubblewrap is not installed; commands will run without the namespace sandbox."
    echo "      install it with: sudo apt install bubblewrap"
fi
if [ ! -f .env ]; then
    cp .env.example .env
    chmod 600 .env
    echo "created .env from .env.example; add your GROQ_API_KEYS there (or run Ollama locally)."
fi

.venv/bin/agentlayer doctor || true
echo
echo "Ready. Activate with:  source .venv/bin/activate   then run:  agentlayer"
