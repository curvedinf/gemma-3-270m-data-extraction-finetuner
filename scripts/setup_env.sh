#!/usr/bin/env bash

set -euo pipefail

# Bootstraps the project's Python environment and dotenv file.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VENV_DIR="${REPO_ROOT}/.venv"
PYTHON_BIN="${PYTHON:-python3}"

log() {
    printf '[setup_env] %s\n' "$1"
}

abort() {
    printf '[setup_env][error] %s\n' "$1" >&2
    exit 1
}

command -v "$PYTHON_BIN" >/dev/null 2>&1 || abort "Python executable '${PYTHON_BIN}' not found. Set PYTHON=<path> before running."

log "Using Python: ${PYTHON_BIN}"

if [ ! -d "$VENV_DIR" ]; then
    log "Creating virtual environment at ${VENV_DIR}"
    "$PYTHON_BIN" -m venv "$VENV_DIR"
else
    log "Virtual environment already exists at ${VENV_DIR}"
fi

PYTHON_VENV="${VENV_DIR}/bin/python"
if [ ! -x "$PYTHON_VENV" ]; then
    abort "Python binary missing in virtual environment (${PYTHON_VENV})"
fi

log "Upgrading pip"
"$PYTHON_VENV" -m pip install --upgrade pip

REQ_FILE="${REPO_ROOT}/requirements.txt"
log "Ensuring ROCm nightly PyTorch stack"
"$PYTHON_VENV" -m pip uninstall -y torch torchvision torchaudio >/dev/null 2>&1 || true
"$PYTHON_VENV" -m pip install --pre --force-reinstall \
    --index-url https://download.pytorch.org/whl/nightly/rocm7.0 \
    torch torchvision torchaudio

if [ -f "$REQ_FILE" ]; then
    log "Installing dependencies from requirements.txt"
    "$PYTHON_VENV" -m pip install -r "$REQ_FILE"
else
    log "requirements.txt not found; skipping dependency installation"
fi

ENV_TEMPLATE="${REPO_ROOT}/.env.example"
ENV_FILE="${REPO_ROOT}/.env"
if [ -f "$ENV_TEMPLATE" ] && [ ! -f "$ENV_FILE" ]; then
    log "Creating .env from .env.example"
    cp "$ENV_TEMPLATE" "$ENV_FILE"
    log "Update ${ENV_FILE} with environment-specific secrets before running Fabric tasks"
else
    log "Skipping .env creation (template missing or .env already present)"
fi

log "Environment setup complete. Activate with 'source .venv/bin/activate'"
