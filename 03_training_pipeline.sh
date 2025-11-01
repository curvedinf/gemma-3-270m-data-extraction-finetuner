#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_BIN="${REPO_ROOT}/.venv/bin"
FAB_BIN="${VENV_BIN}/fab"
TRAIN_CONFIG="${TRAIN_CONFIG:-configs/training.yaml}"

# Load ROCm environment overrides if present
if [ -f "${REPO_ROOT}/.env" ]; then
    set -a
    # shellcheck disable=SC1090
    source "${REPO_ROOT}/.env"
    set +a
fi

if [ ! -x "${FAB_BIN}" ]; then
    echo "[03_training_pipeline] Virtual environment not found. Run ./01_setup_env.sh first." >&2
    exit 1
fi

if ! "${FAB_BIN}" -l 2>/dev/null | grep -q "train.prepare"; then
    echo "[03_training_pipeline] Train tasks are unavailable (likely due to missing GPU support). Skipping." >&2
    exit 0
fi

echo "[03_training_pipeline] Preparing run (config=${TRAIN_CONFIG})"
"${FAB_BIN}" train.prepare --config "${TRAIN_CONFIG}"

if [ -n "${RESUME_FROM:-}" ]; then
    echo "[03_training_pipeline] Resuming training from ${RESUME_FROM} (config=${TRAIN_CONFIG})"
    "${FAB_BIN}" train.resume --checkpoint "${RESUME_FROM}" --config "${TRAIN_CONFIG}"
else
    echo "[03_training_pipeline] Starting training run (config=${TRAIN_CONFIG})"
    "${FAB_BIN}" train.run --config "${TRAIN_CONFIG}"
fi

echo "[03_training_pipeline] Done."
