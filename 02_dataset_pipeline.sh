#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_BIN="${REPO_ROOT}/.venv/bin"
FAB_BIN="${VENV_BIN}/fab"

if [ ! -x "${FAB_BIN}" ]; then
    echo "[02_dataset_pipeline] Virtual environment not found. Run ./01_setup_env.sh first." >&2
    exit 1
fi

SCHEMA_VERSION="${SCHEMA_VERSION:-v1}"
DATASET_CONFIG="${DATASET_CONFIG:-configs/datasets.yaml}"

echo "[02_dataset_pipeline] Cleaning dataset (schema_version=${SCHEMA_VERSION})"
"${FAB_BIN}" dataset.clean --schema-version "${SCHEMA_VERSION}"

echo "[02_dataset_pipeline] Creating splits (config=${DATASET_CONFIG})"
"${FAB_BIN}" dataset.split --config "${DATASET_CONFIG}"

echo "[02_dataset_pipeline] Computing stats (config=${DATASET_CONFIG})"
"${FAB_BIN}" dataset.stats --config "${DATASET_CONFIG}"

echo "[02_dataset_pipeline] Done."
