#!/usr/bin/env bash

# Entry point that activates the env, pins GPU 2, and logs output while running
# the GPT-2 pipeline inside tmux.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

if [[ ! -d "2OE_env" ]]; then
  echo "2OE_env virtualenv not found in ${ROOT_DIR}" >&2
  exit 1
fi

source 2OE_env/bin/activate
export CUDA_VISIBLE_DEVICES=2
export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

LOGFILE="${ROOT_DIR}/phase3_gpt2.log"
echo "Logging to ${LOGFILE}"
exec > >(tee -a "${LOGFILE}") 2>&1

bash phase3/run_phase3_gpt2.sh
