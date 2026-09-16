#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
POLICY_PYTHON="${ROBOT_POLICY_PYTHON:-/home/wangpc/miniconda3/envs/starVLA/bin/python}"
CHECKPOINT="${CKPT:?Set CKPT to a robot_policy .pt checkpoint}"
PORT="${PORT:-10093}"
DEVICE="${DEVICE:-cuda}"
PRECISION="${PRECISION:-bf16}"

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

arguments=(
  -m robot_policy.deployment.server
  --ckpt_path "${CHECKPOINT}"
  --port "${PORT}"
  --device "${DEVICE}"
  --precision "${PRECISION}"
)
if [[ -n "${PREPARED_PATH:-}" ]]; then
  arguments+=(--prepared-path "${PREPARED_PATH}")
fi
exec "${POLICY_PYTHON}" "${arguments[@]}" "$@"

