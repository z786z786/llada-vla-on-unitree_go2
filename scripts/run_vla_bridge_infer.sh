#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

if [[ $# -lt 1 ]]; then
  echo "usage: $0 \"instruction text\" [extra llada_vla_infer_v2 args...]" >&2
  exit 2
fi

INSTRUCTION="$1"
shift

PY_DEPS_DIR="${REPO_ROOT}/.pydeps_vla"
CHECKPOINT_PATH="${REPO_ROOT}/go2_vla_collector/models/baseline_v2_min10_image_instruction_best.pt"
SSH_KEY_PATH="${REPO_ROOT}/.ssh_go2_vla/id_ed25519"
REMOTE_BRIDGE_BIN="/home/unitree/unitree_go2/go2_vla_collector/native/build/go2_bridge"
REMOTE_HOST="unitree@192.168.123.18"
REMOTE_NIC="eth0"

export PYTHONPATH="${PY_DEPS_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

if [[ -f "${SSH_KEY_PATH}" ]]; then
  exec python3 "${REPO_ROOT}/go2_vla_collector/tools/llada_vla_infer_v2.py" \
    --checkpoint-path "${CHECKPOINT_PATH}" \
    --instruction "${INSTRUCTION}" \
    --mode observe \
    --bridge-command "ssh -i ${SSH_KEY_PATH} -o StrictHostKeyChecking=no ${REMOTE_HOST} ${REMOTE_BRIDGE_BIN} --network-interface ${REMOTE_NIC}" \
    "$@"
fi

exec python3 "${REPO_ROOT}/go2_vla_collector/tools/llada_vla_infer_v2.py" \
  --checkpoint-path "${CHECKPOINT_PATH}" \
  --instruction "${INSTRUCTION}" \
  --mode observe \
  --bridge-bin "${REMOTE_BRIDGE_BIN}" \
  --network-interface "${REMOTE_NIC}" \
  "$@"
