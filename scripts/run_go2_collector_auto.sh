#!/usr/bin/env bash
set -euo pipefail

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(cd "$(dirname "${SCRIPT_PATH}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
COLLECTOR_BIN="${PROJECT_ROOT}/native/build/go2_collector"

NETWORK_INTERFACE="${NETWORK_INTERFACE:-}"
INPUT_BACKEND="${INPUT_BACKEND:-evdev}"
WIRELESS_MOTION_MODE="${WIRELESS_MOTION_MODE:-collector}"
INPUT_DEVICE="${INPUT_DEVICE:-}"
CAPTURE_MODE="${CAPTURE_MODE:-trajectory}"
WEB_UI_ENABLED="${WEB_UI_ENABLED:-1}"
WEB_PORT="${WEB_PORT:-8080}"
LOCAL_IPV4_CIDR="${LOCAL_IPV4_CIDR:-192.168.123.222/24}"
AUTO_CONFIGURE_NETWORK=1
AUTO_REEXEC_INPUT_GROUP=1

SCENE_ID=""
OPERATOR_ID=""
INSTRUCTION=""
TASK_FAMILY=""
TARGET_TYPE=""
TARGET_DESCRIPTION=""
TARGET_INSTANCE_ID=""
TASK_TAGS=""
COLLECTOR_NOTES=""

EXTRA_ARGS=()

usage() {
  cat <<'EOF'
用法：
  ./scripts/run_go2_collector_auto.sh --scene-id SCENE --operator-id OP --instruction TEXT [options] [-- extra collector args]

必填参数：
  --scene-id TEXT
  --operator-id TEXT
  --instruction TEXT

可选参数：
  --network-interface IFACE   默认：自动检测首选 Go2 有线网卡
  --input-backend MODE        默认：evdev
  --wireless-motion-mode MODE 默认：collector（仅 wireless_controller 生效）
  --input-device PATH         默认：留空并自动检测键盘
  --capture-mode MODE         默认：trajectory
  --web-ui / --no-web-ui      默认：开启 Web UI
  --web-port INT              默认：8080
  --local-ipv4-cidr CIDR      默认：192.168.123.222/24
  --task-family TEXT
  --target-type TEXT
  --target-description TEXT
  --target-instance-id TEXT
  --task-tags CSV
  --collector-notes TEXT
  --no-auto-network           不要自动拉起/配置 Go2 网卡
  --no-auto-input-group       不要自动通过 sg input 重新执行
  -h, --help

示例：
  ./scripts/run_go2_collector_auto.sh \
    --scene-id earth \
    --operator-id wxh \
    --instruction "go to the door" \
    --task-family goal_navigation \
    --target-type door \
    --target-description "gray iron door"

  ./scripts/run_go2_collector_auto.sh \
    --network-interface enxc817f52dae18 \
    --scene-id hall_b \
    --operator-id wxh \
    --instruction "follow the person in front of you" \
    --task-family visual_following \
    --target-type person \
    --target-description "person with black jacket"
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --network-interface)
      NETWORK_INTERFACE="$2"
      shift 2
      ;;
    --input-backend)
      INPUT_BACKEND="$2"
      shift 2
      ;;
    --wireless-motion-mode)
      WIRELESS_MOTION_MODE="$2"
      shift 2
      ;;
    --input-device)
      INPUT_DEVICE="$2"
      shift 2
      ;;
    --capture-mode)
      CAPTURE_MODE="$2"
      shift 2
      ;;
    --web-ui)
      WEB_UI_ENABLED=1
      shift
      ;;
    --no-web-ui)
      WEB_UI_ENABLED=0
      shift
      ;;
    --web-port)
      WEB_PORT="$2"
      shift 2
      ;;
    --local-ipv4-cidr)
      LOCAL_IPV4_CIDR="$2"
      shift 2
      ;;
    --scene-id)
      SCENE_ID="$2"
      shift 2
      ;;
    --operator-id)
      OPERATOR_ID="$2"
      shift 2
      ;;
    --instruction)
      INSTRUCTION="$2"
      shift 2
      ;;
    --task-family)
      TASK_FAMILY="$2"
      shift 2
      ;;
    --target-type)
      TARGET_TYPE="$2"
      shift 2
      ;;
    --target-description)
      TARGET_DESCRIPTION="$2"
      shift 2
      ;;
    --target-instance-id)
      TARGET_INSTANCE_ID="$2"
      shift 2
      ;;
    --task-tags)
      TASK_TAGS="$2"
      shift 2
      ;;
    --collector-notes)
      COLLECTOR_NOTES="$2"
      shift 2
      ;;
    --no-auto-network)
      AUTO_CONFIGURE_NETWORK=0
      shift
      ;;
    --no-auto-input-group)
      AUTO_REEXEC_INPUT_GROUP=0
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    --)
      shift
      EXTRA_ARGS+=("$@")
      break
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

require_file() {
  local path="$1"
  local description="$2"
  if [[ ! -e "$path" ]]; then
    echo "$description not found: $path" >&2
    exit 1
  fi
}

has_group() {
  local group_name="$1"
  id -nG | tr ' ' '\n' | grep -qx "$group_name"
}

detect_go2_interface() {
  local iface
  local carrier

  if [[ -n "${NETWORK_INTERFACE}" ]]; then
    echo "${NETWORK_INTERFACE}"
    return 0
  fi

  for iface in /sys/class/net/enx* /sys/class/net/eth* /sys/class/net/enp*; do
    [[ -e "${iface}" ]] || continue
    iface="$(basename "${iface}")"
    [[ "${iface}" == "enp0s5" ]] && continue
    carrier="$(cat "/sys/class/net/${iface}/carrier" 2>/dev/null || echo 0)"
    if [[ "${carrier}" == "1" ]]; then
      echo "${iface}"
      return 0
    fi
  done

  for iface in /sys/class/net/enx* /sys/class/net/eth* /sys/class/net/enp*; do
    [[ -e "${iface}" ]] || continue
    iface="$(basename "${iface}")"
    [[ "${iface}" == "enp0s5" ]] && continue
    echo "${iface}"
    return 0
  done

  return 1
}

ensure_input_group() {
  if [[ "${INPUT_BACKEND}" != "evdev" || "${AUTO_REEXEC_INPUT_GROUP}" -eq 0 ]]; then
    return 0
  fi
  if has_group input; then
    return 0
  fi
  if [[ "${GO2_COLLECTOR_REEXEC_INPUT:-0}" == "1" ]]; then
    echo "Current shell still lacks input group after re-exec attempt." >&2
    echo "Try: newgrp input" >&2
    exit 1
  fi

  local -a reexec_args=("${SCRIPT_PATH}")
  reexec_args+=("$@")
  echo "Current shell is not using group 'input'; re-running through sg input."
  exec sg input -c "GO2_COLLECTOR_REEXEC_INPUT=1 $(printf '%q ' "${reexec_args[@]}")"
}

ensure_go2_network() {
  local iface="$1"
  local carrier

  require_file "/sys/class/net/${iface}" "Network interface"
  carrier="$(cat "/sys/class/net/${iface}/carrier" 2>/dev/null || echo 0)"
  if [[ "${carrier}" != "1" ]]; then
    echo "Network interface ${iface} has no carrier. Check the Go2 cable / bridge first." >&2
    exit 1
  fi

  if [[ "${AUTO_CONFIGURE_NETWORK}" -eq 0 ]]; then
    return 0
  fi

  if ip -4 addr show dev "${iface}" | grep -q 'inet '; then
    return 0
  fi

  echo "Configuring ${iface} with ${LOCAL_IPV4_CIDR} ..."
  sudo ip link set "${iface}" up
  sudo ip addr replace "${LOCAL_IPV4_CIDR}" dev "${iface}"
}

build_collector_cmd() {
  local -n cmd_ref=$1
  cmd_ref=(
    "${COLLECTOR_BIN}"
    --network-interface "${NETWORK_INTERFACE}"
    --input-backend "${INPUT_BACKEND}"
    --capture-mode "${CAPTURE_MODE}"
    --scene-id "${SCENE_ID}"
    --operator-id "${OPERATOR_ID}"
    --instruction "${INSTRUCTION}"
  )

  if [[ "${WEB_UI_ENABLED}" == "1" ]]; then
    cmd_ref+=(--web-ui --web-port "${WEB_PORT}")
  fi
  if [[ "${INPUT_BACKEND}" == "wireless_controller" ]]; then
    cmd_ref+=(--wireless-motion-mode "${WIRELESS_MOTION_MODE}")
  fi
  if [[ "${INPUT_BACKEND}" == "evdev" && -n "${INPUT_DEVICE}" ]]; then
    cmd_ref+=(--input-device "${INPUT_DEVICE}")
  fi
  if [[ -n "${TASK_FAMILY}" ]]; then
    cmd_ref+=(--task-family "${TASK_FAMILY}")
  fi
  if [[ -n "${TARGET_TYPE}" ]]; then
    cmd_ref+=(--target-type "${TARGET_TYPE}")
  fi
  if [[ -n "${TARGET_DESCRIPTION}" ]]; then
    cmd_ref+=(--target-description "${TARGET_DESCRIPTION}")
  fi
  if [[ -n "${TARGET_INSTANCE_ID}" ]]; then
    cmd_ref+=(--target-instance-id "${TARGET_INSTANCE_ID}")
  fi
  if [[ -n "${TASK_TAGS}" ]]; then
    cmd_ref+=(--task-tags "${TASK_TAGS}")
  fi
  if [[ -n "${COLLECTOR_NOTES}" ]]; then
    cmd_ref+=(--collector-notes "${COLLECTOR_NOTES}")
  fi
  if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
    cmd_ref+=("${EXTRA_ARGS[@]}")
  fi
}

if [[ -z "${SCENE_ID}" || -z "${OPERATOR_ID}" || -z "${INSTRUCTION}" ]]; then
  echo "--scene-id、--operator-id 和 --instruction 为必填参数。" >&2
  usage >&2
  exit 1
fi

require_file "${COLLECTOR_BIN}" "Collector binary"
if [[ "${INPUT_BACKEND}" == "evdev" && -n "${INPUT_DEVICE}" ]]; then
  require_file "${INPUT_DEVICE}" "Input device"
fi

NETWORK_INTERFACE="$(detect_go2_interface)"
if [[ -z "${NETWORK_INTERFACE}" ]]; then
  echo "Failed to auto-detect the Go2 ethernet interface. Pass --network-interface explicitly." >&2
  exit 1
fi

ensure_input_group "$@"
ensure_go2_network "${NETWORK_INTERFACE}"

declare -a CMD
build_collector_cmd CMD

echo "Using network interface: ${NETWORK_INTERFACE}"
if [[ "${AUTO_CONFIGURE_NETWORK}" -eq 1 ]]; then
  echo "IPv4 status for ${NETWORK_INTERFACE}:"
  ip -4 addr show dev "${NETWORK_INTERFACE}" | sed 's/^/  /'
fi
echo "即将运行 collector："
printf '  %q' "${CMD[@]}"
printf '\n'

cd "${PROJECT_ROOT}"
exec "${CMD[@]}"
