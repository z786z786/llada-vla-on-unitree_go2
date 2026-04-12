#!/usr/bin/env bash
set -euo pipefail

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(cd "$(dirname "${SCRIPT_PATH}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DEFAULT_DEST_ROOT="${PROJECT_ROOT}/data"

SOURCE_HOST="unitree@192.168.123.18"
SOURCE_ROOT="/home/unitree/unitree_go2/go2_vla_collector/data"
DEST_ROOT="${DEFAULT_DEST_ROOT}"
SESSION_ID=""
DRY_RUN=0

SSH_OPTIONS=(-o StrictHostKeyChecking=accept-new)
SESSION_REGEX='^[0-9]{8}_[0-9]{6}$'

usage() {
  cat <<EOF
用法：
  ./scripts/sync_orin_sessions.sh [options]

说明：
  从 Orin 拉取 go2_vla_collector 原始 session 到本机 data 目录。
  默认只同步顶层形如 YYYYMMDD_HHMMSS 的 collector session。
  不会同步 llada_vla_converted、outputs、build、single_action 等目录。

可选参数：
  --source-host HOST   默认：${SOURCE_HOST}
  --source-root PATH   默认：${SOURCE_ROOT}
  --dest-root PATH     默认：${DEFAULT_DEST_ROOT}
  --session-id ID      只同步指定 session，例如 20260411_171528
  --dry-run            只预览，不实际写入本地
  -h, --help

示例：
  ./scripts/sync_orin_sessions.sh --dry-run
  ./scripts/sync_orin_sessions.sh
  ./scripts/sync_orin_sessions.sh --session-id 20260411_171528
EOF
}

require_command() {
  local name="$1"
  if ! command -v "${name}" >/dev/null 2>&1; then
    echo "缺少命令：${name}" >&2
    exit 1
  fi
}

quote_for_single() {
  printf "%s" "$1" | sed "s/'/'\\\\''/g"
}

join_by() {
  local delimiter="$1"
  shift || true
  local first=1
  local item
  for item in "$@"; do
    if [[ ${first} -eq 1 ]]; then
      printf "%s" "${item}"
      first=0
    else
      printf "%s%s" "${delimiter}" "${item}"
    fi
  done
}

print_session_list() {
  local title="$1"
  shift || true
  echo "${title}"
  if [[ $# -eq 0 ]]; then
    echo "  - (none)"
    return 0
  fi
  local session
  for session in "$@"; do
    echo "  - ${session}"
  done
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --source-host)
      SOURCE_HOST="$2"
      shift 2
      ;;
    --source-root)
      SOURCE_ROOT="$2"
      shift 2
      ;;
    --dest-root)
      DEST_ROOT="$2"
      shift 2
      ;;
    --session-id)
      SESSION_ID="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "未知参数：$1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

require_command ssh
require_command rsync

if [[ -n "${SESSION_ID}" && ! "${SESSION_ID}" =~ ${SESSION_REGEX} ]]; then
  echo "session id 格式不合法：${SESSION_ID}" >&2
  exit 1
fi

DEST_ROOT="$(readlink -f "${DEST_ROOT}")"
mkdir -p "${DEST_ROOT}"

TMP_DIR="$(mktemp -d)"
FILTER_FILE="${TMP_DIR}/rsync-filter.rules"
RSYNC_LOG="${TMP_DIR}/rsync.log"
trap 'rm -rf "${TMP_DIR}"' EXIT

REMOTE_ROOT_ESCAPED="$(quote_for_single "${SOURCE_ROOT}")"
REMOTE_FIND_CMD="find '${REMOTE_ROOT_ESCAPED}' -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort"

declare -a remote_entries=()
set +e
REMOTE_LIST_OUTPUT="$(ssh "${SSH_OPTIONS[@]}" "${SOURCE_HOST}" "${REMOTE_FIND_CMD}")"
SSH_STATUS=$?
set -e

if [[ ${SSH_STATUS} -ne 0 ]]; then
  echo "无法连接远端或列出 session：${SOURCE_HOST}:${SOURCE_ROOT}" >&2
  exit "${SSH_STATUS}"
fi

while IFS= read -r line; do
  [[ -n "${line}" ]] || continue
  remote_entries+=("${line}")
done <<< "${REMOTE_LIST_OUTPUT}"

declare -a remote_sessions=()
for entry in "${remote_entries[@]}"; do
  if [[ "${entry}" =~ ${SESSION_REGEX} ]]; then
    remote_sessions+=("${entry}")
  fi
done

if [[ ${#remote_sessions[@]} -eq 0 ]]; then
  echo "远端未找到可同步的 collector session。"
  echo "检查目录：${SOURCE_HOST}:${SOURCE_ROOT}"
  exit 0
fi

declare -a selected_sessions=()
if [[ -n "${SESSION_ID}" ]]; then
  found=0
  for session in "${remote_sessions[@]}"; do
    if [[ "${session}" == "${SESSION_ID}" ]]; then
      selected_sessions+=("${session}")
      found=1
      break
    fi
  done
  if [[ ${found} -eq 0 ]]; then
    echo "远端不存在指定 session：${SESSION_ID}" >&2
    echo "检查目录：${SOURCE_HOST}:${SOURCE_ROOT}" >&2
    exit 1
  fi
else
  selected_sessions=("${remote_sessions[@]}")
fi

declare -A existed_before=()
declare -a new_candidates=()
declare -a existing_candidates=()

for session in "${selected_sessions[@]}"; do
  if [[ -d "${DEST_ROOT}/${session}" ]]; then
    existed_before["${session}"]=1
    existing_candidates+=("${session}")
  else
    existed_before["${session}"]=0
    new_candidates+=("${session}")
  fi
done

{
  echo "+ /"
  for session in "${selected_sessions[@]}"; do
    echo "+ /${session}/"
    echo "+ /${session}/***"
  done
  echo "- *"
} > "${FILTER_FILE}"

echo "源目录：${SOURCE_HOST}:${SOURCE_ROOT}"
echo "目标目录：${DEST_ROOT}"
if [[ ${DRY_RUN} -eq 1 ]]; then
  echo "模式：dry-run（只预览，不写入本地）"
else
  echo "模式：同步执行"
fi
print_session_list "远端候选 session（${#remote_sessions[@]}）：" "${remote_sessions[@]}"
print_session_list "本次选中 session（${#selected_sessions[@]}）：" "${selected_sessions[@]}"
print_session_list "本地尚不存在（${#new_candidates[@]}）：" "${new_candidates[@]}"
print_session_list "本地已存在，若远端有新文件会补齐（${#existing_candidates[@]}）：" "${existing_candidates[@]}"

declare -a rsync_args=(
  -a
  -i
  --human-readable
  --prune-empty-dirs
  --out-format=%i'|'%n%L
  --filter="merge ${FILTER_FILE}"
  -e "ssh $(join_by " " "${SSH_OPTIONS[@]}")"
)

if [[ ${DRY_RUN} -eq 1 ]]; then
  rsync_args+=(-n)
fi

set +e
rsync "${rsync_args[@]}" "${SOURCE_HOST}:${SOURCE_ROOT}/" "${DEST_ROOT}/" | tee "${RSYNC_LOG}"
RSYNC_STATUS=${PIPESTATUS[0]}
set -e

if [[ ${RSYNC_STATUS} -ne 0 ]]; then
  echo "rsync 同步失败，退出码：${RSYNC_STATUS}" >&2
  exit "${RSYNC_STATUS}"
fi

declare -A changed_sessions_map=()
while IFS= read -r line; do
  [[ "${line}" == *"|"* ]] || continue
  path="${line#*|}"
  session="${path%%/*}"
  if [[ "${session}" =~ ${SESSION_REGEX} ]]; then
    changed_sessions_map["${session}"]=1
  fi
done < "${RSYNC_LOG}"

declare -a changed_sessions=()
declare -a new_sessions=()
declare -a updated_sessions=()
for session in "${selected_sessions[@]}"; do
  if [[ -n "${changed_sessions_map[${session}]:-}" ]]; then
    changed_sessions+=("${session}")
    if [[ "${existed_before[${session}]}" == "1" ]]; then
      updated_sessions+=("${session}")
    else
      new_sessions+=("${session}")
    fi
  fi
done

if [[ ${DRY_RUN} -eq 1 ]]; then
  print_session_list "将新增的 session（${#new_sessions[@]}）：" "${new_sessions[@]}"
  print_session_list "将补齐/更新的 session（${#updated_sessions[@]}）：" "${updated_sessions[@]}"
  if [[ ${#changed_sessions[@]} -eq 0 ]]; then
    echo "dry-run 结果：本地已是最新，没有需要同步的 session。"
  fi
else
  print_session_list "本次新增的 session（${#new_sessions[@]}）：" "${new_sessions[@]}"
  print_session_list "本次补齐/更新的 session（${#updated_sessions[@]}）：" "${updated_sessions[@]}"
  if [[ ${#changed_sessions[@]} -eq 0 ]]; then
    echo "同步完成：没有检测到新增或更新，本地已是最新。"
  else
    echo "同步完成：共处理 ${#changed_sessions[@]} 个 session。"
  fi
fi

echo
echo "后续可在本机继续执行："
echo "  cd ${PROJECT_ROOT}"
echo "  python3 scripts/sanity_check_dataset.py --data-root data --output-dir outputs/sanity_check"
echo "  python3 tools/validate_bc_dataset.py --dataset-root data --report-path outputs/validate_report.json"
echo "  python3 tools/convert_llada_vla_dataset.py --raw-root data --output-root data/llada_vla_converted --overwrite"
