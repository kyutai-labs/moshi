#!/usr/bin/env bash
set -euo pipefail

# ─────────────────────────────────────────────────────────────────────────────
# Windows / Git Bash hardening (GH-35)
# ─────────────────────────────────────────────────────────────────────────────
UNAME_S="$(uname -s 2>/dev/null || echo "")"
IS_WINDOWS=0
case "$UNAME_S" in
  MINGW*|MSYS*|CYGWIN*) IS_WINDOWS=1 ;;
esac

# Python detection: prefer python3, fallback to python (common on Windows)
pick_python() {
  if [[ -n "${PYTHON_BIN:-}" ]]; then
    command -v "$PYTHON_BIN" >/dev/null 2>&1 && { echo "$PYTHON_BIN"; return; }
  fi
  if command -v python3 >/dev/null 2>&1; then echo "python3"; return; fi
  if command -v python  >/dev/null 2>&1; then echo "python"; return; fi
  echo ""
}

PYTHON_BIN="$(pick_python)"
[[ -n "$PYTHON_BIN" ]] || { echo "ralph: python not found (need python3 or python in PATH)" >&2; exit 1; }
export PYTHON_BIN

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
CONFIG="$SCRIPT_DIR/config.env"
FLOWCTL="$SCRIPT_DIR/flowctl"
FLOWCTL_PY="$SCRIPT_DIR/flowctl.py"

fail() { echo "ralph: $*" >&2; exit 1; }

# Pre-scan for --config before sourcing (main arg loop runs after config is loaded)
_config_found=0
_prev=""
for _arg in "$@"; do
  if [[ "$_prev" == "--config" ]]; then
    [[ "$_arg" != --* ]] || fail "--config requires a path, not a flag"
    CONFIG="$_arg"
    _config_found=1
    break
  fi
  _prev="$_arg"
done
[[ "$_prev" == "--config" && "$_config_found" -eq 0 ]] && fail "--config requires a path"
unset _prev _arg _config_found

log() {
  # Machine-readable logs: only show when UI disabled
  [[ "${UI_ENABLED:-1}" != "1" ]] && echo "ralph: $*"
  return 0
}

# Ensure flowctl is runnable even when NTFS exec bit / shebang handling is flaky on Windows
ensure_flowctl_wrapper() {
  # If flowctl exists and is executable, use it
  if [[ -f "$FLOWCTL" && -x "$FLOWCTL" ]]; then
    return 0
  fi

  # On Windows or if flowctl not executable, create a wrapper that calls Python explicitly
  if [[ -f "$FLOWCTL_PY" ]]; then
    local wrapper="$SCRIPT_DIR/flowctl-wrapper.sh"
    cat > "$wrapper" <<SH
#!/usr/bin/env bash
set -euo pipefail
DIR="\$(cd "\$(dirname "\${BASH_SOURCE[0]}")" && pwd)"
PY="\${PYTHON_BIN:-python3}"
command -v "\$PY" >/dev/null 2>&1 || PY="python"
exec "\$PY" "\$DIR/flowctl.py" "\$@"
SH
    chmod +x "$wrapper" 2>/dev/null || true
    FLOWCTL="$wrapper"
    export FLOWCTL
    return 0
  fi

  fail "missing flowctl (expected $SCRIPT_DIR/flowctl or $SCRIPT_DIR/flowctl.py)"
}

ensure_flowctl_wrapper

# ─────────────────────────────────────────────────────────────────────────────
# Presentation layer (human-readable output)
# ─────────────────────────────────────────────────────────────────────────────
UI_ENABLED="${RALPH_UI:-1}"  # set RALPH_UI=0 to disable

# Timing
START_TIME="$(date +%s)"

elapsed_time() {
  local now elapsed mins secs
  now="$(date +%s)"
  elapsed=$((now - START_TIME))
  mins=$((elapsed / 60))
  secs=$((elapsed % 60))
  printf "%d:%02d" "$mins" "$secs"
}

# Stats tracking
STATS_TASKS_DONE=0

# Colors (disabled if not tty or NO_COLOR set)
if [[ -t 1 && -z "${NO_COLOR:-}" ]]; then
  C_RESET='\033[0m'
  C_BOLD='\033[1m'
  C_DIM='\033[2m'
  C_BLUE='\033[34m'
  C_GREEN='\033[32m'
  C_YELLOW='\033[33m'
  C_RED='\033[31m'
  C_CYAN='\033[36m'
  C_MAGENTA='\033[35m'
else
  C_RESET='' C_BOLD='' C_DIM='' C_BLUE='' C_GREEN='' C_YELLOW='' C_RED='' C_CYAN='' C_MAGENTA=''
fi

# Watch mode: "", "tools", "verbose"
WATCH_MODE=""

ui() {
  [[ "$UI_ENABLED" == "1" ]] || return 0
  echo -e "$*"
}

# Get title from epic/task JSON
get_title() {
  local json="$1"
  "$PYTHON_BIN" - "$json" <<'PY'
import json, sys
try:
    data = json.loads(sys.argv[1])
    print(data.get("title", "")[:40])
except:
    print("")
PY
}

# Count progress (done/total tasks for scoped epics)
get_progress() {
  "$PYTHON_BIN" - "$ROOT_DIR" "${EPICS_FILE:-}" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
epics_file = sys.argv[2] if len(sys.argv) > 2 else ""
flow_dir = root / ".flow"

# Get scoped epics or all
scoped = []
if epics_file:
    try:
        scoped = json.load(open(epics_file))["epics"]
    except:
        pass

epics_dir = flow_dir / "epics"
tasks_dir = flow_dir / "tasks"
if not epics_dir.exists():
    print("0|0|0|0")
    sys.exit(0)

epic_ids = []
for f in sorted(epics_dir.glob("fn-*.json")):
    eid = f.stem
    if not scoped or eid in scoped:
        epic_ids.append(eid)

epics_done = sum(1 for e in epic_ids if json.load(open(epics_dir / f"{e}.json")).get("status") == "done")
tasks_total = 0
tasks_done = 0
if tasks_dir.exists():
    for tf in tasks_dir.glob("*.json"):
        try:
            t = json.load(open(tf))
            epic_id = tf.stem.rsplit(".", 1)[0]
            if not scoped or epic_id in scoped:
                tasks_total += 1
                if t.get("status") == "done":
                    tasks_done += 1
        except:
            pass
print(f"{epics_done}|{len(epic_ids)}|{tasks_done}|{tasks_total}")
PY
}

# Get git diff stats
get_git_stats() {
  local base_branch="${1:-main}"
  local stats
  stats="$(git -C "$ROOT_DIR" diff --shortstat "$base_branch"...HEAD 2>/dev/null || true)"
  if [[ -z "$stats" ]]; then
    echo ""
    return
  fi
  "$PYTHON_BIN" - "$stats" <<'PY'
import re, sys
s = sys.argv[1]
files = re.search(r"(\d+) files? changed", s)
ins = re.search(r"(\d+) insertions?", s)
dels = re.search(r"(\d+) deletions?", s)
f = files.group(1) if files else "0"
i = ins.group(1) if ins else "0"
d = dels.group(1) if dels else "0"
print(f"{f} files, +{i} -{d}")
PY
}

ui_header() {
  ui ""
  ui "${C_BOLD}${C_BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${C_RESET}"
  ui "${C_BOLD}${C_BLUE}  🤖 Ralph Autonomous Loop${C_RESET}"
  ui "${C_BOLD}${C_BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${C_RESET}"
}

ui_config() {
  local git_branch progress_info epics_done epics_total tasks_done tasks_total
  git_branch="$(git -C "$ROOT_DIR" rev-parse --abbrev-ref HEAD 2>/dev/null || echo "unknown")"
  progress_info="$(get_progress)"
  IFS='|' read -r epics_done epics_total tasks_done tasks_total <<< "$progress_info"

  ui ""
  ui "${C_DIM}   Branch:${C_RESET} ${C_BOLD}$git_branch${C_RESET}"
  ui "${C_DIM}   Progress:${C_RESET} Epic ${epics_done}/${epics_total} ${C_DIM}•${C_RESET} Task ${tasks_done}/${tasks_total}"

  local plan_display="$PLAN_REVIEW" work_display="$WORK_REVIEW" completion_display="$COMPLETION_REVIEW"
  [[ "$PLAN_REVIEW" == "rp" ]] && plan_display="RepoPrompt"
  [[ "$PLAN_REVIEW" == "codex" ]] && plan_display="Codex"
  [[ "$WORK_REVIEW" == "rp" ]] && work_display="RepoPrompt"
  [[ "$WORK_REVIEW" == "codex" ]] && work_display="Codex"
  [[ "$COMPLETION_REVIEW" == "rp" ]] && completion_display="RepoPrompt"
  [[ "$COMPLETION_REVIEW" == "codex" ]] && completion_display="Codex"
  ui "${C_DIM}   Reviews:${C_RESET} Plan=$plan_display ${C_DIM}•${C_RESET} Work=$work_display ${C_DIM}•${C_RESET} Completion=$completion_display"
  [[ -n "${EPICS:-}" ]] && ui "${C_DIM}   Scope:${C_RESET} $EPICS"
  ui ""
}

ui_version_check() {
  local meta_file="$ROOT_DIR/.flow/meta.json"
  local plugin_file="$SCRIPT_DIR/../.claude-plugin/plugin.json"
  [[ -f "$meta_file" ]] || return 0
  [[ -f "$plugin_file" ]] || return 0
  local setup_ver plugin_ver
  setup_ver="$(jq -r '.setup_version // empty' "$meta_file" 2>/dev/null)" || return 0
  plugin_ver="$(jq -r '.version // empty' "$plugin_file" 2>/dev/null)" || return 0
  [[ -z "$setup_ver" ]] && return 0
  [[ "$setup_ver" == "$plugin_ver" ]] && return 0
  ui "${C_YELLOW}   ⚠ Plugin updated to v${plugin_ver}. Run /flow-next:setup to refresh local scripts (current: v${setup_ver}).${C_RESET}"
  ui ""
}

ui_iteration() {
  local iter="$1" status="$2" epic="${3:-}" task="${4:-}" title="" item_json=""
  local elapsed
  elapsed="$(elapsed_time)"
  ui ""
  ui "${C_BOLD}${C_CYAN}🔄 Iteration $iter${C_RESET}                                              ${C_DIM}[${elapsed}]${C_RESET}"
  if [[ "$status" == "plan" ]]; then
    item_json="$("$FLOWCTL" show "$epic" --json 2>/dev/null || true)"
    title="$(get_title "$item_json")"
    ui "   ${C_DIM}Epic:${C_RESET} ${C_BOLD}$epic${C_RESET} ${C_DIM}\"$title\"${C_RESET}"
    ui "   ${C_DIM}Phase:${C_RESET} ${C_YELLOW}Planning${C_RESET}"
  elif [[ "$status" == "work" ]]; then
    item_json="$("$FLOWCTL" show "$task" --json 2>/dev/null || true)"
    title="$(get_title "$item_json")"
    ui "   ${C_DIM}Task:${C_RESET} ${C_BOLD}$task${C_RESET} ${C_DIM}\"$title\"${C_RESET}"
    ui "   ${C_DIM}Phase:${C_RESET} ${C_MAGENTA}Implementation${C_RESET}"
  elif [[ "$status" == "completion_review" ]]; then
    item_json="$("$FLOWCTL" show "$epic" --json 2>/dev/null || true)"
    title="$(get_title "$item_json")"
    ui "   ${C_DIM}Epic:${C_RESET} ${C_BOLD}$epic${C_RESET} ${C_DIM}\"$title\"${C_RESET}"
    ui "   ${C_DIM}Phase:${C_RESET} ${C_GREEN}Completion Review${C_RESET}"
  fi
}

ui_plan_review() {
  local mode="$1" epic="$2"
  if [[ "$mode" == "rp" ]]; then
    ui ""
    ui "   ${C_YELLOW}📝 Plan Review${C_RESET}"
    ui "      ${C_DIM}Sending to reviewer via RepoPrompt...${C_RESET}"
  elif [[ "$mode" == "codex" ]]; then
    ui ""
    ui "   ${C_YELLOW}📝 Plan Review${C_RESET}"
    ui "      ${C_DIM}Sending to reviewer via Codex...${C_RESET}"
  fi
}

ui_impl_review() {
  local mode="$1" task="$2"
  if [[ "$mode" == "rp" ]]; then
    ui ""
    ui "   ${C_MAGENTA}🔍 Implementation Review${C_RESET}"
    ui "      ${C_DIM}Sending to reviewer via RepoPrompt...${C_RESET}"
  elif [[ "$mode" == "codex" ]]; then
    ui ""
    ui "   ${C_MAGENTA}🔍 Implementation Review${C_RESET}"
    ui "      ${C_DIM}Sending to reviewer via Codex...${C_RESET}"
  fi
}

ui_completion_review() {
  local mode="$1" epic="$2"
  if [[ "$mode" == "rp" ]]; then
    ui ""
    ui "   ${C_GREEN}✅ Epic Completion Review${C_RESET}"
    ui "      ${C_DIM}Verifying spec compliance via RepoPrompt...${C_RESET}"
  elif [[ "$mode" == "codex" ]]; then
    ui ""
    ui "   ${C_GREEN}✅ Epic Completion Review${C_RESET}"
    ui "      ${C_DIM}Verifying spec compliance via Codex...${C_RESET}"
  fi
}

ui_task_done() {
  local task="$1" git_stats=""
  STATS_TASKS_DONE=$((STATS_TASKS_DONE + 1))
  init_branches_file 2>/dev/null || true
  local base_branch
  base_branch="$(get_base_branch 2>/dev/null || echo "main")"
  git_stats="$(get_git_stats "$base_branch")"
  if [[ -n "$git_stats" ]]; then
    ui "   ${C_GREEN}✓${C_RESET} ${C_BOLD}$task${C_RESET} ${C_DIM}($git_stats)${C_RESET}"
  else
    ui "   ${C_GREEN}✓${C_RESET} ${C_BOLD}$task${C_RESET}"
  fi
}

ui_retry() {
  local task="$1" attempts="$2" max="$3"
  ui "   ${C_YELLOW}↻ Retry${C_RESET} ${C_DIM}(attempt $attempts/$max)${C_RESET}"
}

ui_blocked() {
  local task="$1"
  ui "   ${C_RED}🚫 Task blocked:${C_RESET} $task ${C_DIM}(max attempts reached)${C_RESET}"
}

ui_complete() {
  local elapsed progress_info epics_done epics_total tasks_done tasks_total
  elapsed="$(elapsed_time)"
  progress_info="$(get_progress)"
  IFS='|' read -r epics_done epics_total tasks_done tasks_total <<< "$progress_info"

  ui ""
  ui "${C_BOLD}${C_GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${C_RESET}"
  ui "${C_BOLD}${C_GREEN}  ✅ Ralph Complete${C_RESET}                                        ${C_DIM}[${elapsed}]${C_RESET}"
  ui ""
  ui "   ${C_DIM}Tasks:${C_RESET} ${tasks_done}/${tasks_total} ${C_DIM}•${C_RESET} ${C_DIM}Epics:${C_RESET} ${epics_done}/${epics_total}"
  ui "${C_BOLD}${C_GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${C_RESET}"
  ui ""
}

ui_fail() {
  local reason="${1:-}" elapsed
  elapsed="$(elapsed_time)"
  ui ""
  ui "${C_BOLD}${C_RED}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${C_RESET}"
  ui "${C_BOLD}${C_RED}  ❌ Ralph Failed${C_RESET}                                          ${C_DIM}[${elapsed}]${C_RESET}"
  [[ -n "$reason" ]] && ui "     ${C_DIM}$reason${C_RESET}"
  ui "${C_BOLD}${C_RED}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${C_RESET}"
  ui ""
}

ui_waiting() {
  ui "   ${C_DIM}⏳ Claude working...${C_RESET}"
}

[[ -f "$CONFIG" ]] || fail "config file not found: $CONFIG"
[[ -x "$FLOWCTL" ]] || fail "missing flowctl"

# shellcheck disable=SC1090
set -a
source "$CONFIG"
set +a

MAX_ITERATIONS="${MAX_ITERATIONS:-25}"
MAX_TURNS="${MAX_TURNS:-}"  # empty = no limit; Claude stops via promise tags
MAX_ATTEMPTS_PER_TASK="${MAX_ATTEMPTS_PER_TASK:-5}"
WORKER_TIMEOUT="${WORKER_TIMEOUT:-3600}"  # 1hr default; safety guard against runaway workers
BRANCH_MODE="${BRANCH_MODE:-new}"
PLAN_REVIEW="${PLAN_REVIEW:-none}"
WORK_REVIEW="${WORK_REVIEW:-none}"
COMPLETION_REVIEW="${COMPLETION_REVIEW:-none}"
CODEX_SANDBOX="${CODEX_SANDBOX:-auto}"  # Codex sandbox mode; flowctl reads this env var
REQUIRE_PLAN_REVIEW="${REQUIRE_PLAN_REVIEW:-0}"
YOLO="${YOLO:-0}"
EPICS="${EPICS:-}"
export CODEX_SANDBOX  # Ensure available to Claude worker for flowctl codex commands

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --watch)
      if [[ "${2:-}" == "verbose" ]]; then
        WATCH_MODE="verbose"
        shift
      else
        WATCH_MODE="tools"
      fi
      shift
      ;;
    --config)
      # Already processed in pre-scan; just consume args
      shift
      ;;
    --help|-h)
      echo "Usage: ralph.sh [options]"
      echo ""
      echo "Options:"
      echo "  --config <path>  Use alternate config file (default: config.env)"
      echo "  --watch          Show tool calls in real-time"
      echo "  --watch verbose  Show tool calls + model responses"
      echo "  --help, -h       Show this help"
      echo ""
      echo "Environment variables:"
      echo "  EPICS            Comma/space-separated epic IDs to work on"
      echo "  MAX_ITERATIONS   Max loop iterations (default: 25)"
      echo "  YOLO             Set to 1 to skip permissions (required for unattended)"
      echo ""
      echo "See config.env for more options."
      exit 0
      ;;
    *)
      fail "Unknown option: $1 (use --help for usage)"
      ;;
  esac
done

# Set up signal trap for clean Ctrl+C handling
# Must kill all child processes including timeout and claude
cleanup() {
  trap - SIGINT SIGTERM  # Prevent re-entry
  # Kill all child processes
  pkill -P $$ 2>/dev/null
  # Kill process group as fallback
  kill -- -$$ 2>/dev/null
  exit 130
}
trap cleanup SIGINT SIGTERM

CLAUDE_BIN="${CLAUDE_BIN:-claude}"

# Detect timeout command (GNU coreutils). On macOS: brew install coreutils
# Use --foreground to keep child in same process group for signal handling
if command -v timeout >/dev/null 2>&1 && timeout --foreground 0 true 2>/dev/null; then
  TIMEOUT_CMD="timeout --foreground"
elif command -v gtimeout >/dev/null 2>&1 && gtimeout --foreground 0 true 2>/dev/null; then
  TIMEOUT_CMD="gtimeout --foreground"
elif command -v timeout >/dev/null 2>&1; then
  TIMEOUT_CMD="timeout"
elif command -v gtimeout >/dev/null 2>&1; then
  TIMEOUT_CMD="gtimeout"
else
  TIMEOUT_CMD=""
  echo "ralph: warning: timeout command not found; worker timeout disabled (brew install coreutils)" >&2
fi

sanitize_id() {
  local v="$1"
  v="${v// /_}"
  v="${v//\//_}"
  v="${v//\\/__}"
  echo "$v"
}

get_actor() {
  if [[ -n "${FLOW_ACTOR:-}" ]]; then echo "$FLOW_ACTOR"; return; fi
  if actor="$(git -C "$ROOT_DIR" config user.email 2>/dev/null)"; then
    [[ -n "$actor" ]] && { echo "$actor"; return; }
  fi
  if actor="$(git -C "$ROOT_DIR" config user.name 2>/dev/null)"; then
    [[ -n "$actor" ]] && { echo "$actor"; return; }
  fi
  echo "${USER:-unknown}"
}

rand4() {
  "$PYTHON_BIN" - <<'PY'
import secrets
print(secrets.token_hex(2))
PY
}

# Portable file truncation (ZSH-safe: bare `> file` hangs on macOS default shell)
truncate_file() {
  : > "$1"
}

render_template() {
  local path="$1"
  "$PYTHON_BIN" - "$path" <<'PY'
import os, sys
path = sys.argv[1]
text = open(path, encoding="utf-8").read()
keys = ["EPIC_ID","TASK_ID","PLAN_REVIEW","WORK_REVIEW","COMPLETION_REVIEW","BRANCH_MODE","BRANCH_MODE_EFFECTIVE","REQUIRE_PLAN_REVIEW","REVIEW_RECEIPT_PATH","RALPH_ITERATION"]
for k in keys:
    text = text.replace("{{%s}}" % k, os.environ.get(k, ""))
print(text)
PY
}

json_get() {
  local key="$1"
  local json="$2"
  "$PYTHON_BIN" - "$key" "$json" <<'PY'
import json, sys
key = sys.argv[1]
data = json.loads(sys.argv[2])
val = data.get(key)
if val is None:
    print("")
elif isinstance(val, bool):
    print("1" if val else "0")
else:
    print(val)
PY
}

ensure_attempts_file() {
  [[ -f "$1" ]] || echo "{}" > "$1"
}

bump_attempts() {
  "$PYTHON_BIN" - "$1" "$2" <<'PY'
import json, sys, os
path, task = sys.argv[1], sys.argv[2]
data = {}
if os.path.exists(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
count = int(data.get(task, 0)) + 1
data[task] = count
with open(path, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, sort_keys=True)
print(count)
PY
}

write_epics_file() {
  "$PYTHON_BIN" - "$1" <<'PY'
import json, sys
raw = sys.argv[1]
parts = [p.strip() for p in raw.replace(",", " ").split() if p.strip()]
print(json.dumps({"epics": parts}, indent=2, sort_keys=True))
PY
}

# Clean format for branches and run dirs (no PII: no hostname, email, or PID)
RUN_ID="$(date -u +%Y%m%d-%H%M%S)-$(rand4)"
# Verbose format for debugging (internal use only in logs)
RUN_ID_FULL="$(date -u +%Y%m%dT%H%M%SZ)-$(hostname -s 2>/dev/null || hostname)-$(sanitize_id "$(get_actor)")-$$-$(rand4)"
RUN_DIR="$SCRIPT_DIR/runs/$RUN_ID"
mkdir -p "$RUN_DIR"
ATTEMPTS_FILE="$RUN_DIR/attempts.json"
ensure_attempts_file "$ATTEMPTS_FILE"
BRANCHES_FILE="$RUN_DIR/branches.json"
RECEIPTS_DIR="$RUN_DIR/receipts"
mkdir -p "$RECEIPTS_DIR"
PROGRESS_FILE="$RUN_DIR/progress.txt"
{
  echo "# Ralph Progress Log"
  echo "Run: $RUN_ID"
  echo "Full ID: $RUN_ID_FULL"
  echo "Started: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "---"
} > "$PROGRESS_FILE"

extract_tag() {
  local tag="$1"
  "$PYTHON_BIN" - "$tag" <<'PY'
import re, sys
tag = sys.argv[1]
text = sys.stdin.read()
matches = re.findall(rf"<{tag}>(.*?)</{tag}>", text, flags=re.S)
print(matches[-1] if matches else "")
PY
}

# Extract assistant text from stream-json log (for tag extraction in watch mode)
extract_text_from_stream_json() {
  local log_file="$1"
  "$PYTHON_BIN" - "$log_file" <<'PY'
import json, sys
path = sys.argv[1]
out = []
try:
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            if ev.get("type") != "assistant":
                continue
            msg = ev.get("message") or {}
            for blk in (msg.get("content") or []):
                if blk.get("type") == "text":
                    out.append(blk.get("text", ""))
except Exception:
    pass
print("\n".join(out))
PY
}

append_progress() {
  local verdict="$1"
  local promise="$2"
  local plan_review_status="${3:-}"
  local task_status="${4:-}"
  local completion_review_status="${5:-}"
  local receipt_exists="0"
  if [[ -n "${REVIEW_RECEIPT_PATH:-}" && -f "$REVIEW_RECEIPT_PATH" ]]; then
    receipt_exists="1"
  fi
  {
    echo "## $(date -u +%Y-%m-%dT%H:%M:%SZ) - iter $iter"
    echo "status=$status epic=${epic_id:-} task=${task_id:-} reason=${reason:-}"
    echo "claude_rc=$claude_rc"
    echo "verdict=${verdict:-}"
    echo "promise=${promise:-}"
    echo "receipt=${REVIEW_RECEIPT_PATH:-} exists=$receipt_exists"
    echo "plan_review_status=${plan_review_status:-}"
    echo "completion_review_status=${completion_review_status:-}"
    echo "task_status=${task_status:-}"
    echo "iter_log=$iter_log"
    echo "last_output:"
    tail -n 10 "$iter_log" || true
    echo "---"
  } >> "$PROGRESS_FILE"
}

# Write completion marker to progress.txt (MUST match find_active_runs() detection in flowctl.py)
write_completion_marker() {
  local reason="${1:-DONE}"
  {
    echo ""
    echo "completion_reason=$reason"
    echo "promise=COMPLETE"  # CANONICAL - must match flowctl.py substring search
  } >> "$PROGRESS_FILE"
}

# Check PAUSE/STOP sentinel files
check_sentinels() {
  local pause_file="$RUN_DIR/PAUSE"
  local stop_file="$RUN_DIR/STOP"

  # Check for stop first (exit immediately, keep file for audit)
  if [[ -f "$stop_file" ]]; then
    log "STOP sentinel detected, exiting gracefully"
    ui_fail "STOP sentinel detected"
    write_completion_marker "STOPPED"
    exit 0
  fi

  # Check for pause (log once, wait in loop, re-check STOP while waiting)
  if [[ -f "$pause_file" ]]; then
    log "PAUSED - waiting for resume..."
    while [[ -f "$pause_file" ]]; do
      # Re-check STOP while paused so external stop works
      if [[ -f "$stop_file" ]]; then
        log "STOP sentinel detected while paused, exiting gracefully"
        ui_fail "STOP sentinel detected"
        write_completion_marker "STOPPED"
        exit 0
      fi
      sleep 5
    done
    log "Resumed"
  fi
}

init_branches_file() {
  if [[ -f "$BRANCHES_FILE" ]]; then return; fi
  local base_branch
  base_branch="$(git -C "$ROOT_DIR" rev-parse --abbrev-ref HEAD 2>/dev/null || true)"
  "$PYTHON_BIN" - "$BRANCHES_FILE" "$base_branch" <<'PY'
import json, sys
path, base = sys.argv[1], sys.argv[2]
data = {"base_branch": base, "run_branch": ""}
with open(path, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, sort_keys=True)
PY
}

get_base_branch() {
  "$PYTHON_BIN" - "$BRANCHES_FILE" <<'PY'
import json, sys
try:
    with open(sys.argv[1], encoding="utf-8") as f:
        data = json.load(f)
    print(data.get("base_branch", ""))
except FileNotFoundError:
    print("")
PY
}

get_run_branch() {
  "$PYTHON_BIN" - "$BRANCHES_FILE" <<'PY'
import json, sys
try:
    with open(sys.argv[1], encoding="utf-8") as f:
        data = json.load(f)
    print(data.get("run_branch", ""))
except FileNotFoundError:
    print("")
PY
}

set_run_branch() {
  "$PYTHON_BIN" - "$BRANCHES_FILE" "$1" <<'PY'
import json, sys
path, branch = sys.argv[1], sys.argv[2]
data = {"base_branch": "", "run_branch": ""}
try:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
except FileNotFoundError:
    pass
data["run_branch"] = branch
with open(path, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, sort_keys=True)
PY
}

list_epics_from_file() {
  "$PYTHON_BIN" - "$EPICS_FILE" <<'PY'
import json, sys
path = sys.argv[1]
if not path:
    sys.exit(0)
try:
    data = json.load(open(path, encoding="utf-8"))
except FileNotFoundError:
    sys.exit(0)
epics = data.get("epics", []) or []
print(" ".join(epics))
PY
}

epic_all_tasks_done() {
  "$PYTHON_BIN" - "$1" <<'PY'
import json, sys
try:
    data = json.loads(sys.argv[1])
except json.JSONDecodeError:
    print("0")
    sys.exit(0)
tasks = data.get("tasks", []) or []
if not tasks:
    print("0")
    sys.exit(0)
for t in tasks:
    if t.get("status") != "done":
        print("0")
        sys.exit(0)
print("1")
PY
}

# Get list of open (non-done) epic IDs from flowctl epics --json
list_open_epics() {
  local tmpfile
  tmpfile="$(mktemp)"
  "$FLOWCTL" epics --json 2>/dev/null > "$tmpfile"
  "$PYTHON_BIN" - "$tmpfile" <<'PY'
import sys, json
try:
    with open(sys.argv[1]) as f:
        data = json.load(f)
    for e in data.get('epics', []):
        if e.get('status') != 'done':
            print(e.get('id', ''))
except: pass
PY
  rm -f "$tmpfile"
}

maybe_close_epics() {
  local epics json status all_done review_status
  if [[ -n "$EPICS_FILE" ]]; then
    # Scoped run: use epic list from file
    epics="$(list_epics_from_file)"
  else
    # Unscoped run: get all open epics from flowctl
    epics="$(list_open_epics)"
  fi
  [[ -z "$epics" ]] && return 0
  for epic in $epics; do
    json="$("$FLOWCTL" show "$epic" --json 2>/dev/null || true)"
    [[ -z "$json" ]] && continue
    status="$(json_get status "$json")"
    [[ "$status" == "done" ]] && continue
    all_done="$(epic_all_tasks_done "$json")"
    if [[ "$all_done" == "1" ]]; then
      # Gate on completion review if enabled
      if [[ "$COMPLETION_REVIEW" != "none" ]]; then
        review_status="$(json_get completion_review_status "$json")"
        if [[ "$review_status" != "ship" ]]; then
          # Don't close - selector will return completion_review status
          continue
        fi
        # Also verify receipt exists (ralph.sh enforces, not just guard)
        if ! verify_receipt "$RECEIPTS_DIR/completion-${epic}.json" "completion_review" "$epic"; then
          continue
        fi
      fi
      "$FLOWCTL" epic close "$epic" --json >/dev/null 2>&1 || true
    fi
  done
}

verify_receipt() {
  local path="$1"
  local kind="$2"
  local id="$3"
  [[ -f "$path" ]] || return 1
  "$PYTHON_BIN" - "$path" "$kind" "$id" <<'PY'
import json, sys
path, kind, rid = sys.argv[1], sys.argv[2], sys.argv[3]
try:
    data = json.load(open(path, encoding="utf-8"))
except Exception:
    sys.exit(1)
if data.get("type") != kind:
    sys.exit(1)
if data.get("id") != rid:
    sys.exit(1)
sys.exit(0)
PY
}

# Read verdict field from receipt file (returns empty string if not found/error)
read_receipt_verdict() {
  local path="$1"
  [[ -f "$path" ]] || return 0
  "$PYTHON_BIN" - "$path" <<'PY'
import json, sys
try:
    data = json.load(open(sys.argv[1], encoding="utf-8"))
    print(data.get("verdict", ""))
except Exception:
    pass
PY
}

# Create/switch to run branch (once at start, all epics work here)
ensure_run_branch() {
  if [[ "$BRANCH_MODE" != "new" ]]; then
    return
  fi
  init_branches_file
  local branch
  branch="$(get_run_branch)"
  if [[ -n "$branch" ]]; then
    # Already on run branch (resumed run)
    git -C "$ROOT_DIR" checkout "$branch" >/dev/null 2>&1 || true
    return
  fi
  # Create new run branch from current position
  branch="ralph-${RUN_ID}"
  set_run_branch "$branch"
  git -C "$ROOT_DIR" checkout -b "$branch" >/dev/null 2>&1
}

EPICS_FILE=""
if [[ -n "${EPICS// }" ]]; then
  EPICS_FILE="$RUN_DIR/run.json"
  write_epics_file "$EPICS" > "$EPICS_FILE"
fi

ui_header
ui_config
ui_version_check

# Create run branch once at start (all epics work on same branch)
ensure_run_branch

iter=1
while (( iter <= MAX_ITERATIONS )); do
  iter_log="$RUN_DIR/iter-$(printf '%03d' "$iter").log"

  # Check for pause/stop at start of iteration (before work selection)
  check_sentinels

  # Close any epics with all tasks done BEFORE calling selector
  # This ensures dependent epics become unblocked in the same iteration
  maybe_close_epics

  selector_args=("$FLOWCTL" next --json)
  [[ -n "$EPICS_FILE" ]] && selector_args+=(--epics-file "$EPICS_FILE")
  [[ "$REQUIRE_PLAN_REVIEW" == "1" ]] && selector_args+=(--require-plan-review)
  [[ "$COMPLETION_REVIEW" != "none" ]] && selector_args+=(--require-completion-review)

  selector_json="$("${selector_args[@]}")"
  status="$(json_get status "$selector_json")"
  epic_id="$(json_get epic "$selector_json")"
  task_id="$(json_get task "$selector_json")"
  reason="$(json_get reason "$selector_json")"

  log "iter $iter status=$status epic=${epic_id:-} task=${task_id:-} reason=${reason:-}"
  ui_iteration "$iter" "$status" "${epic_id:-}" "${task_id:-}"

  if [[ "$status" == "none" ]]; then
    if [[ "$reason" == "blocked_by_epic_deps" ]]; then
      log "blocked by epic deps"
    fi
    # maybe_close_epics already called at start of iteration
    ui_complete
    write_completion_marker "NO_WORK"
    exit 0
  fi

  # Export iteration for receipt tracking
  export RALPH_ITERATION="$iter"

  if [[ "$status" == "plan" ]]; then
    export EPIC_ID="$epic_id"
    export PLAN_REVIEW
    export REQUIRE_PLAN_REVIEW
    export FLOW_REVIEW_BACKEND="$PLAN_REVIEW"  # Skills read this
    if [[ "$PLAN_REVIEW" != "none" ]]; then
      export REVIEW_RECEIPT_PATH="$RECEIPTS_DIR/plan-${epic_id}.json"
    else
      unset REVIEW_RECEIPT_PATH
    fi
    log "plan epic=$epic_id review=$PLAN_REVIEW receipt=${REVIEW_RECEIPT_PATH:-} require=$REQUIRE_PLAN_REVIEW"
    ui_plan_review "$PLAN_REVIEW" "$epic_id"
    prompt="$(render_template "$SCRIPT_DIR/prompt_plan.md")"
  elif [[ "$status" == "work" ]]; then
    epic_id="${task_id%%.*}"
    export TASK_ID="$task_id"
    BRANCH_MODE_EFFECTIVE="$BRANCH_MODE"
    if [[ "$BRANCH_MODE" == "new" ]]; then
      BRANCH_MODE_EFFECTIVE="current"
    fi
    export BRANCH_MODE_EFFECTIVE
    export WORK_REVIEW
    export FLOW_REVIEW_BACKEND="$WORK_REVIEW"  # Skills read this
    if [[ "$WORK_REVIEW" != "none" ]]; then
      export REVIEW_RECEIPT_PATH="$RECEIPTS_DIR/impl-${task_id}.json"
    else
      unset REVIEW_RECEIPT_PATH
    fi
    log "work task=$task_id review=$WORK_REVIEW receipt=${REVIEW_RECEIPT_PATH:-} branch=$BRANCH_MODE_EFFECTIVE"
    ui_impl_review "$WORK_REVIEW" "$task_id"
    prompt="$(render_template "$SCRIPT_DIR/prompt_work.md")"
  elif [[ "$status" == "completion_review" ]]; then
    export EPIC_ID="$epic_id"
    export COMPLETION_REVIEW
    export FLOW_REVIEW_BACKEND="$COMPLETION_REVIEW"  # Skills read this
    if [[ "$COMPLETION_REVIEW" != "none" ]]; then
      export REVIEW_RECEIPT_PATH="$RECEIPTS_DIR/completion-${epic_id}.json"
    else
      unset REVIEW_RECEIPT_PATH
    fi
    log "completion_review epic=$epic_id review=$COMPLETION_REVIEW receipt=${REVIEW_RECEIPT_PATH:-}"
    ui_completion_review "$COMPLETION_REVIEW" "$epic_id"
    prompt="$(render_template "$SCRIPT_DIR/prompt_completion.md")"
  else
    fail "invalid selector status: $status"
  fi

  export FLOW_RALPH="1"
  claude_args=(-p)
  # Always use stream-json for logs (TUI needs it), watch mode only controls terminal display
  claude_args+=(--output-format stream-json)

  # Autonomous mode system prompt - critical for preventing drift
  claude_args+=(--append-system-prompt "AUTONOMOUS MODE ACTIVE (FLOW_RALPH=1). You are running unattended. CRITICAL RULES:
1. EXECUTE COMMANDS EXACTLY as shown in prompts. Do not paraphrase or improvise.
2. VERIFY OUTCOMES by running the verification commands (flowctl show, git status).
3. NEVER CLAIM SUCCESS without proof. If flowctl done was not run, the task is NOT done.
4. COPY TEMPLATES VERBATIM - receipt JSON must match exactly including all fields.
5. USE SKILLS AS SPECIFIED - invoke /flow-next:impl-review, do not improvise review prompts.
Violations break automation and leave the user with incomplete work. Be precise, not creative.")

  [[ -n "${MAX_TURNS:-}" ]] && claude_args+=(--max-turns "$MAX_TURNS")
  [[ "$YOLO" == "1" ]] && claude_args+=(--dangerously-skip-permissions)
  [[ -n "${FLOW_RALPH_CLAUDE_PLUGIN_DIR:-}" ]] && claude_args+=(--plugin-dir "$FLOW_RALPH_CLAUDE_PLUGIN_DIR")
  [[ -n "${FLOW_RALPH_CLAUDE_MODEL:-}" ]] && claude_args+=(--model "$FLOW_RALPH_CLAUDE_MODEL")
  [[ -n "${FLOW_RALPH_CLAUDE_SESSION_ID:-}" ]] && claude_args+=(--session-id "$FLOW_RALPH_CLAUDE_SESSION_ID")
  [[ -n "${FLOW_RALPH_CLAUDE_PERMISSION_MODE:-}" ]] && claude_args+=(--permission-mode "$FLOW_RALPH_CLAUDE_PERMISSION_MODE")
  [[ "${FLOW_RALPH_CLAUDE_NO_SESSION_PERSISTENCE:-}" == "1" ]] && claude_args+=(--no-session-persistence)
  if [[ -n "${FLOW_RALPH_CLAUDE_DEBUG:-}" ]]; then
    if [[ "${FLOW_RALPH_CLAUDE_DEBUG}" == "1" ]]; then
      claude_args+=(--debug)
    else
      claude_args+=(--debug "$FLOW_RALPH_CLAUDE_DEBUG")
    fi
  fi
  [[ "${FLOW_RALPH_CLAUDE_VERBOSE:-}" == "1" ]] && claude_args+=(--verbose)

  # Block Explore subagent auto-delegation - causes READ-ONLY failures in autonomous mode
  # Worker already has disallowedTools: Task but CLI-level is more reliable (precedence 2 vs 6)
  # See: https://code.claude.com/docs/en/sub-agents#disable-specific-subagents
  claude_args+=(--disallowedTools "Task(Explore)")

  ui_waiting
  claude_out=""
  set +e
  if [[ "$WATCH_MODE" == "verbose" ]]; then
    # Full output: stream through filter with --verbose to show text/thinking
    [[ ! " ${claude_args[*]} " =~ " --verbose " ]] && claude_args+=(--verbose)
    echo ""
    if [[ -n "$TIMEOUT_CMD" ]]; then
      $TIMEOUT_CMD "$WORKER_TIMEOUT" "$CLAUDE_BIN" "${claude_args[@]}" "$prompt" 2>&1 | tee "$iter_log" | "$SCRIPT_DIR/watch-filter.py" --verbose
    else
      "$CLAUDE_BIN" "${claude_args[@]}" "$prompt" 2>&1 | tee "$iter_log" | "$SCRIPT_DIR/watch-filter.py" --verbose
    fi
    claude_rc=${PIPESTATUS[0]}
    claude_out="$(cat "$iter_log")"
  elif [[ "$WATCH_MODE" == "tools" ]]; then
    # Filtered output: stream-json through watch-filter.py
    # Add --verbose only if not already set (needed for tool visibility)
    [[ ! " ${claude_args[*]} " =~ " --verbose " ]] && claude_args+=(--verbose)
    if [[ -n "$TIMEOUT_CMD" ]]; then
      $TIMEOUT_CMD "$WORKER_TIMEOUT" "$CLAUDE_BIN" "${claude_args[@]}" "$prompt" 2>&1 | tee "$iter_log" | "$SCRIPT_DIR/watch-filter.py"
    else
      "$CLAUDE_BIN" "${claude_args[@]}" "$prompt" 2>&1 | tee "$iter_log" | "$SCRIPT_DIR/watch-filter.py"
    fi
    claude_rc=${PIPESTATUS[0]}
    # Log contains stream-json; verdict/promise extraction handled by fallback logic
    claude_out="$(cat "$iter_log")"
  else
    # Default: quiet mode (stream-json to log, no terminal display)
    # --verbose required for stream-json with --print
    [[ ! " ${claude_args[*]} " =~ " --verbose " ]] && claude_args+=(--verbose)
    if [[ -n "$TIMEOUT_CMD" ]]; then
      $TIMEOUT_CMD "$WORKER_TIMEOUT" "$CLAUDE_BIN" "${claude_args[@]}" "$prompt" > "$iter_log" 2>&1
    else
      "$CLAUDE_BIN" "${claude_args[@]}" "$prompt" > "$iter_log" 2>&1
    fi
    claude_rc=$?
    claude_out="$(cat "$iter_log")"
  fi
  set -e

  # Handle timeout (exit code 124 from timeout command)
  worker_timeout=0
  if [[ -n "$TIMEOUT_CMD" && "$claude_rc" -eq 124 ]]; then
    timeout_id="${task_id:-$epic_id}"
    echo "ralph: worker timed out after ${WORKER_TIMEOUT}s (phase=$status id=$timeout_id iter=$iter)" >> "$iter_log"
    echo "ralph: hint: increase WORKER_TIMEOUT in config.env (current=${WORKER_TIMEOUT}s, try 3600 for complex tasks)" >> "$iter_log"
    log "worker timeout after ${WORKER_TIMEOUT}s phase=$status id=$timeout_id iter=$iter"
    worker_timeout=1
  fi

  log "claude rc=$claude_rc log=$iter_log"

  force_retry=$worker_timeout
  plan_review_status=""
  task_status=""
  impl_receipt_ok="1"
  if [[ "$status" == "plan" && ( "$PLAN_REVIEW" == "rp" || "$PLAN_REVIEW" == "codex" ) ]]; then
    if ! verify_receipt "$REVIEW_RECEIPT_PATH" "plan_review" "$epic_id"; then
      echo "ralph: missing plan review receipt; forcing retry" >> "$iter_log"
      log "missing plan receipt; forcing retry"
      # Delete corrupted/partial receipt so next attempt starts clean
      rm -f "$REVIEW_RECEIPT_PATH" 2>/dev/null || true
      "$FLOWCTL" epic set-plan-review-status "$epic_id" --status needs_work --json >/dev/null 2>&1 || true
      force_retry=1
    fi
    epic_json="$("$FLOWCTL" show "$epic_id" --json 2>/dev/null || true)"
    plan_review_status="$(json_get plan_review_status "$epic_json")"
  fi
  completion_review_status=""
  completion_receipt_ok="1"
  if [[ "$status" == "completion_review" && ( "$COMPLETION_REVIEW" == "rp" || "$COMPLETION_REVIEW" == "codex" ) ]]; then
    if ! verify_receipt "$REVIEW_RECEIPT_PATH" "completion_review" "$epic_id"; then
      echo "ralph: missing completion review receipt; forcing retry" >> "$iter_log"
      log "missing completion receipt; forcing retry"
      completion_receipt_ok="0"
      # Delete corrupted/partial receipt so next attempt starts clean
      rm -f "$REVIEW_RECEIPT_PATH" 2>/dev/null || true
      "$FLOWCTL" epic set-completion-review-status "$epic_id" --status needs_work --json >/dev/null 2>&1 || true
      force_retry=1
    fi
    epic_json="$("$FLOWCTL" show "$epic_id" --json 2>/dev/null || true)"
    completion_review_status="$(json_get completion_review_status "$epic_json")"
    if [[ "$completion_review_status" == "ship" && "$completion_receipt_ok" == "1" ]]; then
      # Completion review passed - epic can now be closed by maybe_close_epics next iteration
      log "completion_review epic=$epic_id SHIP (will close next iteration)"
      force_retry=0
    elif [[ "$completion_review_status" == "needs_work" ]]; then
      # Review found gaps - skill should have handled fix loop but if we get here, retry
      log "completion_review epic=$epic_id NEEDS_WORK; forcing retry"
      force_retry=1
    fi
  fi
  receipt_verdict=""
  if [[ "$status" == "work" && ( "$WORK_REVIEW" == "rp" || "$WORK_REVIEW" == "codex" ) ]]; then
    if ! verify_receipt "$REVIEW_RECEIPT_PATH" "impl_review" "$task_id"; then
      echo "ralph: missing impl review receipt; forcing retry" >> "$iter_log"
      log "missing impl receipt; forcing retry"
      impl_receipt_ok="0"
      # Delete corrupted/partial receipt so next attempt starts clean
      rm -f "$REVIEW_RECEIPT_PATH" 2>/dev/null || true
      force_retry=1
    else
      # Receipt is valid - read the verdict field
      receipt_verdict="$(read_receipt_verdict "$REVIEW_RECEIPT_PATH")"
    fi
  fi

  # Extract verdict/promise for progress log (not displayed in UI)
  # Always parse stream-json since we always use that format now
  claude_text="$(extract_text_from_stream_json "$iter_log")"
  verdict="$(printf '%s' "$claude_text" | extract_tag verdict)"
  promise="$(printf '%s' "$claude_text" | extract_tag promise)"

  # Fallback: derive verdict from flowctl status for logging
  if [[ -z "$verdict" && -n "$plan_review_status" ]]; then
    case "$plan_review_status" in
      ship) verdict="SHIP" ;;
      needs_work) verdict="NEEDS_WORK" ;;
    esac
  fi
  if [[ -z "$verdict" && -n "$completion_review_status" ]]; then
    case "$completion_review_status" in
      ship) verdict="SHIP" ;;
      needs_work) verdict="NEEDS_WORK" ;;
    esac
  fi

  if [[ "$status" == "work" ]]; then
    task_json="$("$FLOWCTL" show "$task_id" --json 2>/dev/null || true)"
    task_status="$(json_get status "$task_json")"
    if [[ "$task_status" == "done" ]]; then
      if [[ "$impl_receipt_ok" == "0" ]]; then
        # Task marked done but receipt missing/invalid - can't trust done status
        # Reset to todo so flowctl next picks it up again (prevents task jumping)
        echo "ralph: task done but receipt missing; resetting to todo" >> "$iter_log"
        log "task $task_id: resetting done→todo (receipt missing)"
        if "$FLOWCTL" task reset "$task_id" --json >/dev/null 2>&1; then
          task_status="todo"
        else
          # Fatal: if reset fails, we'd silently skip this task forever (task jumping)
          echo "ralph: FATAL: failed to reset task $task_id; aborting to prevent task jumping" >> "$iter_log"
          ui_fail "Failed to reset $task_id after missing receipt; aborting to prevent task jumping"
          write_completion_marker "FAILED"
          exit 1
        fi
        force_retry=1
      else
        # Receipt is structurally valid - now check the verdict
        if [[ "$receipt_verdict" == "NEEDS_WORK" ]]; then
          # Task marked done but review said NEEDS_WORK - must retry
          echo "ralph: receipt verdict is NEEDS_WORK; resetting task to todo" >> "$iter_log"
          log "task $task_id: receipt verdict=NEEDS_WORK despite done status; resetting"
          if "$FLOWCTL" task reset "$task_id" --json >/dev/null 2>&1; then
            task_status="todo"
          else
            echo "ralph: FATAL: failed to reset task $task_id; aborting" >> "$iter_log"
            ui_fail "Failed to reset $task_id after NEEDS_WORK verdict; aborting"
            write_completion_marker "FAILED"
            exit 1
          fi
          verdict="NEEDS_WORK"
          force_retry=1
        else
          ui_task_done "$task_id"
          # Use receipt verdict if available, otherwise derive from task completion
          [[ -n "$receipt_verdict" ]] && verdict="$receipt_verdict"
          [[ -z "$verdict" ]] && verdict="SHIP"
          # If we timed out but can prove completion (done + receipt valid + verdict OK), don't retry
          force_retry=0
        fi
      fi
    else
      echo "ralph: task not done; forcing retry" >> "$iter_log"
      log "task $task_id status=$task_status; forcing retry"
      force_retry=1
    fi
  fi
  append_progress "$verdict" "$promise" "$plan_review_status" "$task_status" "$completion_review_status"

  # NEVER honor COMPLETE from worker output (GH-73: premature completion bug)
  # Workers are single-task/single-epic scope. Completion detection happens via
  # the selector returning status=none at the top of the loop. Workers should
  # NEVER output COMPLETE (both prompt_work.md and prompt_plan.md forbid it).
  # If Claude outputs COMPLETE anyway, log it and continue - let selector decide.
  if echo "$claude_text" | grep -q "<promise>COMPLETE</promise>"; then
    echo "ralph: WARNING: COMPLETE promise ignored (invalid in $status context)" >> "$iter_log"
    log "COMPLETE ignored (invalid in $status context) - letting selector decide"
  fi

  exit_code=0
  if echo "$claude_text" | grep -q "<promise>FAIL</promise>"; then
    exit_code=1
  elif echo "$claude_text" | grep -q "<promise>RETRY</promise>"; then
    exit_code=2
  elif [[ "$force_retry" == "1" ]]; then
    exit_code=2
  elif [[ "$claude_rc" -ne 0 && "$task_status" != "done" && "$verdict" != "SHIP" ]]; then
    # Only fail on non-zero exit code if task didn't complete and verdict isn't SHIP
    # This prevents false failures from transient errors (telemetry, model fallback, etc.)
    exit_code=1
  fi

  if [[ "$exit_code" -eq 1 ]]; then
    log "exit=fail"
    ui_fail "Claude returned FAIL promise"
    write_completion_marker "FAILED"
    exit 1
  fi

  if [[ "$exit_code" -eq 2 && "$status" == "work" ]]; then
    if [[ "$worker_timeout" -eq 0 ]]; then
      # Real failure - count against attempts budget
      attempts="$(bump_attempts "$ATTEMPTS_FILE" "$task_id")"
      log "retry task=$task_id attempts=$attempts"
      ui_retry "$task_id" "$attempts" "$MAX_ATTEMPTS_PER_TASK"
      if (( attempts >= MAX_ATTEMPTS_PER_TASK )); then
        reason_file="$RUN_DIR/block-${task_id}.md"
        {
          echo "Auto-blocked after ${attempts} attempts."
          echo "Run: $RUN_ID"
          echo "Full ID: $RUN_ID_FULL"
          echo "Task: $task_id"
          echo ""
          echo "Last output:"
          tail -n 40 "$iter_log" || true
        } > "$reason_file"
        "$FLOWCTL" block "$task_id" --reason-file "$reason_file" --json || true
        ui_blocked "$task_id"
      fi
    else
      # Timeout is infrastructure issue, not code failure - don't count against attempts
      log "timeout retry task=$task_id (not counting against attempts)"
      ui "   ${C_YELLOW}↻ Timeout retry${C_RESET} ${C_DIM}(not counted)${C_RESET}"
    fi
  fi

  # Check for pause/stop after Claude returns (before next iteration)
  check_sentinels

  sleep 2
  iter=$((iter + 1))
done

ui_fail "Max iterations ($MAX_ITERATIONS) reached"
echo "ralph: max iterations reached" >&2
write_completion_marker "MAX_ITERATIONS"
exit 1
