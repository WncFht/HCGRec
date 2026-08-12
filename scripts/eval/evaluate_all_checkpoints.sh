#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

PYTHON_BIN="python"
SIDECAR_MODULE="hcgrec.evaluate_all_checkpoints_sidecar"
SIDECAR_PY="${REPO_ROOT}/src/hcgrec/evaluate_all_checkpoints_sidecar.py"
MANAGER_DIR="${REPO_ROOT}/log/evaluate_all_checkpoints"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/eval/evaluate_all_checkpoints.sh once [sidecar args...]
  bash scripts/eval/evaluate_all_checkpoints.sh run [--instance NAME] [sidecar args...]
  bash scripts/eval/evaluate_all_checkpoints.sh start [--instance NAME] [sidecar args...]
  bash scripts/eval/evaluate_all_checkpoints.sh stop [--instance NAME]
  bash scripts/eval/evaluate_all_checkpoints.sh status [--instance NAME]
  bash scripts/eval/evaluate_all_checkpoints.sh tail [--instance NAME]

Examples:
  bash scripts/eval/evaluate_all_checkpoints.sh once --dry-run --include-rl 0
  bash scripts/eval/evaluate_all_checkpoints.sh once --model-filter Games --force-reeval 1
  bash scripts/eval/evaluate_all_checkpoints.sh run --instance remote_eval --poll-interval-seconds 60
EOF
}

require_sidecar() {
  if [[ ! -f "$SIDECAR_PY" ]]; then
    echo "[ERROR] sidecar script not found: $SIDECAR_PY"
    exit 1
  fi
}

is_running() {
  local pid_file="$1"
  if [[ ! -f "$pid_file" ]]; then
    return 1
  fi

  local pid=""
  pid="$(cat "$pid_file" 2>/dev/null || true)"
  [[ -z "$pid" ]] && return 1
  kill -0 "$pid" 2>/dev/null
}

if [[ $# -eq 0 ]]; then
  set -- once
fi

COMMAND="$1"
shift

case "$COMMAND" in
  once)
    require_sidecar
    exec "$PYTHON_BIN" -m "$SIDECAR_MODULE" once "$@"
    ;;

  run|start|stop|status|tail)
    require_sidecar
    mkdir -p "$MANAGER_DIR"

    INSTANCE="default"
    FORWARD_ARGS=()
    while [[ $# -gt 0 ]]; do
      case "$1" in
        --instance)
          INSTANCE="$2"
          shift 2
          ;;
        --instance=*)
          INSTANCE="${1#*=}"
          shift
          ;;
        *)
          FORWARD_ARGS+=("$1")
          shift
          ;;
      esac
    done

    INSTANCE_SAFE="$(echo "$INSTANCE" | tr '/ ' '__')"
    PID_FILE="$MANAGER_DIR/${INSTANCE_SAFE}.pid"
    LOG_FILE="$MANAGER_DIR/${INSTANCE_SAFE}.log"

    case "$COMMAND" in
      run)
        exec "$PYTHON_BIN" -m "$SIDECAR_MODULE" watch "${FORWARD_ARGS[@]}"
        ;;

      start)
        if is_running "$PID_FILE"; then
          echo "[INFO] watcher already running. instance=$INSTANCE pid=$(cat "$PID_FILE")"
          echo "[INFO] log=$LOG_FILE"
          exit 0
        fi

        nohup "$PYTHON_BIN" -m "$SIDECAR_MODULE" watch "${FORWARD_ARGS[@]}" >> "$LOG_FILE" 2>&1 &
        PID=$!
        echo "$PID" > "$PID_FILE"
        echo "[INFO] watcher started. instance=$INSTANCE pid=$PID"
        echo "[INFO] log=$LOG_FILE"
        ;;

      stop)
        if ! is_running "$PID_FILE"; then
          echo "[INFO] watcher not running. instance=$INSTANCE"
          rm -f "$PID_FILE"
          exit 0
        fi

        PID="$(cat "$PID_FILE")"
        kill "$PID" 2>/dev/null || true
        for _ in {1..20}; do
          if ! kill -0 "$PID" 2>/dev/null; then
            break
          fi
          sleep 0.2
        done
        if kill -0 "$PID" 2>/dev/null; then
          echo "[WARN] process still alive, sending SIGKILL pid=$PID"
          kill -9 "$PID" 2>/dev/null || true
        fi
        rm -f "$PID_FILE"
        echo "[INFO] watcher stopped. instance=$INSTANCE"
        ;;

      status)
        if is_running "$PID_FILE"; then
          echo "[INFO] watcher running. instance=$INSTANCE pid=$(cat "$PID_FILE")"
          echo "[INFO] log=$LOG_FILE"
          exit 0
        fi

        echo "[INFO] watcher not running. instance=$INSTANCE"
        echo "[INFO] expected pid file: $PID_FILE"
        exit 1
        ;;

      tail)
        touch "$LOG_FILE"
        echo "[INFO] tailing log: $LOG_FILE"
        exec tail -n 100 -f "$LOG_FILE"
        ;;
    esac
    ;;

  -h|--help|help)
    usage
    ;;

  *)
    echo "[ERROR] unknown command: $COMMAND"
    usage
    exit 1
    ;;
esac
