#!/usr/bin/env bash
# Status of the background Ollama coaching job + your labelling progress.
# Usage:  ./status.sh        (from the project root)
#         ./status.sh -w     watch mode, refreshes every 60s

cd "$(dirname "$0")" || exit 1
PY=.venv/bin/python

show() {
  echo "────────────────────────────────────────────────"
  echo " $(date '+%a %d %b %H:%M:%S')"
  echo "────────────────────────────────────────────────"

  # --- Ollama coaching job -------------------------------------------------
  # Find it by name rather than a remembered PID, so this keeps working
  # across restarts.
  PID=$(pgrep -f "coaching.py --generator ollama" | head -1)

  if [ -n "$PID" ]; then
    ELAPSED=$(ps -o etime= -p "$PID" | tr -d ' ')
    echo "ollama job : RUNNING  (pid $PID, $ELAPSED elapsed)"

    # Recover the log path from the process's own stdout handle -- no need to
    # know where it was redirected.
    LOG=$(lsof -p "$PID" 2>/dev/null | awk '$4 ~ /^1[wu]$/ {print $NF}' | head -1)
    if [ -n "$LOG" ] && [ -r "$LOG" ]; then
      LAST=$(grep -oE "[0-9,]+/[0-9,]+ notes \([0-9.]+/s\)" "$LOG" | tail -1)
      echo "progress   : ${LAST:-no milestone yet (they print every 500 notes)}"
      echo "log        : $LOG"
    fi
  else
    echo "ollama job : not running"
    if [ -f data/labels/coaching_notes.csv ]; then
      echo -n "last result: "
      $PY -c "import pandas as pd;d=pd.read_csv('data/labels/coaching_notes.csv');print(len(d),'notes |',d.generated_by.value_counts().to_dict())" 2>/dev/null
    fi
  fi

  # --- Is Ollama itself busy? ---------------------------------------------
  CPU=$(ps aux | awk '/[l]lama-server/ {print $3; exit}')
  echo "ollama cpu : ${CPU:-0}%  $([ "${CPU%%.*}" -gt 5 ] 2>/dev/null && echo '(generating)' || echo '(idle)')"

  # --- Your labelling ------------------------------------------------------
  echo
  $PY - <<'PYEOF' 2>/dev/null
import pandas as pd, pathlib
q = pathlib.Path("data/labels/to_label.csv")
h = pathlib.Path("data/labels/human_labels.csv")
if q.exists():
    total = len(pd.read_csv(q))
    if h.exists():
        done = pd.read_csv(h).drop_duplicates("interaction_id", keep="last")
        bar = "█" * int(len(done)/total*30) + "░" * (30 - int(len(done)/total*30))
        print(f"labelling  : {len(done)}/{total}  [{bar}] {len(done)/total:.0%}")
        print(f"your mix   : {done.human_label.value_counts().to_dict()}")
    else:
        print(f"labelling  : 0/{total} (not started)")
PYEOF
  echo
}

if [ "$1" = "-w" ]; then
  while true; do clear; show; echo "refreshing every 60s — Ctrl-C to stop"; sleep 60; done
else
  show
fi
