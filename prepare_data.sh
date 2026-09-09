#!/bin/bash

# Set the base directory and log file
BASE_DIR="/pgsql/boem-webapp/BOEM-webapp"
LOG_FILE="/pgsql/boem-webapp/prepare-data-cronlog.txt"
CONDA_PATH="/home/retrieverdash/miniconda3/etc/profile.d/conda.sh"

cd "$BASE_DIR" || exit 1

# NOTE: this used to be `echo ... > "$LOG_FILE"`, which truncated the log on
# every run, so this text log only ever showed the most recent run — there
# was no way to tell whether/when prepare.py had last run successfully.
# It now appends. This text log is still just a raw stdout dump for a human
# to skim; app/data/logs/prepare_runs.jsonl (written by prepare.py itself,
# via app/utils/run_log.py) is the structured, per-run history meant for
# later analysis. Rotate this file (e.g. logrotate, or truncate manually)
# if it grows too large — it's append-only now and will grow forever otherwise.
echo "Get latest git updates and prepare data $(date)" >> "$LOG_FILE"
source "$CONDA_PATH"
conda activate boem-webapp
python prepare.py >> "$LOG_FILE" 2>&1
echo "Cron ran at $(date) - BOEM webapp preparation complete (exit code $?)" >> "$LOG_FILE"
