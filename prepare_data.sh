#!/bin/bash

# Set the base directory and log file
BASE_DIR="/pgsql/boem-webapp/BOEM-webapp"
LOG_FILE="/pgsql/boem-webapp/prepare-data-cronlog.txt"
CONDA_PATH="/home/retrieverdash/miniconda3/etc/profile.d/conda.sh"

cd "$BASE_DIR" || exit 1
echo "Get latest git updates and prepare data $(date)" > "$LOG_FILE"
source "$CONDA_PATH"
conda activate boem-webapp
python prepare.py >> "$LOG_FILE"
echo "Cron ran at $(date) - BOEM webapp preparation complete" >> "$LOG_FILE"
