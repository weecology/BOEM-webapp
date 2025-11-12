#!/bin/bash

BASE_DIR="/pgsql/boem-webapp/BOEM-webapp"
LOG_FILE="/pgsql/boem-webapp/update-repo-cronlog.txt"
PROTECTED_FILE="app/data/annotations.csv"
COMMIT_TRACKER="/pgsql/boem-webapp/last-annotations-commit-date.txt"

cd "$BASE_DIR" || exit 1

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" >> "$LOG_FILE"
}

log "=== Script run initiated ==="

###############################################################
# 1. Update repository (using soft reset to avoid file changes)
###############################################################
log "Fetching remote changes..."
if ! git fetch origin >> "$LOG_FILE" 2>&1; then
    log "ERROR: Failed to fetch from origin"
    exit 1
fi

# Check if we're behind remote
if git diff --quiet HEAD origin/main; then
    log "Repository is already up-to-date"
else
    log "Remote has updates. Updating with soft reset (no file changes)..."
    git reset --soft origin/main >> "$LOG_FILE" 2>&1
    log "Repository updated successfully"
fi

#######################################################
# 2. Daily commit logic (once per day if changes exist)
#######################################################
NOW_DATE=$(date '+%Y-%m-%d')

# Check if already committed today
if [ -f "$COMMIT_TRACKER" ] && [ "$(cat "$COMMIT_TRACKER")" = "$NOW_DATE" ]; then
    log "Daily commit already performed today ($NOW_DATE). Skipping."
else
    # Check if file has changes
    if [ -f "$PROTECTED_FILE" ] && ! git diff --quiet HEAD -- "$PROTECTED_FILE" 2>/dev/null; then
        log "Changes detected in $PROTECTED_FILE. Performing daily commit..."
        git add "$PROTECTED_FILE"
        COMMIT_MSG="Annotations update for $NOW_DATE"

        if git commit -m "$COMMIT_MSG" >> "$LOG_FILE" 2>&1; then
            log "Committed $PROTECTED_FILE successfully"
            if git push origin main >> "$LOG_FILE" 2>&1; then
                log "Pushed daily commit to remote successfully"
            else
                log "WARNING: Failed to push daily commit to remote"
            fi
            # Record commit date
            echo "$NOW_DATE" > "$COMMIT_TRACKER"
        else
            log "ERROR: Failed to commit $PROTECTED_FILE"
        fi
    else
        log "No changes in $PROTECTED_FILE. No commit needed today."
    fi
fi

log "=== Script run completed ==="
