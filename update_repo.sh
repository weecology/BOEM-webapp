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

# Setup on weecologydeploy user
source ~/.weecologybot/githubdeploytoken.txt
git config user.email "weecologydeploy@weecology.org"
git config user.name "Weecology Deploy Bot"
git remote add deploy https://${GITHUBTOKEN}@github.com/weecology/BOEM-webapp.git 2>&1 || echo "Deploy already added"

#####################################################################
# 1. Update repository (safe process without touching PROTECTED_FILE)
#####################################################################
log "Fetching remote changes..."
if ! git fetch deploy >> "$LOG_FILE" 2>&1; then
    log "ERROR: Failed to fetch from deploy"
    exit 1
fi

# Check if we're behind remote main branch
if git diff --quiet HEAD deploy/main; then
    log "Repository is already up-to-date"
else
    log "Local branch is behind deploy/main. Starting safe update process..."
    
    # Safety check: If last commit is "WIP", previous run failed
    LAST_COMMIT_MSG=$(git log -1 --pretty=%B)
    if [ "$LAST_COMMIT_MSG" = "WIP" ]; then
        log "ERROR: Previous update run failed! Last commit is still 'WIP'."
        log "ERROR: ***********rebase process did not complete successfully."
        exit 1
    fi
    
    # Step 1: Commit PROTECTED_FILE as WIP to preserve it
    if [ -f "$PROTECTED_FILE" ] && ! git diff --quiet HEAD -- "$PROTECTED_FILE" 2>/dev/null; then
        git add "$PROTECTED_FILE"
        
        if git commit -m "WIP" >> "$LOG_FILE" 2>&1; then
            log "WIP commit successful. Cleaning other uncommitted changes..."
            # Clean out all other uncommitted changes
            git reset --hard HEAD >> "$LOG_FILE" 2>&1

            log "Rebasing onto deploy/main..."
            if git rebase deploy/main >> "$LOG_FILE" 2>&1; then
                log "Rebase successful"
                # Step 3: Undo the WIP commit but keep the changes
                # Check if the last commit is "WIP"
                LAST_COMMIT_MSG=$(git log -1 --pretty=%B)
                if [ "$LAST_COMMIT_MSG" = "WIP" ]; then
                    log "Removing WIP commit while preserving changes..."
                    git reset --soft HEAD~1 >> "$LOG_FILE" 2>&1
                    log "WIP commit removed, changes preserved"
                fi
                
                # Step 4: Unstage everything
                git reset >> "$LOG_FILE" 2>&1
                log "Changes unstaged. PROTECTED_FILE remains untouched in working directory."
                log "Repository updated successfully (PROTECTED_FILE preserved)"
            else
                log "ERROR: Rebase failed. Repository may need manual intervention."
                # If rebase failed, we might need to abort
                if git rebase --abort >> "$LOG_FILE" 2>&1; then
                    log "Rebase aborted. Repository state restored."
                fi
            fi
        else
            log "WARNING: WIP commit failed. Skipping update to preserve PROTECTED_FILE."
        fi
    else
        # PROTECTED_FILE has no changes, just reset --hard to update
        log "PROTECTED_FILE has no changes. Resetting to deploy/main..."
        git reset --hard deploy/main >> "$LOG_FILE" 2>&1
        log "Repository updated successfully"
    fi
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
            if git push deploy main >> "$LOG_FILE" 2>&1; then
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
echo "" >> "$LOG_FILE"
