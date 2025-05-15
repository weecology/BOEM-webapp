#!/bin/bash


BASE_DIR="/pgsql/boem-webapp/BOEM-webapp"
LOG_FILE="/pgsql/boem-webapp/update-repo-cronlog.txt"

cd "$BASE_DIR" || exit 1
echo "Updating repo on $(date)" > "$LOG_FILE"

# Check for updates
changed=0
git remote update && git status -uno | grep -q 'Your branch is behind' && changed=1

if [ $changed = 1 ]; then
    git reset --hard origin/main && git pull
    echo "Updated $(date)" >> "$LOG_FILE"
else
    echo "Up-to-date $(date)" >> "$LOG_FILE"
fi
