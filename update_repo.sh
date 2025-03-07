changed=0
git remote update && git status -uno | grep -q 'Your branch is behind' && changed=1
if [ $changed = 1 ]; then
    git reset --hard origin/main && git pull
    echo "Updated $(date)" >> ../boem_cron.log
else
    echo "Up-to-date $(date)" >> ../boem_cron.log
fi
