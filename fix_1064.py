#!/usr/bin/env python3
"""
Fix DeepForest Issue #1064 in the simplest way possible.

Setup:
1. pip install PyGithub gitpython python-dotenv
2. Create .env file with: GITHUB_TOKEN=your_github_token
3. Run this script
"""

import os
from github import Github
from dotenv import load_dotenv

# Load token
load_dotenv()
token = os.getenv("GITHUB_TOKEN")
if not token:
    print("❌ Add GITHUB_TOKEN to .env file")
    print("Get token at: https://github.com/settings/tokens/new (select 'repo' scope)")
    exit(1)

# Connect and get issue
print("Fetching issue #1064...")
g = Github(token)
repo = g.get_repo("weecology/DeepForest")
issue = repo.get_issue(1064)

print(f"\n📋 Issue #{issue.number}: {issue.title}")
print(f"Labels: {[l.name for l in issue.labels]}")
print(f"\nDescription:\n{'-'*60}")
print(issue.body)
print(f"{'-'*60}\n")

# Now we need to see what the issue is asking for to write the fix
# Once you run this and see the issue, we can add the fix code