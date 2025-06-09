# 🚀 START HERE - DeepForest Issue #1064 Solver

## Run These Commands Now (Copy & Paste)

### 1️⃣ Set Up Environment (2 minutes)

```bash
# Create project and virtual environment
mkdir -p ~/deepforest-agent && cd ~/deepforest-agent
python3 -m venv venv
source venv/bin/activate

# Install packages
pip install --upgrade pip
pip install PyGithub gitpython python-dotenv
```

### 2️⃣ Get Your GitHub Token (2 minutes)

1. Open this link: **https://github.com/settings/tokens/new**
2. Settings:
   - **Note:** DeepForest Agent
   - **Expiration:** 90 days (or your preference)
   - **Scopes:** Check these boxes:
     - ✅ `repo` (all checkboxes under it)
     - ✅ `workflow`
3. Click **"Generate token"** (green button at bottom)
4. **COPY THE TOKEN NOW!** (starts with `ghp_`)

### 3️⃣ Save Your Token (1 minute)

```bash
# Create config file with your token
cat > .env << EOF
GITHUB_TOKEN=PASTE_YOUR_TOKEN_HERE
EOF

# Edit to add your actual token
nano .env
# (Replace PASTE_YOUR_TOKEN_HERE with your token, then Ctrl+X, Y, Enter)
```

### 4️⃣ Create Simple Issue Solver (1 minute)

```bash
# Create a simple solver script
cat > solve_issue_1064.py << 'EOF'
#!/usr/bin/env python3
import os
from github import Github
from dotenv import load_dotenv

# Load token
load_dotenv()
token = os.getenv("GITHUB_TOKEN")

if not token or token == "PASTE_YOUR_TOKEN_HERE":
    print("❌ Please add your GitHub token to .env file!")
    exit(1)

# Connect to GitHub
print("🔗 Connecting to GitHub...")
g = Github(token)

# Get issue details
print("📋 Fetching issue #1064...")
repo = g.get_repo("weecology/DeepForest")
issue = repo.get_issue(1064)

print(f"\n📌 Issue #{issue.number}: {issue.title}")
print(f"🏷️  Labels: {[l.name for l in issue.labels]}")
print(f"\n📝 Description:")
print("-" * 50)
print(issue.body[:500] + "..." if len(issue.body) > 500 else issue.body)
print("-" * 50)

print("\n✅ Setup working! You can access DeepForest issues.")
print("\n🎯 Next: Would you like me to help create a fix for this issue?")
EOF

chmod +x solve_issue_1064.py
```

### 5️⃣ Test Your Setup (10 seconds)

```bash
# Make sure you're in the virtual environment
source venv/bin/activate

# Run the test
python solve_issue_1064.py
```

## 🎉 Success Checklist

If everything worked, you should see:
- ✅ Issue #1064 details printed
- ✅ The issue title and description
- ✅ "Setup working!" message

## 🚨 If Something Goes Wrong

### "No module named 'github'"
```bash
source venv/bin/activate  # Make sure (venv) appears in your prompt
pip install PyGithub
```

### "Please add your GitHub token"
```bash
nano .env  # Add your token and save
```

### "401 Unauthorized"
Your token might be wrong. Get a new one from https://github.com/settings/tokens/new

## 🎯 What's Next?

Once the test works, you can:

1. **See the full agent scripts** I created earlier
2. **Run the advanced solver** to automatically fix issue #1064
3. **Create your first automated PR**

Ready? Let me know when the test script works! 🚀