# 🌲 DeepForest Agent - Complete Beginner's Guide

Welcome! This guide will help you set up and use your first GitHub agent to solve issue #1064 on DeepForest.

## 📋 What You'll Need

1. **A GitHub Account** (you already have this as the DeepForest lead)
2. **Python 3.8+** installed on your system
3. **About 10 minutes** to set everything up

## 🚀 Step 1: Set Up Your Environment

Since you got an environment error, we need to create a virtual environment first:

```bash
# 1. Create a new directory for the agent
mkdir ~/deepforest-agent
cd ~/deepforest-agent

# 2. Create a Python virtual environment
python3 -m venv venv

# 3. Activate the virtual environment
source venv/bin/activate

# You should see (venv) at the beginning of your terminal prompt
```

## 📦 Step 2: Install Required Packages

Now install the packages inside your virtual environment:

```bash
# Make sure you see (venv) in your prompt!
pip install --upgrade pip
pip install PyGithub gitpython python-dotenv openai requests
```

## 🔑 Step 3: Get Your GitHub Token

You need a GitHub Personal Access Token to let the agent work with GitHub:

1. Go to: https://github.com/settings/tokens/new
2. Give it a name like "DeepForest Agent"
3. Select these scopes:
   - ✅ **repo** (Full control of private repositories)
   - ✅ **workflow** (Update GitHub Action workflows)
4. Click "Generate token"
5. **COPY THE TOKEN NOW!** (You won't see it again)

## 📝 Step 4: Create Your Configuration

Create a file called `.env` in your project directory:

```bash
# Create the .env file
cat > .env << 'EOF'
# Paste your GitHub token here
GITHUB_TOKEN=paste_your_token_here

# Optional: If you have an OpenAI API key
OPENAI_API_KEY=your_openai_key_here
EOF

# Edit the file to add your actual token
nano .env  # or use your favorite editor
```

## 🎯 Step 5: Download the Agent Scripts

Copy all the Python scripts I created into your `~/deepforest-agent` directory:

```bash
# Make sure you're in the right directory
cd ~/deepforest-agent

# The scripts you need:
# - github_pr_agent.py (basic agent)
# - ml_github_agent.py (ML-specific agent)
# - deepforest_issue_solver.py (DeepForest-specific solver)
# - deepforest_advanced_solver.py (advanced solver)
```

## 🏃 Step 6: Run Your First Agent Command

Let's start with just analyzing issue 1064:

```bash
# Make sure your virtual environment is activated
source venv/bin/activate

# First, let's just look at the issue without making changes
python deepforest_advanced_solver.py 1064 --analyze-only
```

This will:
- Connect to GitHub
- Fetch issue 1064
- Analyze it
- Show you what it found

## ✨ Step 7: Create Your First PR

If the analysis looks good, you can create a PR:

```bash
# This will analyze and create a PR
python deepforest_issue_solver.py
```

The script will:
1. Show you issue 1064 details
2. Ask for confirmation
3. Create a branch
4. Make the fixes
5. Create a pull request

## 🛠️ Troubleshooting

### "GITHUB_TOKEN not set"
- Make sure your `.env` file has the token
- Make sure you're in the right directory

### "Permission denied"
- Check that your GitHub token has the right permissions
- As the repo owner, you should have full access

### "Module not found"
- Make sure your virtual environment is activated: `source venv/bin/activate`
- Reinstall packages: `pip install PyGithub gitpython python-dotenv`

## 💡 Quick Commands Reference

```bash
# Always start by activating your environment
cd ~/deepforest-agent
source venv/bin/activate

# Analyze an issue
python deepforest_advanced_solver.py 1064 --analyze-only

# Create a PR for an issue
python deepforest_issue_solver.py

# Check your GitHub token works
python -c "from github import Github; g = Github('your_token'); print(g.get_user().login)"
```

## 🎓 Understanding What Happens

When you run the agent:

1. **Connects to GitHub** using your token
2. **Fetches the issue** (#1064) from weecology/DeepForest
3. **Analyzes the issue** to understand what needs fixing
4. **Generates code changes** based on the issue type
5. **Creates a new branch** (like `fix-issue-1064`)
6. **Commits the changes** with a descriptive message
7. **Pushes to GitHub** and creates a pull request

## 🚦 Your First Run Checklist

- [ ] Virtual environment created and activated
- [ ] Packages installed (PyGithub, gitpython, etc.)
- [ ] GitHub token created and saved in `.env`
- [ ] Agent scripts downloaded to your directory
- [ ] Test with `--analyze-only` first
- [ ] Review the proposed changes
- [ ] Create your first automated PR!

## 📞 Need Help?

If you get stuck:
1. Check you're in the virtual environment (see `(venv)` in prompt)
2. Verify your GitHub token works
3. Try the analyze-only mode first
4. The scripts will save fixes locally if PR creation fails

Good luck with your first agent-powered PR! 🚀