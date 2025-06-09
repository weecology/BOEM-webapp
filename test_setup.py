#!/usr/bin/env python3
"""
Test script to verify your GitHub agent setup is working correctly.
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

print("🧪 Testing your DeepForest Agent Setup")
print("=" * 40)

# Test 1: Check Python version
print("\n1️⃣ Checking Python version...")
python_version = sys.version_info
if python_version.major >= 3 and python_version.minor >= 8:
    print(f"✅ Python {python_version.major}.{python_version.minor} - Good!")
else:
    print(f"❌ Python {python_version.major}.{python_version.minor} - Need 3.8+")

# Test 2: Check required packages
print("\n2️⃣ Checking required packages...")
packages = {
    "github": "PyGithub",
    "git": "GitPython", 
    "dotenv": "python-dotenv",
    "openai": "openai"
}

all_good = True
for module, package in packages.items():
    try:
        __import__(module)
        print(f"✅ {package} - Installed")
    except ImportError:
        print(f"❌ {package} - Not installed")
        all_good = False

# Test 3: Check GitHub token
print("\n3️⃣ Checking GitHub token...")
github_token = os.getenv("GITHUB_TOKEN")
if github_token and github_token != "YOUR_TOKEN_HERE":
    # Test the token
    try:
        from github import Github
        g = Github(github_token)
        user = g.get_user()
        username = user.login
        print(f"✅ GitHub token valid - Logged in as: {username}")
        
        # Check DeepForest access
        try:
            repo = g.get_repo("weecology/DeepForest")
            print(f"✅ Can access DeepForest repository")
            
            # Check if user has write access
            if username in [c.login for c in repo.get_collaborators()]:
                print(f"✅ You have collaborator access to DeepForest")
            else:
                print(f"ℹ️  You don't have direct write access - you'll need to fork first")
                
        except Exception as e:
            print(f"❌ Cannot access DeepForest repo: {e}")
            
    except Exception as e:
        print(f"❌ GitHub token invalid: {e}")
else:
    print("❌ GitHub token not set or still default value")
    print("   👉 Edit .env and add your token")

# Test 4: Check OpenAI (optional)
print("\n4️⃣ Checking OpenAI API key (optional)...")
openai_key = os.getenv("OPENAI_API_KEY")
if openai_key and openai_key.strip():
    print("✅ OpenAI API key found (not tested)")
else:
    print("ℹ️  No OpenAI key - agent will work but without AI analysis")

# Test 5: Quick issue fetch
print("\n5️⃣ Testing issue fetch...")
if github_token and github_token != "YOUR_TOKEN_HERE":
    try:
        from github import Github
        g = Github(github_token)
        repo = g.get_repo("weecology/DeepForest")
        issue = repo.get_issue(1064)
        print(f"✅ Successfully fetched issue #1064: {issue.title[:50]}...")
    except Exception as e:
        print(f"❌ Could not fetch issue: {e}")

# Summary
print("\n" + "=" * 40)
print("📊 SUMMARY")
print("=" * 40)

if all_good and github_token and github_token != "YOUR_TOKEN_HERE":
    print("✅ Your setup is ready!")
    print("\n🚀 You can now run:")
    print("   python deepforest_issue_solver.py")
    print("\nOr for analysis only:")
    print("   python deepforest_advanced_solver.py 1064 --analyze-only")
else:
    print("❌ Setup incomplete. Please:")
    if not all_good:
        print("   1. Install missing packages: pip install PyGithub gitpython python-dotenv")
    if not github_token or github_token == "YOUR_TOKEN_HERE":
        print("   2. Add your GitHub token to .env file")
        print("      - Get token at: https://github.com/settings/tokens/new")
        print("      - Edit .env: nano .env")