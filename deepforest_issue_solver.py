#!/usr/bin/env python3
"""
Script to solve issue #1064 on DeepForest repository.
This will analyze the issue and create an appropriate fix.
"""

import os
import sys
from github import Github
from github_pr_agent import GitHubPRAgent
from ml_github_agent import MLGitHubAgent
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def analyze_and_fix_deepforest_issue():
    """Analyze and fix issue 1064 on DeepForest."""
    
    # Get credentials
    github_token = os.getenv("GITHUB_TOKEN")
    openai_key = os.getenv("OPENAI_API_KEY")
    
    if not github_token:
        print("❌ Please set GITHUB_TOKEN environment variable")
        return
    
    # Repository details
    REPO_NAME = "weecology/DeepForest"
    REPO_URL = "https://github.com/weecology/DeepForest.git"
    ISSUE_NUMBER = 1064
    
    print(f"🌲 DeepForest Issue Solver")
    print(f"📋 Working on issue #{ISSUE_NUMBER}\n")
    
    # Since DeepForest is an ML project, use the ML-specific agent
    agent = MLGitHubAgent(github_token, openai_key)
    
    # First, let's get and analyze the issue
    try:
        # Get issue details
        g = Github(github_token)
        repo = g.get_repo(REPO_NAME)
        issue = repo.get_issue(ISSUE_NUMBER)
        
        print(f"Issue #{ISSUE_NUMBER}: {issue.title}")
        print(f"State: {issue.state}")
        print(f"Labels: {[l.name for l in issue.labels]}")
        print(f"\nDescription:")
        print("-" * 50)
        print(issue.body)
        print("-" * 50)
        
        # Analyze the issue
        issue_details = {
            "number": issue.number,
            "title": issue.title,
            "body": issue.body,
            "labels": [l.name for l in issue.labels]
        }
        
        # Get AI analysis if OpenAI key is available
        if openai_key:
            print("\n🤖 Analyzing issue with AI...")
            analysis = agent.analyze_ml_issue(issue_details)
            print(f"Analysis type: {analysis['type']}")
            if analysis['type'] == 'ml':
                print(f"ML Analysis: {analysis['analysis']}")
        
        # Ask for confirmation before proceeding
        print("\n⚡ Ready to create a fix for this issue?")
        print("This will:")
        print("1. Create a new branch")
        print("2. Make necessary code changes")
        print("3. Commit and push the changes")
        print("4. Create a pull request")
        
        response = input("\nProceed? (y/n): ")
        
        if response.lower() == 'y':
            # Create the PR
            print("\n🚀 Creating fix...")
            
            # Since you're the lead developer, you might want to customize the branch name
            branch_name = f"fix-issue-{ISSUE_NUMBER}"
            
            # Process the issue
            pr_url = agent.process_ml_issue(
                repo_url=REPO_URL,
                repo_name=REPO_NAME,
                issue_number=ISSUE_NUMBER
            )
            
            print(f"\n✅ Pull request created: {pr_url}")
            print("\n📝 Next steps:")
            print("1. Review the generated changes")
            print("2. Test the changes locally")
            print("3. Update the PR if needed")
            print("4. Merge when ready!")
            
        else:
            print("\n❌ Operation cancelled")
            
            # Option to manually specify changes
            print("\n💡 Would you like to specify custom changes instead? (y/n): ")
            if input().lower() == 'y':
                create_custom_fix(agent, REPO_URL, REPO_NAME, ISSUE_NUMBER, issue)
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\n💡 Tip: Make sure you have push access to the repository")


def create_custom_fix(agent, repo_url, repo_name, issue_number, issue):
    """Create a custom fix with user-specified changes."""
    
    print("\n📝 Custom Fix Mode")
    print("Please specify the files you want to change and their content.")
    
    changes = {}
    
    while True:
        file_path = input("\nEnter file path (or 'done' to finish): ")
        if file_path.lower() == 'done':
            break
        
        print(f"Enter content for {file_path} (end with 'EOF' on a new line):")
        lines = []
        while True:
            line = input()
            if line == 'EOF':
                break
            lines.append(line)
        
        changes[file_path] = '\n'.join(lines)
    
    if changes:
        print(f"\n📦 Creating PR with {len(changes)} file changes...")
        
        # Use the base agent for custom changes
        from github_pr_agent import GitHubPRAgent
        base_agent = GitHubPRAgent(os.getenv("GITHUB_TOKEN"))
        
        try:
            pr_url = base_agent.process_issue(
                repo_url=repo_url,
                repo_name=repo_name,
                issue_number=issue_number,
                example_changes=changes
            )
            
            print(f"\n✅ Pull request created: {pr_url}")
            
        except Exception as e:
            print(f"\n❌ Error creating PR: {e}")


def check_deepforest_structure():
    """Check DeepForest repository structure to understand the codebase."""
    
    print("\n📂 DeepForest Repository Structure:")
    print("- src/deepforest/ - Main package code")
    print("- tests/ - Unit tests")
    print("- docs/ - Documentation")
    print("- setup.cfg, pyproject.toml - Package configuration")
    print("\n💡 Common file locations:")
    print("- Models: src/deepforest/models/")
    print("- Data utilities: src/deepforest/data.py")
    print("- Training code: src/deepforest/train.py")
    print("- Prediction code: src/deepforest/predict.py")


if __name__ == "__main__":
    print("🌲 DeepForest Issue #1064 Solver\n")
    
    # Check if we should show repo structure
    if len(sys.argv) > 1 and sys.argv[1] == "--info":
        check_deepforest_structure()
    else:
        analyze_and_fix_deepforest_issue()