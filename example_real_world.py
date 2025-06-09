#!/usr/bin/env python3
"""
Real-world example: Contributing to an open source ML project.

This example shows how to:
1. Fork a repository (manual step)
2. Find good first issues
3. Analyze and fix them automatically
4. Create pull requests
"""

import os
import sys
from github import Github
from github_pr_agent import GitHubPRAgent
from ml_github_agent import MLGitHubAgent
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def find_good_first_issues(repo_name: str, github_token: str):
    """Find issues labeled as 'good first issue' in a repository."""
    g = Github(github_token)
    repo = g.get_repo(repo_name)
    
    # Search for good first issues
    issues = repo.get_issues(state='open', labels=['good first issue'])
    
    print(f"\n🔍 Found {issues.totalCount} 'good first issue' in {repo_name}:\n")
    
    good_issues = []
    for issue in issues[:5]:  # Show first 5
        print(f"Issue #{issue.number}: {issue.title}")
        print(f"  Labels: {[l.name for l in issue.labels]}")
        print(f"  URL: {issue.html_url}\n")
        good_issues.append(issue)
    
    return good_issues


def contribute_to_scikit_learn_example():
    """Example: Contributing to scikit-learn documentation."""
    
    # Get credentials
    github_token = os.getenv("GITHUB_TOKEN")
    if not github_token:
        print("❌ Please set GITHUB_TOKEN environment variable")
        return
    
    # Note: You should fork the repository first!
    # Go to https://github.com/scikit-learn/scikit-learn and click "Fork"
    
    YOUR_USERNAME = "your-github-username"  # Replace with your username
    
    print("📚 Example: Contributing to scikit-learn\n")
    print("Prerequisites:")
    print("1. Fork scikit-learn/scikit-learn to your account")
    print("2. Replace YOUR_USERNAME in this script\n")
    
    # Find good first issues
    issues = find_good_first_issues("scikit-learn/scikit-learn", github_token)
    
    if not issues:
        print("No good first issues found")
        return
    
    # Example: Fix a documentation issue
    # This is a hypothetical example - adapt based on actual issues
    agent = GitHubPRAgent(github_token)
    
    # Process the first documentation issue
    for issue in issues:
        if any(label.name in ['Documentation', 'good first issue'] for label in issue.labels):
            print(f"\n🔧 Processing issue #{issue.number}: {issue.title}")
            
            # Example fix for a documentation issue
            example_changes = {
                "doc/fix_example.rst": f"""
Documentation Fix for Issue #{issue.number}
==========================================

This fix addresses the documentation issue reported.

Example Code
------------

.. code-block:: python

    # Fixed example code
    from sklearn.ensemble import RandomForestClassifier
    
    # Create and train model
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    # Make predictions
    predictions = clf.predict(X_test)

Notes
-----
- Fixed formatting issues
- Added proper imports
- Improved code clarity
"""
            }
            
            try:
                # Process on your fork
                pr_url = agent.process_issue(
                    repo_url=f"https://github.com/{YOUR_USERNAME}/scikit-learn.git",
                    repo_name=f"{YOUR_USERNAME}/scikit-learn",
                    issue_number=issue.number,
                    example_changes=example_changes
                )
                
                print(f"✅ Created PR: {pr_url}")
                print("\n📝 Next steps:")
                print("1. Review the changes in your fork")
                print("2. Create a pull request to the main repository")
                print("3. Reference the original issue in your PR")
                
            except Exception as e:
                print(f"❌ Error: {e}")
            
            break


def contribute_to_pytorch_example():
    """Example: Contributing to PyTorch."""
    
    github_token = os.getenv("GITHUB_TOKEN")
    openai_key = os.getenv("OPENAI_API_KEY")
    
    if not github_token:
        print("❌ Please set GITHUB_TOKEN environment variable")
        return
    
    YOUR_USERNAME = "your-github-username"  # Replace with your username
    
    print("🔥 Example: Contributing to PyTorch\n")
    
    # Use ML-specific agent for PyTorch
    ml_agent = MLGitHubAgent(github_token, openai_key)
    
    # Find ML-specific issues
    g = Github(github_token)
    repo = g.get_repo("pytorch/pytorch")
    
    # Look for performance or model-related issues
    issues = repo.get_issues(state='open', labels=['module: nn'])
    
    print("Found neural network module issues:")
    for issue in issues[:3]:
        print(f"Issue #{issue.number}: {issue.title}")
        
        # Analyze if it's an ML-specific issue
        issue_details = {
            "number": issue.number,
            "title": issue.title,
            "body": issue.body,
            "labels": [l.name for l in issue.labels]
        }
        
        analysis = ml_agent.analyze_ml_issue(issue_details)
        if analysis["type"] == "ml":
            print(f"  ✅ ML Issue detected!")
            print(f"  Analysis: {analysis.get('analysis', {})}")
            
            # You could process this issue automatically
            # pr_url = ml_agent.process_ml_issue(...)


def find_beginner_friendly_ml_projects():
    """Find ML projects that are beginner-friendly."""
    
    github_token = os.getenv("GITHUB_TOKEN")
    if not github_token:
        print("❌ Please set GITHUB_TOKEN environment variable")
        return
    
    g = Github(github_token)
    
    print("🎯 Beginner-friendly ML projects:\n")
    
    # Popular ML projects with good first issues
    beginner_projects = [
        "huggingface/transformers",
        "keras-team/keras",
        "dmlc/xgboost",
        "explosion/spaCy",
        "facebookresearch/detectron2",
        "albumentations-team/albumentations",
        "pytorch/vision",
        "tensorflow/models"
    ]
    
    for project_name in beginner_projects:
        try:
            repo = g.get_repo(project_name)
            good_first_issues = repo.get_issues(
                state='open', 
                labels=['good first issue']
            )
            
            if good_first_issues.totalCount > 0:
                print(f"📦 {project_name}")
                print(f"   ⭐ Stars: {repo.stargazers_count}")
                print(f"   🎯 Good first issues: {good_first_issues.totalCount}")
                print(f"   🔗 URL: {repo.html_url}\n")
                
        except Exception as e:
            continue


def automated_contribution_workflow():
    """Complete automated workflow for contributing."""
    
    print("🤖 Automated Contribution Workflow\n")
    
    # Configuration
    config = {
        "target_repo": "your-username/your-ml-project",  # Your fork
        "upstream_repo": "original-owner/original-project",  # Original repo
        "issue_labels": ["good first issue", "help wanted"],
        "max_issues_to_process": 3
    }
    
    github_token = os.getenv("GITHUB_TOKEN")
    openai_key = os.getenv("OPENAI_API_KEY")
    
    if not github_token:
        print("❌ Please set GITHUB_TOKEN environment variable")
        return
    
    # Create agents
    general_agent = GitHubPRAgent(github_token, openai_key)
    ml_agent = MLGitHubAgent(github_token, openai_key)
    
    # Find issues
    g = Github(github_token)
    upstream_repo = g.get_repo(config["upstream_repo"])
    
    issues_processed = 0
    
    for label in config["issue_labels"]:
        if issues_processed >= config["max_issues_to_process"]:
            break
            
        issues = upstream_repo.get_issues(state='open', labels=[label])
        
        for issue in issues:
            if issues_processed >= config["max_issues_to_process"]:
                break
            
            print(f"\n📋 Processing issue #{issue.number}: {issue.title}")
            
            # Determine which agent to use
            issue_text = f"{issue.title} {issue.body}".lower()
            is_ml_issue = any(keyword in issue_text for keyword in 
                            ["model", "training", "accuracy", "loss", "neural"])
            
            try:
                if is_ml_issue:
                    print("  🤖 Using ML-specific agent")
                    pr_url = ml_agent.process_ml_issue(
                        repo_url=f"https://github.com/{config['target_repo']}.git",
                        repo_name=config['target_repo'],
                        issue_number=issue.number
                    )
                else:
                    print("  📝 Using general agent")
                    pr_url = general_agent.process_issue(
                        repo_url=f"https://github.com/{config['target_repo']}.git",
                        repo_name=config['target_repo'],
                        issue_number=issue.number
                    )
                
                print(f"  ✅ Created PR: {pr_url}")
                issues_processed += 1
                
            except Exception as e:
                print(f"  ❌ Error: {e}")
                continue
    
    print(f"\n✨ Processed {issues_processed} issues!")


def main():
    """Main entry point with menu."""
    
    print("🚀 GitHub PR Agent - Real World Examples\n")
    print("Choose an example:")
    print("1. Find good first issues in scikit-learn")
    print("2. Find beginner-friendly ML projects")
    print("3. Contribute to scikit-learn (example)")
    print("4. Analyze PyTorch issues")
    print("5. Run automated contribution workflow")
    print("0. Exit\n")
    
    choice = input("Enter your choice (0-5): ")
    
    if choice == "1":
        github_token = os.getenv("GITHUB_TOKEN")
        if github_token:
            find_good_first_issues("scikit-learn/scikit-learn", github_token)
        else:
            print("Please set GITHUB_TOKEN environment variable")
    
    elif choice == "2":
        find_beginner_friendly_ml_projects()
    
    elif choice == "3":
        contribute_to_scikit_learn_example()
    
    elif choice == "4":
        contribute_to_pytorch_example()
    
    elif choice == "5":
        automated_contribution_workflow()
    
    elif choice == "0":
        print("Goodbye! 👋")
        sys.exit(0)
    
    else:
        print("Invalid choice")


if __name__ == "__main__":
    main()