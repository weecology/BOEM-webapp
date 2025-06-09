#!/usr/bin/env python3
"""
GitHub PR Agent: Automate the process of checking out branches, 
analyzing issues, and creating pull requests.

Requirements:
    pip install PyGithub gitpython requests openai python-dotenv
"""

import os
import json
import tempfile
import shutil
from typing import Optional, Dict, Any
from datetime import datetime

from github import Github
from git import Repo
import openai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class GitHubPRAgent:
    """Agent for automating GitHub pull request workflows."""
    
    def __init__(self, github_token: str, openai_api_key: Optional[str] = None):
        """
        Initialize the GitHub PR Agent.
        
        Args:
            github_token: GitHub personal access token
            openai_api_key: OpenAI API key for AI-powered issue analysis
        """
        self.github = Github(github_token)
        self.openai_api_key = openai_api_key
        if openai_api_key:
            openai.api_key = openai_api_key
    
    def clone_repository(self, repo_url: str, local_path: str) -> Repo:
        """
        Clone a repository to a local path.
        
        Args:
            repo_url: URL of the repository to clone
            local_path: Local directory to clone into
            
        Returns:
            GitPython Repo object
        """
        print(f"Cloning repository: {repo_url}")
        
        # Clean up if directory exists
        if os.path.exists(local_path):
            shutil.rmtree(local_path)
        
        # Clone the repository
        repo = Repo.clone_from(repo_url, local_path)
        print(f"Repository cloned to: {local_path}")
        return repo
    
    def checkout_branch(self, repo: Repo, branch_name: str, base_branch: str = "main") -> None:
        """
        Checkout or create a new branch.
        
        Args:
            repo: GitPython Repo object
            branch_name: Name of the branch to checkout/create
            base_branch: Base branch to create from (default: main)
        """
        try:
            # Try to checkout existing branch
            repo.git.checkout(branch_name)
            print(f"Checked out existing branch: {branch_name}")
        except:
            # Create new branch from base branch
            repo.git.checkout(base_branch)
            repo.git.checkout('-b', branch_name)
            print(f"Created and checked out new branch: {branch_name}")
    
    def get_issue_details(self, repo_name: str, issue_number: int) -> Dict[str, Any]:
        """
        Get details of a GitHub issue.
        
        Args:
            repo_name: Repository name in format "owner/repo"
            issue_number: Issue number
            
        Returns:
            Dictionary containing issue details
        """
        print(f"Fetching issue #{issue_number} from {repo_name}")
        
        repo = self.github.get_repo(repo_name)
        issue = repo.get_issue(issue_number)
        
        issue_details = {
            "number": issue.number,
            "title": issue.title,
            "body": issue.body,
            "labels": [label.name for label in issue.labels],
            "state": issue.state,
            "created_at": issue.created_at,
            "user": issue.user.login
        }
        
        print(f"Issue title: {issue.title}")
        return issue_details
    
    def analyze_issue_with_ai(self, issue_details: Dict[str, Any]) -> Dict[str, Any]:
        """
        Use AI to analyze an issue and suggest code changes.
        
        Args:
            issue_details: Dictionary containing issue information
            
        Returns:
            Dictionary with AI analysis and suggestions
        """
        if not self.openai_api_key:
            print("No OpenAI API key provided, skipping AI analysis")
            return {"analysis": "Manual review required", "suggestions": []}
        
        prompt = f"""
        Analyze the following GitHub issue and suggest how to fix it:
        
        Title: {issue_details['title']}
        Description: {issue_details['body']}
        Labels: {', '.join(issue_details['labels'])}
        
        Please provide:
        1. A brief analysis of the issue
        2. Suggested code changes or implementation approach
        3. Files that might need to be modified
        """
        
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a helpful software engineer analyzing GitHub issues."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000
            )
            
            analysis = response.choices[0].message.content
            return {"analysis": analysis, "suggestions": []}
        except Exception as e:
            print(f"AI analysis failed: {e}")
            return {"analysis": "AI analysis failed", "suggestions": []}
    
    def make_code_changes(self, repo_path: str, changes: Dict[str, str]) -> None:
        """
        Apply code changes to files in the repository.
        
        Args:
            repo_path: Path to the local repository
            changes: Dictionary mapping file paths to new content
        """
        for file_path, content in changes.items():
            full_path = os.path.join(repo_path, file_path)
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            
            # Write the changes
            with open(full_path, 'w') as f:
                f.write(content)
            
            print(f"Modified: {file_path}")
    
    def commit_changes(self, repo: Repo, commit_message: str) -> None:
        """
        Commit all changes in the repository.
        
        Args:
            repo: GitPython Repo object
            commit_message: Commit message
        """
        # Add all changes
        repo.git.add(A=True)
        
        # Commit
        repo.index.commit(commit_message)
        print(f"Committed changes: {commit_message}")
    
    def push_branch(self, repo: Repo, branch_name: str) -> None:
        """
        Push a branch to the remote repository.
        
        Args:
            repo: GitPython Repo object
            branch_name: Name of the branch to push
        """
        origin = repo.remote(name='origin')
        origin.push(branch_name)
        print(f"Pushed branch: {branch_name}")
    
    def create_pull_request(self, repo_name: str, title: str, body: str, 
                          head_branch: str, base_branch: str = "main") -> str:
        """
        Create a pull request on GitHub.
        
        Args:
            repo_name: Repository name in format "owner/repo"
            title: PR title
            body: PR description
            head_branch: Branch with changes
            base_branch: Target branch (default: main)
            
        Returns:
            URL of the created pull request
        """
        print(f"Creating pull request: {title}")
        
        repo = self.github.get_repo(repo_name)
        pr = repo.create_pull(
            title=title,
            body=body,
            head=head_branch,
            base=base_branch
        )
        
        print(f"Pull request created: {pr.html_url}")
        return pr.html_url
    
    def process_issue(self, repo_url: str, repo_name: str, issue_number: int,
                     example_changes: Optional[Dict[str, str]] = None) -> str:
        """
        Full workflow: Clone repo, analyze issue, make changes, create PR.
        
        Args:
            repo_url: URL of the repository
            repo_name: Repository name in format "owner/repo"
            issue_number: Issue number to address
            example_changes: Optional dictionary of file changes to apply
            
        Returns:
            URL of the created pull request
        """
        # Create a temporary directory for the repo
        with tempfile.TemporaryDirectory() as temp_dir:
            # 1. Clone the repository
            repo = self.clone_repository(repo_url, temp_dir)
            
            # 2. Get issue details
            issue_details = self.get_issue_details(repo_name, issue_number)
            
            # 3. Analyze issue with AI (optional)
            ai_analysis = self.analyze_issue_with_ai(issue_details)
            print(f"\nAI Analysis:\n{ai_analysis['analysis']}\n")
            
            # 4. Create a new branch
            branch_name = f"fix-issue-{issue_number}"
            self.checkout_branch(repo, branch_name)
            
            # 5. Make code changes
            if example_changes:
                self.make_code_changes(temp_dir, example_changes)
            else:
                # In a real scenario, you would implement logic to make actual changes
                # based on the issue and AI analysis
                print("No specific changes provided, creating example change")
                example_changes = {
                    "FIXES.md": f"# Fix for Issue #{issue_number}\n\n"
                               f"This file documents the fix for: {issue_details['title']}\n\n"
                               f"## Analysis\n{ai_analysis['analysis']}\n"
                }
                self.make_code_changes(temp_dir, example_changes)
            
            # 6. Commit changes
            commit_message = f"Fix issue #{issue_number}: {issue_details['title']}"
            self.commit_changes(repo, commit_message)
            
            # 7. Push the branch
            self.push_branch(repo, branch_name)
            
            # 8. Create pull request
            pr_title = f"Fix: {issue_details['title']} (#{issue_number})"
            pr_body = f"""
## Description
This PR addresses issue #{issue_number}: {issue_details['title']}

## Issue Analysis
{ai_analysis['analysis']}

## Changes Made
- Added fix for the reported issue
- Updated relevant documentation

## Related Issue
Fixes #{issue_number}

## Testing
- [ ] Tests pass locally
- [ ] Code follows project style guidelines
- [ ] Documentation has been updated

## Additional Notes
This PR was created automatically by a GitHub PR Agent.
"""
            
            pr_url = self.create_pull_request(
                repo_name=repo_name,
                title=pr_title,
                body=pr_body,
                head_branch=branch_name,
                base_branch="main"
            )
            
            return pr_url


# Example usage
if __name__ == "__main__":
    # Example: Using the agent with a real repository
    
    # You need to set these environment variables or pass them directly
    GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")  # Your GitHub personal access token
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # Optional: for AI analysis
    
    if not GITHUB_TOKEN:
        print("Error: GITHUB_TOKEN environment variable not set")
        print("Please create a GitHub personal access token with repo permissions")
        exit(1)
    
    # Initialize the agent
    agent = GitHubPRAgent(GITHUB_TOKEN, OPENAI_API_KEY)
    
    # Example: Process an issue on a repository
    # Note: Replace these with actual values
    REPO_URL = "https://github.com/username/repository.git"  # Replace with actual repo
    REPO_NAME = "username/repository"  # Replace with actual owner/repo
    ISSUE_NUMBER = 1  # Replace with actual issue number
    
    # Example code changes (in real scenario, these would be generated based on issue)
    example_changes = {
        "src/fix.py": """# Automated fix for the issue
def fixed_function():
    '''This function fixes the reported issue'''
    return "Issue fixed!"
""",
        "README.md": """# Updated README
This repository has been updated to fix the reported issue.
"""
    }
    
    try:
        # Process the issue and create a PR
        pr_url = agent.process_issue(
            repo_url=REPO_URL,
            repo_name=REPO_NAME,
            issue_number=ISSUE_NUMBER,
            example_changes=example_changes
        )
        
        print(f"\n✅ Success! Pull request created: {pr_url}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("Make sure you have the correct permissions and the repository exists")