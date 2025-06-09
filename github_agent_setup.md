# GitHub PR Agent Setup Guide

This guide shows you how to set up and use an automated agent to check out branches, analyze issues, and create pull requests on open source repositories.

## 🚀 Quick Start

### 1. Prerequisites

- Python 3.8 or higher
- Git installed on your system
- GitHub account with a personal access token
- (Optional) OpenAI API key for AI-powered issue analysis

### 2. Installation

```bash
# Clone this repository or create a new directory
mkdir github-pr-agent
cd github-pr-agent

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install PyGithub gitpython python-dotenv openai
```

### 3. Setup Credentials

Create a `.env` file in your project directory:

```env
# Required: GitHub Personal Access Token
# Create at: https://github.com/settings/tokens
# Scopes needed: repo, workflow
GITHUB_TOKEN=your_github_token_here

# Optional: For AI-powered issue analysis
OPENAI_API_KEY=your_openai_api_key_here
```

### 4. Basic Usage

```python
from github_pr_agent import GitHubPRAgent
import os

# Load credentials
github_token = os.getenv("GITHUB_TOKEN")
openai_key = os.getenv("OPENAI_API_KEY")

# Create agent
agent = GitHubPRAgent(github_token, openai_key)

# Process an issue and create a PR
pr_url = agent.process_issue(
    repo_url="https://github.com/owner/repository.git",
    repo_name="owner/repository",
    issue_number=123
)

print(f"Created PR: {pr_url}")
```

## 📋 Step-by-Step Workflow

### Step 1: Find an Issue

```python
# Get issue details
issue_details = agent.get_issue_details("owner/repository", 123)
print(f"Issue: {issue_details['title']}")
print(f"Description: {issue_details['body']}")
```

### Step 2: Analyze the Issue

```python
# Use AI to analyze the issue (if OpenAI key provided)
analysis = agent.analyze_issue_with_ai(issue_details)
print(f"AI Analysis: {analysis['analysis']}")
```

### Step 3: Clone and Branch

```python
import tempfile
from git import Repo

# Clone repository
with tempfile.TemporaryDirectory() as temp_dir:
    repo = agent.clone_repository(
        "https://github.com/owner/repository.git", 
        temp_dir
    )
    
    # Create a new branch
    branch_name = f"fix-issue-{issue_number}"
    agent.checkout_branch(repo, branch_name)
```

### Step 4: Make Changes

```python
# Define your code changes
changes = {
    "src/fix.py": """# Fix for the issue
def improved_function():
    '''This fixes the reported bug'''
    return "Fixed!"
""",
    "tests/test_fix.py": """# Test for the fix
def test_improved_function():
    assert improved_function() == "Fixed!"
"""
}

# Apply changes
agent.make_code_changes(temp_dir, changes)
```

### Step 5: Commit and Push

```python
# Commit changes
agent.commit_changes(repo, f"Fix issue #{issue_number}")

# Push to GitHub
agent.push_branch(repo, branch_name)
```

### Step 6: Create Pull Request

```python
# Create PR
pr_url = agent.create_pull_request(
    repo_name="owner/repository",
    title=f"Fix: Issue #{issue_number}",
    body="""## Description
This PR fixes the reported issue.

## Changes
- Fixed the bug
- Added tests

Fixes #123
""",
    head_branch=branch_name,
    base_branch="main"
)
```

## 🤖 ML-Specific Agent

For machine learning repositories, use the specialized `MLGitHubAgent`:

```python
from ml_github_agent import MLGitHubAgent

# Create ML-specific agent
ml_agent = MLGitHubAgent(github_token, openai_key)

# Process ML issue
pr_url = ml_agent.process_ml_issue(
    repo_url="https://github.com/owner/ml-project.git",
    repo_name="owner/ml-project",
    issue_number=42
)
```

The ML agent automatically:
- Analyzes ML-specific issues (overfitting, performance, etc.)
- Generates model improvements
- Creates hyperparameter tuning scripts
- Adds data preprocessing enhancements
- Includes experiment tracking

## 🔧 Advanced Configuration

### Custom Issue Analysis

```python
class CustomGitHubAgent(GitHubPRAgent):
    def analyze_issue_custom(self, issue_details):
        # Your custom analysis logic
        if "performance" in issue_details['title'].lower():
            return {
                "type": "performance",
                "suggestions": ["Add caching", "Optimize algorithms"]
            }
        return super().analyze_issue_with_ai(issue_details)
```

### Automated Workflow

```python
# Process multiple issues
issue_numbers = [123, 124, 125]

for issue_num in issue_numbers:
    try:
        pr_url = agent.process_issue(
            repo_url="https://github.com/owner/repo.git",
            repo_name="owner/repo",
            issue_number=issue_num
        )
        print(f"✅ Created PR for issue #{issue_num}: {pr_url}")
    except Exception as e:
        print(f"❌ Failed to process issue #{issue_num}: {e}")
```

## 🛡️ Best Practices

1. **Test Locally First**: Always test your changes locally before creating PRs
2. **Respect Rate Limits**: GitHub API has rate limits (5000 requests/hour for authenticated requests)
3. **Review Generated Code**: Always review AI-generated code before submitting
4. **Follow Project Guidelines**: Respect the project's contribution guidelines
5. **Small, Focused PRs**: Create small, focused PRs that address one issue

## 🔍 Debugging

Enable verbose logging:

```python
import logging

logging.basicConfig(level=logging.DEBUG)
```

Common issues:

1. **Authentication Error**: Check your GitHub token has correct scopes
2. **Permission Denied**: Ensure you have push access (fork the repo first)
3. **Branch Already Exists**: Delete old branches or use unique names

## 📚 Example Projects

Here are some examples of using the agent on real repositories:

### Example 1: Documentation Fix

```python
# Fix documentation issues
changes = {
    "README.md": """# Project Name

Fixed typos and improved documentation clarity.
""",
    "docs/setup.md": """# Setup Guide

Improved installation instructions.
"""
}
```

### Example 2: Bug Fix

```python
# Fix a specific bug
changes = {
    "src/utils.py": """def fixed_function(data):
    # Added null check to fix issue #123
    if data is None:
        return []
    return process_data(data)
"""
}
```

### Example 3: ML Model Improvement

```python
# For ML repositories
changes = {
    "model.py": """class ImprovedModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Added dropout for better generalization
        self.dropout = nn.Dropout(0.3)
""",
    "train.py": """# Added early stopping
early_stopping = EarlyStopping(patience=10)
"""
}
```

## 🤝 Contributing

Feel free to extend the agent with:
- Support for other version control systems
- Integration with different AI providers
- Custom analysis rules
- Automated testing before PR creation

## 📄 License

This project is open source and available under the MIT License.

## 🔗 Resources

- [GitHub API Documentation](https://docs.github.com/en/rest)
- [PyGithub Documentation](https://pygithub.readthedocs.io/)
- [GitPython Documentation](https://gitpython.readthedocs.io/)
- [OpenAI API Documentation](https://platform.openai.com/docs/)

---

Happy automating! 🚀