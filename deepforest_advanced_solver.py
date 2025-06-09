#!/usr/bin/env python3
"""
Advanced DeepForest Issue Solver with intelligent analysis and fix generation.
"""

import os
import re
import json
from typing import Dict, Any, List
from github import Github
from ml_github_agent import MLGitHubAgent
from github_pr_agent import GitHubPRAgent
from dotenv import load_dotenv

load_dotenv()


class DeepForestIssueSolver:
    """Specialized solver for DeepForest issues."""
    
    def __init__(self, github_token: str, openai_key: str = None):
        self.github_token = github_token
        self.openai_key = openai_key
        self.ml_agent = MLGitHubAgent(github_token, openai_key)
        self.general_agent = GitHubPRAgent(github_token, openai_key)
        self.g = Github(github_token)
        self.repo = self.g.get_repo("weecology/DeepForest")
    
    def analyze_issue(self, issue_number: int) -> Dict[str, Any]:
        """Analyze a DeepForest issue and determine the fix strategy."""
        
        issue = self.repo.get_issue(issue_number)
        
        # Extract issue details
        issue_text = f"{issue.title} {issue.body}".lower()
        labels = [l.name for l in issue.labels]
        
        # Categorize the issue
        issue_type = self._categorize_issue(issue_text, labels)
        
        # Determine files that might need changes
        affected_files = self._identify_affected_files(issue_text, issue_type)
        
        return {
            "number": issue.number,
            "title": issue.title,
            "body": issue.body,
            "labels": labels,
            "type": issue_type,
            "affected_files": affected_files,
            "strategy": self._determine_fix_strategy(issue_type, issue_text)
        }
    
    def _categorize_issue(self, issue_text: str, labels: List[str]) -> str:
        """Categorize the issue type."""
        
        # Check labels first
        if "bug" in labels:
            return "bug"
        elif "enhancement" in labels or "feature request" in labels:
            return "enhancement"
        elif "documentation" in labels:
            return "documentation"
        
        # Check text content
        if any(word in issue_text for word in ["error", "bug", "crash", "fail"]):
            return "bug"
        elif any(word in issue_text for word in ["add", "feature", "enhance", "improve"]):
            return "enhancement"
        elif any(word in issue_text for word in ["docs", "documentation", "readme"]):
            return "documentation"
        elif any(word in issue_text for word in ["model", "accuracy", "training", "performance"]):
            return "model_improvement"
        
        return "general"
    
    def _identify_affected_files(self, issue_text: str, issue_type: str) -> List[str]:
        """Identify files that might need to be changed."""
        
        files = []
        
        # Check for specific file mentions
        file_pattern = r'`([^`]+\.py)`|([a-zA-Z_]+\.py)'
        matches = re.findall(file_pattern, issue_text)
        for match in matches:
            file_name = match[0] or match[1]
            if file_name:
                # Try to find the full path
                if "deepforest" in file_name:
                    files.append(file_name)
                else:
                    files.append(f"src/deepforest/{file_name}")
        
        # Add common files based on issue type
        if issue_type == "bug":
            if "predict" in issue_text:
                files.append("src/deepforest/predict.py")
            if "train" in issue_text:
                files.append("src/deepforest/train.py")
            if "model" in issue_text:
                files.append("src/deepforest/models.py")
        elif issue_type == "documentation":
            files.append("README.md")
            if "api" in issue_text:
                files.append("docs/api.rst")
        elif issue_type == "model_improvement":
            files.extend([
                "src/deepforest/models.py",
                "src/deepforest/train.py",
                "src/deepforest/evaluate.py"
            ])
        
        return list(set(files))  # Remove duplicates
    
    def _determine_fix_strategy(self, issue_type: str, issue_text: str) -> str:
        """Determine the strategy for fixing the issue."""
        
        if issue_type == "bug":
            if "import" in issue_text or "module" in issue_text:
                return "fix_imports"
            elif "type" in issue_text or "dtype" in issue_text:
                return "fix_type_error"
            elif "attribute" in issue_text:
                return "fix_attribute_error"
            else:
                return "general_bug_fix"
        elif issue_type == "enhancement":
            if "parameter" in issue_text or "argument" in issue_text:
                return "add_parameter"
            elif "method" in issue_text or "function" in issue_text:
                return "add_method"
            else:
                return "general_enhancement"
        elif issue_type == "documentation":
            return "update_documentation"
        elif issue_type == "model_improvement":
            return "improve_model"
        
        return "general_fix"
    
    def generate_fix(self, analysis: Dict[str, Any]) -> Dict[str, str]:
        """Generate code fixes based on the analysis."""
        
        fixes = {}
        strategy = analysis["strategy"]
        
        if strategy == "fix_imports":
            fixes.update(self._generate_import_fix(analysis))
        elif strategy == "fix_type_error":
            fixes.update(self._generate_type_fix(analysis))
        elif strategy == "update_documentation":
            fixes.update(self._generate_doc_fix(analysis))
        elif strategy == "add_parameter":
            fixes.update(self._generate_parameter_fix(analysis))
        elif strategy == "improve_model":
            fixes.update(self._generate_model_improvement(analysis))
        else:
            # Use AI to generate fixes if available
            if self.openai_key:
                ml_analysis = self.ml_agent.analyze_ml_issue(analysis)
                if ml_analysis["type"] == "ml":
                    fixes = self.ml_agent.generate_ml_fix(ml_analysis)
        
        return fixes
    
    def _generate_import_fix(self, analysis: Dict[str, Any]) -> Dict[str, str]:
        """Generate fixes for import issues."""
        
        # Example fix for common import issues
        fixes = {}
        
        if "rasterio" in analysis["body"].lower():
            fixes["src/deepforest/__init__.py"] = """# DeepForest Package
from deepforest.main import deepforest
from deepforest.model import Model
from deepforest.utilities import read_file

# Add optional imports with proper error handling
try:
    import rasterio
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False
    import warnings
    warnings.warn("rasterio not installed. Some functionality may be limited.")

__version__ = "1.3.0"
__all__ = ["deepforest", "Model", "read_file", "HAS_RASTERIO"]
"""
        
        return fixes
    
    def _generate_type_fix(self, analysis: Dict[str, Any]) -> Dict[str, str]:
        """Generate fixes for type errors."""
        
        fixes = {}
        
        # Example: Add type checking and conversion
        if any(f.endswith("predict.py") for f in analysis["affected_files"]):
            fixes["src/deepforest/predict.py"] = """# Prediction utilities with type safety
import numpy as np
import torch
from typing import Union, List, Optional

def predict_image(model, image: Union[np.ndarray, torch.Tensor], 
                 patch_size: int = 400) -> List[dict]:
    '''Predict bounding boxes for an image with type safety.
    
    Args:
        model: Trained DeepForest model
        image: Input image as numpy array or torch tensor
        patch_size: Size of patches for prediction
        
    Returns:
        List of predictions with bounding boxes
    '''
    # Ensure correct type
    if isinstance(image, torch.Tensor):
        image = image.numpy()
    
    if not isinstance(image, np.ndarray):
        raise TypeError(f"Image must be numpy array or torch tensor, got {type(image)}")
    
    # Ensure correct dtype
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
    
    # Continue with prediction...
    return model.predict_image(image, patch_size=patch_size)
"""
        
        return fixes
    
    def _generate_doc_fix(self, analysis: Dict[str, Any]) -> Dict[str, str]:
        """Generate documentation fixes."""
        
        fixes = {}
        
        # Update README if mentioned
        if "readme" in analysis["body"].lower():
            fixes["README.md"] = f"""# DeepForest

[![Documentation Status](https://readthedocs.org/projects/deepforest/badge/?version=latest)](https://deepforest.readthedocs.io/en/latest/?badge=latest)

## What is DeepForest?

DeepForest is a python package for training and predicting ecological objects in airborne imagery. 

## Quick Start

```python
from deepforest import main
from deepforest import get_data

# Load the pre-built model
model = main.deepforest()
model.use_release()

# Predict on an image
img_path = get_data("OSBS_029.tif")
predictions = model.predict_image(path=img_path, return_plot=True)
```

## Installation

```bash
pip install deepforest
```

For development:
```bash
git clone https://github.com/weecology/DeepForest.git
cd DeepForest
pip install -e .
```

## Fixed in this PR

This PR addresses issue #{analysis['number']}: {analysis['title']}

## Documentation

Full documentation is available at [deepforest.readthedocs.io](https://deepforest.readthedocs.io/)
"""
        
        return fixes
    
    def _generate_parameter_fix(self, analysis: Dict[str, Any]) -> Dict[str, str]:
        """Generate fixes for adding new parameters."""
        
        fixes = {}
        
        # Example: Add a new parameter to a method
        if "confidence" in analysis["body"].lower():
            fixes["src/deepforest/main.py"] = """# Main DeepForest class with new parameter
class deepforest:
    def predict_image(self, path=None, image=None, return_plot=False, 
                     color=(0, 165, 255), thickness=1, 
                     confidence_threshold=0.4):  # New parameter
        '''Predict bounding boxes for an image.
        
        Args:
            path: Path to image file
            image: Numpy array of image
            return_plot: Return image with boxes drawn
            color: Color of bounding boxes
            thickness: Thickness of box lines
            confidence_threshold: Minimum confidence score for predictions
            
        Returns:
            DataFrame of predictions
        '''
        # Implementation with confidence filtering
        predictions = self._predict_image_internal(path, image)
        
        # Filter by confidence
        if confidence_threshold > 0:
            predictions = predictions[predictions.score >= confidence_threshold]
        
        if return_plot:
            # Draw boxes implementation
            pass
            
        return predictions
"""
        
        return fixes
    
    def _generate_model_improvement(self, analysis: Dict[str, Any]) -> Dict[str, str]:
        """Generate model improvement code."""
        
        # Use the ML agent's model generation
        return {
            "src/deepforest/model_improvements.py": self.ml_agent._generate_model_code(analysis),
            "src/deepforest/augmentations.py": """# Data augmentation for DeepForest
import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_augmentations(augment=True):
    '''Get augmentation pipeline for training.
    
    Args:
        augment: Whether to apply augmentations
        
    Returns:
        Albumentations composition
    '''
    if augment:
        transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.5
            ),
            A.GaussNoise(var_limit=(10, 50), p=0.3),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))
    else:
        transform = A.Compose([
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])
    
    return transform
"""
        }
    
    def solve_issue(self, issue_number: int):
        """Complete workflow to solve a DeepForest issue."""
        
        print(f"🌲 Analyzing DeepForest issue #{issue_number}...")
        
        # Analyze the issue
        analysis = self.analyze_issue(issue_number)
        
        print(f"\n📋 Issue: {analysis['title']}")
        print(f"Type: {analysis['type']}")
        print(f"Strategy: {analysis['strategy']}")
        print(f"Affected files: {', '.join(analysis['affected_files'])}")
        
        # Generate fixes
        print("\n🔧 Generating fixes...")
        fixes = self.generate_fix(analysis)
        
        if not fixes:
            print("⚠️  No automatic fixes generated. Manual intervention required.")
            return None
        
        print(f"\n📝 Generated fixes for {len(fixes)} files:")
        for file_path in fixes:
            print(f"  - {file_path}")
        
        # Create the PR
        print("\n🚀 Creating pull request...")
        
        try:
            pr_url = self.general_agent.process_issue(
                repo_url="https://github.com/weecology/DeepForest.git",
                repo_name="weecology/DeepForest",
                issue_number=issue_number,
                example_changes=fixes
            )
            
            print(f"\n✅ Pull request created: {pr_url}")
            return pr_url
            
        except Exception as e:
            print(f"\n❌ Error creating PR: {e}")
            
            # Save fixes locally
            self._save_fixes_locally(fixes, issue_number)
            return None
    
    def _save_fixes_locally(self, fixes: Dict[str, str], issue_number: int):
        """Save fixes to local files for manual review."""
        
        output_dir = f"deepforest_fixes_issue_{issue_number}"
        os.makedirs(output_dir, exist_ok=True)
        
        for file_path, content in fixes.items():
            # Create subdirectories
            full_path = os.path.join(output_dir, file_path)
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            
            # Write file
            with open(full_path, 'w') as f:
                f.write(content)
        
        print(f"\n💾 Fixes saved locally to: {output_dir}/")
        print("You can review and manually apply these changes.")


def main():
    """Main entry point."""
    
    import argparse
    
    parser = argparse.ArgumentParser(description="Solve DeepForest issues")
    parser.add_argument("issue_number", type=int, help="Issue number to solve")
    parser.add_argument("--analyze-only", action="store_true", 
                       help="Only analyze, don't create PR")
    args = parser.parse_args()
    
    # Get credentials
    github_token = os.getenv("GITHUB_TOKEN")
    openai_key = os.getenv("OPENAI_API_KEY")
    
    if not github_token:
        print("❌ Please set GITHUB_TOKEN environment variable")
        return
    
    # Create solver
    solver = DeepForestIssueSolver(github_token, openai_key)
    
    if args.analyze_only:
        # Just analyze
        analysis = solver.analyze_issue(args.issue_number)
        print(json.dumps(analysis, indent=2))
    else:
        # Solve the issue
        solver.solve_issue(args.issue_number)


if __name__ == "__main__":
    main()