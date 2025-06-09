#!/usr/bin/env python3
"""
ML-Specific GitHub PR Agent: Specialized for machine learning repositories.

This agent can:
- Analyze ML-related issues (model performance, data preprocessing, etc.)
- Suggest ML-specific fixes (hyperparameter tuning, architecture changes)
- Create PRs with ML experiment results

Requirements:
    pip install PyGithub gitpython openai python-dotenv scikit-learn numpy pandas
"""

import os
import json
import re
from typing import Dict, Any, List, Optional
import tempfile

from github import Github
from git import Repo
import openai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class MLGitHubAgent:
    """Specialized GitHub agent for ML repositories."""
    
    def __init__(self, github_token: str, openai_api_key: Optional[str] = None):
        self.github = Github(github_token)
        self.openai_api_key = openai_api_key
        if openai_api_key:
            openai.api_key = openai_api_key
    
    def analyze_ml_issue(self, issue_details: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze ML-specific issues using AI.
        
        Args:
            issue_details: GitHub issue details
            
        Returns:
            Analysis results with ML-specific suggestions
        """
        if not self.openai_api_key:
            return {"type": "manual", "analysis": "Manual review required"}
        
        # Check for ML-specific keywords
        ml_keywords = ["accuracy", "loss", "overfitting", "underfitting", 
                      "performance", "model", "training", "dataset", "hyperparameter"]
        
        issue_text = f"{issue_details['title']} {issue_details['body']}".lower()
        is_ml_issue = any(keyword in issue_text for keyword in ml_keywords)
        
        if not is_ml_issue:
            return {"type": "general", "analysis": "Not an ML-specific issue"}
        
        prompt = f"""
        Analyze this machine learning issue and provide specific technical suggestions:
        
        Title: {issue_details['title']}
        Description: {issue_details['body']}
        
        Please identify:
        1. Type of ML problem (classification, regression, etc.)
        2. Potential causes of the issue
        3. Specific code changes needed
        4. Hyperparameters to tune (if applicable)
        5. Data preprocessing suggestions (if applicable)
        
        Format your response as JSON with keys: problem_type, causes, code_changes, hyperparameters, preprocessing
        """
        
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an ML engineer analyzing GitHub issues. Respond in JSON format."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000
            )
            
            analysis = json.loads(response.choices[0].message.content)
            return {"type": "ml", "analysis": analysis}
        except Exception as e:
            print(f"AI analysis failed: {e}")
            return {"type": "error", "analysis": str(e)}
    
    def generate_ml_fix(self, issue_analysis: Dict[str, Any]) -> Dict[str, str]:
        """
        Generate ML-specific code fixes based on issue analysis.
        
        Args:
            issue_analysis: Analysis results from analyze_ml_issue
            
        Returns:
            Dictionary of file paths to code content
        """
        if issue_analysis["type"] != "ml":
            return {}
        
        analysis = issue_analysis["analysis"]
        changes = {}
        
        # Generate model improvement code if needed
        if "model" in str(analysis.get("code_changes", "")):
            changes["model_improvements.py"] = self._generate_model_code(analysis)
        
        # Generate hyperparameter tuning code if needed
        if analysis.get("hyperparameters"):
            changes["hyperparameter_tuning.py"] = self._generate_tuning_code(analysis)
        
        # Generate data preprocessing improvements if needed
        if analysis.get("preprocessing"):
            changes["data_preprocessing.py"] = self._generate_preprocessing_code(analysis)
        
        # Always generate an experiment script
        changes["run_experiment.py"] = self._generate_experiment_script(analysis)
        
        return changes
    
    def _generate_model_code(self, analysis: Dict[str, Any]) -> str:
        """Generate improved model code."""
        return '''"""
Model improvements based on issue analysis.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np


class ImprovedModel(nn.Module):
    """Improved model architecture addressing reported issues."""
    
    def __init__(self, input_dim, hidden_dim=128, output_dim=10, dropout_rate=0.3):
        super(ImprovedModel, self).__init__()
        
        # Add batch normalization to address training instability
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        self.fc3 = nn.Linear(hidden_dim // 2, output_dim)
        
        # Initialize weights using Xavier initialization
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        
        x = self.fc3(x)
        return x


class ModelWithRegularization(nn.Module):
    """Model with L2 regularization to prevent overfitting."""
    
    def __init__(self, input_dim, hidden_dim=128, output_dim=10):
        super(ModelWithRegularization, self).__init__()
        
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, output_dim)
        )
    
    def forward(self, x):
        return self.model(x)
    
    def l2_regularization(self):
        """Calculate L2 regularization term."""
        l2_reg = 0
        for param in self.parameters():
            l2_reg += torch.norm(param, 2)
        return l2_reg


def create_model(config):
    """Factory function to create model based on configuration."""
    model_type = config.get('model_type', 'improved')
    
    if model_type == 'improved':
        return ImprovedModel(
            input_dim=config['input_dim'],
            hidden_dim=config.get('hidden_dim', 128),
            output_dim=config['output_dim'],
            dropout_rate=config.get('dropout_rate', 0.3)
        )
    elif model_type == 'regularized':
        return ModelWithRegularization(
            input_dim=config['input_dim'],
            hidden_dim=config.get('hidden_dim', 128),
            output_dim=config['output_dim']
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
'''
    
    def _generate_tuning_code(self, analysis: Dict[str, Any]) -> str:
        """Generate hyperparameter tuning code."""
        return '''"""
Hyperparameter tuning for model optimization.
"""

import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import cross_val_score
import numpy as np
from model_improvements import create_model


def objective(trial):
    """Optuna objective function for hyperparameter tuning."""
    
    # Suggest hyperparameters
    config = {
        'input_dim': 784,  # Example: MNIST
        'output_dim': 10,
        'hidden_dim': trial.suggest_int('hidden_dim', 32, 512, step=32),
        'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.5),
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True),
        'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64, 128]),
        'weight_decay': trial.suggest_float('weight_decay', 1e-5, 1e-2, log=True),
        'model_type': trial.suggest_categorical('model_type', ['improved', 'regularized'])
    }
    
    # Create model
    model = create_model(config)
    
    # Training setup
    optimizer = optim.Adam(
        model.parameters(), 
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    # Train and evaluate (simplified for example)
    # In real implementation, you would load your dataset here
    val_accuracy = train_and_evaluate(model, optimizer, config)
    
    return val_accuracy


def train_and_evaluate(model, optimizer, config):
    """Train model and return validation accuracy."""
    # This is a placeholder - implement actual training logic
    # based on your specific dataset and requirements
    
    # Example training loop structure:
    criterion = nn.CrossEntropyLoss()
    
    # Simulate training (replace with actual implementation)
    epochs = 10
    val_accuracy = 0.85 + np.random.random() * 0.1  # Placeholder
    
    return val_accuracy


def run_hyperparameter_search(n_trials=100):
    """Run Optuna hyperparameter optimization."""
    
    # Create study
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=42)
    )
    
    # Optimize
    study.optimize(objective, n_trials=n_trials)
    
    # Print results
    print(f"Best trial:")
    print(f"  Accuracy: {study.best_value:.4f}")
    print(f"  Params: {study.best_params}")
    
    # Save results
    with open('hyperparameter_results.json', 'w') as f:
        json.dump({
            'best_value': study.best_value,
            'best_params': study.best_params,
            'n_trials': n_trials
        }, f, indent=2)
    
    return study.best_params


if __name__ == "__main__":
    best_params = run_hyperparameter_search(n_trials=50)
    print(f"Best parameters found: {best_params}")
'''
    
    def _generate_preprocessing_code(self, analysis: Dict[str, Any]) -> str:
        """Generate data preprocessing code."""
        return '''"""
Data preprocessing improvements to address reported issues.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
import torch
from torch.utils.data import Dataset, DataLoader


class ImprovedDataProcessor:
    """Improved data preprocessing pipeline."""
    
    def __init__(self, scaling_method='standard', apply_pca=False, n_components=None):
        self.scaling_method = scaling_method
        self.apply_pca = apply_pca
        self.n_components = n_components
        
        # Initialize scalers
        if scaling_method == 'standard':
            self.scaler = StandardScaler()
        elif scaling_method == 'minmax':
            self.scaler = MinMaxScaler()
        elif scaling_method == 'robust':
            self.scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaling method: {scaling_method}")
        
        self.pca = PCA(n_components=n_components) if apply_pca else None
        self.feature_selector = None
    
    def fit(self, X, y=None):
        """Fit preprocessing pipeline."""
        # Fit scaler
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit PCA if enabled
        if self.apply_pca:
            X_scaled = self.pca.fit_transform(X_scaled)
        
        return self
    
    def transform(self, X):
        """Transform data using fitted preprocessing pipeline."""
        # Scale data
        X_scaled = self.scaler.transform(X)
        
        # Apply PCA if enabled
        if self.apply_pca and self.pca is not None:
            X_scaled = self.pca.transform(X_scaled)
        
        return X_scaled
    
    def fit_transform(self, X, y=None):
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)


class AugmentedDataset(Dataset):
    """Dataset with data augmentation for better generalization."""
    
    def __init__(self, X, y, augment=True, noise_level=0.01):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
        self.augment = augment
        self.noise_level = noise_level
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]
        
        # Apply augmentation during training
        if self.augment:
            # Add Gaussian noise
            noise = torch.randn_like(x) * self.noise_level
            x = x + noise
            
            # Random dropout (zeroing out features)
            if torch.rand(1) < 0.1:  # 10% chance
                mask = torch.rand_like(x) > 0.1  # Drop 10% of features
                x = x * mask
        
        return x, y


def create_data_loaders(X_train, y_train, X_val, y_val, 
                       batch_size=32, augment_train=True):
    """Create PyTorch data loaders with preprocessing."""
    
    # Create datasets
    train_dataset = AugmentedDataset(X_train, y_train, augment=augment_train)
    val_dataset = AugmentedDataset(X_val, y_val, augment=False)
    
    # Create loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    return train_loader, val_loader


def analyze_data_issues(X, y):
    """Analyze potential data quality issues."""
    issues = []
    
    # Check for missing values
    if isinstance(X, pd.DataFrame):
        missing = X.isnull().sum().sum()
        if missing > 0:
            issues.append(f"Found {missing} missing values")
    
    # Check for class imbalance
    if y is not None:
        unique, counts = np.unique(y, return_counts=True)
        imbalance_ratio = counts.max() / counts.min()
        if imbalance_ratio > 3:
            issues.append(f"Class imbalance detected (ratio: {imbalance_ratio:.2f})")
    
    # Check for outliers
    if isinstance(X, np.ndarray):
        z_scores = np.abs((X - X.mean(axis=0)) / X.std(axis=0))
        outliers = (z_scores > 3).sum()
        if outliers > 0:
            issues.append(f"Found {outliers} potential outliers")
    
    return issues


if __name__ == "__main__":
    # Example usage
    print("Data preprocessing module loaded successfully")
    
    # Example: analyze synthetic data
    X_example = np.random.randn(1000, 20)
    y_example = np.random.randint(0, 2, 1000)
    
    issues = analyze_data_issues(X_example, y_example)
    print(f"Data issues found: {issues}")
'''
    
    def _generate_experiment_script(self, analysis: Dict[str, Any]) -> str:
        """Generate experiment script to test the fixes."""
        return '''"""
Experiment script to test model improvements.
"""

import os
import json
import argparse
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from model_improvements import create_model
from data_preprocessing import ImprovedDataProcessor, create_data_loaders
from hyperparameter_tuning import run_hyperparameter_search


class ExperimentRunner:
    """Run and track ML experiments."""
    
    def __init__(self, experiment_name, config):
        self.experiment_name = experiment_name
        self.config = config
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create experiment directory
        self.exp_dir = f"experiments/{experiment_name}_{self.timestamp}"
        os.makedirs(self.exp_dir, exist_ok=True)
        
        # Initialize tensorboard
        self.writer = SummaryWriter(f"{self.exp_dir}/tensorboard")
        
        # Save config
        with open(f"{self.exp_dir}/config.json", 'w') as f:
            json.dump(config, f, indent=2)
    
    def train_epoch(self, model, train_loader, optimizer, criterion, epoch):
        """Train for one epoch."""
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            
            # Add L2 regularization if model supports it
            if hasattr(model, 'l2_regularization'):
                loss += self.config.get('l2_lambda', 0.01) * model.l2_regularization()
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            if batch_idx % 10 == 0:
                print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.4f}')
        
        accuracy = 100. * correct / total
        avg_loss = total_loss / len(train_loader)
        
        return avg_loss, accuracy
    
    def evaluate(self, model, val_loader, criterion):
        """Evaluate model on validation set."""
        model.eval()
        total_loss = 0
        correct = 0
        total = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in val_loader:
                output = model(data)
                loss = criterion(output, target)
                
                total_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
                
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        
        accuracy = 100. * correct / total
        avg_loss = total_loss / len(val_loader)
        
        return avg_loss, accuracy, all_predictions, all_targets
    
    def plot_results(self, predictions, targets):
        """Plot confusion matrix and save it."""
        cm = confusion_matrix(targets, predictions)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(f"{self.exp_dir}/confusion_matrix.png")
        plt.close()
    
    def run_experiment(self, train_loader, val_loader):
        """Run the full experiment."""
        # Create model
        model = create_model(self.config)
        print(f"Model architecture:\n{model}")
        
        # Setup training
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(
            model.parameters(),
            lr=self.config.get('learning_rate', 0.001),
            weight_decay=self.config.get('weight_decay', 0.0001)
        )
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=5, factor=0.5
        )
        
        # Training loop
        best_val_acc = 0
        train_losses = []
        val_losses = []
        train_accs = []
        val_accs = []
        
        for epoch in range(self.config.get('epochs', 50)):
            # Train
            train_loss, train_acc = self.train_epoch(
                model, train_loader, optimizer, criterion, epoch
            )
            
            # Evaluate
            val_loss, val_acc, predictions, targets = self.evaluate(
                model, val_loader, criterion
            )
            
            # Update scheduler
            scheduler.step(val_loss)
            
            # Log metrics
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Loss/val', val_loss, epoch)
            self.writer.add_scalar('Accuracy/train', train_acc, epoch)
            self.writer.add_scalar('Accuracy/val', val_acc, epoch)
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accs.append(train_acc)
            val_accs.append(val_acc)
            
            print(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), f"{self.exp_dir}/best_model.pth")
                print(f"Saved best model with validation accuracy: {val_acc:.2f}%")
        
        # Final evaluation
        model.load_state_dict(torch.load(f"{self.exp_dir}/best_model.pth"))
        val_loss, val_acc, predictions, targets = self.evaluate(
            model, val_loader, criterion
        )
        
        # Save final results
        results = {
            'best_val_accuracy': best_val_acc,
            'final_val_accuracy': val_acc,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_accuracies': train_accs,
            'val_accuracies': val_accs,
            'classification_report': classification_report(
                targets, predictions, output_dict=True
            )
        }
        
        with open(f"{self.exp_dir}/results.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        # Plot results
        self.plot_results(predictions, targets)
        self.plot_training_curves(train_losses, val_losses, train_accs, val_accs)
        
        return results
    
    def plot_training_curves(self, train_losses, val_losses, train_accs, val_accs):
        """Plot training curves."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss curves
        ax1.plot(train_losses, label='Train Loss')
        ax1.plot(val_losses, label='Val Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        
        # Accuracy curves
        ax2.plot(train_accs, label='Train Accuracy')
        ax2.plot(val_accs, label='Val Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(f"{self.exp_dir}/training_curves.png")
        plt.close()


def main():
    """Main experiment entry point."""
    parser = argparse.ArgumentParser(description='Run ML experiment')
    parser.add_argument('--tune', action='store_true', help='Run hyperparameter tuning first')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--experiment-name', type=str, default='ml_fix', help='Experiment name')
    args = parser.parse_args()
    
    # Configuration
    config = {
        'input_dim': 784,  # Example: MNIST
        'output_dim': 10,
        'hidden_dim': 128,
        'dropout_rate': 0.3,
        'learning_rate': 0.001,
        'weight_decay': 0.0001,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'model_type': 'improved'
    }
    
    # Run hyperparameter tuning if requested
    if args.tune:
        print("Running hyperparameter tuning...")
        best_params = run_hyperparameter_search(n_trials=20)
        config.update(best_params)
    
    # Create dummy data (replace with your actual data loading)
    print("Loading data...")
    X_train = np.random.randn(1000, 784)
    y_train = np.random.randint(0, 10, 1000)
    X_val = np.random.randn(200, 784)
    y_val = np.random.randint(0, 10, 200)
    
    # Preprocess data
    processor = ImprovedDataProcessor(scaling_method='standard')
    X_train = processor.fit_transform(X_train)
    X_val = processor.transform(X_val)
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        X_train, y_train, X_val, y_val, 
        batch_size=config['batch_size']
    )
    
    # Run experiment
    runner = ExperimentRunner(args.experiment_name, config)
    results = runner.run_experiment(train_loader, val_loader)
    
    print(f"\nExperiment completed!")
    print(f"Best validation accuracy: {results['best_val_accuracy']:.2f}%")
    print(f"Results saved to: {runner.exp_dir}")


if __name__ == "__main__":
    main()
'''
    
    def process_ml_issue(self, repo_url: str, repo_name: str, issue_number: int) -> str:
        """
        Process an ML-specific issue and create a PR.
        
        Args:
            repo_url: Repository URL
            repo_name: Repository name (owner/repo)
            issue_number: Issue number
            
        Returns:
            Pull request URL
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            # Clone repository
            repo = Repo.clone_from(repo_url, temp_dir)
            
            # Get issue details
            github_repo = self.github.get_repo(repo_name)
            issue = github_repo.get_issue(issue_number)
            issue_details = {
                "number": issue.number,
                "title": issue.title,
                "body": issue.body,
                "labels": [label.name for label in issue.labels]
            }
            
            # Analyze ML issue
            analysis = self.analyze_ml_issue(issue_details)
            print(f"Issue analysis: {analysis}")
            
            # Generate ML-specific fixes
            code_changes = self.generate_ml_fix(analysis)
            
            # Create branch
            branch_name = f"ml-fix-issue-{issue_number}"
            repo.git.checkout('-b', branch_name)
            
            # Apply changes
            for file_path, content in code_changes.items():
                full_path = os.path.join(temp_dir, file_path)
                os.makedirs(os.path.dirname(full_path), exist_ok=True)
                with open(full_path, 'w') as f:
                    f.write(content)
            
            # Commit and push
            repo.git.add(A=True)
            repo.index.commit(f"Fix ML issue #{issue_number}: {issue.title}")
            repo.remote('origin').push(branch_name)
            
            # Create PR
            pr_body = f"""
## 🤖 ML Issue Fix

This PR addresses issue #{issue_number}: **{issue.title}**

### 📊 Issue Analysis
{json.dumps(analysis.get('analysis', {}), indent=2)}

### 🔧 Changes Made
- ✨ Improved model architecture with better regularization
- 📈 Added hyperparameter tuning capabilities
- 🔄 Enhanced data preprocessing pipeline
- 📝 Created experiment runner for reproducible results

### 🧪 Testing
Run the experiment script to validate improvements:
```bash
python run_experiment.py --tune --epochs 50
```

### 📋 Checklist
- [x] Code follows ML best practices
- [x] Added proper documentation
- [x] Included hyperparameter tuning
- [x] Created reproducible experiments
- [ ] Validated on test dataset
- [ ] Performance metrics improved

### 🎯 Expected Improvements
Based on the analysis, these changes should address:
- Model performance issues
- Training stability
- Generalization capability

Fixes #{issue_number}

---
*This PR was automatically generated by the ML GitHub Agent* 🚀
"""
            
            pr = github_repo.create_pull(
                title=f"[ML Fix] {issue.title} (#{issue_number})",
                body=pr_body,
                head=branch_name,
                base="main"
            )
            
            return pr.html_url


# Example usage
if __name__ == "__main__":
    # Load credentials
    GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    
    if not GITHUB_TOKEN:
        print("Please set GITHUB_TOKEN environment variable")
        exit(1)
    
    # Create ML agent
    agent = MLGitHubAgent(GITHUB_TOKEN, OPENAI_API_KEY)
    
    # Example: Process an ML issue
    # pr_url = agent.process_ml_issue(
    #     repo_url="https://github.com/example/ml-project.git",
    #     repo_name="example/ml-project",
    #     issue_number=42
    # )
    # print(f"Created PR: {pr_url}")
    
    print("ML GitHub Agent ready!")
    print("Use agent.process_ml_issue() to automatically fix ML issues")