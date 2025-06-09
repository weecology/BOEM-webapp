#!/bin/bash
# Quick start script for DeepForest GitHub Agent

echo "🌲 DeepForest GitHub Agent - Quick Start"
echo "========================================"
echo ""

# Step 1: Create and enter project directory
echo "Step 1: Creating project directory..."
mkdir -p ~/deepforest-agent
cd ~/deepforest-agent
echo "✅ Created ~/deepforest-agent"
echo ""

# Step 2: Create virtual environment
echo "Step 2: Creating Python virtual environment..."
python3 -m venv venv
echo "✅ Virtual environment created"
echo ""

# Step 3: Activate and install packages
echo "Step 3: Installing packages..."
source venv/bin/activate
pip install --upgrade pip --quiet
pip install PyGithub gitpython python-dotenv openai requests --quiet
echo "✅ Packages installed"
echo ""

# Step 4: Create .env template
echo "Step 4: Creating configuration template..."
cat > .env << 'EOF'
# Your GitHub Personal Access Token
# Get it from: https://github.com/settings/tokens/new
GITHUB_TOKEN=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

# Optional: OpenAI API Key
OPENAI_API_KEY=
EOF
echo "✅ Created .env file"
echo ""

# Step 5: Download all the agent scripts
echo "Step 5: Downloading agent scripts..."
echo "(Copy the Python scripts created earlier to this directory)"
echo ""

echo "🎯 NEXT STEPS:"
echo "=============="
echo ""
echo "1. Get your GitHub token:"
echo "   👉 Open: https://github.com/settings/tokens/new"
echo "   👉 Name: 'DeepForest Agent'"
echo "   👉 Select scopes: ✅ repo, ✅ workflow"
echo "   👉 Click 'Generate token'"
echo "   👉 COPY THE TOKEN!"
echo ""
echo "2. Add your token:"
echo "   👉 Run: nano .env"
echo "   👉 Replace YOUR_TOKEN_HERE with your actual token"
echo "   👉 Save and exit (Ctrl+X, Y, Enter)"
echo ""
echo "3. Test your setup:"
echo "   👉 Run: python3 test_setup.py"
echo ""
echo "📁 Current directory: $(pwd)"
echo "🐍 Virtual env: ACTIVATED"
echo ""