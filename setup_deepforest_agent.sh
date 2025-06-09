#!/bin/bash
# Setup script for DeepForest GitHub Agent

echo "🌲 DeepForest GitHub Agent Setup"
echo "================================"

# Create project directory
echo "📁 Creating project directory..."
mkdir -p ~/deepforest-agent
cd ~/deepforest-agent

# Create virtual environment
echo "🐍 Creating Python virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "✅ Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "📦 Upgrading pip..."
pip install --upgrade pip

# Install required packages
echo "📦 Installing required packages..."
pip install PyGithub gitpython python-dotenv openai requests

# Create .env file template
echo "📝 Creating .env template..."
cat > .env.template << 'EOF'
# GitHub Personal Access Token
# Get yours at: https://github.com/settings/tokens/new
# Required scopes: repo, workflow
GITHUB_TOKEN=your_github_token_here

# OpenAI API Key (Optional but recommended for better analysis)
# Get yours at: https://platform.openai.com/api-keys
OPENAI_API_KEY=your_openai_api_key_here
EOF

echo ""
echo "✅ Setup complete!"
echo ""
echo "📋 Next steps:"
echo "1. Copy .env.template to .env: cp .env.template .env"
echo "2. Edit .env and add your GitHub token"
echo "3. Run: source venv/bin/activate (to activate the virtual environment)"
echo "4. You're ready to use the agent!"
echo ""
echo "🔑 To get a GitHub token:"
echo "   Visit: https://github.com/settings/tokens/new"
echo "   Select scopes: 'repo' and 'workflow'"
echo ""