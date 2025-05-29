#!/bin/bash

# Setup script for AMT project using uv
# This script sets up a shared environment for levanterForAnticipation and anticipation-lancelot

set -e  # Exit on any error

echo "🚀 Setting up AMT project environment with uv..."

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "❌ uv is not installed. Please install it first:"
    echo "   curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# Create uv project if it doesn't exist
if [ ! -f "pyproject.toml" ]; then
    echo "📝 Creating uv project..."
    uv init --name amt-project --python 3.10
    
    # Add config for VS Code/Pylance compatibility
    echo "🔧 Configuring for VS Code/static analyzer compatibility..."
    cat >> pyproject.toml << 'EOF'

[tool.uv]
config-settings = { editable_mode = "compat" }
EOF
fi

echo "🐍 Setting up Python 3.10 environment..."
uv python install 3.10

# Install common dependencies first
echo "📦 Installing core dependencies..."

# Core ML/scientific computing stack
uv add "numpy>=1.22.4"
uv add "torch>=2.0.1"
uv add "matplotlib>=3.7.0"
uv add "tqdm>=4.65.0"

# For anticipation-lancelot
echo "🎵 Installing anticipation-lancelot dependencies..."
uv add "midi2audio==0.1.1"
uv add "mido==1.2.10"

# Install anticipation-lancelot in editable mode
echo "🔧 Installing anticipation-lancelot package..."
uv add --editable "./anticipation-lancelot"

# For levanterForAnticipation - handle potential conflicts
echo "🧠 Installing levanterForAnticipation dependencies..."

# Install transformers with a compatible version that works for both
uv add "transformers>=4.22.0,<5.0.0"

# JAX ecosystem (user needs to install JAX separately based on their system)
echo "⚠️  Note: You'll need to install JAX separately based on your system:"
echo "   For CPU: uv add 'jax[cpu]'"
echo "   For CUDA: uv add 'jax[cuda12]' (or cuda11 for older CUDA)"

# Other levanter dependencies
uv add "equinox>=0.10.7"
uv add "jaxtyping>=0.2.20"
uv add "optax"
uv add "wandb"
uv add "draccus>=0.6"
uv add "pyarrow>=11.0.0"
uv add "zstandard>=0.20.0"
uv add "datasets==2.11.0"
uv add "gcsfs<2023.10.0"
uv add "braceexpand>=0.1.7"
uv add "jmp>=0.0.3"
uv add "fsspec<2023.10.0"
uv add "tensorstore==0.1.45"
uv add "pytimeparse>=1.1.8"
uv add "humanfriendly==10.0"
uv add "safetensors[numpy]"
uv add "tblib>=1.7.0,<2.0.0"
uv add "dataclasses-json"
uv add "ray[default]"
uv add "pydantic<2"
uv add "rich>=13"

# Install haliax from git (as specified in levanter)
echo "🔗 Installing haliax from git..."
uv add "haliax @ git+https://github.com/stanford-crfm/haliax.git"

# Install levanterForAnticipation in editable mode
echo "🔧 Installing levanterForAnticipation package..."
uv add --editable "./levanterForAnticipation"

echo "✅ Environment setup complete!"
echo ""
echo "🎯 Next steps:"
echo "1. Install JAX for your system:"
echo "   uv add 'jax[cpu]'     # For CPU"
echo "   uv add 'jax[cuda12]'  # For CUDA 12"
echo "   uv add 'jax[cuda11]'  # For CUDA 11"
echo ""
echo "2. Activate the environment:"
echo "   source .venv/bin/activate"
echo ""
echo "3. Or run commands with:"
echo "   uv run python your_script.py"
echo ""
echo "🔍 To check the installation:"
echo "   uv run python -c \"import anticipation; import levanter; print('✅ All packages imported successfully!')\"" 