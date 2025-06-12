# Workspace for Levanter and Anticipation Repos

**Reference:** [Anticipatory Music Transformer](https://arxiv.org/abs/2306.08620) &nbsp; ([Blog](https://crfm.stanford.edu/2023/06/16/anticipatory-music-transformer.html))

## Setup Instructions

1. Clone this repository:
```bash
git clone --recurse-submodules https://github.com/josmithiii/amt.git
cd amt
```

2. Set up a uv environment for the `levanterForAnticipation` and `anticipation-lancelot` submodule repositories:
```bash
bash setup.sh
```

This will:
- Create a Python 3.10 virtual environment using uv
- Install both repos in editable mode with compatible dependencies
- Configure the environment for VS Code/Pylance compatibility
- Automatically fix any JAX compatibility issues

3. Verify the installation:
```bash
uv run python -c "import anticipation; import levanter; print('✅ All packages imported successfully!')"
```

4. Use the environment:
```bash
# Activate the environment
source .venv/bin/activate

# Or run commands directly
uv run python your_script.py
```

## Submodules

1. Siqi's forks:
  - `levanter-siqi/`: Levanter modified by Siqi for Anticipation use:
  - `anticipation-siqi/`: Siqi's fork of Anticipation by John Thickstun
2. Lancelot's forks:
  - `levanterForAnticipation/`: Levanter modified by Lancelot for Anticipation use:
  - `anticipation-lancelot/`: Lancelot's fork of Anticipation by John Thickstun
- For switching submodule sets, see https://chatgpt.com/share/684a6fcc-1a64-800f-8927-7c052fa36db3
- To switch submodules:
```bash
cp dot-gitmodules-[siqi|lancelot] .gitmodules
git submodule sync # Syncs the config
git submodule update --init --recursive --remote
```
- Update submodules:
```bash
git submodule update --remote --merge
```

## Development

This workspace is configured for use with Cursor IDE. The `uv` environment uses:
- Python 3.10
- Compatible versions of NumPy (1.x), JAX (0.4.x), and other dependencies
- Editable installs for both main repositories
- VS Code/Pylance compatibility mode for static analysis
