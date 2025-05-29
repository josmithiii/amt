## Setup Instructions

1. Clone this repository:
```bash
git clone <amt repository-url>
cd amt
```

2. Set up a uv environment for the `levanterForAnticipation` and `anticipation-lancelot` repositories:
```bash
bash setup.sh
```

This will:
- Create a Python 3.10 virtual environment using uv
- Install both packages in editable mode with compatible dependencies
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

## Project Structure

- `levanterForAnticipation/`: Modified Levanter for anticipation experiments
- `anticipation-lancelot/`: Anticipation-related code and models
- `setup.sh`: Automated setup script using uv

## Development

This workspace is configured for use with Cursor IDE. The environment uses:
- Python 3.10
- Compatible versions of NumPy (1.x), JAX (0.4.x), and other dependencies
- Editable installs for both main repositories
- VS Code/Pylance compatibility mode for static analysis
