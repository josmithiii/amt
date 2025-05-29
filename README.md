## Setup Instructions

1. Clone this repository:
```bash
git clone <amt repository-url>
cd amt
```

2. Set up a virtual environment for the `levanterForAnticipation` and `anticipation-lancelot` repositories:
```bash
bash setup.sh
```

3. Install JAX for your system:
- CPU: `uv add 'jax[cpu]'`
- CUDA 12: `uv add 'jax[cuda12]'`
- CUDA 11: `uv add 'jax[cuda11]'`

4. Use the environment:
- Activate: `source .venv/bin/activate`
- Or run directly: `uv run python your_script.py`

## Development

This workspace is configured for use with Cursor IDE. The `.cursorrules` file contains workspace-specific settings for Python development.
