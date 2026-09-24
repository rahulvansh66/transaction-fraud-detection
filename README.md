# transaction-fraud-detection
## Running the notebooks

Dependencies are managed with [uv](https://docs.astral.sh/uv/); the environment lives in `.venv/` at the repo root.

1. Install everything (including dev tools such as `ipykernel`):
   ```powershell
   uv sync
   ```
2. Register the kernel once (optional, for Jupyter/JupyterLab outside VS Code):
   ```powershell
   uv run python -m ipykernel install --user --name transaction-fraud-detection --display-name "Python (transaction-fraud-detection)"
   ```
3. Select the kernel in VS Code: open a notebook, click **Select Kernel** (top-right) →
   **Python Environments…** → **.venv**. If it is not listed, run
   **Developer: Reload Window**, or **Python: Select Interpreter** →
   `.venv\Scripts\python.exe`.
4. Verify: a cell running `import sys; sys.executable` should print a path inside `.venv`.

Headless run (e.g. to check a notebook end to end):
```powershell
uv run jupyter nbconvert --to notebook --execute notebooks/<name>.ipynb --output-dir <tmp-dir>
```
