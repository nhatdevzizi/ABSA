# Commands

Run these commands from the project root.

```powershell
uv sync
uv run python -m uit_vsfc --help
uv run python -m uit_vsfc.data.prepare
uv run python -m uit_vsfc tune
uv run python -m uit_vsfc train --model TF_LR --fraction 5 --seed 42
uv run python -m uit_vsfc evaluate --model TF_LR --fraction 5 --seed 42
```

The preparation command verifies source manifests and exports frozen splits. It rewrites `configs/baseline.json` and exported baseline data. It does not sample new splits.

Tune before training. Train before evaluation. Evaluation reads the test sets.

PowerShell shortcut: `./scripts/baseline.ps1 -Command train -Model TF_LR`.

`python main.py` remains a compatible entry point. Custom configurations can use `--config PATH`. Paths are relative to the configuration directory plus its optional `project_root` field.

The baseline pipeline records source hashes. Existing run receipts from the old layout cannot be resumed with this layout. Preserve old outputs and use a new output directory for new runs.
