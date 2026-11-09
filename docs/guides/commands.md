# Commands

Run these commands from the project root.

```powershell
uv sync
uv run python -m uit_vsfc --help
uv run python -m uit_vsfc.data.prepare
uv run python -m uit_vsfc tune
uv run python -m uit_vsfc train --model TF_LR --fraction 5 --seed 42
uv run python -m uit_vsfc evaluate --model TF_LR --fraction 5 --seed 42
uv run python -m uit_vsfc.baselines.tfidf_emotion_transfer --help
```

The preparation command verifies source manifests and exports frozen splits. It rewrites `configs/baseline.json` and exported baseline data. It does not sample new splits.

Tune before training. Train before evaluation. Evaluation reads the test sets.

`python main.py` remains a compatible entry point. Custom configurations can use `--config PATH`. Paths are relative to the configuration directory plus its optional `project_root` field.

The default `outputs/` folder contains `results.csv` and `summary.csv`. The commands use the sibling `.outputs-state/` folder for tuning, checkpoints, predictions, and run logs. A custom output directory uses a sibling named `.<output-directory-name>-state`.

Start with tuning when the local run state is empty. The baseline checks source hashes before reusing run state.
