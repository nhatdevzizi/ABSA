# Vietnamese sentiment experiments

The project compares three-class sentiment models on frozen Vietnamese datasets.

| Folder | Contents |
| --- | --- |
| `uit_vsfc/models/` | BiLSTM and frozen ViSoBERT plus BiLSTM |
| `uit_vsfc/baselines/` | TF-IDF logistic regression and SVM workflow |
| `uit_vsfc/data/` | Frozen split export, loaders, and batch preparation |
| `uit_vsfc/evaluation/` | Shared metrics and result summaries |
| `configs/` | Experiment configuration |
| `scripts/` | Command shortcuts |
| `docs/` | Model rules and command instructions |
| `notebooks/` | Original experiment notebook; historical reference |
| `data/` | Existing datasets and manifests |
| `outputs/` | Generated checkpoints, predictions, and results |

See [commands](docs/commands.md) and [model rules](docs/model_rules.md).

Python dependencies and interpreter requirements stay in `pyproject.toml`. `uv.lock` fixes dependency versions. The local environment stays in `.venv/`; these files stay at the root for tool discovery. No environment variables are currently required. Local `.env` files are ignored and are not loaded by the code.

The neural model definitions are available, but the command workflow currently runs the classical baselines only. Dataset files and model behavior are preserved by the folder refactor.
