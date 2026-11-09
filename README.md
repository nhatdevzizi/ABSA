# Vietnamese sentiment experiments

The project compares three-class sentiment models on frozen Vietnamese datasets.

| Folder | Contents |
| --- | --- |
| `uit_vsfc/baselines/` | [Direct sentiment](uit_vsfc/baselines/tfidf_direct_sentiment.py) and [emotion-to-sentiment transfer](uit_vsfc/baselines/tfidf_emotion_transfer.py) |
| `uit_vsfc/data/` | Frozen split export |
| `uit_vsfc/evaluation/` | Shared metrics and result summaries |
| `configs/` | Experiment configuration |
| `docs/guides/` | Current commands and baseline model rules |
| `docs/reports/` | The two newest LaTeX reports and their assets |
| `data/` | Existing datasets and manifests |
| `outputs/` | Final metrics in `results.csv` and aggregate metrics in `summary.csv` |

Use the [command guide](docs/guides/commands.md) with the [baseline model rules](docs/guides/model_rules.md).

Python dependencies and interpreter requirements stay in `pyproject.toml`. `uv.lock` fixes dependency versions. The local environment stays in `.venv/`; these files stay at the root for tool discovery. No environment variables are currently required. Local `.env` files are ignored and are not loaded by the code.
