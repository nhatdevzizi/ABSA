from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score
import numpy as np
import argparse
from pathlib import Path
import pandas as pd

CORE_COLUMNS = ["id", "y_true", "y_pred", "p_neg", "p_neu", "p_pos"]
METRIC_NAMES = ("macro_f1", "accuracy", "weighted_f1", "balanced_acc", "ece",
                "f1_neg", "f1_neu", "f1_pos", "truncation_rate")


def save_csv(path, frame):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def evaluate_file(path, expected=None):
    frame = pd.read_csv(path, dtype={"id": str}, keep_default_na=False)
    if list(frame.columns[:6]) != CORE_COLUMNS:
        raise ValueError(f"{path}: required core columns must come first, in order")
    if expected is not None:
        if frame.id.tolist() != expected.id.tolist() or frame.y_true.tolist() != expected.label.tolist():
            raise ValueError(f"{path}: IDs or labels differ from the frozen split")
    return evaluate_predictions(frame)


def update_results(path, model, dataset, fraction, seed, infer_mode, metrics):
    columns = ["model", "dataset", "fraction", "seed", "infer_mode", "metric", "value"]
    path = Path(path)
    results = pd.read_csv(path, keep_default_na=False) if path.exists() else pd.DataFrame(columns=columns)
    same = ((results.model == model) & (results.dataset == dataset)
            & (results.fraction.astype(str) == str(fraction))
            & (results.seed.astype(str) == str(seed)) & (results.infer_mode == infer_mode))
    rows = [dict(model=model, dataset=dataset, fraction=fraction, seed=seed,
                 infer_mode=infer_mode, metric=metric, value=value)
            for metric, value in metrics.items() if metric != "confusion_matrix"]
    save_csv(path, pd.concat([results.loc[~same], pd.DataFrame(rows)], ignore_index=True))


def summarize(path):
    path = Path(path)
    results = pd.read_csv(path, keep_default_na=False)
    keys = ["model", "dataset", "fraction", "infer_mode", "metric"]
    if results.duplicated(keys + ["seed"]).any():
        raise ValueError("Duplicate final metrics for a run")
    summary = results.groupby(keys, sort=False).value.agg(["count", "mean", "std"]).reset_index()
    summary = summary.rename(columns={"count": "n_seeds", "std": "sample_std"})
    expected = {100: 5, 5: 5, 10: 3, 1: 3}
    summary["expected_seeds"] = summary.fraction.map(expected)
    summary.loc[summary.model.str.contains("Hard"), "expected_seeds"] = 1
    summary["complete"] = summary.n_seeds == summary.expected_seeds
    save_csv(path.with_name("summary.csv"), summary)
    return summary


def evaluate_predictions(frame):
    """Shared evaluator for standard prediction columns; class order is 0,1,2."""
    if frame.empty or frame.id.eq("").any() or frame.id.duplicated().any():
        raise ValueError("Prediction IDs must be nonempty and unique")
    truth, predicted = frame.y_true, frame.y_pred
    if not truth.isin([0, 1, 2]).all() or not predicted.isin([0, 1, 2]).all():
        raise ValueError("Prediction labels must be 0,1,2")
    probabilities = frame[["p_neg", "p_neu", "p_pos"]].to_numpy()
    if not np.isfinite(probabilities).all() or (probabilities < 0).any() or (probabilities > 1).any():
        raise ValueError("Probabilities must be finite and between zero and one")
    if not np.allclose(probabilities.sum(axis=1), 1):
        raise ValueError("Probability rows must sum to one")
    per_class = f1_score(truth, predicted, labels=[0, 1, 2], average=None, zero_division=0)
    confidence = probabilities.max(axis=1)
    # ECE measures confidence in the probability argmax.
    correct = probabilities.argmax(axis=1) == truth.to_numpy()
    bins = np.minimum((confidence * 10).astype(int), 9)
    ece = sum(np.mean(bins == index) * abs(correct[bins == index].mean() - confidence[bins == index].mean())
              for index in range(10) if np.any(bins == index))
    return {
        "macro_f1": f1_score(truth, predicted, labels=[0, 1, 2], average="macro", zero_division=0),
        "accuracy": accuracy_score(truth, predicted),
        "weighted_f1": f1_score(truth, predicted, labels=[0, 1, 2], average="weighted", zero_division=0),
        "balanced_acc": balanced_accuracy_score(truth, predicted), "ece": float(ece),
        "f1_neg": per_class[0], "f1_neu": per_class[1], "f1_pos": per_class[2],
        "confusion_matrix": confusion_matrix(truth, predicted, labels=[0, 1, 2]).tolist(),
        **({"truncation_rate": float(frame.truncated.mean())} if "truncated" in frame else {}),
    }


def measure(y_true, y_pred):
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def main():
    parser = argparse.ArgumentParser(description="Shared evaluator for saved prediction CSV files")
    parser.add_argument("prediction", nargs="?", type=Path)
    parser.add_argument("--results", type=Path, default=Path("outputs/results.csv"))
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args()
    if args.prediction:
        # Identity follows the shared filename convention, including optional LA mode.
        parts = args.prediction.stem.split("__")
        if len(parts) not in (4, 5) or not parts[2].startswith("f") or not parts[3].startswith("s"):
            parser.error("Prediction filename must follow the shared model/dataset/fraction/seed convention")
        model, dataset, fraction, seed = parts[:4]
        if dataset not in ("neu_esc_dev", "neu_esc_test", "uit_vsfc_test"):
            parser.error("Unknown evaluation dataset")
        if fraction[1:] not in ("1", "5", "10", "100"):
            parser.error("Fraction must be 1,5,10,100")
        allowed_seeds = ("42", "43", "44", "45", "46") if fraction[1:] in ("5", "100") else ("42", "43", "44")
        if seed[1:] not in (*allowed_seeds, "NA"):
            parser.error("Invalid seed for fraction")
        if len(parts) == 5 and parts[4] not in ("bal", "prior"):
            parser.error("Unknown LA inference mode")
        metrics = evaluate_file(args.prediction)
        update_results(args.results, model, dataset, int(fraction[1:]), seed[1:],
                       parts[4] if len(parts) == 5 else "na", metrics)
        matrix_path = args.prediction.with_suffix(".confusion.csv")
        save_csv(matrix_path, pd.DataFrame(metrics["confusion_matrix"],
                                         columns=["pred_neg", "pred_neu", "pred_pos"]))
        print(f"Updated {args.results} from {args.prediction}")
    elif not args.summary:
        parser.error("Provide a prediction CSV or --summary")
    if args.summary:
        summarize(args.results)


if __name__ == "__main__":
    main()
