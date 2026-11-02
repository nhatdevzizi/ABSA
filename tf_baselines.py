"""Classical baselines on frozen, preprocessed sentiment splits."""

import argparse
import hashlib
import importlib.metadata
import json
import platform
import re
import subprocess
import time
import unicodedata
import shutil
import uuid
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.exceptions import ConvergenceWarning

from measurer import evaluate_predictions, evaluate_file, save_csv, summarize, update_results

MODELS = ("TF_LR", "TF_SVM")
SEEDS = {5: range(42, 47), 100: range(42, 47), 1: range(42, 45), 10: range(42, 45)}
PREPROCESS_VERSION = "nfc-lower-repeat3to2-v1"
SOURCE_FILES = ("tf_baselines.py", "measurer.py", "main.py", "prepare_baseline_data.py",
                "pyproject.toml", "uv.lock")


def settings(value):
    """Record estimator defaults as well as explicitly selected parameters."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return [settings(item) for item in value]
    if isinstance(value, dict):
        return {key: settings(item) for key, item in value.items()}
    if hasattr(value, "get_params"):
        return {"type": type(value).__name__, "params": settings(value.get_params(deep=False))}
    if isinstance(value, StratifiedKFold):
        return {"type": "StratifiedKFold", "n_splits": value.n_splits,
                "shuffle": value.shuffle, "random_state": value.random_state}
    return f"{value.__module__}.{value.__qualname__}" if callable(value) else str(value)


def fit_model(estimator, frame):
    # An unconverged fit must not silently become the final research checkpoint.
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        return estimator.fit(frame.text, frame.label)


def verify_run(folder, config, model, fraction, seed):
    metadata = json.loads((folder / "config.yaml").read_text(encoding="utf-8"))
    verify_files(metadata["data_sha256"])
    verify_source(metadata)
    verify_files({folder / "checkpoint.joblib": metadata["checkpoint_sha256"]})
    if (metadata["model"], metadata["fraction"], metadata["seed"]) != (model, fraction, seed):
        raise ValueError("Run identity does not match saved configuration")
    if metadata["split"] != config["splits"][str(fraction)][str(seed)] or metadata["tests"] != config["tests"]:
        raise ValueError("Configured paths changed; use the original configuration or a new output directory")
    return metadata


def normalize(text):
    return re.sub(r"(.)\1{2,}", r"\1\1", unicodedata.normalize("NFC", text).lower())


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def load_split(path, training=False):
    frame = pd.read_csv(path, dtype={"id": str, "text": str}, keep_default_na=False)
    if not {"id", "text", "label"}.issubset(frame.columns):
        raise ValueError(f"{path}: expected columns id,text,label")
    if frame.empty or frame.id.eq("").any() or frame.id.duplicated().any():
        raise ValueError(f"{path}: IDs must be nonempty and unique; split cannot be empty")
    if frame.text.str.strip().eq("").any() or not frame.label.isin([0, 1, 2]).all():
        raise ValueError(f"{path}: text must be nonempty; labels must be mapped to 0,1,2")
    frame["label"] = frame.label.astype(int)
    if training and set(frame.label) != {0, 1, 2}:
        raise ValueError(f"{path}: training requires all three classes")
    return frame


def check_disjoint(left, right, check_ids=True):
    if check_ids and set(left.id) & set(right.id):
        raise ValueError("NEU-ESC split IDs overlap")
    if set(left.text.map(normalize)) & set(right.text.map(normalize)):
        raise ValueError("Normalized text overlaps across dataset splits")


def make_model(model, c_value, seed):
    vectorizer = TfidfVectorizer(preprocessor=normalize, lowercase=False,
                                 ngram_range=(1, 2), token_pattern=r"(?u)\b\w+\b|[^\w\s]")
    if model == "TF_LR":
        classifier = LogisticRegression(C=c_value, max_iter=2000, random_state=seed)
        return Pipeline([("tfidf", vectorizer), ("classifier", classifier)])
    else:
        # Fit vocabulary/IDF within each calibration fold, too.
        return CalibratedClassifierCV(
            Pipeline([("tfidf", vectorizer),
                      ("classifier", LinearSVC(C=c_value, max_iter=10000, random_state=seed))]),
            cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=seed),
            method="sigmoid", ensemble=False,
        )


def predictions(model, frame):
    if list(model.classes_) != [0, 1, 2]:
        raise ValueError("Unexpected classifier class order")
    probabilities = model.predict_proba(frame.text)
    return pd.DataFrame({"id": frame.id, "y_true": frame.label,
                         "y_pred": model.predict(frame.text), "p_neg": probabilities[:, 0],
                         "p_neu": probabilities[:, 1], "p_pos": probabilities[:, 2],
                         "truncated": 0})


def provenance(paths):
    result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    return {
        "data_sha256": {str(path): digest(path) for path in paths},
        "source_sha256": {name: digest(Path(__file__).with_name(name))
                          for name in SOURCE_FILES},
        "working_tree_dirty": bool(subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, check=False).stdout.strip()),
        "preprocess_version": PREPROCESS_VERSION,
        "commit": result.stdout.strip() or "unknown", "python": platform.python_version(),
        "libraries": {name: importlib.metadata.version(name)
                      for name in ("scikit-learn", "numpy", "pandas", "joblib")}, "device": "cpu",
    }


def verify_files(hashes):
    for path, expected in hashes.items():
        if digest(path) != expected:
            raise ValueError(f"File changed since selection/training: {path}")


def verify_source(metadata):
    verify_files({Path(__file__).with_name(name): value for name, value in metadata["source_sha256"].items()})


def frozen_inputs(config):
    if "preparation_manifest" not in config:
        return {}
    path = Path(config["preparation_manifest"])
    manifest = json.loads(path.read_text(encoding="utf-8"))
    # Exports in preparation.json are relative to the project/config directory.
    root = Path(config.get("config_root", Path.cwd()))
    return {str((root / name).resolve()): value for name, value in manifest["export_sha256"].items()}


def check_frozen(config, paths):
    frozen = frozen_inputs(config)
    if frozen:
        if any(str(Path(path).resolve()) not in frozen for path in paths):
            raise ValueError("Configured split is absent from the frozen preparation manifest")
        verify_files({path: frozen[str(Path(path).resolve())] for path in paths})


def tune(config, model):
    target = Path(config["output_dir"]) / "tuning" / f"{model}.json"
    if target.exists():
        saved = json.loads(target.read_text(encoding="utf-8"))
        verify_files(saved["data_sha256"])
        verify_source(saved)
        full = config["splits"]["100"]["42"]
        if set(saved["data_sha256"]) != {full["train"], full["dev"]}:
            raise ValueError("Tuning paths changed")
        print(f"Skipping existing {model} tuning")
        return "skipped"
    split = config["splits"]["100"]["42"]
    check_frozen(config, [split["train"], split["dev"]])
    train, dev = load_split(split["train"], True), load_split(split["dev"])
    check_disjoint(train, dev)
    scores = []
    started = time.perf_counter()
    for c_value in (0.1, 1.0, 10.0):
        estimator = fit_model(make_model(model, c_value, 42), train)
        scores.append({"C": c_value, "macro_f1": evaluate_predictions(predictions(estimator, dev))["macro_f1"]})
    best = max(scores, key=lambda row: row["macro_f1"])
    save_json(target, {
        "model": model, "fraction": 100, "seed": 42, "C": best["C"], "candidates": scores,
        "seconds": time.perf_counter() - started, **provenance([split["train"], split["dev"]]),
    })
    print(f"{model}: selected C={best['C']} using full NEU-ESC dev only")


def run_dir(config, model, fraction, seed):
    return Path(config["output_dir"]) / "runs" / model / f"f{fraction}__s{seed}"


def train_run(config, model, fraction, seed):
    folder = run_dir(config, model, fraction, seed)
    if (folder / "config.yaml").exists() and (folder / "log.json").exists():
        metadata = verify_run(folder, config, model, fraction, seed)
        tuning_path = Path(config["output_dir"]) / "tuning" / f"{model}.json"
        verify_files({tuning_path: metadata["tuning_sha256"]})
        if not (folder / "status.json").exists():
            save_json(folder / "status.json", {"status": "trained", "complete": False})
        print(f"Skipping trained {model} f{fraction} s{seed}")
        return "skipped"
    tuning_path = Path(config["output_dir"]) / "tuning" / f"{model}.json"
    tuning = json.loads(tuning_path.read_text(encoding="utf-8"))
    verify_files(tuning["data_sha256"])
    verify_source(tuning)
    full = config["splits"]["100"]["42"]
    if set(tuning["data_sha256"]) != {full["train"], full["dev"]}:
        raise ValueError("Full split paths changed since tuning")
    split = config["splits"][str(fraction)][str(seed)]
    check_frozen(config, [split["train"], split["dev"]])
    train, dev = load_split(split["train"], True), load_split(split["dev"])
    check_disjoint(train, dev)
    for subset, key in ((train, "train"), (dev, "dev")):
        reference = load_split(full[key]).set_index("id")
        if not subset.id.isin(reference.index).all():
            raise ValueError(f"Frozen {key} subset contains IDs outside the full split")
        expected = reference.loc[subset.id, ["text", "label"]].reset_index(drop=True)
        if not subset[["text", "label"]].reset_index(drop=True).equals(expected):
            raise ValueError(f"Frozen {key} subset differs from full split")
    if folder.exists():
        archive = folder.with_name(folder.name + ".interrupted-" + uuid.uuid4().hex[:8])
        folder.rename(archive)
    folder.mkdir(parents=True)
    started = time.perf_counter()
    estimator = fit_model(make_model(model, tuning["C"], seed), train)
    pred = predictions(estimator, dev)
    log = {
        "train_loss": log_loss(train.label, estimator.predict_proba(train.text), labels=[0, 1, 2]),
        "val_loss": log_loss(dev.label, pred[["p_neg", "p_neu", "p_pos"]], labels=[0, 1, 2]),
        "dev_macro_f1": evaluate_predictions(pred)["macro_f1"],
        "seconds": time.perf_counter() - started, "truncation_rate": 0.0,
        "checkpoint_selection": "single classical fit; C selected on full dev",
    }
    joblib.dump(estimator, folder / "checkpoint.joblib.tmp")
    (folder / "checkpoint.joblib.tmp").replace(folder / "checkpoint.joblib")
    metadata = {
        "model": model, "fraction": fraction, "seed": seed, "C": tuning["C"],
        "split": split, "tests": config["tests"], "full_split": full,
        "features": "word/punctuation TF-IDF, ngrams 1-2, L2 normalization, no truncation",
        "estimator": settings(estimator),
        "seed_limitation": "subset selection and training variability are combined",
        "classical_training": "single fit; transformer epoch, padding and GPU schedules do not apply",
        "expected_test_sha256": {path: frozen_inputs(config)[str(Path(path).resolve())]
                                 for path in config["tests"].values()} if frozen_inputs(config) else {},
        "labels": {"0": "negative", "1": "neutral", "2": "positive"},
        "tuning_sha256": digest(tuning_path), "checkpoint_sha256": digest(folder / "checkpoint.joblib"),
        **provenance([split["train"], split["dev"], full["train"], full["dev"]]),
    }
    # JSON is a YAML 1.2 subset; no separate YAML dependency is required.
    save_json(folder / "config.yaml", metadata)
    for name in SOURCE_FILES:
        source = Path(__file__).with_name(name)
        (folder / "source").mkdir(exist_ok=True)
        shutil.copy2(source, folder / "source" / name)
    save_json(folder / "log.json", log)
    save_json(folder / "status.json", {"status": "trained", "complete": False})
    print(f"Trained {model} f{fraction} s{seed}; tests have not been evaluated")


def evaluate_run(config, model, fraction, seed):
    folder = run_dir(config, model, fraction, seed)
    metadata = verify_run(folder, config, model, fraction, seed)
    if (folder / "evaluation.json").exists():
        saved = json.loads((folder / "evaluation.json").read_text(encoding="utf-8"))
        verify_files(saved["data_sha256"])
        verify_files(saved["prediction_sha256"])
        finalize_evaluation(config, model, fraction, seed, folder, metadata)
        print(f"Reused saved predictions for {model} f{fraction} s{seed}")
        return "skipped"
    paths = {"neu_esc_dev": metadata["split"]["dev"], **metadata["tests"]}
    verify_files(metadata.get("expected_test_sha256", {}))
    input_receipt = folder / "evaluation_inputs.json"
    if input_receipt.exists():
        verify_files(json.loads(input_receipt.read_text(encoding="utf-8")))
    else:
        save_json(input_receipt, {path: digest(path) for path in paths.values()})
    frames = {name: load_split(path) for name, path in paths.items()}
    full = metadata["full_split"]
    full_train, full_dev = load_split(full["train"]), load_split(full["dev"])
    check_disjoint(full_train, frames["neu_esc_test"])
    check_disjoint(full_dev, frames["neu_esc_test"])
    ood = frames["uit_vsfc_test"]
    if ood.text.map(normalize).duplicated().any():
        raise ValueError("OOD test contains duplicate normalized text")
    for neu in (full_train, full_dev, frames["neu_esc_test"]):
        check_disjoint(neu, ood, check_ids=False)
    estimator = joblib.load(folder / "checkpoint.joblib")
    output = Path(config["output_dir"])
    pred_dir = output / "preds" / model
    pred_dir.mkdir(parents=True, exist_ok=True)
    prediction_hashes = {}
    for dataset, frame in frames.items():
        target = pred_dir / f"{model}__{dataset}__f{fraction}__s{seed}.csv"
        receipt = target.with_suffix(".sha256.json")
        if target.exists():
            if not receipt.exists():
                raise ValueError(f"Unverified prediction file: {target}")
            verify_files(json.loads(receipt.read_text(encoding="utf-8")))
        else:
            save_csv(target, predictions(estimator, frame))
            save_json(receipt, {str(target): digest(target)})
        evaluate_file(target, expected=frame)
        prediction_hashes[str(target)] = digest(target)
    save_json(folder / "evaluation.json", {
        **provenance(list(paths.values())), "prediction_sha256": prediction_hashes,
    })
    finalize_evaluation(config, model, fraction, seed, folder, metadata)
    print(f"Completed {model} f{fraction} s{seed}")


def finalize_evaluation(config, model, fraction, seed, folder, metadata):
    output = Path(config["output_dir"])
    matrices = {}
    for dataset, path in {"neu_esc_dev": metadata["split"]["dev"], **metadata["tests"]}.items():
        prediction = output / "preds" / model / f"{model}__{dataset}__f{fraction}__s{seed}.csv"
        metrics = evaluate_file(prediction, expected=load_split(path))
        matrices[dataset] = metrics["confusion_matrix"]
        update_results(output / "results.csv", model, dataset, fraction, seed, "na", metrics)
    save_json(folder / "confusion_matrices.json", matrices)
    summarize(output / "results.csv")
    update_tracker(config, dict(model=model, fraction=fraction, seed=seed,
                                command="evaluate", status="complete", error=""))
    save_json(folder / "status.json", {"status": "complete", "complete": True})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["tune", "train", "evaluate"])
    parser.add_argument("--config", default="baseline_config.json")
    parser.add_argument("--model", choices=[*MODELS, "all"], default="all")
    parser.add_argument("--fraction", type=int, choices=SEEDS)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["config_root"] = str(config_path.parent)
    if "preparation_manifest" in config:
        config["preparation_manifest"] = str((config_path.parent / config["preparation_manifest"]).resolve())
    if config.get("label_schema") != "negative=0,neutral=1,positive=2":
        parser.error("Set label_schema to negative=0,neutral=1,positive=2 after preprocessing")
    if set(config["tests"]) != {"neu_esc_test", "uit_vsfc_test"}:
        parser.error("Provide NEU-ESC test and cleaned UIT-VSFC test only")
    for seeds in config["splits"].values():
        for split in seeds.values():
            for key in ("train", "dev"):
                split[key] = str((config_path.parent / split[key]).resolve())
    config["tests"] = {key: str((config_path.parent / value).resolve()) for key, value in config["tests"].items()}
    config["output_dir"] = str((config_path.parent / config["output_dir"]).resolve())
    if args.seed is not None and args.fraction is None:
        parser.error("--seed requires --fraction")
    if args.fraction is not None and args.seed is not None and args.seed not in SEEDS[args.fraction]:
        parser.error("Seed is not permitted for this fraction")
    if args.command == "tune" and (args.fraction is not None or args.seed is not None):
        parser.error("Tuning always uses fraction 100 and seed 42")
    models = MODELS if args.model == "all" else [args.model]
    if args.command == "tune":
        for model in models:
            record_attempt(config, model, 100, 42, "tune", lambda: tune(config, model))
        return
    for fraction in [args.fraction] if args.fraction is not None else SEEDS:
        for seed in [args.seed] if args.seed is not None else SEEDS[fraction]:
            for model in models:
                action = train_run if args.command == "train" else evaluate_run
                record_attempt(config, model, fraction, seed, args.command,
                               lambda: action(config, model, fraction, seed))


def record_attempt(config, model, fraction, seed, command, action):
    row = dict(model=model, fraction=fraction, seed=seed, command=command,
               started_utc=datetime.now(timezone.utc).isoformat(), status="running", error="")
    update_tracker(config, row)
    try:
        skipped = action() == "skipped"
        status_path = run_dir(config, model, fraction, seed) / "status.json"
        row["status"] = (json.loads(status_path.read_text())["status"]
                         if status_path.exists() else "tuned")
        row["skipped"] = skipped
    except BaseException as error:
        row.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        row["finished_utc"] = datetime.now(timezone.utc).isoformat()
        update_tracker(config, row)
        path = Path(config["output_dir"]) / "attempts.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row) + "\n")


def update_tracker(config, row):
    path = Path(config["output_dir"]) / "run_tracker.csv"
    tracker = pd.read_csv(path, keep_default_na=False) if path.exists() else pd.DataFrame()
    if not tracker.empty:
        same = ((tracker.model == row["model"]) & (tracker.fraction == row["fraction"])
                & (tracker.seed.astype(str) == str(row["seed"])))
        if row["status"] == "running":
            # A killed process cannot execute finally; retain its unfinished attempt.
            for previous in tracker.loc[same & tracker.status.eq("running")].to_dict("records"):
                previous.update(status="interrupted", error="Previous process ended before recording its result")
                with (path.parent / "attempts.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(previous) + "\n")
        tracker = tracker.loc[~same]
    save_csv(path, pd.concat([tracker, pd.DataFrame([row])], ignore_index=True))


if __name__ == "__main__":
    main()
