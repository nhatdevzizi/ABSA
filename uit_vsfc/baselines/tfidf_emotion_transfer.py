"""Train seven VSMEC emotions, then map predicted emotions to sentiment."""

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, log_loss

from uit_vsfc.baselines.tfidf_direct_sentiment import (
    PROJECT_ROOT, check_disjoint, digest, fit_model, load_split, make_model,
    provenance, record_attempt, run_dir, run_state_dir, save_json, settings, verify_files, verify_manifest_file, verify_source,
)
from uit_vsfc.evaluation.metrics import evaluate_file, save_csv, summarize, update_results

EMOTION_SENTIMENT = {
    "Enjoyment": 2, "Sadness": 0, "Fear": 0, "Anger": 0,
    "Disgust": 0, "Surprise": 1, "Other": 1,
}


def load_emotions(path, subset):
    frame = pd.read_csv(path, dtype={"id": str, "text": str}, keep_default_na=False)
    if not {"id", "text", "label", "subset"}.issubset(frame.columns):
        raise ValueError(f"Missing emotion columns: {path}")
    if (frame.empty or frame.id.eq("").any() or frame.id.duplicated().any()
            or frame.text.str.strip().eq("").any() or set(frame.label) != set(range(7))
            or set(frame.subset) != {subset}):
        raise ValueError(f"Invalid frozen emotion split: {path}")
    return frame


def hard_predictions(estimator, frame, names):
    if list(estimator.classes_) != list(range(7)):
        raise ValueError("Expected seven emotion classes in numeric order")
    mapping = np.array([EMOTION_SENTIMENT[name] for name in names])
    emotion = estimator.predict(frame.text)
    probabilities = estimator.predict_proba(frame.text)
    grouped = np.column_stack([probabilities[:, mapping == label].sum(axis=1)
                               for label in range(3)])
    # The hard rule maps the winning emotion, not the summed-probability argmax.
    return pd.DataFrame({"id": frame.id, "y_true": frame.label,
                         "y_pred": mapping[emotion], "p_neg": grouped[:, 0],
                         "p_neu": grouped[:, 1], "p_pos": grouped[:, 2],
                         "y_pred_7": emotion})


def run(config, model):
    folder = run_dir(config, model, 100, "NA")
    pred_dir = run_state_dir(config) / "preds" / model
    names = config["emotion_names"]
    if len(names) != 7 or set(names) != set(EMOTION_SENTIMENT):
        raise ValueError("emotion_names must specify the seven confirmed numeric labels")
    paths = {name: PROJECT_ROOT / path for name, path in config["vsmec"].items()}
    tests = {name: PROJECT_ROOT / path for name, path in config["tests"].items()}
    if set(paths) != {"train", "dev", "test"} or set(tests) != {"neu_esc_test", "uit_vsfc_test"}:
        raise ValueError("Expected frozen VSMEC splits and two target tests")
    manifest_path = PROJECT_ROOT / "data/clean/MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for path in paths.values():
        verify_manifest_file(path, manifest[path.name])
    preparation = PROJECT_ROOT / "data/baseline/preparation.json"
    exports = json.loads(preparation.read_text(encoding="utf-8"))["export_sha256"]
    verify_files({path: exports[str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")]
                  for path in tests.values()})
    receipt = provenance([*paths.values(), *tests.values(), manifest_path, preparation])
    receipt["source_sha256"]["uit_vsfc/baselines/tfidf_emotion_transfer.py"] = digest(Path(__file__))
    metadata_path = folder / "config.yaml"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        verify_files(metadata["data_sha256"])
        verify_source(metadata)
        verify_files({folder / "checkpoint.joblib": metadata["checkpoint_sha256"]})
        if metadata["configuration"] != config or metadata["model"] != model:
            raise ValueError("Configuration changed; use a new output directory")
    else:
        if folder.exists() and any(folder.iterdir()):
            raise ValueError(f"Incomplete run exists; preserve it before rerunning: {folder}")
        train_checkpoint(config, model, paths, folder, receipt)
    evaluate_checkpoint(config, model, paths, tests, folder, pred_dir, names)


def train_checkpoint(config, model, paths, folder, receipt):
    metadata_path = folder / "config.yaml"
    train, dev = (load_emotions(paths[key], key) for key in ("train", "dev"))
    check_disjoint(train, dev)
    started = time.perf_counter()
    scores, best, best_score = [], None, -1
    for c_value in config["candidates"]:
        estimator = fit_model(make_model(model.removeprefix("Hard_"), c_value, 0), train)
        score = f1_score(dev.label, estimator.predict(dev.text), labels=range(7),
                         average="macro", zero_division=0)
        scores.append({"C": c_value, "macro_f1_vsmec_dev_7": score})
        print(f"{model}: C={c_value}, VSMEC dev macro-F1={score:.6f}", flush=True)
        if score > best_score:
            best, best_score, best_c = estimator, score, c_value
    folder.mkdir(parents=True, exist_ok=True)
    joblib.dump(best, folder / "checkpoint.joblib")
    save_json(folder / "log.json", {
        "candidates": scores, "selected_C": best_c, "dev_macro_f1_7": best_score,
        "train_loss": log_loss(train.label, best.predict_proba(train.text), labels=range(7)),
        "val_loss": log_loss(dev.label, best.predict_proba(dev.text), labels=range(7)),
        "seconds": time.perf_counter() - started,
    })
    save_json(metadata_path, {
        **receipt, "configuration": config, "model": model, "fraction": 100, "seed": "NA",
        "internal_random_state": 0, "selected_C": best_c, "estimator": settings(best),
        "checkpoint_sha256": digest(folder / "checkpoint.joblib"),
        "selection": "VSMEC dev seven-class macro-F1; first C wins ties; train only",
        "decision": "map predicted emotion; probabilities sum by sentiment group",
    })


def evaluate_checkpoint(config, model, paths, tests, folder, pred_dir, names):
    results = Path(config["output_dir"]) / "results.csv"
    evaluation_path = folder / "evaluation.json"
    if evaluation_path.exists():
        verify_files(json.loads(evaluation_path.read_text(encoding="utf-8"))["prediction_sha256"])
    else:
        frames = {key: load_emotions(path, key) for key, path in paths.items()}
        targets = {key: load_split(path) for key, path in tests.items()}
        for left, right in (("train", "dev"), ("train", "test"), ("dev", "test")):
            check_disjoint(frames[left], frames[right])
        for source in frames.values():
            for target in targets.values():
                check_disjoint(source, target, check_ids=False)
        estimator = joblib.load(folder / "checkpoint.joblib")
        hashes = {}
        for dataset, frame in targets.items():
            path = pred_dir / f"{model}__{dataset}__f100__sNA.csv"
            if path.exists():
                raise ValueError(f"Unverified predictions already exist: {path}")
            save_csv(path, hard_predictions(estimator, frame, names))
            hashes[str(path)] = digest(path)
        frame = frames["test"]
        path = folder / "vsmec_test_predictions.csv"
        save_csv(path, pd.DataFrame({"id": frame.id, "y_true": frame.label,
                                     "y_pred": estimator.predict(frame.text)}))
        hashes[str(path)] = digest(path)
        save_json(evaluation_path, {"prediction_sha256": hashes})
    matrices = {}
    for dataset, path in tests.items():
        prediction = pred_dir / f"{model}__{dataset}__f100__sNA.csv"
        metrics = evaluate_file(prediction, expected=load_split(path))
        # Shared ECE describes grouped probability argmax, not the hard decision.
        metrics.pop("ece")
        matrices[dataset] = metrics["confusion_matrix"]
        update_results(results, model, dataset, 100, "NA", "na", metrics)
    emotion = pd.read_csv(folder / "vsmec_test_predictions.csv")
    update_results(results, model, "vsmec_test", 100, "NA", "na", {
        "macro_f1_vsmec_7": f1_score(emotion.y_true, emotion.y_pred, labels=range(7),
                                    average="macro", zero_division=0)})
    save_json(folder / "confusion_matrices.json", matrices)
    summarize(results)
    save_json(folder / "status.json", {"status": "complete", "complete": True})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/hard_tfidf.json")
    parser.add_argument("--model", choices=["Hard_TF_LR", "Hard_TF_SVM", "all"], default="all")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    config["output_dir"] = str((PROJECT_ROOT / config["output_dir"]).resolve())
    for model in ("Hard_TF_LR", "Hard_TF_SVM") if args.model == "all" else (args.model,):
        record_attempt(config, model, 100, "NA", "hard_train_evaluate", lambda: run(config, model))


if __name__ == "__main__":
    main()
