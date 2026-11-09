"""Extract supplied frozen subset groups without sampling or label remapping."""

import json
from pathlib import Path

import pandas as pd

from uit_vsfc.baselines.tfidf_direct_sentiment import SEEDS, digest, load_split, normalize, save_json, verify_manifest_file


ROOT = Path(__file__).resolve().parents[2]

def prepare():
    source_hashes = {}
    for folder in (ROOT / "data" / "clean", ROOT / "data" / "splits"):
        manifest = json.loads((folder / "MANIFEST.json").read_text(encoding="utf-8"))
        source_hashes[str((folder / "MANIFEST.json").relative_to(ROOT))] = digest(folder / "MANIFEST.json")
        for name, expected in manifest.items():
            path = folder / name
            verify_manifest_file(path, expected)
            source_hashes[str(path.relative_to(ROOT))] = digest(path)

    full_path = ROOT / "data/splits/neuesc_100.csv"
    full = load_split(full_path)
    if set(full.subset) != {"train", "dev", "test"}:
        raise ValueError("Expected train/dev/test groups in full NEU-ESC")
    reference = full.set_index("id")
    frozen_test = full.loc[full.subset == "test"].set_index("id").sort_index()
    frames = {}
    config = {"label_schema": "negative=0,neutral=1,positive=2", "output_dir": "outputs",
              "preparation_manifest": "data/baseline/preparation.json",
              "project_root": "..", "splits": {}, "tests": {}}
    for fraction, seeds in SEEDS.items():
        config["splits"][str(fraction)] = {}
        for seed in seeds:
            # Confirmed by the dataset owner: s0=42 through s4=46.
            source = full_path if fraction == 100 else ROOT / f"data/splits/neuesc_{fraction}_s{seed - 42}.csv"
            frame = load_split(source)
            if set(frame.subset) != {"train", "dev", "test"}:
                raise ValueError(f"Missing or unexpected subset group: {source}")
            if not frame.id.isin(reference.index).all():
                raise ValueError(f"Unknown IDs: {source}")
            columns = ["text", "label", "subset"]
            expected = reference.loc[frame.id, columns].reset_index(drop=True)
            if not frame[columns].reset_index(drop=True).equals(expected):
                raise ValueError(f"Rows differ from full frozen dataset: {source}")
            actual_test = frame.loc[frame.subset == "test"].set_index("id").sort_index()
            if not actual_test.equals(frozen_test):
                raise ValueError(f"Test differs from full frozen test: {source}")
            paths = {}
            for subset in ("train", "dev"):
                name = f"data/baseline/neu_esc_{subset}__f{fraction}__s{seed}.csv"
                frames[name] = frame.loc[frame.subset == subset, ["id", "text", "label"]]
                paths[subset] = name
            config["splits"][str(fraction)][str(seed)] = paths
    name = "data/baseline/neu_esc_test.csv"
    frames[name] = full.loc[full.subset == "test", ["id", "text", "label"]]
    config["tests"]["neu_esc_test"] = name
    ood = load_split(ROOT / "data/clean/vsfc_test_clean.csv")
    if set(ood.subset) != {"test"} or ood.text.map(normalize).duplicated().any():
        raise ValueError("OOD data must contain only unique test examples")
    others = [full.text]
    for subset in ("train", "dev", "test"):
        others.append(pd.read_csv(ROOT / f"data/clean/vsmec_{subset}.csv").text)
    if set(ood.text.map(normalize)) & set(pd.concat(others).map(normalize)):
        raise ValueError("OOD normalized text overlaps NEU-ESC or VSMEC")
    name = "data/baseline/uit_vsfc_test_clean.csv"
    frames[name] = ood[["id", "text", "label"]]
    config["tests"]["uit_vsfc_test"] = name
    for name, frame in frames.items():
        target = ROOT / name
        target.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(target, index=False)
    save_json(ROOT / "data/baseline/preparation.json", {
        "method": "filter existing subset column; no sampling, relabeling, or ID changes",
        "seed_mapping": {f"s{i}": 42 + i for i in range(5)},
        "seed_mapping_source": "dataset owner confirmed in chat",
        "source_sha256": source_hashes,
        "export_sha256": {name: digest(ROOT / name) for name in frames},
        "rows": {name: len(frame) for name, frame in frames.items()},
    })
    save_json(ROOT / "configs/baseline.json", config)
    print(f"Prepared {len(frames)} CSV files; baseline_config.json now uses data/baseline/")


if __name__ == "__main__":
    prepare()
