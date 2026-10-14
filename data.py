from train import build_vocab, collate_batch
from datasets import load_dataset
from torch.utils.data import DataLoader

def load_data(dataset="vsfc"):
    if dataset == "vsfc":
        repo = "uitnlp/vietnamese_students_feedback"
        label_names = ["negative", "neutral", "positive"]
    elif dataset == "vsmec":
        repo = "tridm/UIT-VSMEC"
        label_names = ["Anger", "Disgust", "Enjoyment", "Fear", "Other", "Sadness", "Surprise"]

    base = f"https://huggingface.co/datasets/{repo}/resolve/refs%2Fconvert%2Fparquet/default"
    ds = load_dataset(
        "parquet",
        data_files={
            "train": f"{base}/train/0000.parquet",
            "validation": f"{base}/validation/0000.parquet",
            "test": f"{base}/test/0000.parquet",
        },
    )
    if dataset == "vsmec":
        ds = ds.rename_columns({"Sentence": "sentence", "Emotion": "sentiment"})
        label_ids = {name: index for index, name in enumerate(label_names)}
        # Reuse the existing target field for emotion classification.
        ds = ds.map(lambda row: {"sentiment": label_ids[row["sentiment"]]})

    vocab = build_vocab(ds["train"]["sentence"])
    train_loader = DataLoader(
        ds["train"], batch_size=32, shuffle=True,
        collate_fn=lambda rows: collate_batch(rows, vocab),
    )
    validation_loader = DataLoader(
        ds["validation"], batch_size=32, shuffle=False,
        collate_fn=lambda rows: collate_batch(rows, vocab),
    )
    test_loader = DataLoader(
        ds["test"], batch_size=32, shuffle=False,
        collate_fn=lambda rows: collate_batch(rows, vocab),
    )
    return vocab, train_loader, validation_loader, test_loader, label_names
