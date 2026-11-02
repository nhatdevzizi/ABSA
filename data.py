from train import build_vocab, collate_batch, collate_visobert
from datasets import load_dataset
from torch.utils.data import DataLoader
import functools

def load_data(dataset="vsfc", tokenizer=None):
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

    vocab = build_vocab(ds["train"]["sentence"]) if tokenizer is None else None
    collate = (functools.partial(collate_batch, vocab=vocab) if tokenizer is None
               else functools.partial(collate_visobert, tokenizer=tokenizer))
    train_loader = DataLoader(
        ds["train"], batch_size=32, shuffle=True, collate_fn=collate,
    )
    validation_loader = DataLoader(
        ds["validation"], batch_size=32, shuffle=False, collate_fn=collate,
    )
    test_loader = DataLoader(
        ds["test"], batch_size=32, shuffle=False, collate_fn=collate,
    )
    return vocab, train_loader, validation_loader, test_loader, label_names
