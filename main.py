from datasets import load_dataset
from torch.utils.data import DataLoader
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from train import *
from BiLSTM import BiLSTM

device = "cuda" if torch.cuda.is_available() else "cpu"
base = "https://huggingface.co/datasets/uitnlp/vietnamese_students_feedback/resolve/refs%2Fconvert%2Fparquet/default"
ds = load_dataset(
    "parquet",
    data_files = {
        "train": f"{base}/train/0000.parquet",
        "validation": f"{base}/validation/0000.parquet",
        "test": f"{base}/test/0000.parquet",
    },
)

def main():
    vocab = build_vocab(ds["train"]["sentence"])

    #Future optimization might work on collate_fn
    train_loader = DataLoader(
        ds["train"],
        batch_size=32,
        shuffle = True,
        collate_fn=lambda rows: collate_batch(rows, vocab)
    )
    validation_loader = DataLoader(
        ds["validation"],
        batch_size=32,
        shuffle=False,
        collate_fn=lambda rows: collate_batch(rows, vocab),
    )
    test_loader = DataLoader(
        ds["test"],
        batch_size=32,
        shuffle=False,
        collate_fn=lambda rows: collate_batch(rows, vocab),
    )
    


if __name__ == "__main__":
    main()
