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
    model = BiLSTM(len(vocab)).to(device)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)

    train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
    print(f"Training loss: {train_loss:.4f}")

def train_one_epoch(model, loader, loss_fn, optimizer, device):
    model.train()
    total_loss = 0
    for token_ids, lengths, sentiments, _ in loader:
        token_ids = token_ids.to(device)
        sentiments = sentiments.to(device)

        optimizer.zero_grad()
        scores = model(token_ids, lengths)
        loss = loss_fn(scores, sentiments)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        return total_loss/len(loader)
    


if __name__ == "__main__":
    main()
