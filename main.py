from copy import deepcopy

from datasets import load_dataset
from torch.utils.data import DataLoader
import torch

from train import build_vocab, collate_batch
from BiLSTM import BiLSTM

device = "cuda" if torch.cuda.is_available() else "cpu"
base = "https://huggingface.co/datasets/uitnlp/vietnamese_students_feedback/resolve/refs%2Fconvert%2Fparquet/default"

def main():
    ds = load_dataset(
        "parquet",
        data_files={
            "train": f"{base}/train/0000.parquet",
            "validation": f"{base}/validation/0000.parquet",
            "test": f"{base}/test/0000.parquet",
        },
    )
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

    #Start from here
    model = BiLSTM(len(vocab)).to(device)
    loss_func = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001) #Future optimization on *lr*

    best_accuracy = -1.0
    best_weights = None
    for epoch in range(5):
        train_loss = train(model, train_loader, loss_func, optimizer, device)
        accuracy = validate(model, validation_loader, device)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_weights = deepcopy(model.state_dict())
        print(f"Epoch {epoch + 1}: loss={train_loss:.4f}, validation accuracy={accuracy:.2%}")

    model.load_state_dict(best_weights)
    test_accuracy = test(model, test_loader, device)
    print(f"Test accuracy: {test_accuracy:.2%}")

def train(model, loader, loss_fn, optimizer, device):
    model.train()
    total_loss = 0
    for token_ids, lengths, sentiments, _ in loader:
        token_ids = token_ids.to(device)
        sentiments = sentiments.to(device)

        optimizer.zero_grad()
        scores = model(token_ids, lengths)
        loss = loss_fn(scores, sentiments)
        #Future work with other loss functions might work here
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss/len(loader)

def validate(model, loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for token_ids, lengths, sentiments, _ in loader:
            scores = model(token_ids.to(device), lengths)
            predictions = scores.argmax(dim=1)
            correct += (predictions == sentiments.to(device)).sum().item()
            total += len(sentiments)

    return correct / total

def test(model, loader, device):
    return validate(model, loader, device)

if __name__ == "__main__":
    main()
