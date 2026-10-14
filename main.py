from copy import deepcopy
import argparse
import torch
from BiLSTM import BiLSTM
from measurer import measure

device = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    from data import load_data

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["vsfc", "vsmec"], default="vsfc")
    args = parser.parse_args()
    vocab, train_loader, validation_loader, test_loader, label_names = load_data(args.dataset)
    print(f"Dataset: {args.dataset}, labels: {label_names}")

    model = BiLSTM(len(vocab), num_classes=len(label_names)).to(device)
    loss_func = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3) #Future optimization on *lr*

    best_accuracy = -1.0
    best_weights = None
    for epoch in range(5):
        train_loss = train(model, train_loader, loss_func, optimizer, device)
        metrics = validate(model, validation_loader, device)
        accuracy = metrics["accuracy"]
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_weights = deepcopy(model.state_dict())
        summary = ", ".join(f"{name}={value:.2%}" for name, value in metrics.items())
        print(f"Epoch {epoch + 1}: loss={train_loss:.4f}, validation {summary}")

    model.load_state_dict(best_weights)
    test_metrics = test(model, test_loader, device)
    summary = ", ".join(f"{name}={value:.2%}" for name, value in test_metrics.items())
    print(f"Test {summary}")

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
    y_true = []
    y_pred = []

    with torch.no_grad():
        for token_ids, lengths, sentiments, _ in loader:
            scores = model(token_ids.to(device), lengths)
            predictions = scores.argmax(dim=1)
            y_true.extend(sentiments.cpu().tolist())
            y_pred.extend(predictions.cpu().tolist())

    return measure(y_true, y_pred)

def test(model, loader, device):
    return validate(model, loader, device)

if __name__ == "__main__":
        main()
