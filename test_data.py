import unittest
from unittest.mock import patch

import torch
from datasets import Dataset, DatasetDict

from BiLSTM import BiLSTM
from data import load_data


class DatasetIntegrationTest(unittest.TestCase):
    def test_both_datasets_support_training_batches(self):
        for name, labels in (
            ("vsfc", [0, 1, 2]),
            ("vsmec", ["Anger", "Disgust", "Enjoyment", "Fear", "Other", "Sadness", "Surprise"]),
        ):
            with self.subTest(dataset=name):
                if name == "vsfc":
                    rows = {"sentence": ["good"] * len(labels), "sentiment": labels, "topic": [0] * len(labels)}
                    unseen = {"sentence": ["unseenword"], "sentiment": [0], "topic": [1]}
                else:
                    rows = {"Sentence": ["good"] * len(labels), "Emotion": labels}
                    unseen = {"Sentence": ["unseenword"], "Emotion": ["Surprise"]}
                fixture = DatasetDict({
                    "train": Dataset.from_dict(rows),
                    "validation": Dataset.from_dict(unseen),
                    "test": Dataset.from_dict(unseen),
                })
                with patch("data.load_dataset", return_value=fixture):
                    vocab, train, validation, test, names = load_data(name)

                self.assertNotIn("unseenword", vocab)
                ids, lengths, targets, topics = next(iter(train))
                self.assertEqual(sorted(targets.tolist()), list(range(len(labels))))
                self.assertEqual(topics.tolist(), [0 if name == "vsfc" else -1] * len(labels))
                model = BiLSTM(len(vocab), embedding_dim=8, hidden_dim=8, num_classes=len(names))
                scores = model(ids, lengths)
                self.assertEqual(tuple(scores.shape), (len(labels), len(labels)))
                loss = torch.nn.CrossEntropyLoss()(scores, targets)
                loss.backward()
                self.assertTrue(torch.isfinite(loss).item())
                for loader in (validation, test):
                    ids, _, targets, _ = next(iter(loader))
                    self.assertEqual(ids.tolist(), [[vocab["<unk>"]]])
                    self.assertEqual(targets.tolist(), [0 if name == "vsfc" else 6])


if __name__ == "__main__":
    unittest.main()
