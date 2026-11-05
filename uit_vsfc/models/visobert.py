import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


class ViSoBERTBiLSTM(nn.Module):
    def __init__(self, num_classes: int = 3, encoder=None, hidden_dim: int = 256):
        super().__init__()
        if encoder is None:
            from transformers import AutoModel
            encoder = AutoModel.from_pretrained("uitnlp/visobert", add_pooling_layer=False)
        self.encoder = encoder.requires_grad_(False)
        self.encoder.eval()
        self.lstm = nn.LSTM(encoder.config.hidden_size, hidden_dim, batch_first=True, bidirectional=True)
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)

    def train(self, mode=True):
        super().train(mode)
        self.encoder.eval()
        return self

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
            raise ValueError("input_ids and attention_mask must have matching 2D shapes")
        if not ((attention_mask == 0) | (attention_mask == 1)).all():
            raise ValueError("attention_mask must contain only 0 and 1")
        lengths = attention_mask.sum(dim=1).long().cpu()
        if not (lengths > 0).all():
            raise ValueError("All sequences must have at least one valid token")
        if (attention_mask[:, 1:] > attention_mask[:, :-1]).any():
            raise ValueError("attention_mask must be right-padded")

        # main logic
        with torch.no_grad():
            embedded = self.encoder(input_ids=input_ids, attention_mask=attention_mask.to(input_ids.device)).last_hidden_state
        packed = pack_padded_sequence(embedded, lengths, batch_first=True, enforce_sorted=False)
        _, (hidden, _) = self.lstm(packed)
        return self.classifier(torch.cat((hidden[-2], hidden[-1]), dim=1))
