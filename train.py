import torch
from collections import Counter
from collections.abc import Iterable
from underthesea import word_tokenize
from torch.nn.utils.rnn import pad_sequence

#Filtering data
def tokenize(sentence: str) -> list[str]:
    return word_tokenize(sentence.lower())

def build_vocab(sentences: Iterable[str], min_freq: int = 1) -> dict[str,int]:
    counts = Counter()
    vocab = {"<pad>": 0, "<unk>": 1}
    for sentence in sentences:
        counts.update(tokenize(sentence))
    for token, count in sorted(counts.items()):
        if count >= min_freq:
            vocab[token] = len(vocab)
    return vocab

def encode_sentence(sentence: str, vocab: dict[str,int]) -> list[int]:
    res = []
    for token in tokenize(sentence):
        res.append(vocab.get(token, vocab["<unk>"])) #Default value of vocab is <unk>
    return res

def collate_batch(rows, vocab):
    sequences = []
    for row in rows:
        sequences.append(torch.tensor(encode_sentence(row["sentence"],vocab), dtype = torch.long))

    lengths = torch.tensor([len(sequence) for sequence in sequences])
    token_ids = pad_sequence(sequences, batch_first= True, padding_value= vocab["<pad>"])
    sentiments = torch.tensor([row["sentiment"] for row in rows])
    topics = torch.tensor([row["topic"] for row in rows])
    return token_ids, lengths, sentiments, topics