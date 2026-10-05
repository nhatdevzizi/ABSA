import re
import unicodedata
from collections import Counter
from underthesea import word_tokenize

token = re.compile(r"\w+|[^\w\s]",re.UNICODE)

def tokenize(sentence: str) -> list[str]:
    sentence = word_tokenize(sentence.lower())
    return sentence

def build_vocab(sentence):
    vocab = Counter(tokenize(sentence))



