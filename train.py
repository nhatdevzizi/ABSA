from main import ds
import re
import unicodedata

token = re.compile(r"\w+|[^\w\s]",re.UNICODE)

def tokenize(sentence: str) -> list[str]:
    sentence = unicodedata.normalize("NFC", sentence).lower()
    return token.findall(sentence)



