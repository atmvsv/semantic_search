import re
import string
from typing import List, Set, Final, Dict, Any

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize


class TextPreprocessor:
    def __init__(self, language: str = "english"):
        self.stop_words: Final[Set[str]] = set(stopwords.words(language))
        self.stemmer: Final[PorterStemmer] = PorterStemmer()
        self.punctuation_table: Final[Dict[int, Any]] = str.maketrans("", "", string.punctuation)

    def clean_minimal(self, text: str) -> str:
        text = text.lower().strip()
        text = re.sub(r"\s+", " ", text)
        return text

    def clean_full(self, text: str) -> List[str]:
        text = text.lower()
        text = text.translate(self.punctuation_table)
        tokens = word_tokenize(text)

        return [
            self.stemmer.stem(token)
            for token in tokens
            if token not in self.stop_words and token.isalpha()
        ]