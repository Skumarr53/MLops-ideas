# processing/text_processing.py
import spacy
from typing import List
from omegaconf import DictConfig

class TextProcessor:
    def __init__(self, cfg: DictConfig):
        """Initialize TextProcessor using configuration from Hydra."""
        self.nlp = spacy.load(cfg.spacy.model)
        self.custom_stopwords = set(cfg.stopwords)
        self.chunk_size = cfg.spacy.chunk_size

    def tokenize_and_lemmatize(self, text: str) -> List[str]:
        """Tokenize and lemmatize text, handling large documents in chunks."""
        chunks = [text[i:i + self.chunk_size] for i in range(0, len(text), self.chunk_size)]
        tokens = []
        for chunk in chunks:
            tokens.extend([token.lemma_.lower() for token in self.nlp(chunk)
                           if not token.is_stop and not token.is_punct and token.pos_ != 'NUM'])
        return tokens

    def filter_stopwords(self, tokens: List[str]) -> List[str]:
        """Filter custom stopwords from the tokenized text."""
        return [token for token in tokens if token not in self.custom_stopwords]

    def generate_ngrams(self, tokens: List[str], n: int) -> List[str]:
        """Generate n-grams from tokenized text."""
        return ['_'.join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
    
    def word_tokenize(self, text: str):
        return [token.lemma_.lower() for token in self.nlp(text) if not token.is_stop and not token.is_punct and token.pos_ != 'NUM']

    def match_tokenize(self, text: str):
        ret = []
        for token in self.nlp(text):
            if token.pos_ == 'PROPN' or token.text[0].isupper():
                ret.append(token.text.lower())
            elif not token.is_stop and not token.is_punct and token.pos_ != 'NUM':
                ret.append(token.lemma_.lower())
        return ret

    def find_ngrams(self, input_list, n):
        return zip(*[input_list[i:] for i in range(n)])
