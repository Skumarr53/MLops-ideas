# common/text_processing.py

import spacy
from typing import List

# Load NLP model
nlp = spacy.load("en_core_web_sm", disable=['parser'])
nlp.Defaults.stop_words -= {"bottom", "top", "Bottom", "Top", "call", "down"}
nlp.max_length = 1000000000

def word_tokenize(doc: str) -> List[str]:
    """Tokenizes and lemmatizes the input document."""
    return [token.lemma_.lower() for token in nlp(doc) if not token.is_stop and not token.is_punct and token.pos_ != 'NUM']

def match_tokenize(doc: str) -> List[str]:
    """Tokenizes the input document for named entity recognition."""
    return [token.text.lower() if token.pos_ == 'PROPN' or token.text[0].isupper() else token.lemma_.lower() for token in nlp(doc) if not token.is_stop and not token.is_punct and token.pos_ != 'NUM']
