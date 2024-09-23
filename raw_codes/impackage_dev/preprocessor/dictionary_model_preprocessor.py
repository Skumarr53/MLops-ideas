# Databricks notebook source
!pip install spacy==3.4.4
!pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.4.0/en_core_web_sm-3.4.0.tar.gz


# COMMAND ----------

# Databricks notebook source
!pip install spacy==3.4.4
!pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.4.0/en_core_web_sm-3.4.0.tar.gz

# COMMAND ----------

# MAGIC %run ./text_preprocessor

# COMMAND ----------


# COMMAND ----------
from typing import List, Tuple, Union

"""
NAME : DictionaryModelTextPreprocessor

DESCRIPTION:
This module serves the below functionalities:
                Tokenization
             
"""

class DictionaryModelPreprocessor(TextPreprocessor):
    """Topic X Preprocessing Class"""

    def __init__(self):
        """Initialize the superclass and load the spaCy tokenizer."""
        super().__init__()
        self.spacy_tokenizer = spacy.load('en_core_web_sm')

    def word_tokenizer(self, text: str) -> List[str]:
        """
        Tokenizes the text and performs lemmatization.
        Args:
            text (str): The input text to tokenize.
        Returns:
            List[str]: A list of lemmatized words.
        """
        text = text.lower()
        doc = self.spacy_tokenizer(text)
        token_lemmatized = [token.lemma_ for token in doc]
        filtered_words = [word for word in token_lemmatized if word not in self.stop_words_List]
        return filtered_words

    def preprocess_text(self, text: Union[str, List[str]]) -> Tuple[str, List[str], int]:
        """
        Preprocess the text by removing double quotes, single letter words, and replacing multiple spaces with a single space.
        Args:
            text (Union[str, List[str]]): The input text or list of texts to preprocess.
        Returns:
            Tuple[str, List[str], int]: The preprocessed text, list of input words, and word count.
        """
        text = self.check_datatype(text)
        if text:
            text = self._clean_text(text)
            input_words = self.word_tokenizer(text)
            word_count = len(input_words)
            return text, input_words, word_count
        else:
            return "", [], 0

    def preprocess_text_list(self, text_list: List[str]) -> Tuple[List[str], List[List[str]], List[int]]:
        """
        Preprocess a list of texts by removing double quotes, single letter words, and replacing multiple spaces with a single space.
        Args:
            text_list (List[str]): The list of texts to preprocess.
        Returns:
            Tuple[List[str], List[List[str]], List[int]]: The list of preprocessed texts, list of input words for each text, and list of word counts.
        """
        final_text_list = []
        input_word_list = []
        word_count_list = []

        for text in text_list:
            text = self._clean_text(text)
            token_word_list = self.word_tokenizer(text)
            final_text_list.append(text)
            input_word_list.append(token_word_list)
            word_count_list.append(len(token_word_list))

        return final_text_list, input_word_list, word_count_list

    def _clean_text(self, text: str) -> str:
        """
        Cleans the input text by expanding contractions, removing unwanted characters, and normalizing spaces.
        Args:
            text (str): The input text to clean.
        Returns:
            str: The cleaned text.
        """
        text = self.expand_contractions(re.sub('’', "'", text))
        text = text.strip().lower()
        text = text.replace('"', '')
        text = re.sub(r"\b[a-zA-Z]\b", "", text)
        text = re.sub("[^a-zA-Z]", " ", text)
        text = re.sub("\s+", ' ', text)
        return text.strip()
