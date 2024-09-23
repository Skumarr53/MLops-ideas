# NLP Project Documentation

## Overview

This project is designed for Natural Language Processing (NLP) using Databricks. It includes various preprocessing utilities for text data, leveraging the spaCy library for tokenization, lemmatization, and other text processing functionalities. The modules are structured to facilitate the preprocessing of text for further analysis or model training.

## Table of Contents

- [NLP Project Documentation](#nlp-project-documentation)
  - [Overview](#overview)
  - [Table of Contents](#table-of-contents)
  - [Modules](#modules)
    - [DictionaryModelPreprocessor](#dictionarymodelpreprocessor)
      - [Methods:](#methods)
    - [TextPreprocessor](#textpreprocessor)
      - [Methods:](#methods-1)
    - [TopicXPreprocessor](#topicxpreprocessor)
      - [Methods:](#methods-2)


## Modules

### DictionaryModelPreprocessor

The  `DictionaryModelPreprocessor`  class extends the  `TextPreprocessor`  class and provides functionalities for tokenization.

- **Key Features**:
  - **Word Tokenization**: Tokenizes input text and performs lemmatization.
  - **Text Preprocessing**: Cleans input text by removing unwanted characters and normalizing spaces.

#### Methods:
-  `word_tokenizer(text: str)` : Tokenizes the input text and returns a list of lemmatized words.
-  `preprocess_text(text: Union[str, List[str]])` : Preprocesses the input text and returns the cleaned text, tokenized words, and word count.
-  `preprocess_text_list(text_list: List[str])` : Preprocesses a list of texts and returns cleaned texts, tokenized words, and word counts for each text.

### TextPreprocessor

The  `TextPreprocessor`  class provides general preprocessing functionalities, including expanding contractions, checking for stop words, and cleaning text data.

- **Key Features**:
  - **Expand Contractions**: Expands common contractions in the text.
  - **Stop Word Checks**: Checks for the presence of stop words in the text.
  - **Text Cleaning**: Normalizes spaces, removes unwanted characters, and processes text data.

#### Methods:
-  `expand_contractions(text: str)` : Expands contractions in the input text.
-  `check_stopword(text: str)` : Checks if a word is a stop word.
-  `remove_accented_chars(text: str)` : Removes accented characters from the text.

### TopicXPreprocessor

The  `TopicXPreprocessor`  class extends the  `TextPreprocessor`  class and provides functionalities for tokenization and n-gram generation.

- **Key Features**:
  - **N-gram Generation**: Generates n-grams from a list of words.
  - **Sentence Preprocessing**: Cleans sentences by removing unwanted phrases.

#### Methods:
-  `find_ngrams(input_list: List[str], n: int)` : Generates n-grams from the input list of words.
-  `word_tokenizer(doc: str)` : Tokenizes the document text into a list of words, removing stop words and numbers.

