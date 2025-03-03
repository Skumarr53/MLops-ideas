# Databricks notebook source
# MAGIC %run ./text_preprocessor

# COMMAND ----------

!pip install spacy==3.4.4
!pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.4.0/en_core_web_sm-3.4.0.tar.gz

# COMMAND ----------

import spacy
from spacy.lang.en import English

# COMMAND ----------

"""
NAME : TopicXPreprocessor

DESCRIPTION:
This module serves the below functionalities:
                Tokenization
                Creating N Grams
              
"""

class TopicXPreprocessor(TextPreprocessor):
  
  """Topic X Preprocessing CLASS"""
    
  def __init__(self):
    """Initialize the super class Init method"""
    super().__init__()
    self.nlp = spacy.load("en_core_web_sm", disable = ['parser'])
    # Excluding financially relavant stopwords
    self.nlp.Defaults.stop_words -= {"bottom", "top", "Bottom", "Top", "call"}
    self.nlp.max_length = 1000000000

  def find_ngrams(self,input_list, n):
    """
    generates the ngrams from input list of words
    
    Parameters:
    argument1 (list): input list of words
   
    Returns:
    list of ngrams
    
    """
    return zip(*[input_list[i:] for i in range(n)])

  def process_sentence(self,sentence):
    """
    preprocess the sentence by removing some of unwanted words from text.
    
    Parameters:
    argument1 (str): sentence
   
    Returns (str):
    preprocessed sentence
    
    """
    sentence = sentence.replace("Thank you", "")
    sentence = sentence.replace("thank you", "")
    sentence = sentence.replace("thanks", "")
    sentence = sentence.replace("Thanks", "")
    sentence = sentence.replace("earnings call", "")
    sentence = sentence.replace("earnings release", "")
    sentence = sentence.replace("earnings conference", "")
    if len(sentence.split(" ")) < 5:

        return None

    low = sentence.lower()
    if "good morning" in low or "good afternoon" in low or "good evening" in low:
        return None
    return sentence
  
  def word_tokenizer(self, doc):
    """
    tokenizes document text into list of words by removing stop w ords and numbers
    
    Parameters:
    argument1 (str): text document
   
    Returns:
    list of lemmatized words
    
    """
    return [ent.lemma_.lower() for ent in self.nlp(doc) if not ent.is_stop and not ent.is_punct and ent.pos_ != 'NUM']

  def tokenizer_for_matched_words(self,doc):
    """
    tokenizes matched words by removing stop words and numbers
    
    Parameters:
    argument1 (str): text document
   
    Returns:
    list of lemmatized words
    
    """
    ret = []
    for ent in self.nlp(doc):
      if ent.pos_ == 'PROPN' or ent.text[0].isupper():
        ret.append(ent.text.lower())
        continue
      if not ent.is_stop and not ent.is_punct and ent.pos_ != 'NUM':
        ret.append(ent.lemma_.lower())
    return ret
  
