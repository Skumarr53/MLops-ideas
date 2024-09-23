# Databricks notebook source
# MAGIC %run ./../filesystem/blob_storage_utility

# COMMAND ----------

# MAGIC %run ../dataclasses/dataclasses

# COMMAND ----------

import xml.etree.ElementTree as ET
import re, ast
from tokenizers import normalizers
from tokenizers.normalizers import NFD, StripAccents
from ast import literal_eval
import pickle
import requests
from pyspark.sql.functions import max, min
from datetime import datetime
from pyspark.sql.functions import to_date
import spacy
from spacy.lang.en import English
from dataclasses import dataclass


# COMMAND ----------

"""
NAME : TextPreprocessor

DESCRIPTION:
This module serves the preprocessing functionalities:
                EXPAND CONTRACTIONS
                CHECK FOR NEGATED WORDS
                CHECK FOR STOP WORDS
                CHECK FOR DATATYPE
"""

class TextPreprocessor:
  
  """Preprocessing UTILITY CLASS"""
    
  def __init__(self):
    """Initialize the attributes of the Blob Storage Utility object"""
    self.filecfg = BlobFilenameConfig()
    self.blob_obj = BlobStorageUtility()

    self.config = config
    self.stop_words_List=self.get_words_list(self.filecfg.stop_words_flnm)
    self.contractions=self.read_contraction_words(self.filecfg.contraction_flnm)
    self.negate=self.get_words_list(self.filecfg.negate_words_flnm)
    self.sent_tokenizer = English()
    self.sent_tokenizer.add_pipe("sentencizer")
  
  def get_blob_stroage_path(self, filename):
    """
        Constructs the full path to a file in blob storage.
        Args:
            filename (str): The name of the file.
        Returns:
            str: The full path to the file in blob storage.
    """
    return self.config.file_path_config.model_artifacts_path + self.config.file_path_config.preprocessing_words_path + filename
  
  def get_words_list(self,filename):
    """
    Loads a list of words from a text file in blob storage.
    Args:
        filename (str): The name of the file containing the list of words.
    Returns:
        list: A list of words loaded from the text file.
      """
    file_path = self.get_blob_stroage_path(filename)
    return self.blob_obj.load_list_from_txt(file_path)
    
  def read_contraction_words(self,filename):
    """
    Reads and evaluates a dictionary of contraction words from a text file in blob storage
    Args:
        filename (str): The name of the file containing the contraction words.
    Returns:
        dict: A dictionary of contraction words loaded from the text file.
    """
    file_path = self.get_blob_stroage_path(filename)
    txt_cont = self.blob_obj.load_content_from_txt(file_path)
    return ast.literal_eval(txt_cont)

  
  def expand_contractions(self, text):
    """Expand contractions

    Parameters:
    argument1 (str): text
   
    Returns:
    str:returns text with expanded contractions
    
    """
    for word in text.split():
      if word.lower() in self.contractions:
        text = text.replace(word, self.contractions[word.lower()])
    return text
  
  def check_stopword(self, text):
    """check stopwords present in the text or not

    Parameters:
    argument1 (str): text
   
    Returns:
    bool:returns 0 or 1 flag
    
    """
    if text in self.stop_words_List:
        return 0
    return 1
  
  def remove_accented_chars(self, text):
    """removes accented characters from the text

    Parameters:
    argument1 (str): text
   
    Returns:
    str:cleaned text
    
    """
    normalizer = normalizers.Sequence([NFD(), StripAccents()])
    return normalizer.normalize_str(text)
  
  # Identifies 
  def check_for_acquisition_merger(self, text): 
    """checks for acquisition or merger text

    Parameters:
    argument1 (str): text
   
    Returns:
    bool:flag 0 or 1
    
    """
    if ('acquisition' in ' '.join(text).lower()) or ('merger' in ' '.join(text).lower()):
      return 1
    else:
      return 0
    
  def word_negated(self, word):
    """
    Determine if preceding word is a negation word
    
    Parameters:
    argument1 (str): word text
   
    Returns:
    bool:flag 0 or 1
    
    """
    return word in self.negate
  
   
  def sentence_tokenizer(self,doc):
    """
    tokenize the sentence 
    
    Parameters:
    argument1 (str): sentence
   
    Returns (list):
    list of sentences
    
    """
    return [txt for txt in filter(lambda item: item is not None, [self.process_sentence(sent.text) for sent in self.sent_tokenizer(doc).sents])]
  
  def correctID(self,x):
    """
    Returns the oth element of list or complete list
    
    Parameters:
    argument1 (list): list
   
    Returns:
    list:list
    
    """
    if (len(set(x))==1) & (type(x)==list):
      return [x[0]]  
    else:
      return x
    
  def check_datatype(self, text_list):
    """
    IF THE TEXT LIST HAS TEXT, PROCESS IT, OTHERWISE OUTPUT NAN
    
    Parameters:
    argument1 (list): list
   
    Returns:
    list:list
    
    """
    if (not isinstance(text_list,str) and text_list and ' '.join(text_list).strip(' ')) or (isinstance(text_list,str) and text_list.strip(' ')):
      #Text input is not and empty string or list
      if not isinstance(text_list,str):
        #Text input is a list
        text = ' '.join(text_list)
      else:
        #Text input is a string
        text = text_list
    else:
      text = False
        
    return text
  
  def check_sentence_for_operator(self, sentence):
    """
    Checks the text for operator word
    
    Parameters:
    argument1 (str): sentence
   
    Returns:
    bool:0 or 1
    
    """
    sentence = sentence.lower().split(" ")
    if len(sentence) < 3:
      
        return 0
      
    if "operator" in sentence or "operator," in sentence:
      
        return 0
      
    return 1

  def process_sentence(self, sentence):
    """
    Replaces Thanks or sorry like word with empty
    
    Parameters:
    argument1 (str): sentence
   
    Returns:
    bool:0 or 1
    
    """
    sentence = sentence.replace("Thank you", "")
    sentence = sentence.replace("thank you", "")
    sentence = sentence.replace("thanks", "")
    sentence = sentence.replace("Thanks", "")
    sentence = sentence.replace("Sorry", "")
    sentence = sentence.replace("sorry", "")

    return sentence 
 
    
  def checkCEO(self, string_value):
    """
    Check if some version of 'CEO' is in a string
    
    Parameters:
    argument1 (str): string_value
   
    Returns:
    flag 0 or 1
    
    """
    try:
      if ('ceo' in string_value.lower()) or ('chief executive officer' in string_value.lower()) or ('c.e.o' in string_value.lower()):
        return 1
      else:
        return 0
    except:
      return 0
    
  def check_earnings(self, string_value):
    """
    Check if some version of 'Earnings' is in a string
    
    Parameters:
    argument1 (str): string_value
   
    Returns:
    flag 0 or 1
    
    """
    try:
      if ('earnings' in string_value.lower()) or ('earning' in string_value.lower()):
        return 1
      else:
        return 0
    except:
      return 0
    
  

      
