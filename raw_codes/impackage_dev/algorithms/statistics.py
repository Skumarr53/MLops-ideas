# Databricks notebook source
# MAGIC %run ./../preprocessor/dictionary_model_preprocessor

# COMMAND ----------

import pandas as pd
import re

# COMMAND ----------

"""
NAME : Statistics

DESCRIPTION:
This module serves the stats functionalities:
                GET DOLLAR AMOUNTS
                GET NUMBERS
                COMBINE SCORES
"""

class Statistics:
  
    """Statistics UTILITY CLASS"""
    
    def __init__(self):
      """Initialize the attributes of the Statistics Utility object"""
      self.preprocess_obj=DictionaryModelPreprocessor()
    
    def get_dollar_amounts(self, text_list):
      """reads the dollar amount from the input text
      Parameters:
      argument1 (list): text list

      Returns:
      array, int:returns array with dollar amounts, word_count

      """
      text = self.preprocess_obj.check_datatype(text_list)
      text_tokenized, input_words, word_count = self.preprocess_obj.preprocess_text(text_list)
      if (text) and (word_count > 0):
        dollar_amounts_array = re.findall(r'\$[0-9]+', text)
        return (len(dollar_amounts_array), word_count)
      else:
        return (np.nan, np.nan)

  
    def get_numbers(self, text_list):
      """Function to extract no. of numerical values from the given string (including dollar amounts!)
      Parameters:
      argument1 (list): text list

      Returns:
      array, int:returns array with numbers, word_count

      """
      text = self.preprocess_obj.check_datatype(text_list)
      text_tokenized, input_words, word_count = self.preprocess_obj.preprocess_text(text_list)
      if (text) and (word_count > 0):
        num_array = re.findall(r'[\$]*[0-9]+[.]*[0-9]*', text)
        return (len(num_array), word_count)
      else:
        return (np.nan, np.nan)
      
    def combine_sent(self, x, y):
      if x + y == 0:

        return 0

      else:

        return ((x-y)/(x+y))

    def nan_sum(self,*args):
      """Method to sum the args along with nan value
       Parameters:
       argument1 : list of args 
       
       Returns:
       total

      """ 
      if (len(set(np.isnan(list(args)))) > 1) or (not list(set(np.isnan(list(args))))[0]):
        tot = np.nansum(list(args))
      else:
        tot = np.nan

      return tot
    
    def section_level_bert_sentiment_score(self, sent_label):
      """
      calculate sentiment score using financial bert model generated labal values
      Parameters:
      argument1 (array): sentence label

      Returns:
      array, int:returns score, negative_sent, positive_sent, sentences_count

      """ 
      score = np.nan
      positive_sent = 0
      negative_sent = 0
      sentences_count = 0

      sentences_count = len(sent_label)

      if sentences_count > 0:
          for label in sent_label:
            if label==1:
              positive_sent+=1

            if label==-1:
              negative_sent+=1

          if sentences_count!=0:  
            score = (positive_sent - negative_sent)/sentences_count

          return (score, negative_sent, positive_sent, sentences_count)

      else: 

          return(np.nan, np.nan, np.nan, 0)

    
