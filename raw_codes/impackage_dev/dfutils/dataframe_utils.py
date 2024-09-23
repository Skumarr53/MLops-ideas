# Databricks notebook source
# MAGIC %run ./../utilities/config_utility

# COMMAND ----------

import numpy as np
import pandas as pd

# COMMAND ----------

"""
NAME : DFUtils

DESCRIPTION:
This module serves the dataframe operations functionalities:
                FINDS THE DUPLICATE TEXT PERCENTAGE ON DATAFRAME LIST COLUMN
                REMOVE DUPLICATE TEXT PRESENT IN LIST OF SENTENCES IN DATAFRAME COLUMN
"""


class DFUtils:
  
  """DFUtils CLASS"""
    
  def df_get_percentage_of_duplicate_text_in_sentence_list(self,currdf):
      """
      calculate the percentage of duplicate text present in list of sentences in dataframe column.
      This percentage scores for each filt values are added as new significant_duplicate_ columns.
        
      Parameters:
      argument1 (dataframe): dataframe
    
      Returns:
      dataframe
    
      """
      for label in config.FILT_sections:
        currdf["SIGNIFICANT_DUPLICATE_"+label]=currdf[label].apply(lambda x:round(((len(x)-len(set(x)))/len(x))*100,2) if ((len(x)>0) and (len(set(x))<len(x))) else 0)
      return currdf  

  def df_remove_duplicate_text_from_sentence_list(self,currdf):
      """
      remove duplicate text present in list of sentences in dataframe column.
      This removal is done if the percentage occurance of duplicate text is greater than threshold.
      This threshold value is configured in config utility.py
        
      Parameters:
      argument1 (dataframe): dataframe
    
      Returns:
      dataframe
    
      """
      for label in config.FILT_sections:
        currdf[label]=currdf[[label,"SIGNIFICANT_DUPLICATE_"+label]].apply(lambda x:list(dict.fromkeys(x[0])) if x[1]>config.duplication_threshold else x[0],axis=1)
      return currdf  
    
  def df_column_convert_to_list(self,currdf,filt_sections_list):
      """ converts dataframe column type from numpy ndarray to list
      Parameters:
      argument1 : dataframe

      Returns:
      dataframe

      """
      for label in filt_sections_list:
        currdf[label]=currdf[label].apply(lambda x: x.tolist())
      return currdf 
  
    

    
