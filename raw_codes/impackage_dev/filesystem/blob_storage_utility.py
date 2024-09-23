# Databricks notebook source
# MAGIC %run ./../utilities/config_utility

# COMMAND ----------

# MAGIC %run ./../utilities/customexception

# COMMAND ----------

import ast

# COMMAND ----------

"""
NAME : BlobStorageUtility

DESCRIPTION:
This module serves the below functionalities:
                READING THE CONTENTS FROM THE FILES IN BLOB STORAGE
              
"""


class BlobStorageUtility:
  
  """AZURE BLOB STORAGE HELPER CLASS"""
    
  def load_content_from_txt(self, file_path):
    """
    Reads the entire content of a text file from the given file path.
    Args:
        file_path (str): The path to the text file.
    Returns:
        str: The content of the text file.
    Raises:
        FilesNotLoadedException: If the file is not found at the given path.
    """
    
    try:
        with open(file_path, "r") as f_obj:
          content = f_obj.read()
        return content
    except FileNotFoundError as ex:
        raise FilesNotLoadedException()
  
  def load_list_from_txt(self, file_path, is_lower = True):
    """
    Reads the content of a text file and returns it as a set of lines.
    Args:
        file_path (str): The path to the text file.
        is_lower (bool, optional): If True, converts the content to lowercase. Defaults to True.
    Returns:
        set: A set of lines from the text file.
    Raises:
        Exception: If there is an error reading the file.
        """
    try:
      content = self.load_content_from_txt(file_path)
      if is_lower: content = content.lower()
      words_list = content.split('\n')
      return set(words_list)
    except Exception as e:
      raise 
  
  def read_syllable_count(self, file_path):
    """
    Reads a file containing words and their syllable counts, and returns a dictionary.
    Args:
        file_path (str): The path to the text file.
    Returns:
        dict: A dictionary where keys are words and values are their syllable counts.
    Raises:
        FilesNotLoadedException: If the file is not found at the given path.
    """

    syllables = {}
    try:
        with open(file_path ,'r') as fs_pos_words:
          for line in fs_pos_words:
              word, count = line.split() 
              syllables[word.lower()] = int(count)
        return syllables
    except FileNotFoundError as ex:
        raise FilesNotLoadedException()
    
   
    
