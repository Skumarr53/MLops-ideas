# Databricks notebook source
# MAGIC %run ./../utilities/config_utility

# COMMAND ----------

import pickle
from typing import Any
# Databricks notebook source
# MAGIC %run ./../utilities/config_utility

# COMMAND ----------

"""
NAME : DBFSUtility

DESCRIPTION:
This module serves the below functionalities:
                CREATING THE DIRECTORIES IN DBFS
                REMOVING THE FILES FROM DBFS
                READING THE CONTENTS FROM THE FILES IN DBFS
                WRITING THE CONTENTS FROM THE FILES IN DBFS
                GET THE DIRECTORY CONTENT FROM THE DBFS
"""


class DBFSUtility:
  
  """DATABRICKS FILE SYSTEM HELPER CLASS"""
  
  def __init__(self):
    """intialization method for DBFSUtility class"""
    
    self.my_name = self.get_name()
    self.INIpath = config.file_path_config.dbfs_resource_path.format(self.my_name)
    
    
  def get_directory_content(self, folder_path):
    """Returns list of all the sub folders and files containing in the sub folders

    Parameters:
    argument1 (str): folderPath

    Returns:
    str:list 

    """
    dir_paths = dbutils.fs.ls(folder_path)
    subdir_paths = [self.get_directory_content(p.path) for p in dir_paths if p.isDir() and p.path != folder_path]
    flat_subdir_paths = [p for subdir in subdir_paths for p in subdir]
    return list(map(lambda p: p.path, dir_paths)) + flat_subdir_paths

  def add_directory(self, file_path, new_dir):
    """Creates directory in DBFS.

    Parameters:
    argument1 (str): file path where the directory needs to be created
    argument2 (str): new directory name

    """
    dbutils.fs.mkdirs(file_path + new_dir)

  def remove_file(self, file_path, file_name):
    """Removes file on DBFS.

    Parameters:
    argument1 (str): file path from where the file needs to be removed
    argument2 (str): file name that needs to be removed

    """
    dbutils.fs.rm(file_path + file_name, True)
        
  def read_file(self, file_path, file_name):
    """Reads the file content from DBFS

    Parameters:
    argument1 (str): file path
    argument2 (str): file name
    
    Returns:
    str:file content 

    """
    file_content = ''
    full_path='/dbfs' + file_path + file_name
    try:
      with open(full_path, "r") as fs:
        for line in fs:
          if line != '\n':
            file_content = file_content + line + '\n'
    except FileNotFoundError as ex:
      raise ex
    return file_content
    
  def write_file(self, file_path, file_name, content):
    """Writes file on DBFS.

    Parameters:
    argument1 (str): file path
    argument2 (str): file name
    argument2 (str): content that needs to be written to the file
    """
    full_path='/dbfs' + file_path + file_name
    try:
      with open(full_path, "w") as fs_write:
        fs_write.write(content)
    except IOError as ex:
      raise ex

  def load_from_yaml(self, file_path: str) -> dict:
    """
      Load configuration from a YAML file.
      Args:
          file_path (str): The file path to the YAML file.
      Returns:
          dict: The configuration dictionary loaded from the YAML file.
    """
    with open(file_path, 'r') as file:
      config_dict = yaml.safe_load(file)
    return config_dict
  
  @staticmethod
  def read_from_pickle(file_path: str) -> Any:
    """
    Read an object from a pickle file.

    Args:
        file_path (str): The path to the pickle file.

    Returns:
        Any: The object read from the pickle file.
    """
    obj = pd.read_pickle(file_path)
    logging.info(f"Object read from pickle file at: {file_path}")
    return obj
  
  @staticmethod
  def write_to_pickle(file_path: str, obj: Any) -> None:
    """
    Write an object to a pickle file.

    Args:
        file_path (str): The path to the pickle file.
        obj (Any): The object to write.
    """
    with open(file_path, 'wb') as file:
      pickle.dump(obj, file)
    logging.info(f"Object written to pickle file at: {file_path}")  
      
  def get_name(self):
    """Returns Name from the text widget"""
    y = config.file_path_config.folder_name
    return y
                      
  def __str__(self):
    """Returns a string with class description"""
    
    return 'This is a class including funtions to interact with Databricks file system.'

