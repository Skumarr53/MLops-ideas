# Databricks notebook source
import numpy as np
import pandas as pd
from pyspark.sql.types import *

# COMMAND ----------

"""
NAME : SparkDFUtils

DESCRIPTION:
This module serves the dataframe operations functionalities:
               CONVERT THE SPARK DATAFRAME COLUMN DATA TYPES
"""


class SparkDFUtils:
  
  """SparkDFUtils CLASS"""
    
  def convert_column_to_date_timestamp_type(self,spark_df,column,format):
      """
      converts the spark data frame column to date time data type
        
      Parameters:
      argument1 (dataframe): spark dataframe
      argument2 (str): column name
      argument3 (str): date or date time format
    
      Returns:
      spark dataframe
    
      """
      return spark_df.withColumn(column, F.to_timestamp(spark_df[column],format))

  def convert_column_to_date_type(self,spark_df,column,format):
      """
      converts the spark data frame column to date data type
        
      Parameters:
      argument1 (dataframe): spark dataframe
      argument2 (str): column name
      argument3 (str): date or date time format
    
      Returns:
      spark dataframe
    
      """
      
      return spark_df.withColumn(column, to_date(spark_df[column], format))
  
  def cleanup(self,df_obj):
    """
    Clean up an object by deleting it and running garbage collection.

    Args:
        obj (object): The object to be cleaned up.
    """
    del df_obj
    gc.collect()
                                 

  
    

