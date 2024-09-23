# Databricks notebook source
# MAGIC %run ./../utilities/config_utility

# COMMAND ----------

import numpy as np

# COMMAND ----------

from pyspark.sql.types import *
from pyspark.sql.functions import to_date,to_timestamp

# COMMAND ----------


from pyspark import SparkContext

# COMMAND ----------

"""
NAME : DFTransformUtility

DESCRIPTION:
This module serves the dataframe operations functionalities:
                FINDS THE EQUIVALENT SPARK COLUMN DATA TYPE TO DATAFRAME COLUMN DATA TYPE
                CONVERTS PANDAS DATAFRAME TO SPARK DATAFRAME
"""


class DFTransformUtility:
  
  """DFTransformUtility CLASS"""
    
  def preprocess_spark_df(self,currdf,columns_list):
      """ Preprocess the spark dataframe

      Parameters:
      argument1 (str): schema config name 

      Returns:
      spark dataframe:spark parsed dataframe

      """
      self.spark_parsedDF = self.pandas_to_spark(currdf[columns_list])
      self.spark_parsedDF = self.spark_parsedDF.replace(np.nan, None)
    #   self.spark_parsedDF = self.spark_parsedDF.withColumn("DATE", to_date(self.spark_parsedDF.DATE, 'yyyy-MM-d'))
    #   self.spark_parsedDF = self.spark_parsedDF.withColumn("PARSED_DATETIME_EASTERN_TZ", to_timestamp(self.spark_parsedDF.PARSED_DATETIME_EASTERN_TZ, 'yyyy-MM-dd HH mm ss'))
    #   self.spark_parsedDF = self.spark_parsedDF.withColumn("EVENT_DATETIME_UTC", to_timestamp(self.spark_parsedDF.EVENT_DATETIME_UTC, 'yyyy-MM-dd HH mm ss')) 
      return self.spark_parsedDF
      
  def equivalent_type_fundamentals(self,string, type):
      """ Functions to read a pandas dataframe and return a datatype specification for an equivalent Spark dataframe.

      Parameters:
      argument1 (str): column name 
      argument2 (str): column data type

      Returns:
      spark sql data type

      """
      if type == 'datetime64[ns]': return TimestampType()
      elif type == 'int64': return LongType()
      elif type == 'int32': return IntegerType()
      elif type == 'float64': return FloatType()
      elif 'embed' in string.lower(): return ArrayType(FloatType())
      elif 'len' in string.lower(): return ArrayType(IntegerType())
      elif 'total' in string.lower(): return ArrayType(IntegerType())
      elif 'stats_list' in string.lower(): return ArrayType(MapType(StringType(), IntegerType()))
      elif 'stats' in string.lower(): return MapType(StringType(), IntegerType())
      elif 'sent_scores' in string.lower(): return ArrayType(FloatType())
      elif 'fog_index_per_sent' in string.lower(): return ArrayType(FloatType())
      elif 'sent_labels' in string.lower(): return ArrayType(IntegerType())
      elif '_per_sent' in string.lower(): return ArrayType(IntegerType())
      elif 'sent_weight' in string.lower(): return FloatType()
      elif 'sent_rel' in string.lower(): return FloatType()
      elif 'relevance' in string.lower(): return FloatType()
      elif 'net_sent' in string.lower(): return IntegerType()
      elif 'SENT_FILT_QA' == string or 'SENT_FILT_MD' == string or 'SENT_FILT_CEO_MD'== string or 'SENT_FILT_CEO_QA'== string or 'SENT_FILT_EXEC_MD'==string or 'SENT_FILT_EXEC_QA'==string or 'SENT_FILT_ANL_QA'==string: return FloatType()
      elif 'FILT_QA' == string or 'FILT_MD' == string or 'FILT_CEO_MD' == string or 'FILT_EXEC_MD' == string or 'FILT_EXEC_QA' == string or 'FILT_CEO_QA' == string or 'FILT_ANL_QA' == string: return ArrayType(StringType())
      elif 'SENT_FILT_QA' in string or 'SENT_FILT_MD' in string or 'SENT_FILT_CEO_MD' in string or 'SENT_FILT_CEO_QA' in string or 'SENT_FILT_EXEC_MD' in string or 'SENT_FILT_EXEC_QA' in string or 'SENT_FILT_ANL_QA' in string: return FloatType()
      elif 'SENTS_FILT_QA' in string or 'SENTS_FILT_MD' in string or 'SENTS_FILT_CEO_MD' in string or 'SENTS_FILT_CEO_QA' in string or 'SENTS_FILT_EXEC_MD' in string or 'SENTS_FILT_EXEC_QA' in string or 'SENTS_FILT_ANL_QA' in string: return IntegerType()
      else: return StringType()

  def define_structure(self,string, format_type):
      """ Preprocess the spark dataframe

      Parameters:
      argument1 (str): column name
      argument1 (str): column data type

      Returns:
      struct field with column name and spark sql type

      """
      spark_sql_type = self.equivalent_type_fundamentals(string, format_type)
      return StructField(string, spark_sql_type)


  def pandas_to_spark(self,pandas_df):
      """ converts pandas dataframe to spark dataframe
      Parameters:
      argument1 : dataframe

      Returns:
      spark dataframe

      """
      columns = list(pandas_df.columns)
      types = list(pandas_df.dtypes)
      struct_list = []
      for column, data_type in zip(columns, types): 
        struct_list.append(self.define_structure(column, data_type))
      p_schema = StructType(struct_list)
      return sqlContext.createDataFrame(pandas_df, p_schema)
  
  def create_empty_spark_DF(self):
      """ creates an empty spark dataframe

      Returns:
      empty spark dataframe

      """
      sc = SparkContext.getOrCreate()
      spark = SparkSession(sc)     # Need to use SparkSession(sc) to createDataFrame

      schema = StructType([
          StructField("column1",StringType(),True),
          StructField("column2",StringType(),True)
      ])
      empty = spark.createDataFrame(sc.emptyRDD(), schema)
      return empty
    
  def filter_stats_dictionary(self,stats_dict):
      """filter dict values by removing the none item
      Parameters:
      argument1 : dictionary

      Returns:
      filtered dictonary

      """
      filtered_dicts = {}
      for key,value in stats_dict.items():
        if value is not None:
            filtered_dicts[key] = int(value)
      return filtered_dicts


    

    
