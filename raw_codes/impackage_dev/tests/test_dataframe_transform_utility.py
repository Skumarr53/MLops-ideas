# Databricks notebook source
import pandas as pd

# COMMAND ----------

from pyspark.sql.dataframe import DataFrame
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.types import StructField



# COMMAND ----------

# MAGIC %run "./../postprocessor/dataframe_transform_utility"

# COMMAND ----------

"""
Nutter Fixture for testing the dataframe utility module.
"""

from runtime.nutterfixture import NutterFixture
class DFTransformUtilityFixture(NutterFixture):
   """
   This DFTransformUtility fixture is used for unit testing all the methods that are used in the dataframe_utility.py module 
   """
   def __init__(self):
      """
      helps in intializing all the instance variables

      instance variable: DFTransformUtility is created
      """
             
      # initialize list of lists
      data = [['tom', 10,pd.Timestamp.today(),pd.Timestamp.today(),pd.Timestamp.today()], ['nick', np.nan,pd.Timestamp.today(),pd.Timestamp.today(),pd.Timestamp.today()], ['juli', 14,pd.Timestamp.today(),pd.Timestamp.today(),pd.Timestamp.today()]]
        
      # Create the pandas DataFrame
      self.pandas_DF = pd.DataFrame(data, columns=['Name', 'Age','DATE','PARSED_DATETIME_EASTERN_TZ','EVENT_DATETIME_UTC'])
      self.pandas_DF2 = pd.DataFrame(data, columns=['Name', 'Age','DATE','PARSED_DATETIME_EASTERN_TZ','EVENT_DATETIME_UTC'])
      self.df_transform_utility_obj=DFTransformUtility()
      NutterFixture.__init__(self)
      
   def assertion_pandas_to_spark(self):
      """
      This method is used for unit testing DFTransformUtility.pandas_to_spark method
      """
      assert (isinstance(self.df_transform_utility_obj.pandas_to_spark(self.pandas_DF),DataFrame))
   def assertion_preprocess_spark_df(self):
      """
      This method is used for unit testing DFTransformUtility.preprocess_spark_df method
      """
      column_list=['Age','DATE','PARSED_DATETIME_EASTERN_TZ','EVENT_DATETIME_UTC']
      spark_df=self.df_transform_utility_obj.preprocess_spark_df(self.pandas_DF,column_list)
      assert (len(spark_df.columns)==4 and spark_df.collect()[1][0]==None and dict(spark_df.dtypes)['PARSED_DATETIME_EASTERN_TZ']=='timestamp')

   def assertion_equivalent_type_fundamentals(self):
      """
      This method is used for unit testing DFTransformUtility.equivalent_type method
      """
      assert (self.df_transform_utility_obj.equivalent_type_fundamentals("SENT_FILT_QA",type("SENT_FILT_QA"))==FloatType())

   def assertion_equivalent_type_fundamentals_for_default_type(self):
      """
      This method is used for unit testing DFTransformUtility.equivalent_type method
      """
      assert (self.df_transform_utility_obj.equivalent_type_fundamentals("quant","quant")==StringType())

   def assertion_create_empty_spark_DF(self):
      """
      This method is used for unit testing DFTransformUtility.create_empty_spark_DF method
      """
      assert ((self.df_transform_utility_obj.create_empty_spark_DF()).isEmpty)

   def assertion_filter_stats_dictionary(self):
      """
      This method is used for unit testing DFTransformUtility.create_empty_spark_DF method
      """
      filtered_dict=self.df_transform_utility_obj.filter_stats_dictionary({"ebitda_margin": 1,"margin": 7,"roic": None})
      assert(len(filtered_dict.keys())==2)

   def assertion_define_structure(self):
      """
      This method is used for unit testing DFTransformUtility.define_structure method
      """
      assert(isinstance(self.df_transform_utility_obj.define_structure("values","int64"),StructField))


result = DFTransformUtilityFixture().execute_tests()
print(result.to_string())
