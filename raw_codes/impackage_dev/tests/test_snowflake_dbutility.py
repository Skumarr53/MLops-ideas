# Databricks notebook source
# MAGIC %run "./../database/snowflake_dbutility"

# COMMAND ----------

dbutils.fs.cp('dbfs:/FileStore/MJ/cred.ini','dbfs:/FileStore/Santhosh/resources/')

# COMMAND ----------

dbutils.fs.ls("dbfs:/FileStore/Santhosh/resources/")

# COMMAND ----------

# MAGIC %pip install nutter

# COMMAND ----------

from pyspark.sql import SparkSession
spark = SparkSession.builder.appName('SparkExamples').getOrCreate()

# COMMAND ----------

"""
Nutter Fixture for testing the Snowflake db utility module.
"""

from runtime.nutterfixture import NutterFixture
class SnowFlakeDBUtilityFixture(NutterFixture):
   """
   This SnowFlakeDBUtility fixture is used for unit testing all the methods that are used in the snowflake_dbutility.py module 
   """
   def __init__(self):
      """
      helps in intializing all the instance variables

      instance variable: Config is created
      """
      
      self.snowlfake_db_obj=SnowFlakeDBUtility('prod', 'ROLE_EDS_PROD_DDLADMIN_1', 'WORK')
      NutterFixture.__init__(self)
    
   def assertion_get_url_nonprod(self):
      """
      This method is used for unit testing SnowFlakeDBUtility.get_url method 
      """
      assert ("voyatest." in self.snowlfake_db_obj.get_url("nonprod"))
      
   def assertion_get_url_prod(self):
      """
      This method is used for unit testing SnowFlakeDBUtility.get_url method 
      """
      assert ("voya." in self.snowlfake_db_obj.get_url(""))

   def assertion_write_to_snowflake_table(self):
      """
      This method is used for unit testing SnowFlakeDBUtility.get_url method 
      """
      # Create a spark dataframe
      columns = ["EMP_ID", "NAME"]
      data = [("101", "Ajay")]
      emp_df = spark.createDataFrame(data).toDF(*columns)

      # View the dataframe
      result=self.snowlfake_db_obj.write_to_snowflake_table(emp_df, "AJAY_TEST")
      assert(result == "Load Complete")

   def assertion_read_from_snowflake(self):
      """
      This method is used for unit testing SnowFlakeDBUtility.get_url method 
      """
      query='SELECT * FROM AJAY_TEST;'

      # View the dataframe
      result=self.snowlfake_db_obj.read_from_snowflake(query)
      assert(result.count()>0)

   def assertion_query_snowflake(self):
      """
      This method is used for unit testing SnowFlakeDBUtility.get_url method 
      """
      query='SELECT * FROM AJAY_TEST;'

      # View the dataframe
      result=self.snowlfake_db_obj.query_snowflake(query)
      assert(result=="Query Complete")

   def assertion_truncate_or_merge_table(self):
      """
      This method is used for unit testing SnowFlakeDBUtility.get_url method 
      """
      query='TRUNCATE TABLE IF EXISTS AJAY_TEST;'

      # View the dataframe
      result=self.snowlfake_db_obj.truncate_or_merge_table(query)
      assert(result=="Truncate or Merge Complete")


   def assertion_get_snowflake_auth_options(self):
      """
      This method is used for unit testing SnowFlakeDBUtility.get_snowflake_auth_options method 
      """
      options = self.snowlfake_db_obj.get_snowflake_auth_options()
      assert isinstance(options, dict)
      assert "sfUrl" in options
      assert "sfUser" in options
      assert "sfPassword" in options
      assert "sfDatabase" in options
      assert "sfSchema" in options
      assert "sfTimezone" in options
      assert "sfRole" in options
      assert options["sfUrl"] == self.snowlfake_db_obj.url
      assert options["sfDatabase"] == self.snowlfake_db_obj.db
      assert options["sfSchema"] == self.snowlfake_db_obj.schema
      assert options["sfTimezone"] == "spark"
      assert options["sfRole"] == self.snowlfake_db_obj.role

result = SnowFlakeDBUtilityFixture().execute_tests()
print(result.to_string())
