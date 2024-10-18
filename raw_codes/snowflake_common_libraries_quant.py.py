# Databricks notebook source
# MAGIC %md
# MAGIC <br>__Author__:            'Priya Srinivasan'
# MAGIC <br>__Contact__:           'Priya.Srinivasan@voya.com'
# MAGIC <br>__Revision History__:  

# COMMAND ----------

pip install "snowflake-connector-python[secure-local-storage,pandas]"

# COMMAND ----------

# DBTITLE 1,To enable SQL pushdown to snowflake and set timezone for spark 
sfUtils = sc._jvm.net.snowflake.spark.snowflake.Utils
sc = spark.sparkContext
sc._jvm.net.snowflake.spark.snowflake.SnowflakeConnectorUtils.enablePushdownSession(sc._jvm.org.apache.spark.sql.SparkSession.builder().getOrCreate())
zone = sc._jvm.java.util.TimeZone
zone.setDefault(sc._jvm.java.util.TimeZone.getTimeZone("UTC"))

# COMMAND ----------

# DBTITLE 1,Set timezone for databricks sql 
# MAGIC %sql
# MAGIC SET TIME ZONE 'UTC';

# COMMAND ----------

# DBTITLE 1,Import functions 
import pandas as pd
import snowflake.connector
from snowflake.connector.converter_null import SnowflakeNoConverterToPython
from pyspark.sql.functions import year, month, dayofmonth,lit,trim,concat,col,coalesce
#from snowflake.connector import utils
#from snowflake import utils
import base64
import configparser
import logging
from cryptography.fernet import Fernet
from pyspark.sql import functions as F
import pyspark.sql.functions as f
from pyspark.sql.functions import count as count
from pyspark.sql.window import *
from pyspark.sql.functions import *
from functools import reduce
from pyspark.sql import DataFrame
from pyspark.sql.functions import *

# COMMAND ----------

# DBTITLE 1,Date manipulations for load date
from dateutil import tz
import pytz
from datetime import date
from datetime import datetime
import datetime
import os,sys,csv,urllib,time
from datetime import datetime, date , timedelta , datetime, timezone
from pyspark.sql.functions import col
eastern_tzinfo = pytz.timezone("America/New_York")
def utc_to_local(utc_dt,tz_var):
    return utc_dt.replace(tzinfo=timezone.utc).astimezone(tz=tz_var)

# COMMAND ----------

# DBTITLE 1,Snowflake keypair auth
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization
from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
from pyspark.sql.types import *
import re

# COMMAND ----------

# DBTITLE 1,Get Load date
eastern_tzinfo = pytz.timezone("America/New_York")
load_date_time = utc_to_local(datetime.now(),eastern_tzinfo)
load_date_prep = load_date_time.date() 
load_date = load_date_prep.strftime('%Y-%m-%d') 
load_date_time = load_date_prep.strftime('%Y-%m-%d %H:%M:%S') 
print("Today's load date")
print(load_date)
print(load_date_time)

# COMMAND ----------

# DBTITLE 1,Get info from AKV
key_file=dbutils.secrets.get(scope = "id-secretscope-dbk-pr4707-prod-work", key = "eds-prod-quant-key")
pwd=dbutils.secrets.get(scope = "id-secretscope-dbk-pr4707-prod-work", key = "eds-prod-quant-pwd")

# COMMAND ----------

p_key = serialization.load_pem_private_key(
key_file.encode('ascii'),
password=pwd.encode(),
backend=default_backend()
)
 
pkb = p_key.private_bytes(
    encoding=serialization.Encoding.PEM,
    format=serialization.PrivateFormat.PKCS8,
    encryption_algorithm=serialization.NoEncryption()
)
 
pkb = pkb.decode("UTF-8")
pkb = re.sub("-*(BEGIN|END) PRIVATE KEY-*\n","",pkb).replace("\n","")

# COMMAND ----------

# DBTITLE 1,Setting connection parameters for Snowflake
options = {
"sfUrl" : "voya.east-us-2.privatelink.snowflakecomputing.com",
"sfUser" : "SA_EDS_PROD_QUANT",
"pem_private_key": pkb,  
"sfDatabase" : "EDS_PROD",
"sfSchema" : "QUANT",
"sfWarehouse" : "WH_EDS_PROD_READ",
"sfRole": "ROLE_EDS_PROD_DDLADMIN_QUANT",
"sfTimezone" : "spark"
}
sourceDbName_eds = 'EDS_PROD'
source_format="net.snowflake.spark.snowflake"

# COMMAND ----------

# DBTITLE 1,Read from snowflake EDS_ACCP
def read_from_snowflake_eds(sourceDbName_eds,query):
  df = spark.read \
            .format(source_format) \
            .options(**options) \
            .option("query",  query) \
            .load()
  return df

# COMMAND ----------

# DBTITLE 1,Truncate or Merge table EDS_ACCP
def truncate_merge_table_snowflake_eds(sourceDbName_eds,query):
  df = sfUtils.runQuery(options, query)
  result="Truncate or Merge Complete"
  return result

# COMMAND ----------

df = read_from_snowflake_eds('', "SELECT * FROM EDS_PROD.QUANT_STG.CTS_INSTRUCTOREMBEDDING_H LIMIT 10")

# COMMAND ----------

# DBTITLE 1,Write to EDS_ACCP Snowflake
def write_to_snowflake_table_eds(sourceDbName_eds,df,tablename):
  df\
  .write.format(source_format)\
  .options(**options)\
  .option("dbtable",tablename)\
  .mode("append")\
  .save()
  result="Load Complete"
  return result
