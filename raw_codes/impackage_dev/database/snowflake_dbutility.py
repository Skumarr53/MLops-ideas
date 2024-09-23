# Databricks notebook source
# MAGIC %run ./../utilities/customexception

# COMMAND ----------

# MAGIC %run ./../filesystem/dbfs_utility

# COMMAND ----------

sfUtils = sc._jvm.net.snowflake.spark.snowflake.Utils
sc = spark.sparkContext
sc._jvm.net.snowflake.spark.snowflake.SnowflakeConnectorUtils.enablePushdownSession(sc._jvm.org.apache.spark.sql.SparkSession.builder().getOrCreate())
zone = sc._jvm.java.util.TimeZone
zone.setDefault(sc._jvm.java.util.TimeZone.getTimeZone("UTC"))

# COMMAND ----------

import pandas as pd
import base64
import configparser
import pytz
import datetime
import os,sys,csv,urllib,time

from pyspark.sql import functions as F
from pyspark.sql.functions import lit,col
from pyspark.sql.types import *
from cryptography.fernet import Fernet
from dateutil import tz
from datetime import datetime, date , timedelta , datetime, timezone


# COMMAND ----------

"""
NAME : SnowFlakeDBUtility

DESCRIPTION:
This module serves the below functionalities:
                ENCRYPTS AND DECRYPTS THE USER CREDENTIALS
                WRITES DATA TO SNOWFLAKE TABLE
                READ DATA FROM SNOWFLAKE DB
                QUERY SNOWFLAKE DB
                TRUNCATE OR MERGE TABLES IN SNOWFLAKE DB
"""

class SnowFlakeDBUtility:
  
  """SNOWFLAKE UTILITY CLASS"""
  
  def __init__(self, environment_str, role_str, schema_str):
    """Initialize the parameters describing the snowflake database utility object"""
    
    self.env = environment_str
    self.url = self.get_url(environment_str) 
    self.db = config.snowflake_config.eds_db_prod
    self.schema = schema_str
    self.dbfs_obj=DBFSUtility()
    self.perform_encryption()
    self.paramsIni = None
    self.role = role_str
    self.INIpath = self.dbfs_obj.INIpath
    self.snowflake_auth_options = self.get_snowflake_auth_options()
  
  def get_url(self,environment_str):
    """Returns snowflake prod or no prod urls based on the environment

    Parameters:
    argument1 (str): environment value

    Returns:
    str: snowflake environment url 

    """
    if environment_str == 'nonprod':
      return config.snowflake_config.voyatest_snowflake_nonprod
    else:
      return config.snowflake_config.voyatest_snowflake_prod
  
  def get_snowflake_auth_options(self):
    """
    Generates and returns a dictionary of authentication options for connecting to a Snowflake database.

    This method decrypts the username and password stored in the `paramsIni` attribute, and then constructs a dictionary containing the necessary authentication details for connecting to a Snowflake database.

    Returns:
        dict: A dictionary containing the following keys:
            - "sfUrl": The URL of the Snowflake instance.
            - "sfUser": The decrypted username for authentication.
            - "sfPassword": The decrypted password for authentication.
            - "sfDatabase": The name of the Snowflake database.
            - "sfSchema": The schema within the Snowflake database.
            - "sfTimezone": The timezone setting for the Snowflake connection, set to "spark".
            - "sfRole": The role to be used for the Snowflake connection.
    """
    username_decrypt = self.decrypt_message(self.paramsIni["username"].encode('utf-8'))
    password_decrypt = self.decrypt_message(self.paramsIni["password"].encode('utf-8'))
    
    username=username_decrypt.decode('utf-8')
    password=password_decrypt.decode('utf-8')
    
    options = {
    "sfUrl": self.url,
    "sfUser": username,
    "sfPassword": password,
    "sfDatabase": self.db,
    "sfSchema": self.schema,
    "sfTimezone": "spark",
    "sfRole": self.role
    }
    return options
    
  def utc_to_local(self, utc_date, tz_val):
    """converts the UTC time to local time zone"""
    return utc_date.replace(tzinfo=timezone.utc).astimezone(tz=tz_val)
  
  def generate_key(self):
    """This method generates a new fernet key. The key must be kept safe as it is the most important component to decrypt the ciphertext"""
    self.credkey = Fernet.generate_key()
      
  def encrypt_message(self, message):
    """It encrypts data passed as a parameter to the method. 
    The outcome of this encryption is known as a Fernet token which is basically the ciphertext.

    Parameters:
    argument1 (str): value that needs to be encrypted

    Returns:
    str: encrypted value 

    """
    encoded_message = message.encode()
    fernet_obj= Fernet(self.credkey)
    encrypted_message = fernet_obj.encrypt(encoded_message)
    return encrypted_message

  def decrypt_message(self, encrypted_message):
    """This method decrypts the Fernet token passed as a parameter to the method. 
    On successful decryption the original plaintext is obtained as a result

    Parameters:
    argument1 (str): encrypted values that needs to be decrypted

    Returns:
    str: decrypted value 

    """
    fernet_obj= Fernet(self.credkey)    
    decrypted_message = fernet_obj.decrypt(encrypted_message)
    return decrypted_message
  
  def perform_encryption(self):
    
    """Reads the cred.ini file and gets the user credentials
       Encrypts the user credentials
       Creates the JSON request using the encrypted credentials"""
    
    self.generate_key()

    credendtial_config = configparser.RawConfigParser()
    file_path='/dbfs' + self.dbfs_obj.INIpath + config.snowflake_config.credentials_file
    print(file_path)
    try:
        with open(file_path) as fs:
          credendtial_config.read_file(fs)
        
    except FileNotFoundError as ex:
        raise FilesNotLoadedException()
        
    db_name=str(self.db)

    try:
        username=credendtial_config.get(db_name,"username")
        password=credendtial_config.get(db_name,"password")
    except:
        raise KeyNotFoundException()
        
    
    self.paramsIni = {"db" : db_name, "username" : self.encrypt_message(username).decode('utf-8'), "password" : self.encrypt_message(password).decode('utf-8')}
    
    self.dbfs_obj.remove_file(self.dbfs_obj.INIpath, 'cred.ini')
  
  def write_to_snowflake_table(self, df, tablename):
    """write dataframe to snowflake database using encrypted credentials
    
    Parameters:
    argument1 (str): dataframe that needs to be written to snowflake database
    argument2 (str): table name to which the dataframe need to be written

    """

    try:
      df\
      .write.format("snowflake")\
      .options(**self.snowflake_auth_options)\
      .option("dbtable",tablename)\
      .mode("append")\
      .save()
    except Exception as ex:
      raise ex
    result = "Load Complete"
    return result
  
  def read_from_snowflake(self,  query):
    """read data from snowflake using encrypted credentials
    
    Parameters:
    argument1 (str): query to read data from snowflake
    """
    try:
      df = spark.read \
                .format("snowflake") \
                .options(**self.snowflake_auth_options) \
                .option("query",  query) \
                .load()
    except Exception as ex:
      raise ex
    return df
  
  def query_snowflake(self,  query):
    """query data from snowflake using encrypted credentials
    
    Parameters:
    argument1 (str): query 
    """
    try:
      sfUtils.runQuery(self.snowflake_auth_options, query)
    except Exception as ex:
      raise ex
    result = "Query Complete"
    
    return result
          
  def truncate_or_merge_table(self, query):
    """truncate or merge data from snowflake using encrypted credentials
    
    Parameters:
    argument1 (str): query to truncate or merge data to table
    """
    try:
      df=sfUtils.runQuery(self.snowflake_auth_options, query)
    except Exception as ex:
      raise ex
    result="Truncate or Merge Complete"
    return result
              
  

# COMMAND ----------

# DBTITLE 1,AKV Service ID for Snowflake Access
class SnowFlakeDBUtilityCTS:

  def __init__(self, schema, srcdbname):
    self.schema = schema
    self.srcdbname = srcdbname
    self.url = "voya.east-us-2.privatelink.snowflakecomputing.com"
    self.db_scope = "id-secretscope-dbk-pr4707-prod"
    self.db_user_key = "SA203647-EDS-DLY-FACTSET-PROD-username"
    self.db_pass_key = "SA203647-EDS-DLY-FACTSET-PROD-password"
    self.db_auth_options = self.get_db_auth_options()

  def __repr__(self):
    return f"Schema & DB in object : {self.schema} & {self.srcdbname} repectively"
  
  def get_db_auth_options(self):
    options = {
    "sfUrl": self.url,
    "sfUser": dbutils.secrets.get(scope = self.db_scope, key = self.db_user_key),
    "sfPassword": dbutils.secrets.get(scope=self.db_scope, key= self.db_pass_key),
    "sfDatabase": self.srcdbname,
    "sfSchema": self.schema,
    "sfTimezone": "spark"
    }
    return options
  
  def read_from_snowflake(self,query): 
    df = spark.read \
              .format("snowflake") \
              .options(**self.db_auth_options) \
              .option("query",  query) \
              .load()

    return df
  
  def write_to_snowflake_table(self, df, tablename):
    df\
    .write.format("snowflake")\
    .options(**self.db_auth_options)\
    .option("dbtable", tablename)\
    .mode("append")\
    .save()
    result="Load Complete"
    return result
  
  def truncate_or_merge_table(self, query):
    df=sfUtils.runQuery(self.db_auth_options, query)
    result="Truncate or Merge Complete"
    return result
