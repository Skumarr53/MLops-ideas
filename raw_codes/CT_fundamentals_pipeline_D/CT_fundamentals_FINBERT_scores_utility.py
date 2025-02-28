# Databricks notebook source
# MAGIC %md
# MAGIC # CALL TRANSCRIPT FINBERT SCORES GENERATOR SCRIPT 
# MAGIC   
# MAGIC PRIMARY AUTHOR : Bea Yu (primary author of classes composing NLP methods, other classes called in this processing and "glue code")
# MAGIC
# MAGIC SECONDARY AUTHORS: Partha Kadmbi (primary author of most NLP methods)
# MAGIC
# MAGIC DATE : 04.03.2023
# MAGIC
# MAGIC CONTEXT DESCRIPTION : First notebook in the call transcript NLP processing pipeline to execute ETL from Snowflake nonproduction databases
# MAGIC
# MAGIC CONTEXT NOTEBOOKS :
# MAGIC
# MAGIC - "FINBERT_model.py" including classes to generate finbert scores.
# MAGIC - "dataframe_utility.py" including classes to generate the spark dataframe with spark data types from pandas dataframe.
# MAGIC - "snowflake_dbutility.py" including classes to interact with the snowflake database.
# MAGIC - "logging.py" including classes for custom logging.
# MAGIC - After then notebook has executed:
# MAGIC
# MAGIC   - Run CT_fundamentals_preprocessor_utility.py to generate the FILT values and saved in AJAY_CTS_PREPROCESSED_D table in snowflake.
# MAGIC   - Run CT_fundamentals_TOPICX_utility.py and CT_fundamentals_FINBERT_Scores_Utility.py to generate the topicx and FINBERT scores.
# MAGIC         And saves in topicx scores in AJAY_CTS_TOPICX_SCORES_D table and finbert scores in AJAY_CTS_FINBERT_SCORES_D table of snowflake
# MAGIC   - Run CT_fundamentals_sentiment_scores_utility.py to generate the relevance scores and saved in AJAY_CTS_COMBINED_SCORES_D table in snowflake
# MAGIC
# MAGIC
# MAGIC CONTACT INFO : bea.yu@voya.com, partha.kadambi@voya.com, ajaya.devalla@voya.com

# COMMAND ----------

# MAGIC %run ./../../data-science-nlp-ml-common-code/impackage/mlflowutils/mlflow_model

# COMMAND ----------

# MAGIC %run ./../../data-science-nlp-ml-common-code/impackage/postprocessor/dataframe_transform_utility

# COMMAND ----------

# MAGIC %run ./../../data-science-nlp-ml-common-code/impackage/dfutils/dataframe_utils

# COMMAND ----------

# MAGIC %run ./../../data-science-nlp-ml-common-code/impackage/database/snowflake_dbutility

# COMMAND ----------

# MAGIC %run ./../../data-science-nlp-ml-common-code/impackage/dfutils/spark_df_utils

# COMMAND ----------

import pandas as pd
import numpy as np
import pickle
import datetime
import pytz
from datetime import timedelta
from datetime import datetime


# COMMAND ----------

# DBTITLE 1,Read SnowFlake Credentials
# Object to enable Snowflake access to prod Quant
myDBFS = DBFSUtility()
new_sf =  pd.read_pickle(r'/dbfs' + myDBFS.INIpath + config.snowflake_cred_pkl_file)

# COMMAND ----------

# DBTITLE 1,Read Preprocessed Data
# Read the last set of parsed transcripts that were run in the fundamentals workflow. Get the (latest) parsed datetime from the set.
try:
  currdf = pd.read_parquet(config.CT_preprocessed_parquet_file)
  currdf.head(3)
except Exception as ex:
  raise ex


# COMMAND ----------

# DBTITLE 1,Verify if preprocessed  data is available.
if len(currdf)>0:
    print('The data spans from ' + str(currdf['PARSED_DATETIME_EASTERN_TZ'].min()) + ' to ' + str(currdf['PARSED_DATETIME_EASTERN_TZ'].max()) + 'and has ' + str(currdf.shape[0]) + ' rows and ' + str(currdf.shape[1]) + ' columns.')

else:
    CT_finbert_file = open(config.CT_FINBERT_parquet_file, "wb")
    CT_finbert_file.close()
    currdf.to_parquet(config.CT_FINBERT_parquet_file,engine = 'pyarrow', compression = 'gzip')
    raise Exception("No preprocessed transript.")

# COMMAND ----------

# DBTITLE 1,Get Notebook Environement.
notebook_path=dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()

# COMMAND ----------

# DBTITLE 1,Load the Finbert Model based on ENV.
skip_transition=False
model_name=config.mlflow_FINBERT_model_name
if "@voya.com" in notebook_path:
  skip_transition=True
  env=config.mlflow_stages_dict["Quant"]
  from_env=""
  to_env="" 
elif "Quant_Stg" in notebook_path:
  skip_transition=False
  env=config.mlflow_stages_dict["Quant_Stg"]
  from_env=config.mlflow_transtion_dict["Quant_Stg"]
  to_env=env 
elif "Quant_Live" in notebook_path:
  skip_transition=False
  env=config.mlflow_stages_dict["Quant_Live"]
  from_env=config.mlflow_transtion_dict["Quant_Live"]
  to_env=env 
elif "Quant" in notebook_path:
  skip_transition=True
  env=config.mlflow_stages_dict["Quant"]
  from_env=""
  to_env="" 

mlflow_object=MLFlowModel(model_name,env,skip_transition,"",from_env,to_env)
mlflow_model_object=mlflow_object.load_model()


# COMMAND ----------

# DBTITLE 1,Load FILT_sections (FILT_MD, FILT_QA etc)
filt_sections_list=config.FILT_sections

# COMMAND ----------

# DBTITLE 1,Create DFUtils object and conver column to list.
df_utils=DFUtils()
currdf=df_utils.df_column_convert_to_list(currdf,filt_sections_list)

# COMMAND ----------

# DBTITLE 1,Predict labels and Scores using Loaded Finbert model.
sent_map ={'LABEL_0': 1, 'LABEL_1': -1, 'LABEL_2': 0}
for section in config.FILT_sections:
  currdf['SENT_SCORES_' + section] = currdf[section].apply(lambda x: mlflow_model_object.predict(x))  
  currdf['SENT_LABELS_' + section] = currdf['SENT_SCORES_' + section].apply(lambda x: [val for val, score in x])
  currdf['SENT_SCORES_' + section] = currdf['SENT_SCORES_' + section].apply(lambda x: [score for val, score in x])


# COMMAND ----------

# DBTITLE 1,Write Finbert Data to parquet File.
#SAVE UPDATED NLP OBJECT FOR FUTURE REFERENCE
CT_finbert_file = open(config.CT_FINBERT_parquet_file, "wb")
CT_finbert_file.close()
currdf.to_parquet(config.CT_FINBERT_parquet_file,engine = 'pyarrow', compression = 'gzip')

