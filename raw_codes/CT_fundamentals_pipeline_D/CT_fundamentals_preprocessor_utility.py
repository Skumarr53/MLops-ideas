# Databricks notebook source
# MAGIC %md
# MAGIC # CALL TRANSCRIPT PREPROCESSING SCRIPT 
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
# MAGIC - "topicx_model.py" including classes to generate topicx scores.
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

# MAGIC %run ./../../data-science-nlp-ml-common-code/impackage/preprocessor/topicx_preprocessor

# COMMAND ----------

# MAGIC %run ./../../data-science-nlp-ml-common-code/impackage/postprocessor/dataframe_transform_utility

# COMMAND ----------

# MAGIC %run ./../../data-science-nlp-ml-common-code/impackage/database/snowflake_dbutility

# COMMAND ----------

# MAGIC %run ./../../data-science-nlp-ml-common-code/impackage/dfutils/dataframe_utils

# COMMAND ----------

# MAGIC %run ./../../data-science-nlp-ml-common-code/impackage/dfutils/spark_df_utils

# COMMAND ----------

import spacy
from spacy.lang.en import English
import pandas as pd
import numpy as np
import datetime
import pytz
from datetime import datetime
from datetime import timedelta

from centralized_nlp_package.data_processing import calculate_significant_duplicates
from centralized_nlp_package.text_processing import sentence_tokenizer





# COMMAND ----------

# DBTITLE 1,Read SnowFlake Credentials
# Object to enable Snowflake access to prod Quant
myDBFS = DBFSUtility()
new_sf =  pd.read_pickle(r'/dbfs' + myDBFS.INIpath + config.snowflake_cred_pkl_file)

# COMMAND ----------

# DBTITLE 1,Read Parsed data.
# Read the last set of parsed transcripts that were run in the fundamentals workflow. Get the (latest) parsed datetime from the set.

try:
  currdf = pd.read_parquet(config.CT_parsed_parquet_file)
  currdf.head(3)
except Exception as ex:
  raise ex



# COMMAND ----------

# DBTITLE 1,Convert dtypes for  DATE and PARSED_DATETTIME...
currdf['PARSED_DATETIME_EASTERN_TZ'] = currdf['PARSED_DATETIME_EASTERN_TZ'].astype('datetime64[ns]')
currdf['DATE'] = pd.to_datetime(currdf['DATE'])
currdf.dtypes

# COMMAND ----------

# DBTITLE 1,Verify if Parsed file has data if not exist execution.
# Read the last set of parsed transcripts that were run in the fundamentals workflow. Get the (latest) parsed datetime from the set.
if len(currdf)<=0:
      #SAVE UPDATED NLP OBJECT FOR FUTURE REFERENCE
      CT_preprocessed_file = open(config.CT_preprocessed_parquet_file, "wb")
      CT_preprocessed_file.close()
      currdf.to_parquet(config.CT_preprocessed_parquet_file,engine = 'pyarrow', compression = 'gzip')
      raise Exception('No New Transcript to parse')


# COMMAND ----------

print(len(currdf))

# COMMAND ----------

currdf.sort_values(by = 'PARSED_DATETIME_EASTERN_TZ').head(3)

# COMMAND ----------

currdf.sort_values(by = 'PARSED_DATETIME_EASTERN_TZ').tail(3)

# COMMAND ----------

# DBTITLE 1,Create TopicX Object
topicx_preprocessor=TopicXPreprocessor()

# COMMAND ----------

currdf.head(3)

# COMMAND ----------

# DBTITLE 1,Tokenize the Sentence.
for label,section in config.FILT_labels.items():
      currdf[label] = currdf[section].apply(lambda x: topicx_preprocessor.sentence_tokenizer(x))


## Refactored 
sent_tokenizer = English()
sent_tokenizer.add_pipe("sentencizer")

 for section in ['FILT_MD','FILT_QA']:
    currdf[section] = currdf[section].apply(lambda x: sentence_tokenizer(x, sent_tokenizer))

# COMMAND ----------

# DBTITLE 1,Create DFUtils Object
df_utils=DFUtils()

# COMMAND ----------

# DBTITLE 1,Get % of Duplicate text in the sentences.
currdf=df_utils.df_get_percentage_of_duplicate_text_in_sentence_list(currdf)
# COMMAND ----------

# DBTITLE 1,Add 1 to Colmn of Duplicate % is > than Threshold.
for label in config.FILT_sections:
  currdf["SIGNIFICANT_DUPLICATE_"+label]=currdf["SIGNIFICANT_DUPLICATE_"+label].apply(lambda x:1 if x>config.duplication_threshold else 0)

# COMMAND ----------


calculate_significant_duplicates(currdf, config.FILT_sections, config.duplication_threshold)

currdf.head(2)

# COMMAND ----------

# DBTITLE 1,Write to Preprocessed Parquet file.
#SAVE UPDATED NLP OBJECT FOR FUTURE REFERENCE
CT_preprocessed_file = open(config.CT_preprocessed_parquet_file, "wb")
CT_preprocessed_file.close()
currdf.to_parquet(config.CT_preprocessed_parquet_file,engine = 'pyarrow', compression = 'gzip')

