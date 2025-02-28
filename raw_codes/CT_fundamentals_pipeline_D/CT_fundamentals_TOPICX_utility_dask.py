# Databricks notebook source
# MAGIC %md
# MAGIC # CALL TRANSCRIPT TOPICX SCORES GENERATOR SCRIPT 
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

# MAGIC %pip install dask distributed==2024.3.1
# MAGIC %pip install dask==2024.2.1

# COMMAND ----------

# MAGIC %run ./../../data-science-nlp-ml-common-code/impackage/algorithms/topicx_model

# COMMAND ----------

# MAGIC %run ./../../data-science-nlp-ml-common-code/impackage/postprocessor/dataframe_transform_utility

# COMMAND ----------

# MAGIC %run ./../../data-science-nlp-ml-common-code/impackage/database/snowflake_dbutility

# COMMAND ----------

# MAGIC %run ./../../data-science-nlp-ml-common-code/impackage/dfutils/spark_df_utils

# COMMAND ----------

import datetime
import pytz
from datetime import datetime
import pandas as pd


# COMMAND ----------

import seaborn as sns
from gensim.models import Word2Vec
import spacy
from spacy.lang.en import English
from multiprocessing import Pool
import numpy as np
import tqdm
from tqdm import tqdm
tqdm.pandas()
import sklearn.datasets
import plotly.express as px
from gensim.models import Phrases
from collections import Counter
import dask.dataframe as dd
from dask.distributed import Client
from dask.diagnostics import ProgressBar
import gc
from datetime import timedelta

# Dask client - intended for NC6 v3 GPU instance
client = Client(n_workers=6, threads_per_worker=1)

# COMMAND ----------

# DBTITLE 1,Read SnowFlake credentials.
# Object to enable Snowflake access to prod Quant
myDBFS = DBFSUtility()
new_sf =  pd.read_pickle(r'/dbfs' + myDBFS.INIpath + config.snowflake_cred_pkl_file)

# COMMAND ----------

# DBTITLE 1,Read Preprocessed parquet file.
# Read the last set of parsed transcripts that were run in the fundamentals workflow. Get the (latest) parsed datetime from the set.
try:
  currdf = pd.read_parquet(config.CT_preprocessed_parquet_file)
  currdf.head(3)
except Exception as ex:
  raise ex


# COMMAND ----------

# DBTITLE 1,Raise Exception if no records in preprocessed file.
if len(currdf)>0:
    print('The data spans from ' + str(currdf['PARSED_DATETIME_EASTERN_TZ'].min()) + ' to ' + str(currdf['PARSED_DATETIME_EASTERN_TZ'].max()) + 'and has ' + str(currdf.shape[0]) + ' rows and ' + str(currdf.shape[1]) + ' columns.')

else:
    CT_topicx_file = open(config.CT_TOPICX_parquet_file, "wb")
    CT_topicx_file.close()
    currdf.to_parquet(config.CT_TOPICX_parquet_file,engine = 'pyarrow', compression = 'gzip')      
    raise Exception('No New Transcripts to parse')

# COMMAND ----------

# DBTITLE 1,Convert Earnings_CALL column to Boolean.
currdf['EARNINGS_CALL']=currdf['EARNINGS_CALL'].astype('bool')

# COMMAND ----------

# DBTITLE 1,Read Financial related labels and words.
# Read match list and create match set
try:
  match_df = pd.read_csv(config.CT_fundamentals_path)
except Exception as ex:
  raise ex

# COMMAND ----------

# DBTITLE 1,Create TopicXModel object.
topicx_obj=TopicXModel()

# COMMAND ----------

# DBTITLE 1,Word Set dictionary having unigrams and bigrams of financial words.
word_set_dict = {topic.upper() : topicx_obj.get_match_set(match_df[match_df['label']==topic]['word'].values) for topic in match_df['label'].unique()}

# COMMAND ----------

# DBTITLE 1,Assign config.FILT_sections(Ex : FILT_MD, FILT_QA)
filt_items=config.FILT_sections

# COMMAND ----------

# DBTITLE 1,Get partition_value (Default = 1000)
partition_val=config.partition_value

# COMMAND ----------

# DBTITLE 1,Calculate range_end_index (Stop the execution if the index is exceeded)
range_end_index=len(currdf)//partition_val
print(range_end_index)

# COMMAND ----------

# DBTITLE 1,Calculate offset.
remaining_offset=len(currdf)%partition_val
print(remaining_offset)

# COMMAND ----------

# DBTITLE 1,Create new Empty pandas dataframe.
new_df = pd.DataFrame()

# COMMAND ----------

print(len(currdf))

# COMMAND ----------

# MAGIC %md
# MAGIC #Dask Computiontion steps
# MAGIC <br>__Step1__ : Convert pandas dataframe to daskdataframe using the from_pandas method in dask.
# MAGIC <br>__Step2__ : Generate count of words with respect to financial words loaded in above steps.
# MAGIC <br>__Step3__ : Compute above steps with compute method.
# MAGIC <br>__Step4__ : Append the results from compute to new dataframe.

# COMMAND ----------

global new_df
if range_end_index==0:
  dask_df = dd.from_pandas(currdf, npartitions =6)
  dask_df=topicx_obj.generate_match_count(dask_df,word_set_dict)
  dask_df=dask_df.compute()
  new_df=new_df.append(dask_df,ignore_index=True)
else:
  start_index=0
  end_index=partition_val
  for index in range(0,range_end_index):
    dask_df = dd.from_pandas(currdf.iloc[start_index:end_index], npartitions =6)
    dask_df=topicx_obj.generate_match_count(dask_df,word_set_dict)
    dask_df=dask_df.compute()
    new_df=new_df.append(dask_df,ignore_index=True)
    if end_index==range_end_index*partition_val:
      break
    start_index=end_index
    end_index=end_index+partition_val

# COMMAND ----------

print(len(new_df))

# COMMAND ----------

# MAGIC %md
# MAGIC #OFFset Computation with dask.
# MAGIC <br>__Step1__: Here execution happens only if the number of records exceeds the partition value.

# COMMAND ----------

if remaining_offset>0 and range_end_index>0:
  dask_df = dd.from_pandas(currdf.iloc[end_index:end_index+remaining_offset], npartitions =6)
  dask_df=topicx_obj.generate_match_count(dask_df,word_set_dict)
  dask_df=dask_df.compute()
  new_df=new_df.append(dask_df,ignore_index=True)

# COMMAND ----------

print(len(new_df))

# COMMAND ----------

new_df.head(2)

# COMMAND ----------

print(len(new_df))



# COMMAND ----------

currdf=new_df

# COMMAND ----------

currdf.head(2)

# COMMAND ----------

currdf=topicx_obj.generate_topic_statistics(currdf,word_set_dict)

# COMMAND ----------

currdf["EARNINGS_CALL"] = currdf["EARNINGS_CALL"].replace({True: 1, False: 0})

# COMMAND ----------

currdf.head(2)

# COMMAND ----------

#SAVE UPDATED NLP OBJECT FOR FUTURE REFERENCE
CT_topicx_file = open(config.CT_TOPICX_parquet_file, "wb")
CT_topicx_file.close()
currdf.to_parquet(config.CT_TOPICX_parquet_file,engine = 'pyarrow', compression = 'gzip')  

