# Databricks notebook source
# MAGIC %md
# MAGIC # NLI using transformers

# COMMAND ----------

!pip install transformers==4.40.1

# COMMAND ----------

pip install --upgrade safetensors

# COMMAND ----------

!pip install gensim==4.2.0
!pip install spacy==3.4.4
!pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.4.0/en_core_web_sm-3.4.0.tar.gz
!pip install dask distributed --upgrade
!pip install dask==2023.5.0
!pip install pydantic==2.2.1


# COMMAND ----------

# MAGIC %run ./../../../data-science-nlp-ml-common-code/impackage/utilities/config_utility

# COMMAND ----------

# MAGIC %run ./../../../data-science-nlp-ml-common-code/impackage/database/snowflake_dbutility

# COMMAND ----------

new_sf = SnowFlakeDBUtility(config.schema, config.eds_db_prod)

# COMMAND ----------

import ast
from IPython.display import clear_output
import numpy as np
from dateutil.relativedelta import relativedelta
import pickle 
import pandas as pd

# COMMAND ----------

# Load model directly
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel
from transformers import TextClassificationPipeline
from transformers import ZeroShotClassificationPipeline
import torch

device = 0 if torch.cuda.is_available() else -1

model_1_folder_name = "deberta-v3-large-zeroshot-v2"

model_folder_path = "/dbfs/mnt/access_work/UC25/Libraries/HuggingFace/"

tokenizer_1 = AutoTokenizer.from_pretrained(model_folder_path + model_1_folder_name)
model_1 = AutoModelForSequenceClassification.from_pretrained(model_folder_path + model_1_folder_name)


# COMMAND ----------

class_data_path = "/dbfs/mnt/access_work/UC25/Topic Modeling/NLI Models/"

# COMMAND ----------

# MAGIC %md
# MAGIC # Read raw data

# COMMAND ----------

# MAGIC %md
# MAGIC ## Query data if no csv exists

# COMMAND ----------

data_end_date = datetime.now() 
data_last_month = datetime.now() - relativedelta(months=1)

# COMMAND ----------

lastDateNewQuery = (pd.to_datetime(format(data_last_month, '%m') + "-01-" + format(data_last_month, '%Y'))).strftime('%Y-%m-%d')
currentDateNewQuery = (pd.to_datetime(format(data_end_date, '%m') + "-01-" + format(data_end_date, '%Y'))).strftime('%Y-%m-%d')

mind = "'" + lastDateNewQuery + "'"
maxd = "'" + currentDateNewQuery + "'"
print('The next query spans ' + mind + ' to ' + maxd)

# COMMAND ----------

tsQuery= ("select CALL_ID,ENTITY_ID, DATE, FILT_MD, FILT_QA, CALL_NAME,COMPANY_NAME,EARNINGS_CALL,ERROR,TRANSCRIPT_STATUS,UPLOAD_DT_UTC,VERSION_ID,EVENT_DATETIME_UTC,PARSED_DATETIME_EASTERN_TZ,SENT_LABELS_FILT_MD, SENT_LABELS_FILT_QA from EDS_PROD.QUANT_LIVE.CTS_FUND_COMBINED_SCORES_H  WHERE DATE >= " + mind + " AND DATE < " + maxd + " ORDER BY PARSED_DATETIME_EASTERN_TZ DESC;")

resultspkdf = new_sf.read_from_snowflake(tsQuery)

currdf = resultspkdf.toPandas()

if len(currdf)>0:
    print('The data spans from ' + str(currdf['PARSED_DATETIME_EASTERN_TZ'].min()) + ' to ' + str(currdf['PARSED_DATETIME_EASTERN_TZ'].max()) + 'and has ' + str(currdf.shape[0]) + ' rows and ' + str(currdf.shape[1]) + ' columns.')
else:
    print('No new transcripts to parse.')
    dbutils.notebook.exit(1)
    os._exit(1)

# COMMAND ----------

currdf['CALL_ID'] = currdf['CALL_ID'].apply(lambda x: str(x))

# COMMAND ----------

currdf['FILT_MD'] = currdf['FILT_MD'].apply(ast.literal_eval)
currdf['FILT_QA'] = currdf['FILT_QA'].apply(ast.literal_eval)
currdf['SENT_LABELS_FILT_MD'] = currdf['SENT_LABELS_FILT_MD'].apply(ast.literal_eval)
currdf['SENT_LABELS_FILT_QA'] = currdf['SENT_LABELS_FILT_QA'].apply(ast.literal_eval)
currdf['LEN_FILT_MD'] = currdf['FILT_MD'].apply(len)
currdf['LEN_FILT_QA'] = currdf['FILT_QA'].apply(len)

# COMMAND ----------



# COMMAND ----------

currdf.shape

# COMMAND ----------

currdf = currdf.sort_values(by = 'UPLOAD_DT_UTC').drop_duplicates(subset = ['ENTITY_ID', 'EVENT_DATETIME_UTC'], keep = 'first')

# COMMAND ----------

# MAGIC %md
# MAGIC # Define the candidate label sets for comparison

# COMMAND ----------

topic_set = ["This text is about the company's business or market demand, not accounting metrics, will have a strong looking-forward increase or acceleration in the future", "This text is about the company still shows confidence or strong demand from market in macro difficult time", "This text is about positive sentiment"]

# COMMAND ----------

labels = topic_set

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model1

# COMMAND ----------

# MAGIC %md
# MAGIC ### zero-shot

# COMMAND ----------

def create_text_pair(transcript, inference_template):
  template = inference_template + "{label}."
  text1, text2 = [], []
  for t in transcript:
      for l in labels:
          text1.append(t)
          text2.append(template.format(label=l))
  return text1, text2

# COMMAND ----------

topic_set

# COMMAND ----------

def inference_summary1(text1, text2, inference_result):
  result_dict = {tp +'.': [] for tp in topic_set}
  total_dict = {tp +'.': [] for tp in topic_set}
  for i, sentence in enumerate(text1):
    for s in inference_result[i]:
      if s['label'] == 'entailment':
        if s['score'] > 0.91:
          result_dict[text2[i]].append(sentence)
          total_dict[text2[i]].append(1)
        else:
          total_dict[text2[i]].append(0)
  return result_dict, total_dict

# COMMAND ----------

pl_inference1 = pipeline(task="text-classification", model = model_1, tokenizer = tokenizer_1, device = device)

# COMMAND ----------

currdf['SENT_LABELS_FILT_QA'] = currdf.apply(lambda x: [x['SENT_LABELS_FILT_QA'][i] for i, sent in enumerate(x['FILT_QA']) if not sent.endswith('?')], axis=1)
currdf['FILT_QA'] = currdf['FILT_QA'].apply(lambda x: [sent for sent in x if not sent.endswith('?')])
currdf['LEN_FILT_QA'] = currdf['FILT_QA'].apply(lambda x: len(x))

# COMMAND ----------

inference_template = ""
currdf['TEXT1_MD'] = currdf['FILT_MD'].apply(lambda x: create_text_pair(x, inference_template)[0])
currdf['TEXT2_MD'] = currdf['FILT_MD'].apply(lambda x: create_text_pair(x, inference_template)[1])
currdf['MD_RESULT'] = currdf.apply(lambda x: pl_inference1([f"{x['TEXT1_MD'][i]}</s></s>{x['TEXT2_MD'][i]}" for i in range(len(x['TEXT1_MD'])) ], padding=True, top_k=None, batch_size = 16, truncation = True, max_length = 512), axis=1)
currdf['MD_FINAL'] = currdf.apply(lambda x: inference_summary1(x['TEXT1_MD'], x['TEXT2_MD'], x['MD_RESULT'])[0], axis=1)
currdf['MD_FINAL_TOTAL'] = currdf.apply(lambda x: inference_summary1(x['TEXT1_MD'], x['TEXT2_MD'], x['MD_RESULT'])[1], axis=1)


# COMMAND ----------

inference_template = ""
currdf['TEXT1_QA'] = currdf['FILT_QA'].apply(lambda x: create_text_pair(x, inference_template)[0])
currdf['TEXT2_QA'] = currdf['FILT_QA'].apply(lambda x: create_text_pair(x, inference_template)[1])
currdf['QA_RESULT'] = currdf.apply(lambda x: pl_inference1([f"{x['TEXT1_QA'][i]}</s></s>{x['TEXT2_QA'][i]}" for i in range(len(x['TEXT1_QA'])) ], padding=True, top_k=None, batch_size = 16, truncation = True, max_length = 512), axis=1)
currdf['QA_FINAL'] = currdf.apply(lambda x: inference_summary1(x['TEXT1_QA'], x['TEXT2_QA'], x['QA_RESULT'])[0], axis=1)
currdf['QA_FINAL_TOTAL'] = currdf.apply(lambda x: inference_summary1(x['TEXT1_QA'], x['TEXT2_QA'], x['QA_RESULT'])[1], axis=1)


# COMMAND ----------

def sentscore(a, b, weight = True):
  
  # number of relevant sentences
      
  length = len(a)
  if length==0:
      return None
  if length!=len(b):
      return None
  num = len([x for x in a if x>0])
  
  if num==0:
      return None
   
  if weight==True:
    return np.dot(a,b)/num
  else:
    return np.dot([1 if x>0 else 0 for x in a], b)/num

# COMMAND ----------

currdf

# COMMAND ----------

def extract_inf(row, section, section_len):
  count_col = {}
  rel_col = {}
  extract_col = {}
  for tp, sent in row.items():
    count_col[f'{tp}_COUNT_{section}'] = len(sent)
    if section_len != 0:
      rel_col[f'{tp}_REL_{section}'] = len(sent) / section_len
    else:
      rel_col[f'{tp}_REL_{section}'] = None
    extract_col[f'{tp}_EXTRACT_{section}'] = sent
  return pd.Series({**count_col,**rel_col,**extract_col})

# COMMAND ----------

currdf['TP1_SENT_NLI_MD'] = currdf['MD_FINAL_TOTAL'].apply(lambda x: sentscore(x["This text is about the company's business or market demand, not accounting metrics, will have a strong looking-forward increase or acceleration in the future."], x['This text is about positive sentiment.'], weight = False))
currdf['TP2_SENT_NLI_MD'] = currdf['MD_FINAL_TOTAL'].apply(lambda x: sentscore(x["This text is about the company still shows confidence or strong demand from market in macro difficult time."], x['This text is about positive sentiment.'], weight = False))
currdf['TP1_SENT_FINBERT_MD'] = currdf.apply(lambda x: sentscore(x['MD_FINAL_TOTAL']["This text is about the company's business or market demand, not accounting metrics, will have a strong looking-forward increase or acceleration in the future."], x['SENT_LABELS_FILT_MD'], weight = False), axis =1)
currdf['TP2_SENT_FINBERT_MD'] = currdf.apply(lambda x: sentscore(x['MD_FINAL_TOTAL']["This text is about the company still shows confidence or strong demand from market in macro difficult time."], x['SENT_LABELS_FILT_MD'], weight = False), axis = 1)


# COMMAND ----------

currdf['TP1_SENT_NLI_QA'] = currdf['QA_FINAL_TOTAL'].apply(lambda x: sentscore(x["This text is about the company's business or market demand, not accounting metrics, will have a strong looking-forward increase or acceleration in the future."], x['This text is about positive sentiment.'], weight = False))
currdf['TP2_SENT_NLI_QA'] = currdf['QA_FINAL_TOTAL'].apply(lambda x: sentscore(x["This text is about the company still shows confidence or strong demand from market in macro difficult time."], x['This text is about positive sentiment.'], weight = False))
currdf['TP1_SENT_FINBERT_QA'] = currdf.apply(lambda x: sentscore(x['QA_FINAL_TOTAL']["This text is about the company's business or market demand, not accounting metrics, will have a strong looking-forward increase or acceleration in the future."], x['SENT_LABELS_FILT_QA'], weight = False), axis =1)
currdf['TP2_SENT_FINBERT_QA'] = currdf.apply(lambda x: sentscore(x['QA_FINAL_TOTAL']["This text is about the company still shows confidence or strong demand from market in macro difficult time."], x['SENT_LABELS_FILT_QA'], weight = False), axis = 1)


# COMMAND ----------


currdf_final = pd.concat([currdf[['ENTITY_ID','DATE','CALL_NAME','COMPANY_NAME', 'LEN_FILT_MD', 'LEN_FILT_QA', 'TP1_SENT_NLI_MD', 'TP2_SENT_NLI_MD','TP1_SENT_FINBERT_MD','TP2_SENT_FINBERT_MD','TP1_SENT_NLI_QA', 'TP2_SENT_NLI_QA','TP1_SENT_FINBERT_QA','TP2_SENT_FINBERT_QA']], currdf.apply(lambda x: extract_inf(x['MD_FINAL'], 'FILT_MD', x['LEN_FILT_MD']), axis =1), currdf.apply(lambda x: extract_inf(x['QA_FINAL'], 'FILT_QA', x['LEN_FILT_QA']), axis =1)], axis=1)


# COMMAND ----------

currdf_final

# COMMAND ----------

"/dbfs/mnt/access_work/UC25/Topic Modeling/NLI Models/SF_NLI_" + lastDateNewQuery[5:7] + "_" + lastDateNewQuery[2:4] + "_" + currentDateNewQuery[5:7] + "_" + currentDateNewQuery[2:4] +  "_v2_git.csv"

# COMMAND ----------

currdf_final.to_csv("/dbfs/mnt/access_work/UC25/Topic Modeling/NLI Models/SF_NLI_" + lastDateNewQuery[5:7] + "_" + lastDateNewQuery[2:4] + "_" + currentDateNewQuery[5:7] + "_" + currentDateNewQuery[2:4] +  "_v2_git.csv")

# COMMAND ----------

df_all = currdf_final

# COMMAND ----------

df_all["TP1_EXTRACT_FILT_MD"] = df_all["This text is about the company's business or market demand, not accounting metrics, will have a strong looking-forward increase or acceleration in the future._EXTRACT_FILT_MD"]
df_all["TP1_EXTRACT_FILT_QA"] = df_all["This text is about the company's business or market demand, not accounting metrics, will have a strong looking-forward increase or acceleration in the future._EXTRACT_FILT_QA"]
df_all["TP2_EXTRACT_FILT_MD"] = df_all["This text is about the company still shows confidence or strong demand from market in macro difficult time._EXTRACT_FILT_MD"]
df_all["TP2_EXTRACT_FILT_QA"] = df_all["This text is about the company still shows confidence or strong demand from market in macro difficult time._EXTRACT_FILT_QA"]

# COMMAND ----------

df_all = df_all.drop(columns = ["This text is about positive sentiment._COUNT_FILT_MD", "This text is about positive sentiment._REL_FILT_MD", "This text is about positive sentiment._COUNT_FILT_QA", "This text is about positive sentiment._REL_FILT_QA", "This text is about positive sentiment._EXTRACT_FILT_MD", "This text is about positive sentiment._EXTRACT_FILT_QA", "This text is about the company's business or market demand, not accounting metrics, will have a strong looking-forward increase or acceleration in the future._EXTRACT_FILT_MD","This text is about the company's business or market demand, not accounting metrics, will have a strong looking-forward increase or acceleration in the future._EXTRACT_FILT_QA","This text is about the company still shows confidence or strong demand from market in macro difficult time._EXTRACT_FILT_MD", "This text is about the company still shows confidence or strong demand from market in macro difficult time._EXTRACT_FILT_QA" ])


# COMMAND ----------

df_all

# COMMAND ----------

df_all.columns

# COMMAND ----------

df_all.columns = ['ENTITY_ID', 'DATE', 'CALL_NAME', 'COMPANY_NAME', 'LEN_FILT_MD',
       'LEN_FILT_QA', 'TP1_SENT_NLI_MD', 'TP2_SENT_NLI_MD',
       'TP1_SENT_FINBERT_MD', 'TP2_SENT_FINBERT_MD', 'TP1_SENT_NLI_QA',
       'TP2_SENT_NLI_QA', 'TP1_SENT_FINBERT_QA', 'TP2_SENT_FINBERT_QA',
       "TP1_COUNT_FILT_MD",
       'TP2_COUNT_FILT_MD',
       "TP1_REL_FILT_MD",
       'TP2_REL_FILT_MD',
       "TP1_COUNT_FILT_QA",
       'TP2_COUNT_FILT_QA',
       "TP1_REL_FILT_QA",
       'TP2_REL_FILT_QA',
       'TP1_EXTRACT_FILT_MD', 'TP1_EXTRACT_FILT_QA', 'TP2_EXTRACT_FILT_MD',
       'TP2_EXTRACT_FILT_QA']

# COMMAND ----------

df_all['TP1_EXTRACT_FILT_MD'] = df_all['TP1_EXTRACT_FILT_MD'].apply(lambda x: x if x!=[] else None)

# COMMAND ----------

df_all['TP1_EXTRACT_FILT_QA'] = df_all['TP1_EXTRACT_FILT_QA'].apply(lambda x: x if x!=[] else None)
df_all['TP2_EXTRACT_FILT_QA'] = df_all['TP2_EXTRACT_FILT_QA'].apply(lambda x: x if x!=[] else None)
df_all['TP2_EXTRACT_FILT_MD'] = df_all['TP2_EXTRACT_FILT_MD'].apply(lambda x: x if x!=[] else None)

# COMMAND ----------

df_all

# COMMAND ----------

df_all['DATE'] = pd.to_datetime(df_all['DATE'])

# COMMAND ----------

from pyspark.sql.types import *
def equivalent_type(string, f):
    print(string, f)
    
    if f == 'datetime64[ns]': return TimestampType()
    elif f == 'int64': return LongType()
    elif f == 'int32': return IntegerType()
    elif f == 'float64': return FloatType()
    elif "EXTRACT" in string: return ArrayType(StringType())
    else: return StringType()

def define_structure(string, format_type):
    #try: 
    typo = equivalent_type(string, format_type)
    print(typo)
    #except: typo = StringType()
    return StructField(string, typo)

# Given pandas dataframe, it will return a spark's dataframe.
def pandas_to_spark(pandas_df):
    columns = list(pandas_df.columns)
    types = list(pandas_df.dtypes)
    struct_list = []
    for column, typo in zip(columns, types): 
      struct_list.append(define_structure(column, typo))
    p_schema = StructType(struct_list)
    return sqlContext.createDataFrame(pandas_df, p_schema)

# COMMAND ----------

spark_parsedDF = pandas_to_spark(df_all)
spark_parsedDF = spark_parsedDF.replace(np.nan, None)
spark_parsedDF = spark_parsedDF.withColumn("DATE", F.to_timestamp(spark_parsedDF.DATE, 'yyyy-MM-dd'))

new_sf.db = 'EDS_PROD'
new_sf.schema = 'QUANT'
 
tablename_curr = 'YUJING_SF_SUPER_STOCK_DEV1'
result_curr = new_sf.write_to_snowflake_table(spark_parsedDF, tablename_curr)

# COMMAND ----------


