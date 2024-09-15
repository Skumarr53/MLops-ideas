myDBFS = DBFShelper()
new_sf = pd.read_pickle(r'/dbfs' + myDBFS.iniPath + 'mysf_prod_quant.pkl')

import ast
from IPython.display import clear_output
import numpy as np
from dateutil.relativedelta import relativedelta
import pickle 
import pandas as pd


spark.conf.set("spark.sql.legacy.setCommandRejectsSparkCoreConfs", False)

spark.conf.set('spark.rpc.message.maxSize','1024') # To set the spark configuration  
spark.conf.get('spark.rpc.message.maxSize') # To get the spark configuration   

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

class_data_path = "/dbfs/mnt/access_work/UC25/Topic Modeling/NLI Models/"


# Read start & end ranges. Note that the range does is NOT inclusive of end month; i.e the range ends at the beginning of the end month
minDateNewQuery = (pd.to_datetime(dbutils.widgets.get("Start Month") + "-01-" + dbutils.widgets.get("Start Year"))).strftime('%Y-%m-%d')
maxDateNewQuery = (pd.to_datetime(dbutils.widgets.get("End Month") + "-01-" + dbutils.widgets.get("End Year"))).strftime('%Y-%m-%d')

mind = "'" + minDateNewQuery + "'"
maxd = "'" + maxDateNewQuery + "'"

print('The next query spans ' + mind + ' to ' + maxd)


tsQuery= ("select CALL_ID,ENTITY_ID, DATE, FILT_MD, FILT_QA, CALL_NAME,COMPANY_NAME,EARNINGS_CALL,ERROR,TRANSCRIPT_STATUS,UPLOAD_DT_UTC,VERSION_ID,EVENT_DATETIME_UTC,PARSED_DATETIME_EASTERN_TZ,SENT_LABELS_FILT_MD, SENT_LABELS_FILT_QA from EDS_PROD.QUANT.PARTHA_FUND_CTS_STG_1_VIEW  WHERE DATE >= " + mind + " AND DATE < " + maxd + " ORDER BY PARSED_DATETIME_EASTERN_TZ DESC;")

resultspkdf = new_sf.read_from_snowflake(tsQuery)

currdf = resultspkdf.toPandas()

if len(currdf)>0:
    print('The data spans from ' + str(currdf['PARSED_DATETIME_EASTERN_TZ'].min()) + ' to ' + str(currdf['PARSED_DATETIME_EASTERN_TZ'].max()) + 'and has ' + str(currdf.shape[0]) + ' rows and ' + str(currdf.shape[1]) + ' columns.')
else:
    print('No new transcripts to parse.')
    dbutils.notebook.exit(1)
    os._exit(1)


    currdf['FILT_MD'] = currdf['FILT_MD'].apply(ast.literal_eval)
currdf['FILT_QA'] = currdf['FILT_QA'].apply(ast.literal_eval)
currdf['SENT_LABELS_FILT_MD'] = currdf['SENT_LABELS_FILT_MD'].apply(ast.literal_eval)
currdf['SENT_LABELS_FILT_QA'] = currdf['SENT_LABELS_FILT_QA'].apply(ast.literal_eval)
currdf['LEN_FILT_MD'] = currdf['FILT_MD'].apply(len)
currdf['LEN_FILT_QA'] = currdf['FILT_QA'].apply(len)


currdf = currdf.sort_values(by = 'UPLOAD_DT_UTC').drop_duplicates(subset = ['ENTITY_ID', 'EVENT_DATETIME_UTC'], keep = 'first')


topic_set = ["This text is about consumer strength", "This text is about consumer weakness", "This text is about reduced consumer's spending patterns"]

def create_text_pair(transcript, inference_template, labels):
  template = inference_template + "{label}."
  text1, text2 = [], []
  for t in transcript:
      for l in labels:
          text1.append(t)
          text2.append(template.format(label=l))
  return text1, text2


def create_text_pair(transcript, inference_template, labels):
  template = inference_template + "{label}."
  text1, text2 = [], []
  for t in transcript:
      for l in labels:
          text1.append(t)
          text2.append(template.format(label=l))
  return text1, text2



pl_inference1 = pipeline(task="text-classification", model = model_1, tokenizer = tokenizer_1, device = device)


currdf['SENT_LABELS_FILT_QA'] = currdf.apply(lambda x: [x['SENT_LABELS_FILT_QA'][i] for i, sent in enumerate(x['FILT_QA']) if not sent.endswith('?')], axis=1)
currdf['FILT_QA'] = currdf['FILT_QA'].apply(lambda x: [sent for sent in x if not sent.endswith('?')])
currdf['LEN_FILT_QA'] = currdf['FILT_QA'].apply(lambda x: len(x))


inference_template = ""
currdf['TEXT1_MD'] = currdf['FILT_MD'].apply(lambda x: create_text_pair(x, inference_template, labels)[0])
currdf['TEXT2_MD'] = currdf['FILT_MD'].apply(lambda x: create_text_pair(x, inference_template, labels)[1])
currdf['MD_RESULT'] = currdf.apply(lambda x: pl_inference1([f"{x['TEXT1_MD'][i]}</s></s>{x['TEXT2_MD'][i]}" for i in range(len(x['TEXT1_MD'])) ], padding=True, top_k=None, batch_size = 16, truncation = True, max_length = 512), axis=1)
currdf['MD_FINAL_TOTAL'] = currdf.apply(lambda x: inference_summary1(x['TEXT1_MD'], x['TEXT2_MD'], x['MD_RESULT'], 0.8)[0], axis=1)
currdf['MD_FINAL_SCORE'] = currdf.apply(lambda x: inference_summary1(x['TEXT1_MD'], x['TEXT2_MD'], x['MD_RESULT'], 0.8)[1], axis=1)



inference_template = ""
currdf['TEXT1_QA'] = currdf['FILT_QA'].apply(lambda x: create_text_pair(x, inference_template, labels)[0])
currdf['TEXT2_QA'] = currdf['FILT_QA'].apply(lambda x: create_text_pair(x, inference_template, labels)[1])
currdf['QA_RESULT'] = currdf.apply(lambda x: pl_inference1([f"{x['TEXT1_QA'][i]}</s></s>{x['TEXT2_QA'][i]}" for i in range(len(x['TEXT1_QA'])) ], padding=True, top_k=None, batch_size = 16, truncation = True, max_length = 512), axis=1)
currdf['QA_FINAL_TOTAL'] = currdf.apply(lambda x: inference_summary1(x['TEXT1_QA'], x['TEXT2_QA'], x['QA_RESULT'], 0.8)[0], axis=1)
currdf['QA_FINAL_SCORE'] = currdf.apply(lambda x: inference_summary1(x['TEXT1_QA'], x['TEXT2_QA'], x['QA_RESULT'], 0.8)[1], axis=1)

def extract_inf(row, section, section_len, threshold):
  count_col = {}
  rel_col = {}
  score_col = {}
  total_col = {}
  for tp, score in row.items():
    if section_len != 0:
      score_binary = [1 if s > threshold else 0 for s in score]
      total_col[f'{tp}_TOTAL_{section}'] = score_binary
      count_col[f'{tp}_COUNT_{section}'] = float(sum(score_binary))
      rel_col[f'{tp}_REL_{section}'] = sum(score_binary) / section_len
      score_col[f'{tp}_SCORE_{section}'] = [round(s, 4) for s in score]
    else:
      count_col[f'{tp}_COUNT_{section}'] = None
      rel_col[f'{tp}_REL_{section}'] = None
      total_col[f'{tp}_TOTAL_{section}'] = [] # if change this to None, the table cannot be stored in sf.
      score_col[f'{tp}_SCORE_{section}'] = []

  return pd.Series({**count_col,**rel_col,**score_col, **total_col})

currdf_all = pd.concat([currdf[['ENTITY_ID','CALL_ID','VERSION_ID','DATE','CALL_NAME','COMPANY_NAME', 'LEN_FILT_MD', 'LEN_FILT_QA', 'FILT_MD', 'FILT_QA']],currdf.apply(lambda x: extract_inf(x['MD_FINAL_SCORE'], 'FILT_MD', x['LEN_FILT_MD'], 0.8), axis =1), currdf.apply(lambda x: extract_inf(x['QA_FINAL_SCORE'], 'FILT_QA', x['LEN_FILT_QA'], 0.8), axis =1)], axis=1)

currdf_all['DATE'] = pd.to_datetime(currdf_all['DATE'])


output_path = class_data_path + 'NLI_Demand_' + dbutils.widgets.get("Start Month") + '_' + dbutils.widgets.get("Start Year")[2:] + '_' + dbutils.widgets.get("End Month") + '_' + dbutils.widgets.get("End Year")[2:]


currdf_all.to_csv(output_path, index = False)


new_columns = []
for col in currdf_all.columns:
  if "This text is about reduced consumer's spending patterns." in col:
    col = col.replace("This text is about reduced consumer's spending patterns.", "CONSUMER_SPENDING_PATTERNS")
  elif "This text is about consumer weakness." in col:
    col = col.replace("This text is about consumer weakness.", "CONSUMER_WEAKNESS")
  elif "This text is about consumer strength." in col:
    col = col.replace("This text is about consumer strength.", "CONSUMER_STRENGTH")
  new_columns.append(col)


tsQuery= ("select top 2 * from EDS_PROD.QUANT.YUJING_MASS_NLI_DEMAND_DEV_3 ;")

resultspkdf = new_sf.read_from_snowflake(tsQuery)

currdf_sample = resultspkdf.toPandas()


from pyspark.sql.types import *
def equivalent_type(string, f):
    print(string, f)
    
    if f == 'datetime64[ns]': return TimestampType()
    elif f == 'int64': return LongType()
    elif f == 'int32': return IntegerType()
    elif f == 'float64': return FloatType()
    elif 'FILT_MD' == string: return ArrayType(StringType())
    elif 'FILT_QA' == string: return ArrayType(StringType())
    elif '_total_' in string.lower(): return ArrayType(IntegerType())
    elif '_score_' in string.lower(): return ArrayType(FloatType())

 #   elif f == 'object': return ArrayType()
 #   elif f == 'list': return ArrayType()
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

spark.conf.get("spark.rpc.message.maxSize")

spark_parsedDF = pandas_to_spark(currdf_all[currdf_sample.columns])
spark_parsedDF = spark_parsedDF.replace(np.nan, None)
spark_parsedDF = spark_parsedDF.withColumn("DATE", F.to_timestamp(spark_parsedDF.DATE, 'yyyy-MM-dd'))
                                                                         
new_sf.db = 'EDS_PROD'
new_sf.schema = 'QUANT'
 
tablename_curr = 'YUJING_MASS_NLI_DEMAND_DEV_3'
result_curr = new_sf.write_to_snowflake_table(spark_parsedDF, tablename_curr)

currdf_all.apply(lambda x: [x['FILT_MD'][i] for i, v in enumerate(x['This text is about consumer weakness._TOTAL_FILT_MD'])   if v == 1 ], axis = 1).values

currdf.apply(lambda x: extract_inf(x['MD_FINAL_SCORE'], 'FILT_MD', x['LEN_FILT_MD'], 0.8), axis =1)

currdf['TP1_SENT_NLI_MD'] = currdf['MD_FINAL_TOTAL'].apply(lambda x: sentscore(x["This text is about the company's business or market demand, not accounting metrics, will have a strong looking-forward increase or acceleration in the future."], x['This text is about positive sentiment.'], weight = False))
currdf['TP2_SENT_NLI_MD'] = currdf['MD_FINAL_TOTAL'].apply(lambda x: sentscore(x["This text is about the company still shows confidence or strong demand from market in macro difficult time."], x['This text is about positive sentiment.'], weight = False))
currdf['TP1_SENT_FINBERT_MD'] = currdf.apply(lambda x: sentscore(x['MD_FINAL_TOTAL']["This text is about the company's business or market demand, not accounting metrics, will have a strong looking-forward increase or acceleration in the future."], x['SENT_LABELS_FILT_MD'], weight = False), axis =1)
currdf['TP2_SENT_FINBERT_MD'] = currdf.apply(lambda x: sentscore(x['MD_FINAL_TOTAL']["This text is about the company still shows confidence or strong demand from market in macro difficult time."], x['SENT_LABELS_FILT_MD'], weight = False), axis = 1)



currdf['TP1_SENT_NLI_QA'] = currdf['QA_FINAL_TOTAL'].apply(lambda x: sentscore(x["This text is about the company's business or market demand, not accounting metrics, will have a strong looking-forward increase or acceleration in the future."], x['This text is about positive sentiment.'], weight = False))
currdf['TP2_SENT_NLI_QA'] = currdf['QA_FINAL_TOTAL'].apply(lambda x: sentscore(x["This text is about the company still shows confidence or strong demand from market in macro difficult time."], x['This text is about positive sentiment.'], weight = False))
currdf['TP1_SENT_FINBERT_QA'] = currdf.apply(lambda x: sentscore(x['QA_FINAL_TOTAL']["This text is about the company's business or market demand, not accounting metrics, will have a strong looking-forward increase or acceleration in the future."], x['SENT_LABELS_FILT_QA'], weight = False), axis =1)
currdf['TP2_SENT_FINBERT_QA'] = currdf.apply(lambda x: sentscore(x['QA_FINAL_TOTAL']["This text is about the company still shows confidence or strong demand from market in macro difficult time."], x['SENT_LABELS_FILT_QA'], weight = False), axis = 1)



currdf_final = pd.concat([currdf[['ENTITY_ID','DATE','CALL_NAME','COMPANY_NAME', 'LEN_FILT_MD', 'LEN_FILT_QA', 'TP1_SENT_NLI_MD', 'TP2_SENT_NLI_MD','TP1_SENT_FINBERT_MD','TP2_SENT_FINBERT_MD','TP1_SENT_NLI_QA', 'TP2_SENT_NLI_QA','TP1_SENT_FINBERT_QA','TP2_SENT_FINBERT_QA']], currdf.apply(lambda x: extract_inf(x['MD_FINAL'], 'FILT_MD', x['LEN_FILT_MD']), axis =1), currdf.apply(lambda x: extract_inf(x['QA_FINAL'], 'FILT_QA', x['LEN_FILT_QA']), axis =1)], axis=1)



currdf_final.to_csv("/dbfs/mnt/access_work/UC25/Topic Modeling/NLI Models/SF_NLI_" + dbutils.widgets.get("Start Month") + "_" + dbutils.widgets.get("Start Year") + "_" + dbutils.widgets.get("End Month") + "_" + dbutils.widgets.get("End Year") +  "_v2.csv")

df_all["TP1_EXTRACT_FILT_MD"] = df_all["This text is about the company's business or market demand, not accounting metrics, will have a strong looking-forward increase or acceleration in the future._EXTRACT_FILT_MD"]
df_all["TP1_EXTRACT_FILT_QA"] = df_all["This text is about the company's business or market demand, not accounting metrics, will have a strong looking-forward increase or acceleration in the future._EXTRACT_FILT_QA"]
df_all["TP2_EXTRACT_FILT_MD"] = df_all["This text is about the company still shows confidence or strong demand from market in macro difficult time._EXTRACT_FILT_MD"]
df_all["TP2_EXTRACT_FILT_QA"] = df_all["This text is about the company still shows confidence or strong demand from market in macro difficult time._EXTRACT_FILT_QA"]

df_all = df_all.drop(columns = ["This text is about positive sentiment._COUNT_FILT_MD", "This text is about positive sentiment._REL_FILT_MD", "This text is about positive sentiment._COUNT_FILT_QA", "This text is about positive sentiment._REL_FILT_QA", "This text is about positive sentiment._EXTRACT_FILT_MD", "This text is about positive sentiment._EXTRACT_FILT_QA", "This text is about the company's business or market demand, not accounting metrics, will have a strong looking-forward increase or acceleration in the future._EXTRACT_FILT_MD","This text is about the company's business or market demand, not accounting metrics, will have a strong looking-forward increase or acceleration in the future._EXTRACT_FILT_QA","This text is about the company still shows confidence or strong demand from market in macro difficult time._EXTRACT_FILT_MD", "This text is about the company still shows confidence or strong demand from market in macro difficult time._EXTRACT_FILT_QA" ])


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

df_all['TP1_EXTRACT_FILT_MD'] = df_all['TP1_EXTRACT_FILT_MD'].apply(lambda x: x if x!=[] else None)

df_all['TP1_EXTRACT_FILT_QA'] = df_all['TP1_EXTRACT_FILT_QA'].apply(lambda x: x if x!=[] else None)
df_all['TP2_EXTRACT_FILT_QA'] = df_all['TP2_EXTRACT_FILT_QA'].apply(lambda x: x if x!=[] else None)
df_all['TP2_EXTRACT_FILT_MD'] = df_all['TP2_EXTRACT_FILT_MD'].apply(lambda x: x if x!=[] else None)

spark_parsedDF = pandas_to_spark(df_all)
spark_parsedDF = spark_parsedDF.replace(np.nan, None)
spark_parsedDF = spark_parsedDF.withColumn("DATE", F.to_timestamp(spark_parsedDF.DATE, 'yyyy-MM-dd'))

new_sf.db = 'EDS_PROD'
new_sf.schema = 'QUANT'
 
tablename_curr = 'YUJING_SF_SUPER_STOCK_DEV1'
result_curr = new_sf.write_to_snowflake_table(spark_parsedDF, tablename_curr)