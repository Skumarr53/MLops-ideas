myDBFS = DBFShelper()
new_sf = pd.read_pickle(r'/dbfs' + myDBFS.iniPath + 'mysf_prod_quant.pkl')

import ast
from IPython.display import clear_output
import numpy as np
from dateutil.relativedelta import relativedelta
import pickle 
import pandas as pd


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


data_end_date = datetime.now() 
data_last_month = datetime.now() - relativedelta(months=1)


lastDateNewQuery = (pd.to_datetime(format(data_last_month, '%m') + "-01-" + format(data_last_month, '%Y'))).strftime('%Y-%m-%d')
currentDateNewQuery = (pd.to_datetime(format(data_end_date, '%m') + "-01-" + format(data_end_date, '%Y'))).strftime('%Y-%m-%d')

mind = "'" + lastDateNewQuery + "'"
maxd = "'" + currentDateNewQuery + "'"
print('The next query spans ' + mind + ' to ' + maxd)

tsQuery= ("select CALL_ID,ENTITY_ID, DATE, FILT_MD, FILT_QA, CALL_NAME,COMPANY_NAME,EARNINGS_CALL,ERROR,TRANSCRIPT_STATUS,UPLOAD_DT_UTC,VERSION_ID,EVENT_DATETIME_UTC,PARSED_DATETIME_EASTERN_TZ,SENT_LABELS_FILT_MD, SENT_LABELS_FILT_QA from EDS_PROD.QUANT_LIVE.CTS_FUND_COMBINED_SCORES_H  WHERE DATE >= " + mind + " AND DATE < " + maxd + " ORDER BY PARSED_DATETIME_EASTERN_TZ DESC;")

resultspkdf = new_sf.read_from_snowflake(tsQuery)

currdf = resultspkdf.toPandas()

if len(currdf)>0:
    print('The data spans from ' + str(currdf['PARSED_DATETIME_EASTERN_TZ'].min()) + ' to ' + str(currdf['PARSED_DATETIME_EASTERN_TZ'].max()) + 'and has ' + str(currdf.shape[0]) + ' rows and ' + str(currdf.shape[1]) + ' columns.')
else:
    print('No new transcripts to parse.')
    dbutils.notebook.exit(1)
    os._exit(1)


currdf['CALL_ID'] = currdf['CALL_ID'].apply(lambda x: str(x))

currdf['FILT_MD'] = currdf['FILT_MD'].apply(ast.literal_eval)
currdf['FILT_QA'] = currdf['FILT_QA'].apply(ast.literal_eval)
currdf['SENT_LABELS_FILT_MD'] = currdf['SENT_LABELS_FILT_MD'].apply(ast.literal_eval)
currdf['SENT_LABELS_FILT_QA'] = currdf['SENT_LABELS_FILT_QA'].apply(ast.literal_eval)
currdf['LEN_FILT_MD'] = currdf['FILT_MD'].apply(len)
currdf['LEN_FILT_QA'] = currdf['FILT_QA'].apply(len)

currdf = currdf.sort_values(by = 'UPLOAD_DT_UTC').drop_duplicates(subset = ['ENTITY_ID', 'EVENT_DATETIME_UTC'], keep = 'first')



topic_set = ["This text is about the company's business or market demand, not accounting metrics, will have a strong looking-forward increase or acceleration in the future", "This text is about the company still shows confidence or strong demand from market in macro difficult time", "This text is about positive sentiment"]


labels = topic_set

def create_text_pair(transcript, inference_template):
  template = inference_template + "{label}."
  text1, text2 = [], []
  for t in transcript:
      for l in labels:
          text1.append(t)
          text2.append(template.format(label=l))
  return text1, text2

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

pl_inference1 = pipeline(task="text-classification", model = model_1, tokenizer = tokenizer_1, device = device)


currdf['SENT_LABELS_FILT_QA'] = currdf.apply(lambda x: [x['SENT_LABELS_FILT_QA'][i] for i, sent in enumerate(x['FILT_QA']) if not sent.endswith('?')], axis=1)
currdf['FILT_QA'] = currdf['FILT_QA'].apply(lambda x: [sent for sent in x if not sent.endswith('?')])
currdf['LEN_FILT_QA'] = currdf['FILT_QA'].apply(lambda x: len(x))


inference_template = ""
currdf['TEXT1_MD'] = currdf['FILT_MD'].apply(lambda x: create_text_pair(x, inference_template)[0])
currdf['TEXT2_MD'] = currdf['FILT_MD'].apply(lambda x: create_text_pair(x, inference_template)[1])
currdf['MD_RESULT'] = currdf.apply(lambda x: pl_inference1([f"{x['TEXT1_MD'][i]}</s></s>{x['TEXT2_MD'][i]}" for i in range(len(x['TEXT1_MD'])) ], padding=True, top_k=None, batch_size = 16, truncation = True, max_length = 512), axis=1)
currdf['MD_FINAL'] = currdf.apply(lambda x: inference_summary1(x['TEXT1_MD'], x['TEXT2_MD'], x['MD_RESULT'])[0], axis=1)
currdf['MD_FINAL_TOTAL'] = currdf.apply(lambda x: inference_summary1(x['TEXT1_MD'], x['TEXT2_MD'], x['MD_RESULT'])[1], axis=1)

inference_template = ""
currdf['TEXT1_QA'] = currdf['FILT_QA'].apply(lambda x: create_text_pair(x, inference_template)[0])
currdf['TEXT2_QA'] = currdf['FILT_QA'].apply(lambda x: create_text_pair(x, inference_template)[1])
currdf['QA_RESULT'] = currdf.apply(lambda x: pl_inference1([f"{x['TEXT1_QA'][i]}</s></s>{x['TEXT2_QA'][i]}" for i in range(len(x['TEXT1_QA'])) ], padding=True, top_k=None, batch_size = 16, truncation = True, max_length = 512), axis=1)
currdf['QA_FINAL'] = currdf.apply(lambda x: inference_summary1(x['TEXT1_QA'], x['TEXT2_QA'], x['QA_RESULT'])[0], axis=1)
currdf['QA_FINAL_TOTAL'] = currdf.apply(lambda x: inference_summary1(x['TEXT1_QA'], x['TEXT2_QA'], x['QA_RESULT'])[1], axis=1)

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