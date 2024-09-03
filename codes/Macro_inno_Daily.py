#import seaborn as sns
#from gensim.models import Word2Vec
import spacy
from spacy.lang.en import English
#from transformers import TextClassificationPipeline
import pandas as pd
import numpy as np
import spacy
import tqdm
from tqdm import tqdm
tqdm.pandas()
#import sklearn.datasets
#import plotly.express as px
#from gensim.models import Phrases
from collections import Counter
import dask.dataframe as dd
from dask.distributed import Client
from dask.diagnostics import ProgressBar
import gc

# Dask client - intended for NC6 v3 GPU instance
client = Client(n_workers=8, threads_per_worker=1)


# Object to enable Snowflake access to prod Quant
myDBFS = DBFShelper()
new_sf = pd.read_pickle(r'/dbfs' + myDBFS.iniPath + 'mysf_prod_quant.pkl')


# Query all parsed transcripts parsed after the last known parsed date.
tsQuery= ("select TOP 1200 CALL_ID,ENTITY_ID, FILT_MD, FILT_QA, SENT_LABELS_FILT_MD, SENT_LABELS_FILT_QA, CALL_NAME,COMPANY_NAME,EARNINGS_CALL,ERROR,TRANSCRIPT_STATUS,UPLOAD_DT_UTC,VERSION_ID,EVENT_DATETIME_UTC,PARSED_DATETIME_EASTERN_TZ from EDS_PROD.QUANT.PARTHA_FUND_CTS_STG_1_VIEW t2 where not exists (select 1 from EDS_PROD.QUANT.PARTHA_MACRO_INNO_CTS_STG_1 t1 where  t1.CALL_ID = t2.CALL_ID and t1.ENTITY_ID = t2.ENTITY_ID and t1.VERSION_ID = t2.VERSION_ID) ORDER BY PARSED_DATETIME_EASTERN_TZ DESC;")

resultspkdf = new_sf.read_from_snowflake(tsQuery)

currdf = resultspkdf.toPandas()

if len(currdf)>0:
   # currdf.to_parquet('/dbfs/mnt/access_work/UC25/Backtest/Fundamentals_David_PoC/fundP2_latestProcessed.parquet', compression = 'gzip')
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



nlp = spacy.load("en_core_web_sm", disable = ['parser'])

# Excluding financially relavant stopwords
nlp.Defaults.stop_words -= {"bottom", "top", "Bottom", "Top", "call", "down"}
nlp.max_length = 1000000000


# Lemmatizer - for document text
def wordTokenize(doc):
  
  return [ent.lemma_.lower() for ent in nlp(doc) if not ent.is_stop and not ent.is_punct and ent.pos_ != 'NUM']


# Tokenizer/lemmatizer for match words
def matchTokenize(doc):
  ret = []
  for ent in nlp(doc):
    if ent.pos_ == 'PROPN' or ent.text[0].isupper():
      ret.append(ent.text.lower())
    #  print(ent.text.lower())
    #  print(ent.text)
      continue
    if not ent.is_stop and not ent.is_punct and ent.pos_ != 'NUM':
      ret.append(ent.lemma_.lower())
  return ret


# Creates n grams helper fxn
def find_ngrams(input_list, n):
  return zip(*[input_list[i:] for i in range(n)])


# Create a set of match patterns from match list. This ensures variations such as lemmas & case are handled.
def get_match_set(matches):
  
  bigrams = set([word.lower() for word in matches if len(word.split('_'))==2] + [word.lower().replace(" ", '_') for word in matches if len(word.split(' '))==2] + ['_'.join(matchTokenize(word)) for word in matches if len(word.split(' '))==2])
 
  unigrams = set([matchTokenize(match)[0] for match in matches if ('_' not in match) and (len(match.split(' '))==1)] + [match.lower() for match in matches if ('_' not in match) and (len(match.split(' '))==1)])

#  phrases = set([phrase.lower() for phrase in matches if len(phrase.split(" "))>2] + [' '.join(matchTokenize(phrase)) for phrase in matches if len(phrase.split(" "))>2])

#  Phrase matching correction
  phrases = [phrase.lower() for phrase in matches if len(phrase.split(" "))>2]
  
  #print(matches)
  #print(unigrams, bigrams, phrases)
  return {'original': matches, 'unigrams': unigrams, 'bigrams': bigrams, 'phrases': phrases}
  


# Counting fxn. Phrase matching increases compute time - consider use case before enabling it. Set to true as fxn default.
def match_count_lowStat(texts, match_sets, phrases = True, suppress = None):

  count_dict = {label : {matchw: 0 for matchw in match_set['unigrams'].union(match_set['bigrams']) } for label, match_set in match_sets.items()}
  total_counts = {label: [] for label in match_sets.keys()}

  for text in texts:
    
    counted = {label: 0 for label in match_sets.keys()}
    unigrams = wordTokenize(text)
    bigrams = ['_'.join(g) for g in find_ngrams(unigrams, 2)]
    
    text = text.lower()
    for label, match_set in match_sets.items(): 
      
      if any(item in text for item in suppress[label]):
        counted[label] += 0
        continue
        
      for word in unigrams:
        if word in match_set['unigrams']:
          count_dict[label][word]+=1
          counted[label] += 1

      for word in bigrams:
        if word in match_set['bigrams']:
          count_dict[label][word]+=1
          counted[label] += 1
      
      if phrases:
        if any(phrase in text for phrase in match_set['phrases']):
          counted[label] += 1
          continue

    for label in match_sets.keys():
      
      total_counts[label].append(counted[label])

    
  return {label : {'total': total_counts[label], 'stats' : count_dict[label]} for label in match_sets.keys()}
    


# Used to merge dictionaries that keep track of word counts 
def mergeCount(x):
  
  try:
    merge = Counter(x[0])
 
    for calc in x[1:]:

      merge = merge + Counter(calc)
    if len(merge.keys())==0:
      return {'NO_MATCH': 1}
    return merge
  except:
    return {'ERROR': 1}
  
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

def netscore(a, b):
  
  # number of relevant sentences
      
  length = len(a)
  if length==0:
      return None
  if length!=len(b):
      return None
  num = len([x for x in a if x>0])
  
  if num==0:
      return None
   
  return np.dot([1 if x>0 else 0 for x in a], b)


# Read match list and create match set
match_df = pd.read_csv(dbutils.widgets.get("Match list path"))

word_set_dict = {topic.replace(' ', '_').upper() : get_match_set(match_df[(match_df['label']==topic) & (match_df['negate']==False)]['match'].values) for topic in match_df['label'].unique()}

negate_dict = {topic.replace(' ', '_').upper() : [word.lower() for word in match_df[(match_df['label']==topic) & (match_df['negate']==True)]['match'].values.tolist()] for topic in match_df['label'].unique()}

currdf = dd.from_pandas(currdf, npartitions = 8)
for label, section in {'FILT_MD': 'FILT_MD', 'FILT_QA': 'FILT_QA'}.items():

  currdf['matches_' + label] = currdf[section].apply(lambda x: match_count_lowStat(x, word_set_dict, phrases = False, suppress = negate_dict), meta = ('matches_' + label, object))
  #currdf[label] = currdf['matches_' + label].apply(lambda x: [str(calc['filt']) for calc in x], meta = ('FILT_' + label, object))

with ProgressBar():
  currdf = currdf.compute()


# Dask is not used for below code due to unfixable bugs
# Below code only used to aggregate stats and organize data

#%%time
for label, section in {'FILT_MD': 'MGNT_DISCUSSION', 'FILT_QA': 'QA_SECTION'}.items():
  
 # currdf['len_' + label] = currdf['matches_' + label].apply(lambda x: [calc['len'] for calc in x]) 
 # currdf['raw_len_' + label] = currdf['matches_' + label].apply(lambda x: [calc['raw_len'] for calc in x]) 

  for topic in word_set_dict.keys():
    currdf[topic + '_TOTAL_' + label] = currdf['matches_' + label].apply(lambda x: x[topic]['total'])
    currdf[topic + '_STATS_' + label] = currdf['matches_' + label].apply(lambda x: x[topic]['stats'])
  #  currdf[topic + '_stats_list_' + label] = currdf['matches_' + label].apply(lambda x: [dict(calc[topic]['stats']) for calc in x])
  
  currdf.drop(['matches_' + label], axis = 1, inplace = True)
  gc.collect()
  

# Calculate additional stats derived from count stats & sentiment

#%%time
for label, section in {'FILT_MD': 'MGNT_DISCUSSION', 'FILT_QA': 'QA_SECTION'}.items():
  
  # currdf['sent_' + label] = currdf['sent_labels_' + label].apply(lambda x: float(np.sum(x)/len(x)) if len(x)>0 else None)
  # currdf['net_sent_' + label] = currdf['sent_labels_' + label].apply(lambda x: np.sum(x) if len(x)>0 else None)
  
  for topic in word_set_dict.keys():
  
  # relevance = #sentences detected with topic / #total sentences
    currdf[topic + '_REL_' + label] = currdf[topic + '_TOTAL_' + label].apply(lambda x: len([a for a in x if a>0])/len(x) if len(x)>0 else None)
    currdf[topic + '_COUNT_' + label] = currdf[topic + '_TOTAL_' + label].apply(lambda x: len([a for a in x if a>0]) if len(x)>0 else None)
    currdf[topic + '_SENT_' + label] = currdf[[topic + '_TOTAL_' + label, 'SENT_LABELS_' + label]].apply(lambda x: sentscore(x[0], x[1], weight = False), axis = 1)
    
  # sent_rel = simple multiplication of sentiment and relevance
  #  currdf[topic + '_sent_rel_' + label] = currdf[[topic + '_relevance_' + label,topic + '_sent_' + label]].apply(lambda x: float(x[0] * x[1]) if x[1] else None, axis = 1)
  #  currdf[topic + '_sent_num_weight_' + label] = currdf[[topic + '_total_' + label, 'sent_labels_' + label]].apply(lambda x: sentscore(x[0], x[1], weight = True), axis = 1)
  
  # sent_weight weights sentiment of sentence by number of unique matching words in that sentence
  #  currdf[topic + '_sent_weight_' + label] = currdf[[topic + '_stats_list_' + label, 'sent_labels_' + label]].apply(lambda x: sentscore([sum([1 if val>0 else 0 for val in stat.values()]) for stat in x[0]], x[1]), axis = 1)
  #  currdf[topic + '_sent_weight_rel_' + label] = currdf[[topic + '_relevance_' + label,topic + '_sent_weight_' + label]].apply(lambda x: float(x[0] * x[1]) if x[1] else None, axis = 1)
  
  # net_sent = #pos - #neg sentences
    currdf[topic + '_NET_SENT_' + label] = currdf[[topic + '_TOTAL_' + label, 'SENT_LABELS_' + label]].apply(lambda x: netscore(x[0], x[1]), axis = 1)
    
    
#  currdf.drop(['matches_' + label], axis = 1, inplace = True)

spark_parsedDF = pandas_to_spark(currdf)
spark_parsedDF = spark_parsedDF.replace(np.nan, None)
#spark_parsedDF = spark_parsedDF.withColumn("DATE", F.to_timestamp(spark_parsedDF.DATE, 'yyyy-MM-dd'))
spark_parsedDF = spark_parsedDF.withColumn("PARSED_DATETIME_EASTERN_TZ", F.to_timestamp(spark_parsedDF.PARSED_DATETIME_EASTERN_TZ, 'yyyy-MM-dd HH mm ss'))
spark_parsedDF = spark_parsedDF.withColumn("EVENT_DATETIME_UTC", F.to_timestamp(spark_parsedDF.EVENT_DATETIME_UTC, 'yyyy-MM-dd HH mm ss'))                                                                            
new_sf.db = 'EDS_PROD'
new_sf.schema = 'QUANT'
 
tablename_curr = 'PARTHA_MACRO_INNO_CTS_STG_1'
result_curr = new_sf.write_to_snowflake_table(spark_parsedDF, tablename_curr)