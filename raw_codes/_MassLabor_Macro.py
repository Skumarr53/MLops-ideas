# import seaborn as sns
# from gensim.models import Word2Vec
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

tsQuery= ("SELECT CALL_ID,ENTITY_ID, DATE, FILT_MD, FILT_QA, SENT_LABELS_FILT_MD, SENT_LABELS_FILT_QA, CALL_NAME,COMPANY_NAME,EARNINGS_CALL,ERROR,TRANSCRIPT_STATUS,UPLOAD_DT_UTC,VERSION_ID,EVENT_DATETIME_UTC,PARSED_DATETIME_EASTERN_TZ "
          
   "FROM QUANT_LIVE.CTS_FUND_COMBINED_SCORES_H  t2 where not exists (select 1 from EDS_PROD.QUANT.YUJING_MASS_LABOR_MACRO_DEV_2 t1 where  t1.CALL_ID = CAST(t2.CALL_ID AS VARCHAR(16777216)) and t1.ENTITY_ID = t2.ENTITY_ID and t1.VERSION_ID = t2.VERSION_ID) ORDER BY PARSED_DATETIME_EASTERN_TZ DESC; ")
          

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

currdf['SENT_LABELS_FILT_QA'] = currdf.apply(lambda x: [x['SENT_LABELS_FILT_QA'][i] for i, sent in enumerate(x['FILT_QA']) if not sent.endswith('?')], axis=1)

currdf['FILT_QA'] = currdf['FILT_QA'].apply(lambda x: [sent for sent in x if not sent.endswith('?')])


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


sections = ['FILT_MD', 'FILT_QA']


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
    
def match_count_lowStat_singleSent(text, match_sets, phrases = True, suppress = None):
 # print(type(text))
 # print(text)
  count_dict = {label : {matchw: 0 for matchw in match_set['unigrams'].union(match_set['bigrams']) } for label, match_set in match_sets.items()}

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

    
  return {label : {'total': counted[label], 'stats' : count_dict[label]} for label in match_sets.keys()}
    

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
# def statscore(a, b):
  
  
  
#   c = 
  
#   return []


match_df_v0['Refined Keywords'] = match_df_v0['Refined Keywords'].apply(ast.literal_eval)

match_df = match_df_v0[['Subtopic','Refined Keywords']].explode(column='Refined Keywords')

match_df_negate = match_df_v0[~match_df_v0['Negation'].isna()][['Subtopic', 'Negation']]#.apply(lambda x: ast.literal_eval(x['Negation']), axis=1)#.explode(column = 'Negation')

match_df_negate['Negation'] = match_df_negate.apply(lambda x: ast.literal_eval(x['Negation']), axis=1)#.explode(column = 'Negation')

match_df_negate = match_df_negate.explode(column = 'Negation')

match_df_negate['negate'] = True

match_df_negate = match_df_negate.rename(columns = {'Subtopic': 'label', 'Negation': 'match'})

match_df['negate'] = False

match_df = match_df.rename(columns={'Subtopic':'label', 'Refined Keywords':'match'})

match_df = pd.concat([match_df, match_df_negate])




word_set_dict = {topic.replace(' ', '_').upper() : get_match_set(match_df[(match_df['label']==topic) & (match_df['negate']==False)]['match'].values) for topic in match_df['label'].unique()}

negate_dict = {topic.replace(' ', '_').upper() : [word.lower() for word in match_df[(match_df['label']==topic) & (match_df['negate']==True)]['match'].values.tolist()] for topic in match_df['label'].unique()}


negate_dict1 = {k: [] for k in negate_dict.keys()}
for k, v in negate_dict.items():
  for word in v:
    if len(word.split('_'))==2:
      new_word = ' '.join(word.split('_'))
      negate_dict1[k].append(new_word)
    else:
      negate_dict1[k].append(word)



currdf = dd.from_pandas(currdf, npartitions = 32)
for label, section in {'FILT_MD': 'FILT_MD', 'FILT_QA': 'FILT_QA'}.items():

  currdf['matches_' + label] = currdf[section].apply(lambda x: match_count_lowStat(x, word_set_dict, phrases = True, suppress = negate_dict1), meta = ('matches_' + label, object))
  #currdf[label] = currdf['matches_' + label].apply(lambda x: [str(calc['filt']) for calc in x], meta = ('FILT_' + label, object))

# Running Dask compute
with ProgressBar():
  currdf = currdf.compute()


for label, section in {'FILT_MD': 'MGNT_DISCUSSION', 'FILT_QA': 'QA_SECTION'}.items():

  for topic in word_set_dict.keys():
    currdf[topic + '_TOTAL_' + label] = currdf['matches_' + label].apply(lambda x: x[topic]['total'])
    currdf[topic + '_STATS_' + label] = currdf['matches_' + label].apply(lambda x: x[topic]['stats'])

  currdf.drop(['matches_' + label], axis = 1, inplace = True)
  gc.collect()

# Dask is not used for below code due to unfixable bugs
# Below code only used to aggregate stats and organize data


for label, section in {'FILT_MD': 'MGNT_DISCUSSION', 'FILT_QA': 'QA_SECTION'}.items():

  for topic in word_set_dict.keys():
    currdf[topic + '_TOTAL_' + label] = currdf['matches_' + label].apply(lambda x: x[topic]['total'])
    currdf[topic + '_STATS_' + label] = currdf['matches_' + label].apply(lambda x: x[topic]['stats'])

  currdf.drop(['matches_' + label], axis = 1, inplace = True)
  gc.collect()
  
# Calculate additional stats derived from count stats & sentiment

for label, section in {'FILT_MD': 'MGNT_DISCUSSION', 'FILT_QA': 'QA_SECTION'}.items():
  
 
  for topic in word_set_dict.keys():
  
  # relevance = #sentences detected with topic / #total sentences
    currdf[topic + '_REL_' + label] = currdf[topic + '_TOTAL_' + label].apply(lambda x: len([a for a in x if a>0])/len(x) if len(x)>0 else None)
    currdf[topic + '_COUNT_' + label] = currdf[topic + '_TOTAL_' + label].apply(lambda x: len([a for a in x if a>0]) if len(x)>0 else None)
    currdf[topic + '_SENT_' + label] = currdf[[topic + '_TOTAL_' + label, 'SENT_LABELS_' + label]].apply(lambda x: sentscore(x[0], x[1], weight = False), axis = 1)

currdf['DATE'] = pd.to_datetime(currdf['DATE'])


from pyspark.sql.types import *

# Auxiliar functions
def equivalent_type(string, f):
    print(string, f)
   # if 'DATE' == string: return StringType()
    
    if f == 'datetime64[ns]': return TimestampType()
    elif f == 'int64': return LongType()
    elif f == 'int32': return IntegerType()
    elif f == 'float64': return FloatType()
    elif 'FILT_MD' == string: return ArrayType(StringType())
    elif 'FILT_QA' == string: return ArrayType(StringType())
    elif '_len_' in string.lower(): return ArrayType(IntegerType())
    elif '_total_' in string.lower(): return ArrayType(IntegerType())
    elif '_count_' in string.lower(): return IntegerType()
    elif '_stats_' in string.lower(): return MapType(StringType(), IntegerType())
    elif 'sent_scores' in string.lower(): return ArrayType(FloatType())
    elif 'sent_labels' in string.lower(): return ArrayType(IntegerType())
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


spark_parsedDF = pandas_to_spark(currdf)
spark_parsedDF = spark_parsedDF.replace(np.nan, None)
spark_parsedDF = spark_parsedDF.withColumn("DATE", F.to_timestamp(spark_parsedDF.DATE, 'yyyy-MM-dd'))
spark_parsedDF = spark_parsedDF.withColumn("PARSED_DATETIME_EASTERN_TZ", F.to_timestamp(spark_parsedDF.PARSED_DATETIME_EASTERN_TZ, 'yyyy-MM-dd HH mm ss'))
spark_parsedDF = spark_parsedDF.withColumn("EVENT_DATETIME_UTC", F.to_timestamp(spark_parsedDF.EVENT_DATETIME_UTC, 'yyyy-MM-dd HH mm ss'))                                                                            
new_sf.db = 'EDS_PROD'
new_sf.schema = 'QUANT'
 
tablename_curr = 'YUJING_MASS_LABOR_MACRO_DEV_2'
result_curr = new_sf.write_to_snowflake_table(spark_parsedDF, tablename_curr)



currdf.to_parquet('/dbfs/mnt/access_work/UC25/Topic Modeling/NLI Models/MASS_Labor/Backtest/MASS_Labor_Macro_historical_test.parquet', compression = 'gzip')