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
from datetime import date 
#import sklearn.datasets
#import plotly.express as px
#from gensim.models import Phrases
from collections import Counter
import dask.dataframe as dd
from dask.distributed import Client
from dask.diagnostics import ProgressBar
import gc


## IMPORTANT =  Set to number of Cores !!
n_tasks = 32

# Dask client 
client = Client(n_workers=n_tasks, threads_per_worker=1)

# Object to enable Snowflake access to prod Quant
myDBFS = DBFShelper()
new_sf = pd.read_pickle(r'/dbfs' + myDBFS.iniPath + 'mysf_prod_quant.pkl')

# Read start & end ranges. Range not inclusive of end date. 
month = datetime.now().month
year = datetime.now().year

minDateNewQuery = str(year - (1 if month == 1 else 0)) + '-' + (('0' if (month<11 and month>1) else '') + str(month - (1 if month > 1 else -11))) + "-01"
maxDateNewQuery = str(year) + '-' + (('0' if month<10 else '') + str(month)) + "-01"

mind = "'" + minDateNewQuery + "'"
maxd = "'" + maxDateNewQuery + "'"

print('The next query spans ' + mind + ' to ' + maxd)

# Query transcripts.
tsQuery= ("SELECT CALL_ID,ENTITY_ID, FILT_MD, FILT_QA, SENT_LABELS_FILT_MD, SENT_LABELS_FILT_QA, CALL_NAME,COMPANY_NAME,EARNINGS_CALL,ERROR,TRANSCRIPT_STATUS,UPLOAD_DT_UTC,VERSION_ID,EVENT_DATETIME_UTC,PARSED_DATETIME_EASTERN_TZ "
          
   "FROM EDS_PROD.QUANT_LIVE.CTS_FUND_COMBINED_SCORES_H "
          
   "WHERE EVENT_DATETIME_UTC >= " + mind  + " AND EVENT_DATETIME_UTC < " + maxd  + " AND (EARNINGS_CALL > 0) AND (TRANSCRIPT_STATUS = 'CorrectedTranscript');")

resultspkdf = new_sf.read_from_snowflake(tsQuery)

currdf = resultspkdf.toPandas()

if len(currdf)>0:
   # currdf.to_parquet('/dbfs/mnt/access_work/UC25/Backtest/Fundamentals_David_PoC/fundP2_latestProcessed.parquet', compression = 'gzip')
    print('The data spans from ' + str(currdf['PARSED_DATETIME_EASTERN_TZ'].min()) + ' to ' + str(currdf['PARSED_DATETIME_EASTERN_TZ'].max()) + 'and has ' + str(currdf.shape[0]) + ' rows and ' + str(currdf.shape[1]) + ' columns.')
else:
    print('No new transcripts to parse.')
    dbutils.notebook.exit(1)
    os._exit(1)

currdf['CALL_ID'] = currdf['CALL_ID'].apply(lambda x: str(x))

# Convert to Python list from string
currdf['FILT_MD'] = currdf['FILT_MD'].apply(ast.literal_eval)
currdf['FILT_QA'] = currdf['FILT_QA'].apply(ast.literal_eval)
currdf['SENT_LABELS_FILT_MD'] = currdf['SENT_LABELS_FILT_MD'].apply(ast.literal_eval)
currdf['SENT_LABELS_FILT_QA'] = currdf['SENT_LABELS_FILT_QA'].apply(ast.literal_eval)

currdf['LEN_MD'] = currdf['FILT_MD'].apply(len)
currdf['LEN_QA'] = currdf['FILT_QA'].apply(len)



# Remove questions and keep +ve sentiment only
currdf['FILT_MD'] = currdf[['FILT_MD', 'SENT_LABELS_FILT_MD']].apply(lambda x: [y for y,z in zip(x[0], x[1]) if ((z==1) & (not y.endswith('?')))], axis = 1)
currdf['FILT_QA'] = currdf[['FILT_QA', 'SENT_LABELS_FILT_QA']].apply(lambda x: [y for y,z in zip(x[0], x[1]) if ((z==1) & (not y.endswith('?')))], axis = 1)

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
def match_count_noStat(text, match_sets, phrases = True, suppress = None):
  suppressed = {label: False for label in match_sets.keys()}
  unigrams = wordTokenize(text)
  
  vocab = {word: 0 for word in set(unigrams)}
  bigrams = ['_'.join(g) for g in find_ngrams(unigrams, 2)]
  
  #count_dict = {label : {match: 0 for match in match_set['unigrams'].union(match_set['bigrams']) } for label, match_set in match_sets.items()}
  
  total_count = {label : 0 for label, match_set in match_sets.items()}
  
  for label, match_set in match_sets.items(): 
    for word in unigrams:
      if label in suppress.keys():
        if word in suppress[label]:
          suppressed[label] = True
          continue
      if word in match_set['unigrams']:
       # count_dict[label][word]+=1
        total_count[label]+=1
    for word in bigrams:
     # print(word)
      if label in suppress.keys():
        if word in suppress[label]:
          suppressed[label] = True
          continue
      if word in match_set['bigrams']:
       # count_dict[label][word]+=1
        total_count[label]+=1
    
    if phrases:
      total_count[label] = uni_count[label] + bi_count[label] + sum([1 if phrase in text.lower() else 0 for phrase in match_set['phrases']]) 
    
  #  print(suppressed)
    #print(phrase_count)

 # uni_count = sum([1 if word in match_set else 0 for word in unigrams])
 # print({'uni': uni_count, 'bi' : bi_count, 'phrase': phrase_count, 'total': uni_count + bi_count + phrase_count, 'stats' : count_dict})
  
  return {label : total_count[label] if not suppressed[label] else 0 for label in match_sets.keys()}
    

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

currdf = dd.from_pandas(currdf, npartitions = n_tasks)
for label, section in {'FILT_MD': 'FILT_MD', 'FILT_QA': 'FILT_QA'}.items():

  currdf['matches_' + label] = currdf[section].apply(lambda x: match_count_lowStat(x, word_set_dict, suppress = negate_dict), meta = ('matches_' + label, object))



  # Running Dask compute
with ProgressBar():
  currdf = currdf.compute()

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

for label, section in {'FILT_MD': 'MGNT_DISCUSSION', 'FILT_QA': 'QA_SECTION'}.items():
  
 # currdf['sent_' + label] = currdf['SENT_LABEL_' + label].apply(lambda x: float(np.sum(x)/len(x)) if len(x)>0 else None)
 # currdf['net_sent_' + label] = currdf['SENT_LABEL_' + label].apply(lambda x: np.sum(x) if len(x)>0 else None)
  
  for topic in word_set_dict.keys():
  
  # relevance = #sentences detected with topic / #total sentences
    currdf[topic + '_REL_' + label] = currdf[topic + '_TOTAL_' + label].apply(lambda x: len([a for a in x if a>0]) if len(x)>0 else None)
    currdf[topic + '_EXTRACT_' + label] = currdf[[label, topic + '_TOTAL_' + label]].apply(lambda x: ' '.join([y for y,z in zip(x[0], x[1]) if ((z>0))]), axis = 1)
   # currdf[topic + '_SENT_' + label] = currdf[[topic + '_TOTAL_' + label, 'SENT_LABELS_' + label]].apply(lambda x: sentscore(x[0], x[1], weight = False), axis = 1)

    

# Recovery report

rdf = currdf.sort_values('RECOVERY_REL_FILT_MD', ascending= False)[['ENTITY_ID', 'CALL_NAME', 'EVENT_DATETIME_UTC', 'COMPANY_NAME', 'RECOVERY_REL_FILT_MD', 'RECOVERY_STATS_FILT_MD','RECOVERY_EXTRACT_FILT_MD']].drop_duplicates('ENTITY_ID').head(102).dropna(subset = ['RECOVERY_REL_FILT_MD'])
#rdf.to_csv('/dbfs/mnt/access_work/UC25/SF_Equities_Reports/sf_equities_recovery_' + str(month) + '_' + str(year) + '.csv')

# Cycle report

cdf = currdf.sort_values('CYCLE_REL_FILT_MD', ascending= False)[['ENTITY_ID', 'CALL_NAME', 'EVENT_DATETIME_UTC','COMPANY_NAME', 'CYCLE_REL_FILT_MD', 'CYCLE_STATS_FILT_MD','CYCLE_EXTRACT_FILT_MD']].drop_duplicates('ENTITY_ID').head(102).dropna(subset = ['CYCLE_REL_FILT_MD'])

sdf = currdf.sort_values('S&D_REL_FILT_MD', ascending= False)[['ENTITY_ID', 'CALL_NAME', 'EVENT_DATETIME_UTC','COMPANY_NAME', 'S&D_REL_FILT_MD', 'S&D_STATS_FILT_MD','S&D_EXTRACT_FILT_MD']].drop_duplicates('ENTITY_ID').head(102).dropna(subset = ['S&D_REL_FILT_MD'])


# Joining results of top 100 matches for each topic
concatdf = pd.concat([rdf, cdf, sdf])
concatdf.reset_index(inplace = True)
concatdf.drop(['index'], axis = 1, inplace = True)

concatdf = concatdf[(concatdf['RECOVERY_REL_FILT_MD']>0) | (concatdf['CYCLE_REL_FILT_MD']>0) | (concatdf['S&D_REL_FILT_MD']>0)]

2
# Function to switch dictionary key-value pairs and check if values>0
def simpDict(x):
 # print(type(x))
  return {key : val for key, val in x.items() if val!=0}

# Applying function above.
for col in concatdf.columns:
  if 'STATS' in col:
   # print(concatdf[col].values[0].items())
    concatdf[col] = concatdf[col].apply(lambda x: simpDict(x) if type(x)==dict else None)

concatdf['REPORT_DATE'] = pd.to_datetime(maxd[1:-1])
concatdf.to_csv('/dbfs/mnt/access_work/UC25/SF_Equities_Reports/sf_equities_report_' + str(month) + '_' + str(year) + '.csv')


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
    elif 'len' in string.lower(): return ArrayType(IntegerType())
    elif 'total' in string.lower(): return ArrayType(IntegerType())
    elif 'count' in string.lower(): return IntegerType()
    elif 'stats' in string.lower(): return MapType(StringType(), IntegerType())
    elif 'sent_scores' in string.lower(): return ArrayType(FloatType())
    elif 'sent_labels' in string.lower(): return ArrayType(IntegerType())
 #   elif f == 'object': return ArrayType()
 #   elif f == 'list': return ArrayType()
    else: return StringType()

# Given pandas dataframe, it will return a spark's dataframe.
def pandas_to_spark(pandas_df):
    columns = list(pandas_df.columns)
    types = list(pandas_df.dtypes)
    struct_list = []
    for column, typo in zip(columns, types): 
      struct_list.append(define_structure(column, typo))
    p_schema = StructType(struct_list)
    return sqlContext.createDataFrame(pandas_df, p_schema)
  
def define_structure(string, format_type):
  #try: 
  typo = equivalent_type(string, format_type)
  print(typo)
  #except: typo = StringType()
  return StructField(string, typo)

spark_parsedDF = pandas_to_spark(concatdf)
spark_parsedDF = spark_parsedDF.replace(np.nan, None)
#spark_parsedDF = spark_parsedDF.withColumn("DATE", F.to_timestamp(spark_parsedDF.DATE, 'yyyy-MM-dd'))
spark_parsedDF = spark_parsedDF.withColumn("REPORT_DATE", F.to_timestamp(spark_parsedDF.REPORT_DATE, 'yyyy-MM-dd HH mm ss'))
spark_parsedDF = spark_parsedDF.withColumn("EVENT_DATETIME_UTC", F.to_timestamp(spark_parsedDF.EVENT_DATETIME_UTC, 'yyyy-MM-dd HH mm ss'))                                                                            
new_sf.db = 'EDS_PROD'
new_sf.schema = 'QUANT'
 
tablename_curr = 'PARTHA_SF_REPORT_CTS_STG_1'
result_curr = new_sf.write_to_snowflake_table(spark_parsedDF, tablename_curr)