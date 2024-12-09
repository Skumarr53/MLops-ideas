# Databricks notebook source
# MAGIC
# MAGIC %pip install loguru==0.7.2
# MAGIC %pip install hydra-core==1.3
# MAGIC %pip install python-dotenv==1.0.1
# MAGIC %pip install numpy==1.24.4
# MAGIC %pip install cryptography==43.0.1
# MAGIC %pip install gensim==4.3.3
# MAGIC %pip install cython==3.0.11
# MAGIC %pip install spacy==3.4.4 #3.0.4
# MAGIC %pip install thinc==8.1.7
# MAGIC %pip install pandas==2.0.0
# MAGIC %pip install snowflake-connector-python==3.12.2
# MAGIC %pip install transformers==4.46.1
# MAGIC %pip install pyarrow==16.0.0
# MAGIC %pip install datasets==3.1.0
# MAGIC %pip install evaluate==0.4.3
# MAGIC %pip install pyspark==3.5.3
# MAGIC %pip install "dask[dataframe,distributed]"==2023.9.3
# MAGIC %pip install torch==2.0.0
# MAGIC %pip install cymem==2.0.8
# MAGIC %pip install scikit-learn==1.1.0
# MAGIC %pip install typer==0.7.0
# MAGIC %pip install accelerate==0.26.0

# COMMAND ----------

import ast
from centralized_nlp_package.data_access import (
    read_from_snowflake,
    write_dataframe_to_snowflake
)
from centralized_nlp_package.data_processing import (
  check_pd_dataframe_for_records,
    initialize_dask_client,
    df_apply_transformations,
    dask_compute_with_progress,
    pandas_to_spark,
    convert_columns_to_timestamp
)
from centralized_nlp_package.text_processing import (initialize_spacy, get_match_set)

from topic_modelling_package.reports import create_topic_dict, generate_topic_report, replace_separator_in_dict_words

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load dataset

# COMMAND ----------


tsQuery= ("SELECT CALL_ID,ENTITY_ID, DATE, FILT_MD, FILT_QA, SENT_LABELS_FILT_MD,"
         "SENT_LABELS_FILT_QA, CALL_NAME,COMPANY_NAME,EARNINGS_CALL,ERROR,TRANSCRIPT_STATUS,UPLOAD_DT_UTC,VERSION_ID,EVENT_DATETIME_UTC,PARSED_DATETIME_EASTERN_TZ "
          "FROM EDS_PROD.QUANT_LIVE.CTS_FUND_COMBINED_SCORES_H  t2  ORDER BY PARSED_DATETIME_EASTERN_TZ DESC LIMIT 10")
          



# COMMAND ----------

tsQuery

# COMMAND ----------

resultspkdf = read_from_snowflake(tsQuery)

# COMMAND ----------

currdf_old = resultspkdf.toPandas()
currdf_ref = resultspkdf.toPandas()

# COMMAND ----------

currdf_ref.to_csv('/Workspace/Users/santhosh.kumar3@voya.com/MLFlow_and_NLI_finetune/currdf_ref.csv', index=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Checking records

# COMMAND ----------

# MAGIC %md
# MAGIC ### old

# COMMAND ----------

import os
if len(currdf_old)>0:
    print('The data spans from ' + str(currdf_old['PARSED_DATETIME_EASTERN_TZ'].min()) + ' to ' + str(currdf_old['PARSED_DATETIME_EASTERN_TZ'].max()) + 'and has ' + str(currdf_old.shape[0]) + ' rows and ' + str(currdf_old.shape[1]) + ' columns.')
else:
    print('No new transcripts to parse.')
    dbutils.notebook.exit(1)
    os._exit(1)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Reafctored

# COMMAND ----------

check_pd_dataframe_for_records(currdf_old, datetime_col = 'PARSED_DATETIME_EASTERN_TZ')

# COMMAND ----------

# MAGIC %md
# MAGIC ### Apply Transformation

# COMMAND ----------

# MAGIC %md
# MAGIC ### old

# COMMAND ----------


currdf_old['CALL_ID'] = currdf_old['CALL_ID'].apply(lambda x: str(x))
currdf_old['FILT_MD'] = currdf_old['FILT_MD'].apply(ast.literal_eval)
currdf_old['FILT_QA'] = currdf_old['FILT_QA'].apply(ast.literal_eval)
currdf_old['SENT_LABELS_FILT_MD'] = currdf_old['SENT_LABELS_FILT_MD'].apply(ast.literal_eval)
currdf_old['SENT_LABELS_FILT_QA'] = currdf_old['SENT_LABELS_FILT_QA'].apply(ast.literal_eval)
currdf_old['LEN_FILT_MD'] = currdf_old['FILT_MD'].apply(len)
currdf_old['LEN_FILT_QA'] = currdf_old['FILT_QA'].apply(len)
currdf_old['SENT_LABELS_FILT_QA'] = currdf_old.apply(lambda x: [x['SENT_LABELS_FILT_QA'][i] for i, sent in enumerate(x['FILT_QA']) if not sent.endswith('?')], axis=1)
currdf_old['FILT_QA'] = currdf_old['FILT_QA'].apply(lambda x: [sent for sent in x if not sent.endswith('?')])


currdf_old.head(5)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Refactored

# COMMAND ----------

col_inti_tranform = [
    ("FILT_MD", "FILT_MD", ast.literal_eval),
    ("FILT_QA", "FILT_QA", ast.literal_eval),
    ("SENT_LABELS_FILT_QA", "SENT_LABELS_FILT_QA", ast.literal_eval),
    ("SENT_LABELS_FILT_MD", "SENT_LABELS_FILT_MD", ast.literal_eval),
    ('LEN_MD', 'FILT_MD', len),
    ('LEN_QA', 'FILT_QA', len),
    ('SENT_LABELS_FILT_QA', ['SENT_LABELS_FILT_QA','FILT_QA'], (lambda x: [x['SENT_LABELS_FILT_QA'][i] 
                                                                           for i, sent in enumerate(x['FILT_QA']) 
                                                                           if not sent.endswith('?')])),
    ('FILT_QA', 'FILT_QA', lambda x: [sent for sent in x if not sent.endswith('?')])
]

currdf_ref = df_apply_transformations(currdf_ref, col_inti_tranform)
currdf_ref.head(5)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Spacy model initalization

# COMMAND ----------

# MAGIC %md
# MAGIC #### old 

# COMMAND ----------

# MAGIC %pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.5.0/en_core_web_sm-3.5.0.tar.gz

# COMMAND ----------

import spacy 
nlp = spacy.load("en_core_web_sm", disable = ['parser'])
# Excluding financially reavant stopwords
nlp.Defaults.stop_words -= {"bottom", "top", "Bottom", "Top", "call", "down"}
nlp.max_length = 1000000000

# COMMAND ----------

# MAGIC %md
# MAGIC ### refactored

# COMMAND ----------

nlp_ref = initialize_spacy()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Load Topics dictionary

# COMMAND ----------

import pandas as pd

match_df_v0_old = pd.read_csv("/dbfs/mnt/access_work/UC25/Topic Modeling/NLI Models/MASS_labor_macro_dictionaries_final.csv")
match_df_v0_ref = pd.read_csv("/dbfs/mnt/access_work/UC25/Topic Modeling/NLI Models/MASS_labor_macro_dictionaries_final.csv")

# COMMAND ----------

match_df_v0_old.to_csv('/Workspace/Users/santhosh.kumar3@voya.com/MLFlow_and_NLI_finetune/match_df.csv', index=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Match keyword list explosion

# COMMAND ----------

# MAGIC %md
# MAGIC #### old

# COMMAND ----------

match_df_v0_old['Refined Keywords'] = match_df_v0_old['Refined Keywords'].apply(ast.literal_eval)
match_df_old = match_df_v0_old[['Subtopic','Refined Keywords']].explode(column='Refined Keywords')

match_df_old.head(5)

# COMMAND ----------

# MAGIC %md
# MAGIC #### ref

# COMMAND ----------

match_df_v0_ref = df_apply_transformations(match_df_v0_ref, [('Refined Keywords', 'Refined Keywords', ast.literal_eval)])
match_df_ref = match_df_v0_ref[['Subtopic','Refined Keywords']].explode(column='Refined Keywords')

match_df_ref.head(5)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Common code

# COMMAND ----------

match_df_negate = match_df_v0_ref[~match_df_v0_ref['Negation'].isna()][['Subtopic', 'Negation']]#.apply(lambda x: ast.literal_eval(x['Negation']), axis=1)#.explode(column = 'Negation')
match_df_negate = df_apply_transformations(match_df_negate, [('Negation', 'Negation', ast.literal_eval)])
match_df_negate = match_df_negate.explode(column = 'Negation')
match_df_negate['negate'] = True
match_df_negate = match_df_negate.rename(columns = {'Subtopic': 'label', 'Negation': 'match'})
match_df_ref['negate'] = False
match_df_ref = match_df_ref.rename(columns={'Subtopic':'label', 'Refined Keywords':'match'})
match_df_ref = pd.concat([match_df_ref, match_df_negate])

# COMMAND ----------

# MAGIC %md
# MAGIC ### create topic dictionary

# COMMAND ----------

# MAGIC %md
# MAGIC #### old

# COMMAND ----------

match_df_old = match_df_ref.copy()

# COMMAND ----------

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

def get_match_set_old(matches):
  
  bigrams = set([word.lower() for word in matches if len(word.split('_'))==2] + [word.lower().replace(" ", '_') for word in matches if len(word.split(' '))==2] + ['_'.join(matchTokenize(word)) for word in matches if len(word.split(' '))==2])
 
  unigrams = set([matchTokenize(match)[0] for match in matches if ('_' not in match) and (len(match.split(' '))==1)] + [match.lower() for match in matches if ('_' not in match) and (len(match.split(' '))==1)])

#  phrases = set([phrase.lower() for phrase in matches if len(phrase.split(" "))>2] + [' '.join(matchTokenize(phrase)) for phrase in matches if len(phrase.split(" "))>2])

#  Phrase matching correction
  phrases = [phrase.lower() for phrase in matches if len(phrase.split(" "))>2]
  
  #print(matches)
  #print(unigrams, bigrams, phrases)
  return {'original': matches, 'unigrams': unigrams, 'bigrams': bigrams, 'phrases': phrases}
  


# COMMAND ----------

word_set_dict_old = {topic.replace(' ', '_').upper() : get_match_set_old(match_df_old[(match_df_old['label']==topic) & (match_df_old['negate']==False)]['match'].values) for topic in match_df_old['label'].unique()}

negate_dict_old = {topic.replace(' ', '_').upper() : [word.lower() for word in match_df_old[(match_df_old['label']==topic) & (match_df_old['negate']==True)]['match'].values.tolist()] for topic in match_df_old['label'].unique()}


# COMMAND ----------

# MAGIC %md
# MAGIC #### Refactored

# COMMAND ----------

match_df_ref.head()


# COMMAND ----------

word_set_dict_ref, negate_dict_ref = create_topic_dict(match_df_ref)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Replace separator in dicts with underscore

# COMMAND ----------

# MAGIC %md
# MAGIC #### old 

# COMMAND ----------

negate_dict1_old = {k: [] for k in negate_dict_old.keys()}
for k, v in negate_dict_old.items():
  for word in v:
    if len(word.split('_'))==2:
      new_word = ' '.join(word.split('_'))
      negate_dict1_old[k].append(new_word)
    else:
      negate_dict1_old[k].append(word)

# COMMAND ----------

# MAGIC %md
# MAGIC #### refactored

# COMMAND ----------

negate_dict1_ref = replace_separator_in_dict_words(negate_dict_ref)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Genearte Topic report 

# COMMAND ----------

# MAGIC %md
# MAGIC #### old

# COMMAND ----------

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

currdf_old = dd.from_pandas(currdf_old, npartitions = 32)
for label, section in {'FILT_MD': 'FILT_MD', 'FILT_QA': 'FILT_QA'}.items():

  currdf_old['matches_' + label] = currdf_old[section].apply(lambda x: match_count_lowStat(x, word_set_dict_old, phrases = True, suppress = negate_dict1_old), meta = ('matches_' + label, object))
  #currdf_old[label] = currdf_old['matches_' + label].apply(lambda x: [str(calc['filt']) for calc in x], meta = ('FILT_' + label, object))

# Running Dask compute
with ProgressBar():
  currdf_old = currdf_old.compute()


for label, section in {'FILT_MD': 'MGNT_DISCUSSION', 'FILT_QA': 'QA_SECTION'}.items():

  for topic in word_set_dict_old.keys():
    currdf_old[topic + '_TOTAL_' + label] = currdf_old['matches_' + label].apply(lambda x: x[topic]['total'])
    currdf_old[topic + '_STATS_' + label] = currdf_old['matches_' + label].apply(lambda x: x[topic]['stats'])

  currdf_old.drop(['matches_' + label], axis = 1, inplace = True)
  gc.collect()

# Dask is not used for below code due to unfixable bugs
# Below code only used to aggregate stats and organize data


for label, section in {'FILT_MD': 'MGNT_DISCUSSION', 'FILT_QA': 'QA_SECTION'}.items():

  for topic in word_set_dict_old.keys():
    currdf_old[topic + '_TOTAL_' + label] = currdf_old['matches_' + label].apply(lambda x: x[topic]['total'])
    currdf_old[topic + '_STATS_' + label] = currdf_old['matches_' + label].apply(lambda x: x[topic]['stats'])

  currdf_old.drop(['matches_' + label], axis = 1, inplace = True)
  gc.collect()
  
# Calculate additional stats derived from count stats & sentiment

for label, section in {'FILT_MD': 'MGNT_DISCUSSION', 'FILT_QA': 'QA_SECTION'}.items():
  
 
  for topic in word_set_dict_old.keys():
  
  # relevance = #sentences detected with topic / #total sentences
    currdf_old[topic + '_REL_' + label] = currdf_old[topic + '_TOTAL_' + label].apply(lambda x: len([a for a in x if a>0])/len(x) if len(x)>0 else None)
    currdf_old[topic + '_COUNT_' + label] = currdf_old[topic + '_TOTAL_' + label].apply(lambda x: len([a for a in x if a>0]) if len(x)>0 else None)
    currdf_old[topic + '_SENT_' + label] = currdf_old[[topic + '_TOTAL_' + label, 'SENT_LABELS_' + label]].apply(lambda x: sentscore(x[0], x[1], weight = False), axis = 1)


# COMMAND ----------

# MAGIC %md
# MAGIC #### refactored

# COMMAND ----------

currdf = generate_topic_report(currdf_ref, word_set_dict_ref, negate_dict1_ref, stats_list = ['relevance', 'count', 'sentiment'])


# COMMAND ----------

# MAGIC %md
# MAGIC ### Writing Back to Snowflake table 

# COMMAND ----------

# MAGIC %md
# MAGIC #### old

# COMMAND ----------

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

# COMMAND ----------

# MAGIC %md
# MAGIC #### refactored

# COMMAND ----------



# COMMAND ----------

import time 

while 1>0:
  time.sleep(60)

# COMMAND ----------


