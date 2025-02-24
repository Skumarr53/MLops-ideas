I have the following Dask code that processes text columns in a DataFrame. I need to transition this code to PySpark to run it in Databricks. Could you carefully review the code to understand the logic in detail and convert it to PySpark code? Please double-check the converted code for any errors to ensure it works as expected.

code 
```
from spacy.lang.en import English
import spacy 

nlp = spacy.load("en_core_web_sm", disable = ['parser'])

# Excluding financially relavant stopwords
nlp.Defaults.stop_words -= {"bottom", "top", "Bottom", "Top", "call"}
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

# Preprocessing & filtering of CT sentences. Should be customized for use.
def process_sentence(sent):
  
  sent = sent.replace("Thank you", "")
  sent = sent.replace("thank you", "")
  sent = sent.replace("thanks", "")
  sent = sent.replace("Thanks", "")
  sent = sent.replace("earnings call", "")
  sent = sent.replace("earnings release", "")
  sent = sent.replace("earnings conference", "")
  
  # Filter sents with length less than 5 or if introductory remarks are found
  if len(sent.split(" ")) < 5:

      return None
  
  low = sent.lower()
  if "good morning" in low or "good afternoon" in low or "good evening" in low or "welcome" in low:
      return None

  return sent


# Sentence Identification
sent_tokenizer = English()
sent_tokenizer.add_pipe("sentencizer")


def sentTokenize(doc):
  
  return filter(lambda item: item is not None, [process_sentence(sent.text) for sent in sent_tokenizer(doc).sents])


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
def match_count(text, match_sets, phrases = True):
  unigrams = wordTokenize(text)
  
  vocab = {word: 0 for word in set(unigrams)}
  bigrams = ['_'.join(g) for g in find_ngrams(unigrams, 2)]
  
  count_dict = {label : {match: 0 for match in match_set['unigrams'].union(match_set['bigrams']) } for label, match_set in match_sets.items()}
  
  uni_count = {label : 0 for label, match_set in match_sets.items()}
  bi_count = {label : 0 for label, match_set in match_sets.items()}
  
  if phrases:
    phrase_count = {label : 0 for label, match_set in match_sets.items()}
  total_count = {label : 0 for label, match_set in match_sets.items()}
  
  for label, match_set in match_sets.items(): 
    for word in unigrams:
      if word in match_set['unigrams']:
        count_dict[label][word]+=1
        uni_count[label]+=1
    for word in bigrams:
      if word in match_set['bigrams']:
        count_dict[label][word]+=1
        bi_count[label]+=1
    
    if phrases:
      phrase_count[label] = sum([1 if phrase in text.lower() else 0 for phrase in match_set['phrases']])    
      total_count[label] = uni_count[label] + bi_count[label] + phrase_count[label]
    
    else:
      total_count[label] = uni_count[label] + bi_count[label]
    
    #print(phrase_count)

 # uni_count = sum([1 if word in match_set else 0 for word in unigrams])
 # print({'uni': uni_count, 'bi' : bi_count, 'phrase': phrase_count, 'total': uni_count + bi_count + phrase_count, 'stats' : count_dict})
  
  if phrases:
    ret = {label : {'uni': uni_count[label], 'bi' : bi_count[label], 'phrase': phrase_count[label], 'total': total_count[label], 'stats' : count_dict[label]} for label in match_sets.keys()}
  
  else:
    ret = {label : {'uni': uni_count[label], 'bi' : bi_count[label], 'total': total_count[label], 'stats' : count_dict[label]} for label in match_sets.keys()}
  
  ret['len'] = len(unigrams)
  ret['raw_len'] = len(text.split(' '))
 
 ## REMOVE IF NOT NECESSARY
  ret['filt'] = text 
  
  return ret


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
match_df = pd.read_csv("/dbfs/mnt/access_work/UC25/Embeddings/Word lists/fundamental_final_draft.csv")
word_set_dict = {topic : get_match_set(match_df[match_df['label']==topic]['word'].values) for topic in match_df['label'].unique()}

req_cols = ['FILT_MD','FILT_QA', 'FILT_CEO_MD', 'FILT_EXEC_MD', 'FILT_CEO_QA', 'FILT_EXEC_QA', 'FILT_ANL_QA']

import re, ast


currdf1 = dd.from_pandas(currdf1, npartitions = 32)

for col in req_cols:
  currdf1['matches_' + col] = currdf1[col].apply(lambda x:[match_count(sent, word_set_dict, phrases = False) for sent in sentTokenize(x)], meta = ('matches_' + col, object))  ## long text to list of sentences
  currdf1[col] = currdf1['matches_' + col].apply(lambda x: [text_clean(str(calc['filt'])) for calc in x], meta = (col, object))

currdf1 = currdf1.compute()
```

Chat GPT response:
```
# --- Imports and Global Setup ---
import spacy
from spacy.lang.en import English
import numpy as np
import pandas as pd
import re
from collections import Counter
import threading
import json

from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.types import (
    StructType, StructField, ArrayType, StringType
)

# Create SparkSession (if not already created)
spark = SparkSession.builder.getOrCreate()

# --- Lazy Initialization for spaCy in Workers ---
thread_local = threading.local()

def get_spacy_model():
    if not hasattr(thread_local, "nlp"):
        nlp = spacy.load("en_core_web_sm", disable=['parser'])
        # Exclude specific financially relevant stopwords
        nlp.Defaults.stop_words -= {"bottom", "top", "Bottom", "Top", "call"}
        nlp.max_length = 1000000000
        
        # Create sentencizer for sentence tokenization
        sent_tokenizer = English()
        sent_tokenizer.add_pipe("sentencizer")
        thread_local.nlp = nlp
        thread_local.sent_tokenizer = sent_tokenizer
    return thread_local.nlp, thread_local.sent_tokenizer

# --- NLP Functions ---
def wordTokenize(doc):
    nlp, _ = get_spacy_model()
    return [ent.lemma_.lower() for ent in nlp(doc) if not ent.is_stop and not ent.is_punct and ent.pos_ != 'NUM']

def matchTokenize(doc):
    nlp, _ = get_spacy_model()
    ret = []
    for ent in nlp(doc):
        if ent.pos_ == 'PROPN' or (ent.text and ent.text[0].isupper()):
            ret.append(ent.text.lower())
            continue
        if not ent.is_stop and not ent.is_punct and ent.pos_ != 'NUM':
            ret.append(ent.lemma_.lower())
    return ret

def find_ngrams(input_list, n):
    return zip(*[input_list[i:] for i in range(n)])

def process_sentence(sent):
    # Remove unwanted phrases
    sent = sent.replace("Thank you", "")
    sent = sent.replace("thank you", "")
    sent = sent.replace("thanks", "")
    sent = sent.replace("Thanks", "")
    sent = sent.replace("earnings call", "")
    sent = sent.replace("earnings release", "")
    sent = sent.replace("earnings conference", "")
    
    # Filter out short sentences and common greetings
    if len(sent.split(" ")) < 5:
        return None
    low = sent.lower()
    if any(greet in low for greet in ["good morning", "good afternoon", "good evening", "welcome"]):
        return None
    return sent

def sentTokenize(doc):
    # Tokenize document into sentences and filter them
    _, sent_tokenizer = get_spacy_model()
    doc_obj = sent_tokenizer(doc)
    return [s for s in (process_sentence(sent.text) for sent in doc_obj.sents) if s is not None]

def get_match_set(matches):
    # Build sets for unigrams, bigrams, and phrases
    bigrams = set(
        [word.lower() for word in matches if len(word.split('_')) == 2] +
        [word.lower().replace(" ", '_') for word in matches if len(word.split(' ')) == 2] +
        ['_'.join(matchTokenize(word)) for word in matches if len(word.split(' ')) == 2]
    )
    unigrams = set(
        [matchTokenize(match)[0] for match in matches if ('_' not in match) and (len(match.split(' ')) == 1)] +
        [match.lower() for match in matches if ('_' not in match) and (len(match.split(' ')) == 1)]
    )
    phrases = [phrase.lower() for phrase in matches if len(phrase.split(" ")) > 2]
    return {'original': matches, 'unigrams': unigrams, 'bigrams': bigrams, 'phrases': phrases}

def match_count(text, match_sets, phrases=True):
    unigrams = wordTokenize(text)
    bigrams = ['_'.join(g) for g in find_ngrams(unigrams, 2)]
    
    count_dict = {
        label: {match: 0 for match in (match_set['unigrams'].union(match_set['bigrams']))}
        for label, match_set in match_sets.items()
    }
    uni_count = {label: 0 for label in match_sets.keys()}
    bi_count = {label: 0 for label in match_sets.keys()}
    if phrases:
        phrase_count = {label: 0 for label in match_sets.keys()}
    total_count = {label: 0 for label in match_sets.keys()}
    
    for label, match_set in match_sets.items():
        for word in unigrams:
            if word in match_set['unigrams']:
                count_dict[label][word] += 1
                uni_count[label] += 1
        for word in bigrams:
            if word in match_set['bigrams']:
                count_dict[label][word] += 1
                bi_count[label] += 1
        if phrases:
            phrase_count[label] = sum([1 if phrase in text.lower() else 0 for phrase in match_set['phrases']])
            total_count[label] = uni_count[label] + bi_count[label] + phrase_count[label]
        else:
            total_count[label] = uni_count[label] + bi_count[label]
    
    if phrases:
        ret = {
            label: {
                'uni': uni_count[label],
                'bi': bi_count[label],
                'phrase': phrase_count[label],
                'total': total_count[label],
                'stats': count_dict[label]
            } for label in match_sets.keys()
        }
    else:
        ret = {
            label: {
                'uni': uni_count[label],
                'bi': bi_count[label],
                'total': total_count[label],
                'stats': count_dict[label]
            } for label in match_sets.keys()
        }
    
    ret['len'] = len(unigrams)
    ret['raw_len'] = len(text.split(' '))
    ret['filt'] = text  # original text (could be cleaned further if needed)
    return ret

def text_clean(text):
    # Simple cleaning function (customize as needed)
    return text.strip()

# --- Broadcast Match Dictionary ---
# Read the CSV (assumed to be accessible via DBFS)
match_df = pd.read_csv("/dbfs/mnt/access_work/UC25/Embeddings/Word lists/fundamental_final_draft.csv")
# Create a dictionary keyed by 'label' with match sets
word_set_dict = {
    topic: get_match_set(match_df[match_df['label'] == topic]['word'].tolist())
    for topic in match_df['label'].unique()
}
# Broadcast the dictionary to all workers
broadcast_word_set = spark.sparkContext.broadcast(word_set_dict)

# --- UDF for Processing Text Columns ---
def process_column_json(text):
    """
    Process input text by tokenizing sentences, applying match_count on each,
    and cleaning the text. Returns:
      - A JSON string for the list of match_count dictionaries.
      - A list of cleaned sentences.
    """
    if text is None:
        return json.dumps([]), []
    matches = []
    cleaned_texts = []
    sentences = sentTokenize(text)
    for sent in sentences:
        mc = match_count(sent, broadcast_word_set.value, phrases=False)
        matches.append(mc)
        cleaned_texts.append(text_clean(str(mc['filt'])))
    return json.dumps(matches), cleaned_texts

def process_column_struct(text):
    json_matches, cleaned_texts = process_column_json(text)
    return {"matches": json_matches, "cleaned": cleaned_texts}

# Define the return schema for the UDF
result_schema = StructType([
    StructField("matches", StringType(), True),     # JSON string of match_count results
    StructField("cleaned", ArrayType(StringType()), True)  # List of cleaned sentences
])

process_udf = F.udf(process_column_struct, result_schema)

# --- Process DataFrame Columns ---
# Assume 'df' is your existing Spark DataFrame analogous to currdf1.
req_cols = ['FILT_MD', 'FILT_QA', 'FILT_CEO_MD', 'FILT_EXEC_MD', 'FILT_CEO_QA', 'FILT_EXEC_QA', 'FILT_ANL_QA']

for col in req_cols:
    # Apply the UDF and create a temporary struct column
    df = df.withColumn("temp_" + col, process_udf(F.col(col)))
    # Extract match results (as JSON) and cleaned text lists into separate columns
    df = df.withColumn("matches_" + col, F.col("temp_" + col + ".matches"))
    df = df.withColumn(col, F.col("temp_" + col + ".cleaned"))
    df = df.drop("temp_" + col)

# Display the updated DataFrame
df.show(truncate=False)

```


DeepSeek response:
```
from pyspark.sql import functions as F
from pyspark.sql.types import *
import spacy
from spacy.lang.en import English
from collections import Counter
import numpy as np
import pandas as pd

# Initialize spaCy models
nlp = spacy.load("en_core_web_sm", disable=['parser'])
nlp.Defaults.stop_words -= {"bottom", "top", "Bottom", "Top", "call"}
nlp.max_length = 1000000000

sent_tokenizer = English()
sent_tokenizer.add_pipe("sentencizer")

# Broadcast the spaCy model (handled via singleton pattern)
class SpacyManager:
    nlp = None
    sent_tokenizer = None
    
    @classmethod
    def get_nlp(cls):
        if not cls.nlp:
            cls.nlp = spacy.load("en_core_web_sm", disable=['parser'])
            cls.nlp.Defaults.stop_words -= {"bottom", "top", "Bottom", "Top", "call"}
            cls.nlp.max_length = 1000000000
        return cls.nlp
    
    @classmethod
    def get_sent_tokenizer(cls):
        if not cls.sent_tokenizer:
            cls.sent_tokenizer = English()
            cls.sent_tokenizer.add_pipe("sentencizer")
        return cls.sent_tokenizer

# Helper functions
def word_tokenize(doc):
    nlp = SpacyManager.get_nlp()
    return [token.lemma_.lower() for token in nlp(doc) 
            if not token.is_stop and not token.is_punct and token.pos_ != 'NUM']

def match_tokenize(doc):
    nlp = SpacyManager.get_nlp()
    tokens = []
    for token in nlp(doc):
        if token.pos_ == 'PROPN' or token.text[0].isupper():
            tokens.append(token.text.lower())
            continue
        if not token.is_stop and not token.is_punct and token.pos_ != 'NUM':
            tokens.append(token.lemma_.lower())
    return tokens

def find_ngrams(input_list, n):
    return zip(*[input_list[i:] for i in range(n)])

def process_sentence(sent):
    replacements = {
        "Thank you": "", "thank you": "", "thanks": "", "Thanks": "",
        "earnings call": "", "earnings release": "", "earnings conference": ""
    }
    for k, v in replacements.items():
        sent = sent.replace(k, v)
    if len(sent.split()) < 5:
        return None
    low = sent.lower()
    if any(phrase in low for phrase in ["good morning", "good afternoon", "good evening", "welcome"]):
        return None
    return sent

# UDF definitions
def sent_tokenize_udf(doc):
    tokenizer = SpacyManager.get_sent_tokenizer()
    return [process_sentence(sent.text) for sent in tokenizer(doc).sents if process_sentence(sent.text) is not None]

sent_tokenize = F.udf(sent_tokenize_udf, ArrayType(StringType()))

# Read match list
match_pd = pd.read_csv("/dbfs/mnt/access_work/UC25/Embeddings/Word lists/fundamental_final_draft.csv")
word_set_dict = {}
for topic in match_pd['label'].unique():
    words = match_pd[match_pd['label'] == topic]['word'].values
    word_set_dict[topic] = {
        'unigrams': set([match_tokenize(w)[0] if len(w.split()) == 1 else w.lower() for w in words]),
        'bigrams': set(['_'.join(match_tokenize(w)) if len(w.split()) == 2 else w.lower().replace(' ', '_') for w in words]),
        'phrases': [w.lower() for w in words if len(w.split()) > 2]
    }
broadcast_word_set = F.broadcast(word_set_dict)

# Match count UDF
def match_count_udf(text_col, phrases_flag):
    def _match_count(text):
        match_sets = broadcast_word_set.value
        unigrams = word_tokenize(text)
        bigrams = ['_'.join(ngram) for ngram in find_ngrams(unigrams, 2)]
        
        counts = {}
        for label in match_sets:
            uni_count = len([w for w in unigrams if w in match_sets[label]['unigrams']])
            bi_count = len([b for b in bigrams if b in match_sets[label]['bigrams']])
            phrase_count = len([p for p in match_sets[label]['phrases'] if p in text.lower()]) if phrases_flag else 0
            counts[label] = {
                'uni': uni_count,
                'bi': bi_count,
                'total': uni_count + bi_count + (phrase_count if phrases_flag else 0)
            }
        return {
            'filt': text,
            'len': len(unigrams),
            'raw_len': len(text.split()),
            'counts': counts
        }
    return F.udf(_match_count, MapType(StringType(), MapType(StringType(), MapType(StringType(), IntegerType())))

# Text clean UDF
def text_clean_udf():
    return F.udf(lambda x: x.strip().lower() if x else None, StringType())

# Process columns
req_cols = ['FILT_MD','FILT_QA', 'FILT_CEO_MD', 'FILT_EXEC_MD', 'FILT_CEO_QA', 'FILT_EXEC_QA', 'FILT_ANL_QA']

for col in req_cols:
    currdf = currdf.withColumn(f"sentences_{col}", sent_tokenize(F.col(col)))
    
    # Process sentences and matches
    currdf = currdf.withColumn(
        f"matches_{col}",
        F.transform(
            F.col(f"sentences_{col}"),
            match_count_udf(col, phrases_flag=False)
        )
    )
    
    # Clean text
    currdf = currdf.withColumn(
        col,
        F.transform(
            F.col(f"matches_{col}"),
            lambda x: text_clean_udf()(x["filt"])
        )
    )
    
    # Optional: Drop intermediate columns
    currdf = currdf.drop(f"sentences_{col}")

# Display results
currdf.show()
```