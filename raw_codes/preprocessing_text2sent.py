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