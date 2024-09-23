# Databricks notebook source
!pip install gensim==4.2.0


# COMMAND ----------

# MAGIC %run ./../preprocessor/topicx_preprocessor

# COMMAND ----------

import numpy as np
from gensim.models import Phrases
from collections import Counter

# COMMAND ----------

"""
NAME : TopicXModel

DESCRIPTION:
This module serves the below functionalities:
                CREATES FINBERT PIPELINE
                PREDICTION
              
"""
class TopicXModel:
  def __init__(self):
    self.topicx_preprocessor=TopicXPreprocessor()
  
  def get_match_set(self,matches):
    """
    Generates the match set
    
    Parameters:
    argument1 (list): matches
   
    Returns:
    dictionary of unigrams and bigrams
    
    """
    bigrams = set([word.lower() for word in matches if len(word.split('_'))==2] + [word.lower().replace(" ", '_') for word in matches if len(word.split(' '))==2] + ['_'.join(self.topicx_preprocessor.tokenizer_for_matched_words(word)) for word in matches if len(word.split(' '))==2])

    unigrams = set([self.topicx_preprocessor.tokenizer_for_matched_words(match)[0] for match in matches if ('_' not in match) and (len(match.split(' '))==1)] + [match.lower() for match in matches if ('_' not in match) and (len(match.split(' '))==1)])

    phrases = [phrase.lower() for phrase in matches if len(phrase.split(" "))>2]
    return {'original': matches, 'unigrams': unigrams, 'bigrams': bigrams, 'phrases': phrases}
  
  def match_count(self,text, match_sets, phrases = True):
    """
    Generates the count dictionary with matched count of unigrams and bigrams
    
    Parameters:
    argument1 (str): text
    argument2 (dictionary): match sets
   
    Returns:
    dictionary of unigrams and bigrams count with topic label
    
    """
    unigrams = self.topicx_preprocessor.word_tokenizer(text)
    vocab = {word: 0 for word in set(unigrams)}
    bigrams = ['_'.join(g) for g in self.topicx_preprocessor.find_ngrams(unigrams, 2)]

    count_dict = {label : {match: 0 for match in match_set['unigrams'].union(match_set['bigrams']) } for label, match_set in match_sets.items()}

    unigram_count = {label : 0 for label, match_set in match_sets.items()}
    bigram_count = {label : 0 for label, match_set in match_sets.items()}

    if phrases:
      phrase_count = {label : 0 for label, match_set in match_sets.items()}
    total_count = {label : 0 for label, match_set in match_sets.items()}

    for label, match_set in match_sets.items(): 
      for word in unigrams:
        if word in match_set['unigrams']:
          count_dict[label][word]+=1
          unigram_count[label]+=1
      for word in bigrams:
        if word in match_set['bigrams']:
          count_dict[label][word]+=1
          bigram_count[label]+=1

      if phrases:
        phrase_count[label] = sum([1 if phrase in text.lower() else 0 for phrase in match_set['phrases']])    
        total_count[label] = unigram_count[label] + bigram_count[label] + phrase_count[label]

      else:
        total_count[label] = unigram_count[label] + bigram_count[label]

    if phrases:
      ret = {label : {'uni': unigram_count[label], 'bi' : bigram_count[label], 'phrase': phrase_count[label], 'total': total_count[label], 'stats' : count_dict[label]} for label in match_sets.keys()}

    else:
      ret = {label : {'uni': unigram_count[label], 'bi' : bigram_count[label], 'total': total_count[label], 'stats' : count_dict[label]} for label in match_sets.keys()}

    ret['len'] = len(unigrams)
    ret['raw_len'] = len(text.split(' '))

   ## REMOVE IF NOT NECESSARY
    ret['filt'] = text 

    return ret

  def mergeCount(self,x):
    """
    Used to merge dictionaries that keep track of word counts
   
       
    """

    try:
      merge = Counter(x[0])

      for calc in x[1:]:

        merge = merge + Counter(calc)
      if len(merge.keys())==0:
        return {'NO_MATCH': 1}
      return merge
    except:
      return {'ERROR': 1}

  def sentscore(self,a, b, weight = True):

    """ number of relevant sentences"""

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

  def netscore(self,a, b):

    """number of relevant sentences"""

    length = len(a)
    if length==0:
        return None
    if length!=len(b):
        return None
    num = len([x for x in a if x>0])

    if num==0:
        return None

    return np.dot([1 if x>0 else 0 for x in a], b)
  
  def generate_match_count(self,currdf,word_set_dict):
    """
    Generates the match count using the topics data
    
    Parameters:
    argument1 (dataframe): dataframe
   
    Returns:
    dataframe
    
    """
    for label in config.FILT_sections:
        currdf['matches_' + label]=currdf[label].apply(lambda x: [self.match_count(sent, word_set_dict, phrases = False) for sent in x],meta = ('matches_' + label, object))
    return currdf
  
      
  def generate_topic_statistics(self,currdf,word_set_dict):
    """
    Generates the new columns with topic total count stats
    
    Parameters:
    argument1 (dataframe): dataframe
   
    Returns:
    dataframe
    
    """
    for label in config.FILT_sections:
      currdf['LEN_' + label] = currdf['matches_' + label].apply(lambda x: [calc['len'] for calc in x]) 

      currdf['RAW_LEN_' + label] = currdf['matches_' + label].apply(lambda x: [calc['raw_len'] for calc in x]) 

      for topic in word_set_dict.keys():
        currdf[topic + '_TOTAL_' + label] = currdf['matches_' + label].apply(lambda x: [calc[topic]['total'] for calc in x])
        currdf[topic+ '_STATS_' + label] = currdf['matches_' + label].apply(lambda x: dict(self.mergeCount([calc[topic]['stats'] for calc in x])))
        currdf[topic + '_STATS_LIST_' + label] = currdf['matches_' + label].apply(lambda x: [dict(calc[topic]['stats']) for calc in x])
      currdf['NUM_SENTS_' + label] = currdf['LEN_' + label].apply(lambda x: len(x))
 
      currdf.drop(['matches_' + label], axis = 1,inplace=True)
    return currdf
  
  
  def generate_sentence_relevance_score(self,currdf,word_set_dict):
    """
    Generates Relevance scores for sentence wise
    
    Parameters:
    argument1 (dataframe): dataframe
   
    Returns:
    dataframe
    
    """
    for label in config.FILT_sections:
  
      currdf['SENT_' + label] = currdf['SENT_LABELS_' + label].apply(lambda x: float(np.sum(x)/len(x)) if len(x)>0 else None)
      currdf['NET_SENT_' + label] = currdf['SENT_LABELS_' + label].apply(lambda x: np.sum(x) if len(x)>0 else None)

      for topic in word_set_dict.keys():

        currdf[topic + '_RELEVANCE_' + label] = currdf[topic + '_TOTAL_' + label].apply(lambda x: len([a for a in x if a>0])/len(x) if len(x)>0 else None)
        currdf[topic + '_SENT_' + label] = currdf[[topic + '_TOTAL_' + label, 'SENT_LABELS_' + label]].apply(lambda x: self.sentscore(x[0], x[1], weight = False), axis = 1)


        currdf[topic + '_SENT_REL_' + label] = currdf[[topic + '_RELEVANCE_' + label,topic + '_SENT_' + label]].apply(lambda x: float(x[0] * x[1]) if x[1] else None, axis = 1)

        currdf[topic + '_SENT_WEIGHT_' + label] = currdf[[topic + '_STATS_LIST_' + label, 'SENT_LABELS_' + label]].apply(lambda x: self.sentscore([sum([1 if val>0 else 0 for val in stat.values()]) for stat in x[0]], x[1]), axis = 1)
        currdf[topic + '_SENT_WEIGHT_REL_' + label] = currdf[[topic + '_RELEVANCE_' + label,topic + '_SENT_WEIGHT_' + label]].apply(lambda x: float(x[0] * x[1]) if x[1] else None, axis = 1)

      # net_sent = #pos - #neg sentences
        currdf[topic + '_NET_SENT_' + label] = currdf[[topic+ '_TOTAL_' + label, 'SENT_LABELS_' + label]].apply(lambda x: self.netscore(x[0], x[1]),axis = 1)
    return currdf
  
  
    
