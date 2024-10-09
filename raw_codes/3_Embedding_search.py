import seaborn as sns
from gensim.models import Word2Vec
import spacy
from spacy.lang.en import English
import pandas as pd
import numpy as np
import spacy
import tqdm
from tqdm import tqdm
tqdm.pandas()
import sklearn.datasets
import plotly.express as px
from gensim.models import Phrases
from collections import Counter


nlp = spacy.load("en_core_web_sm", disable = ['parser'])

# Excluding certain words from being registered as stop words. These words are relevant in call transcripts/financial documents. 
nlp.Defaults.stop_words -= {"bottom", "top", "Bottom", "Top", "call"}

# To avoid errors with processing lengthy docs. Not necessary here but we keep this for consistency across notebooks.
nlp.max_length = 1000000000


# Returns lemmatized document
def wordTokenize(doc):
  
  return [ent.lemma_.lower() for ent in nlp(doc) if not ent.is_stop and not ent.is_punct and ent.pos_ != 'NUM']


def find_ngrams(input_list, n):
  return zip(*[input_list[i:] for i in range(n)])


# Breaks down sentence into the most representative subset of unigrams/bigrams that belong in model vocabulary.
def get_model_ngrams(x, model):
 # print(x)
  unigrams = wordTokenize(x)
  vocab = {word: 0 for word in set(unigrams)}
  bigrams = [g for g in find_ngrams(unigrams, 2)]
  
  prev_removed = False
  if len(bigrams)>0:
    if '_'.join(bigrams[0]) in model.wv:
      unigrams.remove(bigrams[0][0])
      unigrams.remove(bigrams[0][1])
      unigrams.append('_'.join(bigrams[0]))
      prev_removed = True
  
  for bigram in bigrams[1:]:
    if '_'.join(bigram) in model.wv:
      
      unigrams.remove(bigram[1])
      unigrams.append('_'.join(bigram))
      
      if not prev_removed:
        unigrams.remove(bigram[0])
        prev_removed = True
    
  else:
      prev_removed = False
        
  return unigrams 



# Embeds unigram/bigram/phrase
def embed(x, model):
  
  if '_' in x:
    try:
      return model.wv[x]
    except:
      None
  unigrams = wordTokenize(x)
  vocab = {word: 0 for word in set(unigrams)}
  bigrams = [g for g in find_ngrams(unigrams, 2)]
  
  prev_removed = False
  if len(bigrams)>0:
    if '_'.join(bigrams[0]) in model.wv:
      unigrams.remove(bigrams[0][0])
      unigrams.remove(bigrams[0][1])
      unigrams.append('_'.join(bigrams[0]))
      prev_removed = True
  
  for bigram in bigrams[1:]:
    if '_'.join(bigram) in model.wv:
      
      unigrams.remove(bigram[1])
      unigrams.append('_'.join(bigram))
      
      if not prev_removed:
        unigrams.remove(bigram[0])
        prev_removed = True
    
  else:
      prev_removed = False
        
#  print(unigrams) 
  try:
    return np.mean(np.stack([model.wv[phrase] for phrase in unigrams if phrase in model.wv]), axis = 0)
  except:
    try:
      return model.wv[x]
    except:
      return None
    
  
    
# Finds nearest matching words to set
def nearest(words, model, num_neigh = 50, filename = False, regularize = False):
  
  alist = {'label': [], 'embed': [], 'match': [], 'sim': []}
  for topic in set(words['label']):

    topic_embed = [[word[0], model.wv[word[0]], word[1]] for word in model.wv.most_similar_cosmul(positive = words[words['label']==topic]['match'].apply(lambda x: [y for y in get_model_ngrams(x, model) if y in model.wv] if x not in model.wv else [x]).sum(), topn = num_neigh)]
    topic_embed_norm = [[word[0], model.wv[word[0]], word[1]] for word in model.wv.most_similar(positive = words[words['label']==topic]['match'].apply(lambda x: [y for y in get_model_ngrams(x, model) if y in model.wv] if x not in model.wv else [x]).sum(), topn = num_neigh)]
    
    alist['label'] = alist['label'] + [topic for i in range(num_neigh)]
    if regularize:
      alist['embed'] = alist['embed'] + [embed[1] for embed in topic_embed]
      alist['match'] = alist['match'] + [word[0] for word in topic_embed]
      alist['sim'] = alist['sim'] + [word[2] for word in topic_embed]
    else:
      alist['embed'] = alist['embed'] + [embed[1] for embed in topic_embed_norm]
      alist['match'] = alist['match'] + [word[0] for word in topic_embed_norm]
      alist['sim'] = alist['sim'] + [word[2] for word in topic_embed_norm]

 #   print("Inclusions ", topic, set([word[0] for word in topic_embed]) - set([word[0] for word in topic_embed_norm]))
 #   print("Exclusions ", topic, set([word[0] for word in topic_embed_norm]) - set([word[0] for word in topic_embed]))

  # print(len(alist['Code']), len(alist['embed']), len(alist['match']))
  tdf = pd.DataFrame(alist)
  if filename:
    tdf.to_csv("/dbfs/mnt/access_work/UC25/Embeddings/Output Word lists/" + filename + "_neighbors_n" + str(num_neigh) + ".csv" )
  return tdf



# Create interactive viz. of dataframe
def umap_viz(df, marker_size = None, save_to = None):
  
  mapper = umap.UMAP().fit_transform(np.stack(df['embed']))
#  print(mapper[:,0])
  
  df['x'] = mapper[:,0]
  df['y'] = mapper[:,1]
  fig = px.scatter(df, x = 'x' , y = 'y', color = 'label', hover_data = ['match'])
  fig.update_layout(
    autosize=False,
    width=1000,
    height=800,)
  if marker_size:
    fig.update_traces(marker_size = marker_size)
  if save_to:
    fig.write_html(save_to)
  fig.show()
  
  # Load Word2Vec model and seed list
model = Word2Vec.load(dbutils.widgets.get("Model path"))
seed = pd.read_csv(dbutils.widgets.get("Seed list path"))

  # Embed seed words and set 'seed' flag = True, output seed viz.
seed['embed'] = seed['match'].apply(lambda x: embed(x, model))
seed = seed[seed['embed'].notna()]
seed['seed'] = True
#print(seed)
if len(seed.index) >=8: 
  umap_viz(seed, marker_size = 8, save_to = dbutils.widgets.get('Output list path').split('.')[0].replace('_expanded', '') + '_seed_fig.html')


  def process_dataframe(self, currdf) -> pd.DataFrame:
    currdf['CALL_ID'] = currdf['CALL_ID'].apply(lambda x: str(x))
    currdf["FILT_DATA"] = currdf.apply(lambda row: ast.literal_eval(row['FILT_MD']) + ast.literal_eval(row['FILT_QA']),axis= 1)
    currdf = (currdf[['ENTITY_ID', 'FILT_DATA', 'COMPANY_NAME', 'CALL_NAME', 'UPLOAD_DT_UTC', 'EVENT_DATETIME_UTC','VERSION_ID','CALL_ID']]
                    .sort_values(by = 'UPLOAD_DT_UTC')
                    .drop_duplicates(subset = ['ENTITY_ID', 'COMPANY_NAME', 'CALL_ID'], 
                                    keep = 'first'))
    
    # currdf['FILT_DATA'] = currdf['FILT_DATA'].apply(lambda x: self.text_processor.lemmatize_and_tokenize_sentences(x))
    ddf = dd.from_pandas(currdf, npartitions=16)
    ddf['FILT_DATA'] = ddf['FILT_DATA'].apply(lambda x: self.text_processor.lemmatize_and_tokenize_sentences(x), meta=('FILT_DATA', object))
    processed_df = ddf.compute()
    self.sparkdf_util.cleanup(ddf)
    return  processed_df #processed_df
  

search = nearest(seed, model, regularize = True)
search['seed'] = False

exp = pd.concat([search, seed])
exp.reset_index(inplace = True)
exp.drop('index', axis = 1, inplace = True)


# Save expanded word list
exp.drop(['embed', 'x', 'y'], axis = 1, inplace = True)
exp.to_csv(dbutils.widgets.get('Output list path'))