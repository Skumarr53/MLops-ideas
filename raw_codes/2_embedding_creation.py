import pyarrow.parquet as pq
from dask.distributed import Client
import dask.bag as db
from gensim.models import Word2Vec
from gensim.models import Phrases
import pickle
import dask.dataframe as dd
import dask.array as da
import gensim
import gc
import itertools
import pyarrow.parquet as pq
from pandas.tseries.offsets import MonthBegin, MonthEnd
from dateutil.relativedelta import relativedelta

# Dask parallelization setting. Set to roughly no. of CPU cores 
n_tasks = 32
client = Client(n_workers=n_tasks)


# Object to enable Snowflake access to prod Quant
myDBFS = DBFShelper()
new_sf = pd.read_pickle(r'/dbfs' + myDBFS.iniPath + 'mysf_prod_quant.pkl')


# Get the dates for REL table.
data_end_date = datetime.now() 
data_start_date = datetime.now() - relativedelta(years=5)


minDateNewQuery = (pd.to_datetime(format(data_start_date, '%m') + "-01-" + format(data_start_date, '%Y'))).strftime('%Y-%m-%d')
maxDateNewQuery = (pd.to_datetime(format(data_end_date, '%m') + "-01-" + format(data_end_date, '%Y'))).strftime('%Y-%m-%d')

mind = "'" + minDateNewQuery + "'"
maxd = "'" + maxDateNewQuery + "'"
print(mind, maxd)


tsQuery= ("SELECT FILT_DATA, ENTITY_ID, UPLOAD_DT_UTC,VERSION_ID, EVENT_DATETIME_UTC FROM EDS_PROD.QUANT.YUJING_CT_TL_STG_1 WHERE EVENT_DATETIME_UTC >= " + mind  + " AND EVENT_DATETIME_UTC < " + maxd  + " ;")

resultspkdf = new_sf.read_from_snowflake(tsQuery)

currdf = resultspkdf.toPandas()

currdf = currdf.sort_values(by = 'UPLOAD_DT_UTC').drop_duplicates(subset = ['ENTITY_ID', 'EVENT_DATETIME_UTC'], keep = 'first')

currdf['FILT_DATA'] = currdf['FILT_DATA'].apply(ast.literal_eval)

feed = list(itertools.chain.from_iterable(currdf['FILT_DATA'].tolist()))

# Num of sentences across all docs
print(len(feed))
gc.collect()

# Run model

# Threshold -> Freq. based score assigned to bigrams must cross this value to be considered within model vocab
# Vector size -> Number of dimensions to represent each word. 
# Window -> Modeling rolling window size.
# Min_count -> Min. no. of times a word/bigram needs to appear within corpus to be considered.
# Workers -> No. of threads to use for computation. 

if gen_bigram:
  bigram_transformer = Phrases(feed, threshold = 2)
  model = Word2Vec(sentences= bigram_transformer[feed], vector_size=300, window=5, min_count=45, workers = 32, epochs = 20)#min_count=30
else:
  model = Word2Vec(sentences= feed, vector_size=300, window=5, min_count=10, workers= 16, epochs = 15)

w2vmodel_path = '/dbfs/mnt/access_work/UC25/Topic Modeling/Embedding/word2vec_DATM_' + minDateNewQuery[2:4] + '_' + minDateNewQuery[5:7] + '_' +  maxDateNewQuery[2:4] + '_' + maxDateNewQuery[5:7] + '_v1.model'

model.save(w2vmodel_path)
