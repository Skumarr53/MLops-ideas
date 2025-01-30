# Databricks notebook source
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
# MAGIC %pip install dask==2023.10.1 
# MAGIC %pip install distributed==2023.10.1
# MAGIC %pip install torch==2.0.0
# MAGIC # %pip install cymem==2.0.8
# MAGIC # %pip install scikit-learn==1.1.0
# MAGIC # %pip install typer==0.7.0
# MAGIC # %pip install accelerate==0.26.0

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

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

from topic_modelling_package.reports import create_topic_dict,  replace_separator_in_dict_words #generate_topic_report

from topic_modelling_package.processing import transform_match_keywords_df

# COMMAND ----------

!pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.4.0/en_core_web_sm-3.4.0.tar.gz

# COMMAND ----------

client = initialize_dask_client(n_workers=16, threads_per_worker=1)
nlp = initialize_spacy()

# COMMAND ----------

tsQuery= ("select TOP 100 CALL_ID,ENTITY_ID, FILT_MD, FILT_QA, SENT_LABELS_FILT_MD, SENT_LABELS_FILT_QA, CALL_NAME,COMPANY_NAME,EARNINGS_CALL,ERROR,TRANSCRIPT_STATUS,UPLOAD_DT_UTC,VERSION_ID,EVENT_DATETIME_UTC,PARSED_DATETIME_EASTERN_TZ from EDS_PROD.QUANT_LIVE.CTS_FUND_COMBINED_SCORES_H t2 where not exists (select 1 from EDS_PROD.QUANT.MACRO_INNO_CTS_SANTHOSH_TEST t1 where  t1.CALL_ID = CAST(t2.CALL_ID AS VARCHAR(16777216)) and t1.ENTITY_ID = t2.ENTITY_ID and t1.VERSION_ID = t2.VERSION_ID) ORDER BY PARSED_DATETIME_EASTERN_TZ DESC;")

currdf = read_from_snowflake(tsQuery).toPandas()

# COMMAND ----------

check_pd_dataframe_for_records(currdf)

# COMMAND ----------

import ast 

transformations1 =  [
  ("CALL_ID","CALL_ID", str),
    ("FILT_MD", "FILT_MD", ast.literal_eval),
    ("FILT_QA", "FILT_QA", ast.literal_eval),
    ("SENT_LABELS_FILT_QA", "SENT_LABELS_FILT_QA", ast.literal_eval),
    ("SENT_LABELS_FILT_MD", "SENT_LABELS_FILT_MD", ast.literal_eval),
    ('LEN_FILT_MD', 'FILT_MD', len),
    ('LEN_FILT_QA', 'FILT_QA', len)]
currdf = df_apply_transformations(currdf, transformations1)

# COMMAND ----------

currdf.describe()

# COMMAND ----------

import pandas as pd 
match_df = pd.read_csv("/dbfs/mnt/access_work/UC25/Embeddings/Word lists/macro_inno_prod_v1.csv")
 
match_df.head(5)

# COMMAND ----------

# match_df = transform_match_keywords_df(match_df)

# COMMAND ----------

word_set_dict, negate_dict = create_topic_dict(match_df, nlp)

# COMMAND ----------

for key in negate_dict:
  print(key, len(negate_dict[key]))

# COMMAND ----------

for key in word_set_dict:
  print(key, len(word_set_dict[key]))

# COMMAND ----------

currdf.FILT_MD[0]

# COMMAND ----------

currdf.FILT_MD[0]

# COMMAND ----------

def wordTokenize(doc):
  
  return [ent.lemma_.lower() for ent in nlp(doc) if not ent.is_stop and not ent.is_punct and ent.pos_ != 'NUM']

def find_ngrams(input_list, n):
  return zip(*[input_list[i:] for i in range(n)])

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
# match_count_lowStat(currdf.FILT_MD[0], word_set_dict, phrases = False, suppress = negate_dict)


# COMMAND ----------

# from topic_modelling_package.processing.match_operations import create_match_patterns, count_matches_in_single_sentence

# count_matches_in_single_sentence(currdf.FILT_MD[0], word_set_dict,nlp,phrases = False, suppress = negate_dict)

# COMMAND ----------

currdf.to_csv("/Workspace/Users/santhosh.kumar3@voya.com/DEVELOPMENT/data/currdf_processed_data.csv", index = False)

# COMMAND ----------

import gc
import dask.dataframe as dd
from dask.diagnostics import ProgressBar
from loguru import logger
# from topic_modelling_package.processing.match_operations import create_match_patterns, count_matches_in_single_sentence
# from centralized_nlp_package.data_processing import df_apply_transformations
from typing import Dict, Any, List, Tuple, Optional, Union, Callable
from topic_modelling_package.reports import STATISTICS_MAP

def df_apply_transformations(
    df: Union[pd.DataFrame, dd.DataFrame],
    transformations: List[Tuple[str, Union[str, List[str]], Callable]],
) -> Union[pd.DataFrame, dd.DataFrame]:
    """
    Applies a set of transformations to a DataFrame based on the given list of transformation tuples.

    Each transformation tuple should contain:
        - new_column (str): The new column name to create or transform.
        - columns_to_use (Union[str, List[str]]): Column(s) to use in the transformation.
        - func (Callable): The transformation function.

    Args:
        df (Union[pd.DataFrame, dd.DataFrame]): The DataFrame to apply transformations on.
        transformations (List[Tuple[str, Union[str, List[str]], Callable]]): 
            A list of tuples where each tuple contains:
                - new_column (str): The new column name to create or transform.
                - columns_to_use (Union[str, List[str]]): Column(s) to use in the transformation.
                - func (Callable): The transformation function.

    Returns:
        Union[pd.DataFrame, dd.DataFrame]: The DataFrame with applied transformations.

    Raises:
        ValueError: If a transformation tuple is invalid.
        Exception: Re-raises any exception that occurs during the transformation process after logging.

    Example:
        >>> from centralized_nlp_package.data_processing import df_apply_transformations
        >>> def concat_columns(a, b):
        ...     return f"{a}_{b}"
        >>> data = {'col1': ['A', 'B'], 'col2': ['C', 'D']}
        >>> df = pd.DataFrame(data)
        >>> transformations = [
        ...     ('col3', ['col1', 'col2'], lambda row: concat_columns(row['col1'], row['col2']))
        ... ]
        >>> transformed_df = df_apply_transformations(df, transformations)
        >>> print(transformed_df)
          col1 col2 col3
        0    A    C  A_C
        1    B    D  B_D
    """
    for transformation in transformations:
        if len(transformation) != 3:
            logger.error(f"Invalid transformation tuple: {transformation}. Expected 3 elements.")
            raise ValueError(f"Invalid transformation tuple: {transformation}. Expected 3 elements.")

        new_column, columns_to_use, func = transformation

        if not callable(func):
            logger.error(f"Transformation function for column '{new_column}' is not callable.")
            raise ValueError(f"Transformation function for column '{new_column}' is not callable.")

        try:
            if isinstance(columns_to_use, str):
                # Single column transformation
                print(f"Applying transformation on single column '{columns_to_use}' to create '{new_column}'.")
                if isinstance(df, dd.DataFrame):
                    print(f"processing as dask opertaion.")
                    df[new_column] = df[columns_to_use].apply(func, meta=(new_column, object))
                else:
                    df[new_column] = df[columns_to_use].apply(func)
            elif isinstance(columns_to_use, list):
                # Multiple columns transformation
                print(f"Applying transformation on multiple columns {columns_to_use} to create '{new_column}'.")
                if isinstance(df, dd.DataFrame):
                    df[new_column] = df.apply(lambda row: func(row), axis=1, meta=(new_column, object))
                else:
                    df[new_column] = df.apply(lambda row: func(row), axis=1)
            else:
                logger.error(f"Invalid type for columns_to_use: {columns_to_use}. Expected str or list of str.")
                raise ValueError(f"Invalid type for columns_to_use: {columns_to_use}. Expected str or list of str.")

            print(f"Successfully applied transformation for '{new_column}'.")
        except Exception as e:
            logger.error(f"Error applying transformation for column '{new_column}': {e}")
            raise

    print("All transformations applied successfully.")
    return df
  
def generate_topic_report(
    df,#: pd.DataFrame,
    word_set_dict,#: #Dict[str, Any],
    negate_dict,#: #Dict[str, List[str]],
    nlp,#: spacy.Language,
    stats_list,#: List[str],
    phrases: bool = False,
    dask_partitions = 8,#: int = 4,
    label_column: str = "matches",
) -> pd.DataFrame:
    """
    Generates topic-specific columns for selected statistics.

    This function applies transformations to the DataFrame based on the provided statistics list.
    Supported statistics include:
        - 'total': Total counts of matches.
        - 'stats': Detailed statistics of matches.
        - 'relevance': Relevance scores.
        - 'count': Count of matches.
        - 'extract': Extracted matches.
        - 'sentiment': Sentiment analysis results.

    Args:
        df (pd.DataFrame): DataFrame containing the data to be processed.
        word_set_dict (Dict[str, Any]): Dictionary of word sets for different topics.
        negate_dict (Dict[str, List[str]]): Dictionary for negation handling.
        stats_list (List[str]): List of statistic identifiers to compute. Supported:
            ['total', 'stats', 'relevance', 'count', 'extract', 'sentiment']
        label_column (str, optional): Prefix for match labels in the DataFrame. Defaults to "matches".

    Returns:
        pd.DataFrame: Updated DataFrame with additional report columns for each topic and selected statistics.
    
    Raises:
        ValueError: If an unsupported statistic identifier is provided in stats_list.

    Example:
        >>> import pandas as pd
        >>> data = {
        ...     'matches_FILT_MD': [['good', 'bad'], ['good']],
        ...     'matches_FILT_QA': [['excellent', 'poor'], ['average']],
        ...     'FILT_MD': ['some text', 'other text'],
        ...     'FILT_QA': ['additional text', 'more text']
        ... }
        >>> df = pd.DataFrame(data)
        >>> word_set_dict = {
        ...     'POSITIVE': {'original': ['good'], 'unigrams': {'good'}, 'bigrams': {'good_service'}, 'phrases': []},
        ...     'NEGATIVE': {'original': ['bad'], 'unigrams': {'bad'}, 'bigrams': {'bad_service'}, 'phrases': []}
        ... }
        >>> negate_dict = {
        ...     'POSITIVE': ['not good'],
        ...     'NEGATIVE': ['not bad']
        ... }
        >>> stats_list = ['total', 'count']
        >>> report_df = generate_topic_report(df, word_set_dict, negate_dict, stats_list)
        >>> print(report_df.columns)
        Index(['matches_FILT_MD', 'matches_FILT_QA', 'FILT_MD', 'FILT_QA', 
               'POSITIVE_TOTAL_FILT_MD', 'POSITIVE_COUNT_FILT_MD', 
               'NEGATIVE_TOTAL_FILT_MD', 'NEGATIVE_COUNT_FILT_MD', 
               'POSITIVE_TOTAL_FILT_QA', 'POSITIVE_COUNT_FILT_QA', 
               'NEGATIVE_TOTAL_FILT_QA', 'NEGATIVE_COUNT_FILT_QA'], 
              dtype='object')
    """
    # Validate stats_list
    unsupported_stats = set(stats_list) - set(STATISTICS_MAP.keys())
    if unsupported_stats:
        raise ValueError(f"Unsupported statistics requested: {unsupported_stats}")
    

    ## convert to dataframe to Dask dataframe
    df = dd.from_pandas(df, npartitions = dask_partitions)

    # Initial transformations: match counts
    labels = ["FILT_MD", "FILT_QA"]
    # lab_sec_dict1 = [
    #     # (f"{label_column}_{lab}", lab, lambda x: count_matches_in_single_sentence(x, word_set_dict, nlp,  phrases = phrases, suppress=negate_dict))
    #      (f"{label_column}_{lab}", lab, lambda x: match_count_lowStat(x, word_set_dict,  phrases = phrases, suppress=negate_dict))
    #     for lab in labels
    # ]

    # print("Applying initial match count transformations.")
    # df = df_apply_transformations(df, lab_sec_dict1)

    for label, section in {'FILT_MD': 'FILT_MD', 'FILT_QA': 'FILT_QA'}.items():
      df['matches_' + label] = df[section].apply(lambda x: match_count_lowStat(x, word_set_dict, phrases = False, suppress = negate_dict), meta = ('matches_' + label, object))
    
    gc.collect()

    with ProgressBar():
        df = df.compute()

    # Iterate over labels and topics to apply selected statistics
    for label in labels:
        df['matches_' + label] = df['matches_' + label].apply(ast.literal_eval)
        for topic in word_set_dict.keys():
            lab_sec_dict2 = []
            for stat in stats_list:
                transformation_func = STATISTICS_MAP.get(stat)
                if transformation_func:
                    lab_sec_dict2.append(transformation_func(topic, label, label_column))
                else:
                    print(f"Statistic '{stat}' not found in STATISTICS_MAP.")
            if lab_sec_dict2:
                print(f"Applying transformations for topic '{topic}' and label '{label}'.")
                df = df_apply_transformations(df, lab_sec_dict2)

    # Drop intermediate match columns
    intermediate_cols = [f"{label_column}_{label}" for label in labels]
    df.drop(columns=intermediate_cols, inplace=True, errors='ignore')
    print(f"Dropped intermediate match columns: {intermediate_cols}")

    return df
  
currdf = generate_topic_report(currdf, word_set_dict, negate_dict, nlp, 
                               phrases = False,
                                   stats_list = ['total', 'stats','relevance', 'count', 'sentiment', 'net_sentiment'],
                                   dask_partitions = 8)

# COMMAND ----------

import inspect
source_file_path = inspect.getfile(read_from_snowflake)
source_file_path

# COMMAND ----------

currdf

# COMMAND ----------

