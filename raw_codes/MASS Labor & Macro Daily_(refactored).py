# Databricks notebook source
# MAGIC %md
# MAGIC #### MASS Labor and Macro topics daily dictionary-based pipeline
# MAGIC This notebook is used to update the MASS Labor and Macro company-level daily topic table. The output table will be used for generating weekly time series table for these topics
# MAGIC
# MAGIC Reads from: QUANT.PARTHA_FUND_CTS_STG_1_VIEW (historical backfilling only) and QUANT_LIVE.CTS_FUND_COMBINED_SCORES_H
# MAGIC
# MAGIC Writes to: QUANT.YUJING_MASS_LABOR_MACRO_DEV_2
# MAGIC
# MAGIC Match word list: /dbfs/mnt/access_work/UC25/Topic Modeling/NLI Models/MASS_labor_macro_dictionaries_final_v2.csv
# MAGIC
# MAGIC Recommended cluster: Any Standard D series cluster with 32gb RAM and 8 cores. (14.3 LTS runtime)
# MAGIC
# MAGIC Expected daily runtime: 10-25mins

# COMMAND ----------

# %pip install /dbfs/mnt/access_work/packages/topic_modelling_package-0.1.0-py3-none-any.whl


%pip install pyspark==3.5.0
%pip install loguru==0.7.2
%pip install hydra-core==1.3
%pip install python-dotenv==1.0.1
%pip install numpy==1.25.2
%pip install cryptography==43.0.1
%pip install gensim==4.3.3
%pip install Cython==0.29.32
%pip install spacy==3.5.0 #3.0.4
%pip install thinc==8.1.7
%pip install pandas==2.1.1
%pip install snowflake-connector-python==3.12.2
%pip install transformers==4.46.1
%pip install pyarrow==16.0.0
%pip install datasets==3.1.0
%pip install evaluate==0.4.3
%pip install pyspark==3.5.3
%pip install dask==2023.10.1 
%pip install distributed==2023.10.1
%pip install torch==2.0.0
%pip install mlflow==2.18.0
%pip install mlflow-skinny==2.18.0
# %pip install cymem==2.0.8
# %pip install scikit-learn==1.1.0
# %pip install typer==0.7.0
# %pip install accelerate==0.26.0

# COMMAND ----------

!pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.4.0/en_core_web_sm-3.4.0.tar.gz

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import pandas as pd

def calculate_spearman_correlation(df1: pd.DataFrame, df2: pd.DataFrame):
    """
    Calculate the Spearman correlation coefficient for each feature in two dataframes.

    Parameters:
    - df1: First dataframe
    - df2: Second dataframe

    Returns:
    - A dictionary with feature names as keys and their Spearman correlation coefficients as values.tsQuery
    """
    df1 = df1.reset_index(drop=True)
    df2 = df2.reset_index(drop=True)
    
    # Ensure both dataframes have the same columns
    if set(df1.columns) != set(df2.columns):
        raise ValueError("Dataframes must have the same columns for comparison.")

    # Calculate Spearman correlation for each feature
    correlations = {}
    for column in df1.columns:
      temp_df = pd.DataFrame({
                'df1': df1[column],
                'df2': df2[column]
            })
      
      # Drop rows where either value is NaN
      temp_df_clean = temp_df.dropna()
      valid_samples = len(temp_df_clean)

      if valid_samples < 2:
        print(
            f"Skipping {column}: Only {valid_samples} valid samples "
            f"(minimum required: {2})"
        )
        continue

      try:
        corr, _ = spearmanr(temp_df_clean['df1'], temp_df_clean['df2'])
        correlations[column] = corr
      except:
        pass
        

    return correlations

# COMMAND ----------

import pandas as pd
from scipy.stats import spearmanr
import numpy as np
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    IntegerType,
    LongType,
    FloatType,
    DoubleType,
    BooleanType,
    TimestampType,
    ArrayType,
    MapType,
    DataType,
)

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

from topic_modelling_package.reports import create_topic_dict,  replace_separator_in_dict_words, generate_topic_report

from topic_modelling_package.processing import transform_match_keywords_df

# COMMAND ----------

from dask.distributed import Client

# client = initialize_dask_client(n_workers=8, threads_per_worker=1)

nlp = initialize_spacy()

# COMMAND ----------

tsQuery = ("select CALL_ID,ENTITY_ID, DATE, FILT_MD, FILT_QA, SENT_LABELS_FILT_MD, SENT_LABELS_FILT_QA, CALL_NAME,COMPANY_NAME,EARNINGS_CALL,ERROR,TRANSCRIPT_STATUS,UPLOAD_DT_UTC,VERSION_ID,EVENT_DATETIME_UTC,PARSED_DATETIME_EASTERN_TZ from EDS_PROD.QUANT.CALL_TRANSCRIPT_5K_REF_TESTING ORDER BY PARSED_DATETIME_EASTERN_TZ DESC limit 100")


currdf = read_from_snowflake(tsQuery).toPandas()

# COMMAND ----------

check_pd_dataframe_for_records(currdf)

# COMMAND ----------

import ast 
# from typing import Any, Callable, List, Tuple, Union
# import dask.dataframe as dd


# def df_apply_transformations(
#     df: Union[pd.DataFrame, dd.DataFrame],
#     transformations: List[Tuple[str, Union[str, List[str]], Callable]],
# ) -> Union[pd.DataFrame, dd.DataFrame]:

#     for transformation in transformations:
#         if len(transformation) != 3:
#             print("Invalid transformation tuple: {transformation}. Expected 3 elements.")
#             raise ValueError(f"Invalid transformation tuple: {transformation}. Expected 3 elements.")

#         new_column, columns_to_use, func = transformation

#         if not callable(func):
#             print("Transformation function for column '{new_column}' is not callable.")
#             raise ValueError(f"Transformation function for column '{new_column}' is not callable.")

#         try:
#             if isinstance(columns_to_use, str):
#                 # Single column transformation
#                 # print("Applying transformation on single column '{columns_to_use}' to create '{new_column}'.")
#                 if isinstance(df, dd.DataFrame):
#                     df[new_column] = df[columns_to_use].map(func, meta=(new_column, object))
#                 else:
#                     df[new_column] = df[columns_to_use].apply(func)
#             elif isinstance(columns_to_use, list):
#                 # Multiple columns transformation
#                 print("Applying transformation on multiple columns {columns_to_use} to create '{new_column}'.")
#                 if isinstance(df, dd.DataFrame):
#                     df[new_column] = df.apply(lambda row: func(row), axis=1, meta=(new_column, object))
#                 else:
#                     df[new_column] = df.apply(lambda row: func(row), axis=1)
#             else:
#                 print("Invalid type for columns_to_use: {columns_to_use}. Expected str or list of str.")
#                 raise ValueError(f"Invalid type for columns_to_use: {columns_to_use}. Expected str or list of str.")

#             # print("Successfully applied transformation for '{new_column}'.")
#         except Exception as e:
#             print("Error applying transformation for column '{new_column}': {e}")
#             raise

#     print("All transformations applied successfully.")
#     return df
transformations1 =  [
  ("CALL_ID","CALL_ID", str),
    ("FILT_MD", "FILT_MD", ast.literal_eval),
    ("FILT_QA", "FILT_QA", ast.literal_eval),
    ("SENT_LABELS_FILT_QA", "SENT_LABELS_FILT_QA", ast.literal_eval),
    ("SENT_LABELS_FILT_MD", "SENT_LABELS_FILT_MD", ast.literal_eval),
    ('LEN_FILT_MD', 'FILT_MD', len),
    ('SENT_LABELS_FILT_QA', ['SENT_LABELS_FILT_QA','FILT_QA'], 
     lambda x: [x['SENT_LABELS_FILT_QA'][i] for i, sent in enumerate(x['FILT_QA']) if not sent.endswith('?')]),
     ('FILT_QA', 'FILT_QA', lambda x: [sent for sent in x if not sent.endswith('?')]),
     ('LEN_FILT_QA', 'FILT_QA', len)]
currdf = df_apply_transformations(currdf, transformations1)

# COMMAND ----------

match_df = pd.read_csv("/dbfs/mnt/access_work/UC25/Topic Modeling/NLI Models/MASS_labor_macro_dictionaries_final_v2.csv")

match_df.head(5)

# COMMAND ----------

currdf.to_csv("/Workspace/Users/santhosh.kumar3@voya.com/DEVELOPMENT/refactored_pipelines_testing/currdf.csv")
match_df.to_csv("/Workspace/Users/santhosh.kumar3@voya.com/DEVELOPMENT/refactored_pipelines_testing/match_df.csv")

# COMMAND ----------

match_df = transform_match_keywords_df(match_df)

# COMMAND ----------

word_set_dict, negate_dict = create_topic_dict(match_df, nlp)

# COMMAND ----------

negate_dict = replace_separator_in_dict_words(negate_dict)

# COMMAND ----------

currdf = generate_topic_report(currdf, word_set_dict, negate_dict, nlp, 
                               phrases = True,
                               stats_list = ['total', 'stats','relevance', 'count', 'sentiment'],
                               dask_partitions = 8)

# COMMAND ----------

currdf['DATE'] = pd.to_datetime(currdf['DATE'])

# COMMAND ----------



spark_parsedDF = pandas_to_spark(currdf, column_type_mapping = {'FILT_MD' :  ArrayType(StringType()),
                                                                'FILT_QA' :  ArrayType(StringType()),
                                                                '_len_' :  ArrayType(IntegerType()),
                                                                '_total_' :  ArrayType(IntegerType()),
                                                                '_count_' :  IntegerType(),
                                                                '_stats_' :  MapType(StringType(), IntegerType()),
                                                                'sent_scores' :  ArrayType(FloatType()),
                                                                'sent_labels' :  ArrayType(IntegerType())}, spark = spark)


# COMMAND ----------

import numpy as np
spark_parsedDF = spark_parsedDF.replace(np.nan, None)


spark_parsedDF = convert_columns_to_timestamp(spark_parsedDF, columns_formats = {'DATE': 'yyyy-MM-dd',
                                                                                 'PARSED_DATETIME_EASTERN_TZ': 'yyyy-MM-dd HH mm ss',
                                                                                 'EVENT_DATETIME_UTC': 'yyyy-MM-dd HH mm ss'})

# COMMAND ----------

# MAGIC %md
# MAGIC ### Evaluate results  

# COMMAND ----------

ori_df_raw = read_from_snowflake("select * from EDS_PROD.QUANT.SANTHOSH_MASS_LABandMACRO_DAILY_REF_TEST").toPandas()

numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

ori_df = ori_df_raw.select_dtypes(include=numerics)
ori_df[['ENTITY_ID', 'VERSION_ID']] = ori_df_raw[['ENTITY_ID', 'VERSION_ID']]

ori_df = ori_df.sort_values(by=['ENTITY_ID', 'VERSION_ID'])



# COMMAND ----------

currdf = currdf[ori_df.columns]
currdf = currdf.sort_values(by=['ENTITY_ID', 'VERSION_ID'])

# COMMAND ----------

# MAGIC %md
# MAGIC ### Compare Stats

# COMMAND ----------

ori_df.describe()

# COMMAND ----------

currdf.describe()

# COMMAND ----------

calculate_spearman_correlation(ori_df, currdf)

# COMMAND ----------

