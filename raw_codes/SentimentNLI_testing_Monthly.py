# Databricks notebook source
# MAGIC %md
# MAGIC

# COMMAND ----------

# MAGIC %pip install /dbfs/mnt/access_work/packages/topic_modelling_package-0.1.0-py3-none-any.whl

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

from pyspark.sql import SparkSession 
from pyspark.sql.functions import udf, col, size, explode, collect_list, when, expr
from pyspark.sql.types import ArrayType, StringType, FloatType, IntegerType, StructType, StructField, TimestampType, MapType
import json
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import numpy as np
from pyspark.sql import functions as F
from pyspark.sql.functions import pandas_udf, PandasUDFType

from centralized_nlp_package.data_access import read_from_snowflake
from centralized_nlp_package.data_processing import initialize_spark_session,  convert_columns_to_timestamp ,pd_udf_wrapper
from centralized_nlp_package.nli_utils import initialize_nli_infer_pipeline
from topic_modelling_package.nli_process import  inference_summary, extract_inf, processing_nested_columns, literal_eval_safe #, inference_run

import topic_modelling_package

# COMMAND ----------

from topic_modelling_package.nli_process import parse_json_list, create_text_pairs
MODEL_FOLDER_PATH ="/dbfs/mnt/access_work/UC25/Libraries/HuggingFace/"
MODEL_NAME = "deberta-v3-large-zeroshot-v2"
LABELS = ['This text has the positive sentiment', 'This text has the negative sentiment', 'This text has the neutral sentiment']
labels_broadcast = spark.sparkContext.broadcast(LABELS)
parse_json_udf = udf(parse_json_list, ArrayType(StringType()))


from functools import partial
create_text_pairs_par = partial(create_text_pairs, labels = LABELS)

create_text_pairs_udf = udf(create_text_pairs_par, ArrayType(StructType([
                                                StructField("text", StringType(), nullable=False),
                                                StructField("topic", StringType(), nullable=False)
                                            ])))

# COMMAND ----------

N_PARTITION = 240
spark_ref = initialize_spark_session(app_name = "NLI_Inference", shuffle_partitions = N_PARTITION)

# COMMAND ----------

nli_pipeline = initialize_nli_infer_pipeline(model_path = MODEL_FOLDER_PATH + MODEL_NAME, enable_quantization = False)

# COMMAND ----------

start_date = dbutils.widgets.get("Start Date")
end_date = dbutils.widgets.get("End Date")

# COMMAND ----------

import pandas as pd
minDateNewQuery = start_date 
maxDateNewQuery = end_date

mind = f"'{minDateNewQuery}'"
maxd = f"'{maxDateNewQuery}'"

## call id column is a string  for patha table and call id has casted to str of the cts table
tsQuery = (f"SELECT CAST(CALL_ID AS STRING) AS CALL_ID, ENTITY_ID, DATE, FILT_MD, FILT_QA, CALL_NAME, COMPANY_NAME, "
           f"EARNINGS_CALL, ERROR, TRANSCRIPT_STATUS, UPLOAD_DT_UTC, VERSION_ID, "
           f"EVENT_DATETIME_UTC, PARSED_DATETIME_EASTERN_TZ, SENT_LABELS_FILT_MD, "
           f"SENT_LABELS_FILT_QA "
           f"FROM EDS_PROD.QUANT_LIVE.CTS_FUND_COMBINED_SCORES_H "
           f"WHERE DATE >= {mind} AND DATE < {maxd}" #mind
           f"ORDER BY PARSED_DATETIME_EASTERN_TZ DESC")



currdf_spark = read_from_snowflake(tsQuery)  # Assuming newDBFS is correctly initialized


# COMMAND ----------

tsQuery

# COMMAND ----------

import ast
from centralized_nlp_package.data_processing import sparkdf_apply_transformations

transformations1 = [
    ('FILT_MD', 'FILT_MD', parse_json_udf),
    ('FILT_QA', 'FILT_QA', parse_json_udf),
    ('LEN_FILT_MD', 'FILT_MD', size),
    ('LEN_FILT_QA', 'FILT_QA', size)
    ]

currdf_spark = sparkdf_apply_transformations(currdf_spark, transformations1)

# currdf_spark = (currdf_spark
#                     .dropDuplicates(['ENTITY_ID', 'EVENT_DATETIME_UTC'])
#                     .orderBy(col('UPLOAD_DT_UTC').asc()))


array_int_convert_udf = udf(literal_eval_safe, ArrayType(IntegerType()))
array_float_convert_udf = udf(literal_eval_safe, ArrayType(FloatType()))

transformations2 = [
    ('SENT_LABELS_FILT_MD', 'SENT_LABELS_FILT_MD', array_int_convert_udf),
    ('SENT_LABELS_FILT_QA', 'SENT_LABELS_FILT_QA', array_int_convert_udf),
    ('TEXT_PAIRS_MD', 'FILT_MD', create_text_pairs_udf),
    ('TEXT_PAIRS_QA', 'FILT_QA', create_text_pairs_udf)
]

currdf_spark = sparkdf_apply_transformations(currdf_spark, transformations2)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Run Inference

# COMMAND ----------

currdf_spark.columns

# COMMAND ----------

from typing import Iterator, List, Tuple, Dict, Any
from pyspark.sql.functions import pandas_udf, PandasUDFType
import pandas as pd


def inference_run(
    iterator: Iterator[List[str]],
    nli_pipeline,
    max_length: int = 512,
    batch_size: int = 32,
) -> Iterator[pd.Series]:
    """
    Performs inference on batches of text pairs using the NLI pipeline and yields the results as Pandas Series.
    
    Args:
        iterator (Iterator[List[str]]): An iterator where each element is a list of text pairs.
        max_length (int, optional): Maximum token length for each text input. Defaults to 512.
        batch_size (int, optional): Number of samples per batch for inference. Defaults to 32.
        enable_quantization (bool, optional): Whether to enable model quantization for faster inference. Defaults to False.
    
    Yields:
        Iterator[pd.Series]: An iterator yielding Pandas Series containing inference results for each batch.
    
    Example:
        >>> from topic_modelling_package.nli_process.inference import inference_udf
        >>> texts = [["I love this product</s></s>positive.", "This is bad</s></s>negative."]]
        >>> results = inference_udf(iter(texts))
        >>> for res in results:
        ...     print(res)
        0    [{'label': 'entailment', 'score': 0.9}, {'label': 'contradiction', 'score': 0.1}]
        1    [{'label': 'entailment', 'score': 0.7}, {'label': 'contradiction', 'score': 0.3}]
        dtype: object
    """
    
    for batch_num, batch in enumerate(iterator, start=1):
        # logger.info(f"Processing inference batch {batch_num} with {len(batch)} rows.")
        try:
            # Flatten the list of text pairs in the batch
            batch = batch.tolist()
            flat_text_pairs = [dict(text=pair['text'], text_pair=pair['topic']) for sublist in batch for pair in sublist]
            # logger.debug(f"Batch {batch_num}: Total text pairs to infer: {len(flat_text_pairs)}")
            
            if flat_text_pairs:
                # Perform inference in batch
                results = nli_pipeline(
                    flat_text_pairs,
                    padding=True,
                    top_k=None,
                    batch_size=batch_size,
                    truncation=True,
                    max_length=max_length
                )
                # logger.debug(f"Batch {batch_num}: Inference completed with {len(results)} results.")
            else:
                results = []
                # logger.warning(f"Batch {batch_num}: No text pairs to infer.")
            
            # Split results back to original rows
            split_results = []
            idx = 0
            for pairs in batch:
                pairs = pairs.tolist()
                if pairs:
                    pair_length = len(pairs)
                    split_results.append(results[idx:idx + pair_length])
                    idx += pair_length
                    # logger.debug(f"Batch {batch_num}: Split {pair_length} results for current row.")
                else:
                    split_results.append([])
                    # logger.debug(f"Batch {batch_num}: No pairs in current row. Appended empty list.")
            
            yield pd.Series(split_results)
        except Exception as e:
            # logger.error(f"Error in inference batch {batch_num}: {e}")
            raise Exception(f"Error in inference batch {batch_num}: {e}")

# Define the schema for the inference results
dict_schema = StructType([
    StructField("label", StringType(), False),
    StructField("score", FloatType(), False)
])




# Define the schema for the output of the UDF
inference_schema = ArrayType(ArrayType(dict_schema))

## Define the UDF
inference_udf_init = partial(inference_run, nli_pipeline = nli_pipeline, max_length=512, batch_size=64) 

infernece_udf_func = pd_udf_wrapper(inference_udf_init, inference_schema, udf_type=PandasUDFType.SCALAR_ITER)

currdf_spark = sparkdf_apply_transformations(
    currdf_spark,
    [
        ("MD_RESULT", "TEXT_PAIRS_MD", infernece_udf_func),
        ("QA_RESULT", "TEXT_PAIRS_QA", infernece_udf_func)
    ])


# COMMAND ----------

from topic_modelling_package.nli_process import inference_summary

inference_summary_partial = partial(inference_summary, labels = LABELS)

schema_summary = StructType([
    StructField("total_dict", MapType(StringType(), ArrayType(IntegerType())), False),
    StructField("score_dict", MapType(StringType(), ArrayType(FloatType())), False)
])

summary_udf_ref = udf(lambda texts, inference_result: inference_summary_partial(texts, inference_result), schema_summary)

# COMMAND ----------


transformations2 = [
    ("MD_SUMMARY", ["TEXT_PAIRS_MD", "MD_RESULT"], summary_udf_ref),
    ("QA_SUMMARY", ["TEXT_PAIRS_QA", "QA_RESULT"], summary_udf_ref)
]


currdf_spark = sparkdf_apply_transformations(currdf_spark, transformations2)

currdf_spark = (currdf_spark
    .withColumn("MD_FINAL_TOTAL", F.col("MD_SUMMARY.total_dict")) 
    .withColumn("MD_FINAL_SCORE", F.col("MD_SUMMARY.score_dict")) 
    .withColumn("QA_FINAL_TOTAL", F.col("QA_SUMMARY.total_dict")) 
    .withColumn("QA_FINAL_SCORE", F.col("QA_SUMMARY.score_dict")))

# COMMAND ----------

from topic_modelling_package.nli_process import processing_nested_columns

def get_section_extract_udf(section, threshold):
    par_func = partial(extract_inf, section = section,threshold = threshold)
    return udf(par_func, MapType(StringType(), StringType()))
    
def processing_nested_columns(spark_df,
                              fixed_columns = ['ENTITY_ID', 'CALL_ID', 'VERSION_ID', 'DATE', 'CALL_NAME', 'COMPANY_NAME','LEN_FILT_MD', 'LEN_FILT_QA', 'FILT_MD', 'FILT_QA', 'SENT_LABELS_FILT_MD', 'SENT_LABELS_FILT_QA'],
                               threshold=0.8):

    extract_tranformations = [(f"{ent}_FINAL_SCORE_EXTRACTED", [f"{ent}_FINAL_SCORE",f"LEN_FILT_{ent}"], get_section_extract_udf(f"FILT_{ent}", threshold)) for ent in ['MD', 'QA']]

    spark_df = sparkdf_apply_transformations(spark_df, extract_tranformations)
  
      # Extract the keys from the UDF output and create new columns
    md_final_score_extracted_cols = spark_df.select('MD_FINAL_SCORE_EXTRACTED').first().asDict()['MD_FINAL_SCORE_EXTRACTED'].keys()
    qa_final_score_extracted_cols = spark_df.select('QA_FINAL_SCORE_EXTRACTED').first().asDict()['QA_FINAL_SCORE_EXTRACTED'].keys()


    for col_name in md_final_score_extracted_cols:
        spark_df = spark_df.withColumn(col_name, col('MD_FINAL_SCORE_EXTRACTED').getItem(col_name))

    for col_name in qa_final_score_extracted_cols:
        spark_df = spark_df.withColumn(col_name, col('QA_FINAL_SCORE_EXTRACTED').getItem(col_name))
    spark_df = spark_df.drop('MD_FINAL_SCORE_EXTRACTED', 'QA_FINAL_SCORE_EXTRACTED')

    new_columns = [col.replace('.', '') for col in spark_df.columns]
    for old_col, new_col in zip(spark_df.columns, new_columns):
        spark_df = spark_df.withColumnRenamed(old_col, new_col)
    

    columns_filt =( fixed_columns + 
            [col.replace('.', '') for col in md_final_score_extracted_cols] + 
            [col.replace('.', '') for col in qa_final_score_extracted_cols] )

    spark_df = spark_df.select(*columns_filt)

    return spark_df
  
currdf_spark = processing_nested_columns(currdf_spark)

# COMMAND ----------

from topic_modelling_package.nli_process import convert_column_types, generate_label_columns

# COMMAND ----------

float_type_cols_ref = generate_label_columns(LABELS,  ['COUNT', 'REL'])
array_type_cols_ref = generate_label_columns(LABELS,  ['TOTAL', 'SCORE'])


currdf_spark = convert_column_types(currdf_spark, float_type_cols_ref, array_type_cols_ref)

# COMMAND ----------

currdf_spark.columns

# COMMAND ----------

def get_max_label(pos_scores, neu_scores, neg_scores):
    if not pos_scores or not neu_scores or not neg_scores:
        return []
    max_labels = []
    for pos, neu, neg in zip(pos_scores, neu_scores, neg_scores):
        if pos >= neu and pos >= neg:
            max_labels.append(1)
        elif neu >= pos and neu >= neg:
            max_labels.append(0)
        else:
            max_labels.append(-1)
    return max_labels

# Register the UDF
get_max_label_udf = udf(get_max_label, ArrayType(IntegerType()))

currdf_spark = currdf_spark.withColumn(
    'NLI_SENT_LABELS_FILT_MD',
    get_max_label_udf(
        col('This text has the positive sentiment_SCORE_FILT_MD'),
        col('This text has the neutral sentiment_SCORE_FILT_MD'),
        col('This text has the negative sentiment_SCORE_FILT_MD')
    )
)

currdf_spark = currdf_spark.withColumn(
    'NLI_SENT_LABELS_FILT_QA',
    get_max_label_udf(
        col('This text has the positive sentiment_SCORE_FILT_QA'),
        col('This text has the neutral sentiment_SCORE_FILT_QA'),
        col('This text has the negative sentiment_SCORE_FILT_QA')
    )
)

# COMMAND ----------

from topic_modelling_package.nli_process import rename_columns_by_label_matching

LABELS_MAPPING = {
    "This text has the positive sentiment": "POS",
    "This text has the neutral sentiment": "NEU",
    "This text has the negative sentiment": "NEG"
}

currdf_spark = rename_columns_by_label_matching(currdf_spark, LABELS_MAPPING)

# COMMAND ----------

from pyspark.sql.functions import col, when, array_union, array, concat

# COMMAND ----------

def combine_lists(md_col, qa_col, data_type):
    return when(col(md_col).isNotNull() & col(qa_col).isNotNull(), concat(col(md_col).cast(data_type), col(qa_col).cast(data_type))) \
           .when(col(md_col).isNotNull(), col(md_col).cast(data_type)) \
           .when(col(qa_col).isNotNull(), col(qa_col).cast(data_type)) \
           .otherwise(array().cast(data_type))

# COMMAND ----------

# Apply the function to create new columns
currdf_spark = currdf_spark.withColumn('FILT_ALL', combine_lists('FILT_MD', 'FILT_QA', 'array<string>'))
currdf_spark = currdf_spark.withColumn('SENT_LABELS_FILT_ALL', combine_lists('SENT_LABELS_FILT_MD', 'SENT_LABELS_FILT_QA', 'array<int>'))
currdf_spark = currdf_spark.withColumn('POS_SCORE_FILT_ALL', combine_lists('POS_SCORE_FILT_MD', 'POS_SCORE_FILT_QA', 'array<double>'))
currdf_spark = currdf_spark.withColumn('NEG_SCORE_FILT_ALL', combine_lists('NEG_SCORE_FILT_MD', 'NEG_SCORE_FILT_QA', 'array<double>'))
currdf_spark = currdf_spark.withColumn('NEU_SCORE_FILT_ALL', combine_lists('NEU_SCORE_FILT_MD', 'NEU_SCORE_FILT_QA', 'array<double>'))
currdf_spark = currdf_spark.withColumn('NLI_SENT_LABELS_FILT_ALL', combine_lists('NLI_SENT_LABELS_FILT_MD', 'NLI_SENT_LABELS_FILT_QA', 'array<int>'))

# COMMAND ----------

fixed_cols = ['ENTITY_ID', 'CALL_ID', 'VERSION_ID', 'DATE', 'CALL_NAME',
       'COMPANY_NAME', 'FILT_ALL', 'FIN_SENT_LABELS_FILT_ALL', 'NLI_SENT_LABELS_FILT_ALL',]
columns_order = fixed_cols +  ['POS_SCORE_FILT_ALL','NEU_SCORE_FILT_ALL','NEG_SCORE_FILT_ALL']

# COMMAND ----------

currdf_spark = currdf_spark.withColumnRenamed('SENT_LABELS_FILT_ALL', 'FIN_SENT_LABELS_FILT_ALL')

# COMMAND ----------

import numpy as np

# currdf_spark = currdf_spark.select(columns_order)

currdf_spark = currdf_spark.replace(np.nan, None)


currdf_spark = convert_columns_to_timestamp(currdf_spark, columns_formats = {'DATE': 'yyyy-MM-dd'})

# COMMAND ----------

# for i in range(len(currdf_pd)):
#   print(
#         # len(currdf_pd.SECTION_IDENTIFIER.iloc[i]), 
#         len(currdf_pd.FILT_ALL.iloc[i]), 
#         len(currdf_pd.POS_SCORE_FILT_ALL.iloc[i]), 
#         len(currdf_pd.FIN_SENT_LABELS_FILT_ALL.iloc[i]), 
#         len(currdf_pd.NEG_SCORE_FILT_ALL.iloc[i]),
#         len(currdf_pd.FILT_ALL.iloc[i]) - len(currdf_pd.POS_SCORE_FILT_ALL.iloc[i])
#         )

# COMMAND ----------

  # import time

# while True:
#   time.sleep(60)

# COMMAND ----------

# MAGIC %run /Workspace/Repos/santhosh.kumar3@voya.com/data-science-nlp-ml-common-code/impackage/utilities/config_utility

# COMMAND ----------

# MAGIC %run /Workspace/Repos/santhosh.kumar3@voya.com/data-science-nlp-ml-common-code/impackage/database/snowflake_dbutility

# COMMAND ----------

new_sf = SnowFlakeDBUtility(config.schema, config.eds_db_prod)

# COMMAND ----------

from pyspark.sql.functions import to_json

# COMMAND ----------

# currdf_spark = currdf_spark.withColumn('FILT_ALL', to_json(col('FILT_ALL'))) \
#                             .withColumn('FIN_SENT_LABELS_FILT_ALL', to_json(col('FIN_SENT_LABELS_FILT_ALL'))) \
#                             .withColumn('NLI_SENT_LABELS_FILT_ALL', to_json(col('NLI_SENT_LABELS_FILT_ALL'))) \
#                             .withColumn('POS_SCORE_FILT_ALL', to_json(col('POS_SCORE_FILT_ALL'))) \
#                             .withColumn('NEU_SCORE_FILT_ALL', to_json(col('NEU_SCORE_FILT_ALL'))) \
#                             .withColumn('NEG_SCORE_FILT_ALL', to_json(col('NEG_SCORE_FILT_ALL'))) 

# COMMAND ----------

tablename_curr = 'SANTHOSH_ECALL_NLI_SENTIMENT_SCORE_DEV_2025'
result_curr = new_sf.write_to_snowflake_table(currdf_spark, tablename_curr)

# COMMAND ----------

dbutils.notebook.exit()

# COMMAND ----------

currdf_pd = new_sf.read_from_snowflake("select * from EDS_PROD.QUANT.SANTHOSH_ECALL_NLI_SENTIMENT_SCORE_DEV_2025").toPandas()

# COMMAND ----------

import ast 
for col in currdf_pd.columns:
  try:
    currdf_pd[col] = currdf_pd[col].apply(ast.literal_eval)
  except:
    pass


# COMMAND ----------

currdf_pd.columns

for i in range(len(currdf_pd)):
  md_diff = (len(currdf_pd.FILT_MD.iloc[i])-len(currdf_pd.NEG_SCORE_FILT_MD.iloc[i]))>0 
  qa_diff = (len(currdf_pd.FILT_QA.iloc[i])-len(currdf_pd.NEG_SCORE_FILT_QA.iloc[i]))>0
  if md_diff:
  # if qa_diff:
  # if md_diff or qa_diff:
    print(
          i,
          len(currdf_pd.FILT_MD.iloc[i]), 
          len(currdf_pd.NEG_SCORE_FILT_MD.iloc[i]), 
          (len(currdf_pd.FILT_MD.iloc[i])-len(currdf_pd.NEG_SCORE_FILT_MD.iloc[i])),
          # len(currdf_pd.FILT_QA.iloc[i]), 
          # len(currdf_pd.NEG_SCORE_FILT_QA.iloc[i]),
          # (len(currdf_pd.FILT_QA.iloc[i])-len(currdf_pd.NEG_SCORE_FILT_QA.iloc[i]))
)

# COMMAND ----------

currdf_pd.columns

for i in range(len(currdf_pd)):
  md_diff = (len(currdf_pd.FILT_MD.iloc[i])-len(currdf_pd.NEG_SCORE_FILT_MD.iloc[i]))>0 
  qa_diff = (len(currdf_pd.FILT_QA.iloc[i])-len(currdf_pd.NEG_SCORE_FILT_QA.iloc[i]))>0
  # if md_diff:
  if qa_diff:
  # if md_diff or qa_diff:
    print(
          i,
          # len(currdf_pd.FILT_MD.iloc[i]), 
          # len(currdf_pd.NEG_SCORE_FILT_MD.iloc[i]), 
          # (len(currdf_pd.FILT_MD.iloc[i])-len(currdf_pd.NEG_SCORE_FILT_MD.iloc[i])),
          currdf_pd.FILT_QA.iloc[i],
          # len(currdf_pd.FILT_QA.iloc[i]), 
          len(currdf_pd.NEG_SCORE_FILT_QA.iloc[i]),
          (len(currdf_pd.FILT_QA.iloc[i])-len(currdf_pd.NEG_SCORE_FILT_QA.iloc[i]))
)

# COMMAND ----------

len(currdf_pd)

# COMMAND ----------

import ast

# currdf_pd = new_sf.read_from_snowflake("select * from SANTHOSH_ECALL_NLI_SENTIMENT_SCORE_DEV_4_FULLYEAR_TEST").toPandas().sample(100)

# currdf_pd = currdf_pd.sample(1000)

for col in currdf_pd.columns:
  try:
    currdf_pd[col] = currdf_pd[col].apply(ast.literal_eval)
  except:
    pass


for i in range(len(currdf_pd)):
  if len(currdf_pd.POS_SCORE_FILT_ALL.iloc[i]) != len(currdf_pd.FILT_ALL.iloc[i]):
    print(
          i,
          len(currdf_pd.FILT_ALL.iloc[i]), 
          len(currdf_pd.POS_SCORE_FILT_ALL.iloc[i]), 
          len(currdf_pd.FIN_SENT_LABELS_FILT_ALL.iloc[i]), 
          len(currdf_pd.NEG_SCORE_FILT_ALL.iloc[i]),
          len(currdf_pd.FILT_ALL.iloc[i]) - len(currdf_pd.POS_SCORE_FILT_ALL.iloc[i])
          )

# COMMAND ----------

for i in range(len(currdf_pd)):
  if len(currdf_pd.POS_SCORE_FILT_ALL.iloc[i]) != len(currdf_pd.FILT_ALL.iloc[i]):
    print(
          i,
          len(currdf_pd.FILT_ALL.iloc[i]), 
          len(currdf_pd.POS_SCORE_FILT_ALL.iloc[i]), 
          len(currdf_pd.FIN_SENT_LABELS_FILT_ALL.iloc[i]), 
          len(currdf_pd.NEG_SCORE_FILT_ALL.iloc[i]),
          len(currdf_pd.FILT_ALL.iloc[i]) - len(currdf_pd.POS_SCORE_FILT_ALL.iloc[i])
          )

# COMMAND ----------

currdf_pd.FILT_ALL.iloc[74]

# COMMAND ----------

