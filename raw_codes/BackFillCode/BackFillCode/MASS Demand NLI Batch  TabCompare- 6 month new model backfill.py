# Databricks notebook source
# MAGIC %pip install --upgrade transformers
# MAGIC # %pip install loguru==0.7.2
# MAGIC %pip install mlflow==2.17.0

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Load Modules

# COMMAND ----------

# MAGIC %run "./../data-science-nlp-topic-modeling-pipelines/package_loader/Cloud_DB_module_Azure_SQL_dev_2_Yujing_git.py"

# COMMAND ----------

# MAGIC %run ./../data-science-nlp-ml-common-code/impackage/utilities/config_utility

# COMMAND ----------

# MAGIC %run ./../data-science-nlp-ml-common-code/impackage/database/snowflake_dbutility

# COMMAND ----------

# from loguru import logger
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col, size, explode, collect_list, when, expr
from pyspark.sql.types import ArrayType, StringType, FloatType, IntegerType, StructType, StructField, TimestampType
import json
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import numpy as np
from pyspark.sql import functions as F
from pyspark.sql.functions import pandas_udf, PandasUDFType

# COMMAND ----------

# MAGIC %md
# MAGIC ## Define Run Configuration 

# COMMAND ----------

N_PARTITION = 240
ENABLE_QUANTIZATION = False  # Set to False to disable quantization
BATCH_SIZE = 64
MODEL_REG_NAME = dbutils.widgets.get("Model_registry_name")

# COMMAND ----------

import logging, os
from logging.handlers import RotatingFileHandler

# Configure logging to write to a file in DBFS or another centralized location
# log_path = "/Workspace/Users/yujing.sun@voya.com/logfile.log"
# handler = RotatingFileHandler(log_path, maxBytes=10**7, backupCount=5)
logger = logging.getLogger("NLI_Inference")
logger.setLevel(logging.INFO)
# logger.addHandler(handler)

# COMMAND ----------

# # Step 2: Initialize Spark Session (if not already initialized)
spark = (SparkSession.builder.appName("Optimized_NLI_Inference")
         .config("spark.sql.shuffle.partitions",N_PARTITION)
         .config("spark.executor.resource.gpu.amount", "1")
         .config("spark.task.resource.gpu.amount", "0.8")
         .getOrCreate())

spark.sparkContext.setLogLevel("DEBUG")

# COMMAND ----------

# Step 3: Define Constants and Broadcast Variables
# MODEL_FOLDER_PATH = "/dbfs/mnt/access_work/UC25/Topic Modeling/NLI Models/Fine-tune NLI models/"
# MODEL_NAME = "FineTuned_RD_deberta-v3-large-zeroshot-v2_Consumer_Final"
LABELS = [
    "This text is about a weak consumer or reduced consumption",
    "This text is about a strong consumer or increased consumption"
]
labels_broadcast = spark.sparkContext.broadcast(LABELS)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Define Helper Functions

# COMMAND ----------

# Step 4: Define Helper Functions

# Function to parse JSON strings to lists
def parse_json_list(s):
    try:
        return json.loads(s)
    except Exception as e:
        logger.error(f"JSON parsing error: {e} for input: {s}")
        return []

# UDF to parse JSON lists
parse_json_udf = udf(parse_json_list, ArrayType(StringType()))

# Function to create text pairs for NLI
def create_text_pairs(filt):
    text_pairs = []
    for t in filt:
        for l in labels_broadcast.value:
            text_pairs.append(f"{t}</s></s>{l}.")
    return text_pairs

# UDF to create text pairs
create_text_pairs_udf = udf(create_text_pairs, ArrayType(StringType()))

# COMMAND ----------

# import mlflow

# model_name = dbutils.widgets.get("Model_registry_name")
# print(model_name)

# # Load the model from the staging branch of the model registry
# staging_model = mlflow.transformers.load_model(f"models:/{model_name}/Staging")

# # Access the model and tokenizer
# model = staging_model.model
# tokenizer = staging_model.tokenizer

# COMMAND ----------

# Define the model name
import mlflow

# Function to initialize the NLI pipeline on each executor
def initialize_nli_pipeline(enable_quantization=False, model_name=None, version = 'Staging'):
    try:
        # tokenizer = AutoTokenizer.from_pretrained(MODEL_FOLDER_PATH + MODEL_NAME)
        # model = AutoModelForSequenceClassification.from_pretrained(MODEL_FOLDER_PATH + MODEL_NAME)
        

        # Load the model from the staging branch of the model registry
        staging_model = mlflow.transformers.load_model(f"models:/{model_name}/{version}")

        # Access the model and tokenizer
        model = staging_model.model
        tokenizer = staging_model.tokenizer

        
        if enable_quantization:
            model = torch.quantization.quantize_dynamic(
                model, {torch.nn.Linear}, dtype=torch.qint8
            )
            logger.debug("Model quantization enabled.")
        else:
            logger.debug("Model quantization disabled.")
        
        device = 0 if torch.cuda.is_available() else -1
        nli_pipeline = pipeline(
            task="text-classification",
            model=model,
            tokenizer=tokenizer,
            device=device
        )
        logger.debug(f"NLI pipeline initialized on device: {'GPU' if device == 0 else 'CPU'}")
        return nli_pipeline
    except Exception as e:
        logger.error(f"Failed to initialize NLI pipeline: {e}")
        raise e

# Register the initialization function as a global variable to ensure it's loaded once per executor
nli_pipeline = None

# COMMAND ----------


def inference_summary(texts, inference_result, threshold=0.8):
    score_dict = {tp + '.': [] for tp in LABELS}
    total_dict = {tp + '.': [] for tp in LABELS}
    
    for i, (text_pair, inference) in enumerate(zip(texts, inference_result)):
        text1, text2_label = text_pair.split('</s></s>')
        for s in inference:
            if s['label'] == 'entailment':
                if s['score'] > threshold:
                    total_dict[text2_label].append(1)
                else:
                    total_dict[text2_label].append(0)
                score_dict[text2_label].append(s['score'])
    return total_dict, score_dict

# Define the schema for the UDF
schema_summary = StructType([
    StructField("total_dict", MapType(StringType(), ArrayType(IntegerType())), False),
    StructField("score_dict", MapType(StringType(), ArrayType(FloatType())), False)
])

#EDITED: Define UDF for parsing string representations of lists
parse_udf = udf(lambda x: ast.literal_eval(x) if x else [], ArrayType(StringType()))
summary_udf = udf(lambda texts, inference_result: inference_summary(texts, inference_result), schema_summary)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Fetch Input data

# COMMAND ----------

# MAGIC %md
# MAGIC ### Call transcript data

# COMMAND ----------

new_sf = SnowFlakeDBUtility(config.schema, config.eds_db_prod)

# COMMAND ----------

import pandas as pd
minDateNewQuery = pd.to_datetime(dbutils.widgets.get("Start Date")).strftime('%Y-%m-%d')
maxDateNewQuery = pd.to_datetime(dbutils.widgets.get("End Date")).strftime('%Y-%m-%d')

mind = f"'{minDateNewQuery}'"
maxd = f"'{maxDateNewQuery}'"

print(f'The next query spans {mind} to {maxd}')


#EDITED: Construct and execute the Snowflake query using Spark
tsQuery =  (f"SELECT CAST(CALL_ID AS STRING) AS CALL_ID, ENTITY_ID, DATE, FILT_MD, FILT_QA, CALL_NAME, COMPANY_NAME, "
           f"EARNINGS_CALL, ERROR, TRANSCRIPT_STATUS, UPLOAD_DT_UTC, VERSION_ID, "
           f"EVENT_DATETIME_UTC, PARSED_DATETIME_EASTERN_TZ, SENT_LABELS_FILT_MD, "
           f"SENT_LABELS_FILT_QA "
           f"FROM EDS_PROD.QUANT_LIVE.CTS_FUND_COMBINED_SCORES_H "
           f"WHERE DATE >= {mind} AND DATE < {maxd} " #mind
           f"ORDER BY PARSED_DATETIME_EASTERN_TZ DESC")

currdf_spark = new_sf.read_from_snowflake(tsQuery)  


# COMMAND ----------

# MAGIC %md
# MAGIC ### Load Market Cap Data

# COMMAND ----------

myDBFS_sql = DBFShelper_sql()
myDBFS_sql.get_DBFSdir_content(myDBFS_sql.iniPath)

# azSQL_LinkUp = pd.read_pickle(r'/dbfs/yujing' + myDBFS_sql.iniPath + 'my_azSQL_LinkUp.pkl')
azSQL_LinkUp = pd.read_pickle(r'/dbfs/FileStore/NLP_common/resources/my_azSQL_LinkUp.pkl')
azSQL_LinkUp.databaseName = 'QNT'
market_cap_df = azSQL_LinkUp.read_from_azure_SQL("qnt.p_coe.earnings_calls_mapping_table_mcap")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Input data prepartion

# COMMAND ----------

from pyspark.sql.functions import col, to_date, date_format, rank, when
from pyspark.sql.window import Window
from pyspark.sql import DataFrame


def classify_rank_C3(rank_col):
    return (when(rank_col <= 1500, 'top 1500')
            .otherwise('top 1501-3000'))
    
def filter_and_concat_transcripts(market_cap: DataFrame, currdf: DataFrame) -> DataFrame:
    # Get distinct YEAR_MONTH values
    year_months = currdf.select("YEAR_MONTH").distinct().collect()
    
    # Initialize an empty DataFrame for concatenating results
    concatenated_df = None
    
    for row in year_months:
        year_month = row['YEAR_MONTH']
        
        # Filter market_cap for the top 1500 companies for the current month
        top_companies = (market_cap
                         .filter(col("YEAR_MONTH") == year_month)
                         .orderBy(col("MCAP").desc())
                        #  .limit(1500)
                         .select("factset_entity_id"))
        
        # Filter currdf for transcripts of the top companies for the current month
        filtered_transcripts = (currdf
                                .filter((col("YEAR_MONTH") == year_month) & 
                                        (col("ENTITY_ID").isin([row.factset_entity_id for row in top_companies.collect()]))))
        
        # Concatenate the filtered transcripts to the result DataFrame
        if concatenated_df is None:
            concatenated_df = filtered_transcripts
        else:
            concatenated_df = concatenated_df.union(filtered_transcripts)
    
    return concatenated_df

# Market Data Processing
market_cap_df = market_cap_df.withColumn('YEAR_MONTH', date_format(to_date(col('date'), 'yyyy-MM-dd'), 'yyyy-MM'))
window_spec = Window.partitionBy('YEAR_MONTH').orderBy(col('MCAP').desc())
market_cap_df = market_cap_df.withColumn('MCAP_RANK', rank().over(window_spec))
market_cap_df = market_cap_df.withColumn('MCAP_GROUP', classify_rank_C3(col('MCAP_RANK')))
market_cap_df = market_cap_df.filter(col('MCAP_GROUP') == 'top 1500')

# Call Trsncript Processing
currdf_spark = currdf_spark.withColumn('YEAR_MONTH', date_format(to_date(col('date'), 'yyyy-MM-dd'), 'yyyy-MM'))
currdf_spark = currdf_spark.withColumn('row_num', rank().over(Window.partitionBy('ENTITY_ID', 'DATE').orderBy(col('UPLOAD_DT_UTC').desc()))).filter(col('row_num') == 1).drop('row_num')
currdf_spark = filter_and_concat_transcripts(market_cap_df, currdf_spark)

# COMMAND ----------

# sp_df = concatenated_currdf_df.toPandas()
# mf_df = market_cap_df.toPandas()

# COMMAND ----------

# sp_df = currdf_spark.toPandas()
# mf_df = market_cap_df.toPandas()

# def data_sanity_check(sp_df, mf_df):
#   for yr_mnt in sp_df.YEAR_MONTH.unique():
#     top_1500 = mf_df.factset_entity_id[(mf_df.YEAR_MONTH==yr_mnt) & (mf_df.MCAP_RANK<=1500)]
#     for ent in sp_df.ENTITY_ID[sp_df.YEAR_MONTH==yr_mnt]:
#       if ent not in top_1500.values:
#         print("ERROR: " + ent + " not in top 1500")

# data_sanity_check(sp_df, mf_df)

# COMMAND ----------

# Convert stringified lists to actual arrays
currdf_spark = (currdf_spark 
    .withColumn('FILT_MD', parse_json_udf(col('FILT_MD'))) 
    .withColumn('FILT_QA', parse_json_udf(col('FILT_QA'))) 
    # .withColumn('SENT_LABELS_FILT_MD', parse_json_udf(col('SENT_LABELS_FILT_MD'))) 
    # .withColumn('SENT_LABELS_FILT_QA', parse_json_udf(col('SENT_LABELS_FILT_QA'))) 
    .withColumn('LEN_FILT_MD', size(col('FILT_MD'))) 
    .withColumn('LEN_FILT_QA', size(col('FILT_QA'))) 
    .dropDuplicates(['ENTITY_ID', 'EVENT_DATETIME_UTC']) 
    .orderBy(col('UPLOAD_DT_UTC').asc()))


# COMMAND ----------

'SENT_LABELS_FILT_MD' in currdf_spark.columns

# COMMAND ----------

import ast
array_int_type_cols = ["SENT_LABELS_FILT_MD",
                       "SENT_LABELS_FILT_QA"]                

def literal_eval_safe(data_str):
    try:
        return ast.literal_eval(data_str)
    except (ValueError, SyntaxError):
        return None
      
array_int_convert_udf = udf(literal_eval_safe, ArrayType(IntegerType())) 

for col_name in array_int_type_cols:
    currdf_spark = currdf_spark.withColumn(col_name, array_int_convert_udf(currdf_spark[col_name]))

# COMMAND ----------

currdf_spark.select('SENT_LABELS_FILT_MD', 
                  'SENT_LABELS_FILT_QA').limit(10).show()

# COMMAND ----------

# Create text pairs for MD and QA
currdf_spark = currdf_spark \
    .withColumn('TEXT_PAIRS_MD', create_text_pairs_udf(col('FILT_MD'))) \
    .withColumn('TEXT_PAIRS_QA', create_text_pairs_udf(col('FILT_QA')))

# COMMAND ----------

# Define the schema for the inference results
dict_schema = StructType([
    StructField("label", StringType(), False),
    StructField("score", FloatType(), False)
])


# Define the schema for the output of the UDF
inference_schema = ArrayType(ArrayType(dict_schema))


@pandas_udf(inference_schema, PandasUDFType.SCALAR)
def inference_udf(texts: pd.Series) -> pd.Series:
    global nli_pipeline
    if nli_pipeline is None:
        nli_pipeline = initialize_nli_pipeline(enable_quantization=ENABLE_QUANTIZATION, model_name=MODEL_REG_NAME)

    # Prepare the texts for inference
    # flat_text_pairs = [pair for sublist in batch for pair in sublist]
    #     logger.debug(f"Batch {batch_num}: Total text pairs to infer: {len(flat_text_pairs)}")
    # processed_texts = texts.apply(lambda texts_list: [f"{it['text1']}</s></s>{it['text2']}" for it in texts_list])
    text_list = texts.tolist()
    flat_text_pairs = [pair for sublist in text_list for pair in sublist]
    # Perform inference in batch
    
    results = nli_pipeline(
        flat_text_pairs,
        padding=True,
        top_k=None,
        batch_size=BATCH_SIZE,  # Adjusted batch size
        truncation=True,
        max_length=512
    )
    # results = processed_texts.apply(pl_inference)
    logger.debug(f"Batch: Inference completed with {len(flat_text_pairs)} results.")
    # Split results back to original rows
    split_results = []
    idx = 0
    for pairs in text_list:
        if len(pairs):
            split_results.append(results[idx:idx+len(pairs)])
            idx += len(pairs)
        else:
            split_results.append([])

    return pd.Series(split_results)

# COMMAND ----------

# logger.info(f"Repartitioning DataFrame to {N_PARTITION} partitions for GPU workers.")
currdf_spark = currdf_spark.repartition(N_PARTITION)
# Step 9: Apply Inference UDFs to DataFrame

currdf_spark = (currdf_spark 
    .withColumn('MD_RESULT', inference_udf(col('TEXT_PAIRS_MD'))) 
    .withColumn('QA_RESULT', inference_udf(col('TEXT_PAIRS_QA'))))

currdf_spark.cache()

# COMMAND ----------


currdf_spark = currdf_spark \
    .withColumn("MD_SUMMARY", summary_udf(col("TEXT_PAIRS_MD"),col("MD_RESULT"))) \
    .withColumn("QA_SUMMARY", summary_udf(col("TEXT_PAIRS_QA"),col("QA_RESULT")))

# #EDITED: Extract summary fields
currdf_spark = (currdf_spark
    .withColumn("MD_FINAL_TOTAL", col("MD_SUMMARY.total_dict")) 
    .withColumn("MD_FINAL_SCORE", col("MD_SUMMARY.score_dict")) 
    .withColumn("QA_FINAL_TOTAL", col("QA_SUMMARY.total_dict")) 
    .withColumn("QA_FINAL_SCORE", col("QA_SUMMARY.score_dict")))

# COMMAND ----------

from pdb import set_trace

def extract_inf(row, section, section_len, threshold):
    count_col = {}
    rel_col = {}
    score_col = {}
    total_col = {}

    # set_trace()
    for tp, score in row.items():
        if section_len != 0:
            score_binary = [float(1) if s > threshold else float(0) for s in score]
            total_col[f'{tp}_TOTAL_{section}'] = score_binary 
            count_col[f'{tp}_COUNT_{section}'] = float(sum(score_binary))
            rel_col[f'{tp}_REL_{section}'] = sum(score_binary) / section_len
            score_col[f'{tp}_SCORE_{section}'] = [round(s, 4) for s in score]
        else:
            count_col[f'{tp}_COUNT_{section}'] = None
            rel_col[f'{tp}_REL_{section}'] = None
            total_col[f'{tp}_TOTAL_{section}'] = []
            score_col[f'{tp}_SCORE_{section}'] = []
    # print(count_col.keys())

    return {**count_col, **rel_col, **score_col, **total_col}

# extract_inf(pd_df.MD_FINAL_SCORE.iloc[0],'FILT_MD', pd_df.LEN_FILT_MD.iloc[0], 0.8)

# COMMAND ----------

from pyspark.sql.functions import udf, col, explode
from pyspark.sql.types import StructType, StructField, StringType, FloatType, ArrayType, MapType, DoubleType


# Register the UDF
extract_inf_udf = udf(extract_inf, MapType(StringType(), StringType()))#ArrayType(DoubleType())))

# Apply the UDF for MD_FINAL_SCORE and QA_FINAL_SCORE
currdf_spark = currdf_spark.withColumn(
    'MD_FINAL_SCORE_EXTRACTED',
    extract_inf_udf(col('MD_FINAL_SCORE'), lit('FILT_MD'), col('LEN_FILT_MD'), lit(0.8))
).withColumn(
    'QA_FINAL_SCORE_EXTRACTED',
    extract_inf_udf(col('QA_FINAL_SCORE'), lit('FILT_QA'), col('LEN_FILT_QA'), lit(0.8))
)

# COMMAND ----------


# Extract the keys from the UDF output and create new columns
md_final_score_extracted_cols = currdf_spark.select('MD_FINAL_SCORE_EXTRACTED').first().asDict()['MD_FINAL_SCORE_EXTRACTED'].keys()
qa_final_score_extracted_cols = currdf_spark.select('QA_FINAL_SCORE_EXTRACTED').first().asDict()['QA_FINAL_SCORE_EXTRACTED'].keys()

for col_name in md_final_score_extracted_cols:
    currdf_spark = currdf_spark.withColumn(col_name, col('MD_FINAL_SCORE_EXTRACTED').getItem(col_name))

for col_name in qa_final_score_extracted_cols:
    currdf_spark = currdf_spark.withColumn(col_name, col('QA_FINAL_SCORE_EXTRACTED').getItem(col_name))



# COMMAND ----------

# Drop the intermediate columns
currdf_spark = currdf_spark.drop('MD_FINAL_SCORE_EXTRACTED', 'QA_FINAL_SCORE_EXTRACTED')

new_columns = [col.replace('.', '') for col in currdf_spark.columns]
for old_col, new_col in zip(currdf_spark.columns, new_columns):
    currdf_spark = currdf_spark.withColumnRenamed(old_col, new_col)

columns_filt =(['ENTITY_ID', 'CALL_ID', 'VERSION_ID', 'DATE', 'CALL_NAME', 'COMPANY_NAME', 'UPLOAD_DT_UTC',  'PARSED_DATETIME_EASTERN_TZ', 'LEN_FILT_MD', 'LEN_FILT_QA', 'FILT_MD', 'FILT_QA', 'SENT_LABELS_FILT_MD', 'SENT_LABELS_FILT_QA'] + 
           [col.replace('.', '') for col in md_final_score_extracted_cols] + 
          [col.replace('.', '') for col in qa_final_score_extracted_cols] )
currdf_spark = currdf_spark.select(*columns_filt)

# COMMAND ----------


import ast
from typing import List, Tuple, Union, Callable, Any

# float_type_cols = ["This text is about consumer weakness_COUNT_FILT_MD",
#                     "This text is about consumer strength_COUNT_FILT_MD",
#                     "This text is about reduced consumer's spending patterns_COUNT_FILT_MD",
#                     "This text is about consumer weakness_REL_FILT_MD",
#                     "This text is about consumer strength_REL_FILT_MD",
#                     "This text is about reduced consumer's spending patterns_REL_FILT_MD",
#                     "This text is about consumer weakness_COUNT_FILT_QA",
#                     "This text is about consumer strength_COUNT_FILT_QA",
#                     "This text is about reduced consumer's spending patterns_COUNT_FILT_QA",
#                     "This text is about consumer weakness_REL_FILT_QA",
#                     "This text is about consumer strength_REL_FILT_QA",
#                     "This text is about reduced consumer's spending patterns_REL_FILT_QA"] 

# array_type_cols = ["This text is about consumer weakness_SCORE_FILT_MD",
#                   "This text is about consumer strength_SCORE_FILT_MD",
#                   "This text is about consumer strength_TOTAL_FILT_MD",
#                   "This text is about consumer weakness_TOTAL_FILT_MD",
#                   "This text is about reduced consumer's spending patterns_TOTAL_FILT_MD",
#                   "This text is about reduced consumer's spending patterns_SCORE_FILT_MD",
#                   "This text is about consumer weakness_SCORE_FILT_QA",
#                   "This text is about consumer strength_SCORE_FILT_QA",
#                   "This text is about consumer strength_TOTAL_FILT_QA",
#                   "This text is about consumer weakness_TOTAL_FILT_QA",
#                   "This text is about reduced consumer's spending patterns_TOTAL_FILT_QA",
#                   "This text is about reduced consumer's spending patterns_SCORE_FILT_QA"
#                   ]                

def generate_label_columns(
    labels: List[str],
    metrics: List[str] = ['COUNT', 'REL', 'SCORE', 'TOTAL'],
    sec_filters: List[str] = ['FILT_MD', 'FILT_QA'],
) -> List[str]:
    """
    Generates a list of column names based on the provided labels, metrics, and secondary filters.

    The order of metrics should always follow: 'COUNT', 'REL', 'SCORE', 'TOTAL', and 'EXTRACT'.
    Some metrics may be omitted based on the use case, but the order must remain the same.

    Args:
        labels (List[str]): Labels such as 'consumer_strength', 'consumer_weakness', etc.
        metrics (List[str]): Metrics like 'COUNT', 'REL', 'SCORE', etc. Defaults to ['COUNT', 'REL', 'SCORE', 'TOTAL'].
        sec_filters (List[str], optional): Secondary filters like 'FILT_MD', 'FILT_QA'. Defaults to ['FILT_MD', 'FILT_QA'].

    Returns:
        List[str]: An ordered list of generated column names.

    Example:
        >>> labels = ['consumer_strength', 'consumer_weakness']
        >>> metrics = ['COUNT', 'REL']
        >>> generate_label_columns(labels, metrics)
        ['consumer_strength_COUNT_FILT_MD', 'consumer_strength_REL_FILT_MD',
         'consumer_strength_COUNT_FILT_QA', 'consumer_strength_REL_FILT_QA',
         'consumer_weakness_COUNT_FILT_MD', 'consumer_weakness_REL_FILT_MD',
         'consumer_weakness_COUNT_FILT_QA', 'consumer_weakness_REL_FILT_QA']
    """
    dynamic_columns = []
    for label in labels:
        for sec_filter in sec_filters:
            for metric in metrics:
                # Base column
                column_name = f"{label}_{metric}_{sec_filter}"
                dynamic_columns.append(column_name)
                logger.debug(f"Generated column name: {column_name}")
    logger.info(f"Generated {len(dynamic_columns)} label columns.")
    return dynamic_columns

def filter_columns(columns, substrings):
    return [col for col in columns if any(sub in col for sub in substrings)]


label_columns = generate_label_columns(LABELS)



# Filter columns based on the defined substrings
float_type_cols = filter_columns(label_columns, ['_COUNT_', '_REL_'])
array_type_cols = filter_columns(label_columns, ['_SCORE_', '_TOTAL_'])

# Define a UDF to apply ast.literal_eval
def literal_eval_safe(data_str):
    try:
        return ast.literal_eval(data_str)
    except (ValueError, SyntaxError):
        return None
      
array_convert_udf = udf(literal_eval_safe, ArrayType(DoubleType()))
float_convert_udf = udf(literal_eval_safe, DoubleType())

for col_name in float_type_cols:
    currdf_spark = currdf_spark.withColumn(col_name, float_convert_udf(currdf_spark[col_name]))

for col_name in array_type_cols:
    currdf_spark = currdf_spark.withColumn(col_name, array_convert_udf(currdf_spark[col_name]))

# COMMAND ----------

LABELS

# COMMAND ----------

# Define the mapping of original labels to new labels
LABELS_MAPPING = {
    "This text is about a weak consumer or reduced consumption": "REDUCED_CONSUMPTION",
    "This text is about a strong consumer or increased consumption": "INCREASED_CONSUMPTION",
}

# Create a new list for the updated column names
new_columns = []

# # Iterate through the current DataFrame columns
# for old_col in currdf_spark.columns:
#     # Replace the original label with the new label if it exists in the mapping
#     for original_label, new_label in LABELS_MAPPING.items():
#         if original_label in old_col:
#             new_col = old_col.replace(original_label, new_label)
#             currdf_spark = currdf_spark.withColumnRenamed(old_col, new_col)
#             break  # Exit the loop once a match is found
    
    # new_columns.append(col)
def rename_columns(df, labels_mapping):
    if not isinstance(df, DataFrame):
        raise ValueError("The provided input is not a valid Spark DataFrame.")

    if not isinstance(labels_mapping, dict):
        raise ValueError("The labels_mapping must be a dictionary.")

    try:
        # Create a mapping for renaming
        new_column_names = {}
        for old_col in df.columns:
            for original_label, new_label in labels_mapping.items():
                if original_label in old_col:
                    new_column_names[old_col] = old_col.replace(original_label, new_label)
                    break  # Break after the first match to avoid multiple replacements

        # Rename columns if any new names are generated
        for old_col, new_col in new_column_names.items():
            df = df.withColumnRenamed(old_col, new_col)
            logger.debug(f"Renamed column '{old_col}' to '{new_col}'")

        logger.info("Columns renamed successfully.")
        return df
    except Exception as e:
        logger.error(f"Failed to rename columns: {e}")
        raise e

currdf_spark = rename_columns(currdf_spark, LABELS_MAPPING)

# COMMAND ----------

def extract_matched_sentences(sentences, matches):
    if not sentences:  # Check if the list is empty
        return None
    return [sentence for sentence, match in zip(sentences, matches) if match == 1]

extract_udf = udf(extract_matched_sentences, ArrayType(StringType()))

# Get all columns that match the pattern "_TOTAL_FILT_MD" or "_TOTAL_FILT_QA"
patterns = [re.compile(r".*_TOTAL_FILT_MD"), re.compile(r".*_TOTAL_FILT_QA")]
matched_columns = [col for col in currdf_spark.columns if any(pattern.match(col) for pattern in patterns)]

# Apply UDF to create new columns
for col_name in matched_columns:
    if "_TOTAL_FILT_MD" in col_name:
        new_col_name = col_name.replace("_TOTAL_FILT_MD", "_EXTRACT_FILT_MD")
        currdf_spark = currdf_spark.withColumn(new_col_name, extract_udf(col("FILT_MD"), col(col_name)))
    elif "_TOTAL_FILT_QA" in col_name:
        new_col_name = col_name.replace("_TOTAL_FILT_QA", "_EXTRACT_FILT_QA")
        currdf_spark = currdf_spark.withColumn(new_col_name, extract_udf(col("FILT_QA"), col(col_name)))


# COMMAND ----------

columns_order = ['ENTITY_ID', 'CALL_ID', 'VERSION_ID', 'DATE', 'CALL_NAME',
       'COMPANY_NAME', 'UPLOAD_DT_UTC',  'PARSED_DATETIME_EASTERN_TZ','LEN_FILT_MD', 'LEN_FILT_QA', 'FILT_MD', 'FILT_QA', 'SENT_LABELS_FILT_MD', 'SENT_LABELS_FILT_QA',
       'INCREASED_CONSUMPTION_COUNT_FILT_MD', 'INCREASED_CONSUMPTION_REL_FILT_MD',
       'INCREASED_CONSUMPTION_SCORE_FILT_MD', 'INCREASED_CONSUMPTION_TOTAL_FILT_MD', 
       'REDUCED_CONSUMPTION_COUNT_FILT_MD', 'REDUCED_CONSUMPTION_REL_FILT_MD',
       'REDUCED_CONSUMPTION_SCORE_FILT_MD', 'REDUCED_CONSUMPTION_TOTAL_FILT_MD',
       'INCREASED_CONSUMPTION_COUNT_FILT_QA', 'INCREASED_CONSUMPTION_REL_FILT_QA',
       'INCREASED_CONSUMPTION_SCORE_FILT_QA', 'INCREASED_CONSUMPTION_TOTAL_FILT_QA',
       'REDUCED_CONSUMPTION_COUNT_FILT_QA', 'REDUCED_CONSUMPTION_REL_FILT_QA',
       'REDUCED_CONSUMPTION_SCORE_FILT_QA', 'REDUCED_CONSUMPTION_TOTAL_FILT_QA',
       'INCREASED_CONSUMPTION_EXTRACT_FILT_MD', 'INCREASED_CONSUMPTION_EXTRACT_FILT_QA',
       'REDUCED_CONSUMPTION_EXTRACT_FILT_MD', 'REDUCED_CONSUMPTION_EXTRACT_FILT_QA'
]


# spark_parsedDF = pandas_to_spark(currdf_all[currdf_sample.columns])
currdf_spark = currdf_spark.select(*columns_order)
currdf_spark = currdf_spark.replace(np.nan, None)
currdf_spark = currdf_spark.withColumn("DATE", F.to_timestamp(currdf_spark.DATE, 'yyyy-MM-dd'))

# COMMAND ----------

currdf_spark.count()

# COMMAND ----------

currdf_spark.limit(10).show()

# COMMAND ----------

dev_name = dbutils.widgets.get("DEV_NAME")
tablename_curr = f'{dev_name}_MASS_FT_NLI_DEMAND_DEV_6M_NEW_MODEL'
result_curr = new_sf.write_to_snowflake_table(currdf_spark, tablename_curr)

# COMMAND ----------

# import time

# while True:
#   time.sleep(60)

# COMMAND ----------

