# Databricks notebook source
# MAGIC %md
# MAGIC

# COMMAND ----------

# MAGIC %run "Quant/Call_Transcript_Topic_Modeling/Development/Dynamic Topic Modeling/call_transcript_NLP_BERT_libs_stg5_pl4.py"

# COMMAND ----------

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

N_PARTITION = 240
LIMIT = 100
ENABLE_QUANTIZATION = False  # Set to False to disable quantization
BATCH_SIZE = 64

# COMMAND ----------

import logging, os
from logging.handlers import RotatingFileHandler

# Configure logging to write to a file in DBFS or another centralized location
# Configure logging to write to a file in DBFS or another centralized location
log_path = "/dbfs/Workspace/Users/santhosh.kumar3@voya.com/logfile.log"
handler = RotatingFileHandler(log_path, maxBytes=10**7, backupCount=5)
logger = logging.getLogger("NLI_Inference")
logger.setLevel(logging.INFO)
logger.addHandler(handler)

# Check if the file exists
if not os.path.exists(log_path):
    # Create the file
    with open(log_path, 'w') as file:
        file.write('')  # Optionally write an empty string or some initial content
    print(f'File "{log_path}" created.')
else:
    print(f'File "{log_path}" already exists.')

# COMMAND ----------

# # Step 2: Initialize Spark Session (if not already initialized)
spark = (SparkSession.builder.appName("Optimized_NLI_Inference")
         .config("spark.sql.shuffle.partitions",N_PARTITION)
         .config("spark.executor.resource.gpu.amount", "1")
         .config("spark.task.resource.gpu.amount", "0.8")
         .getOrCreate())

spark.sparkContext.setLogLevel("DEBUG")

# COMMAND ----------

# %sql
# SET TIME ZONE 'UTC';

# COMMAND ----------

# Step 3: Define Constants and Broadcast Variables
MODEL_FOLDER_PATH = "/dbfs/mnt/access_work/UC25/Libraries/HuggingFace/"
MODEL_NAME = "deberta-v3-large-zeroshot-v2"
LABELS = [
    "This text is about consumer strength",
    "This text is about consumer weakness",
    "This text is about reduced consumer's spending patterns"
]
labels_broadcast = spark.sparkContext.broadcast(LABELS)

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

# Function to initialize the NLI pipeline on each executor
def initialize_nli_pipeline(enable_quantization=False):
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_FOLDER_PATH + MODEL_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_FOLDER_PATH + MODEL_NAME)
        
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



# inference_udf = udf(transformer_inference, ArrayType(StructType([
#     StructField("label", StringType(), False),
#     StructField("score", FloatType(), False)
# ])))


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

#EDITED: Read parameters from widgets
# minDateNewQuery = (pd.to_datetime(dbutils.widgets.get("Start Month") + "-01-" + dbutils.widgets.get("Start Year"))).strftime('%Y-%m-%d')
# maxDateNewQuery = (pd.to_datetime(dbutils.widgets.get("End Month") + "-01-" + dbutils.widgets.get("End Year"))).strftime('%Y-%m-%d')
import pandas as pd
minDateNewQuery = pd.to_datetime("01-12-2021").strftime('%Y-%m-%d')
maxDateNewQuery = pd.to_datetime("01-01-2022").strftime('%Y-%m-%d')

mind = f"'{minDateNewQuery}'"
maxd = f"'{maxDateNewQuery}'"

print(f'The next query spans {mind} to {maxd}')

# COMMAND ----------

myDBFS = DBFShelper()
# new_sf = pd.read_pickle(r'/dbfs' + myDBFS.iniPath + 'mysf_prod_quant.pkl')
new_sf = pd.read_pickle(r'/dbfs/FileStore/Yujing/resources/mysf_prod_quant.pkl')


#EDITED: Construct and execute the Snowflake query using Spark
tsQuery = (f"SELECT CALL_ID, ENTITY_ID, DATE, FILT_MD, FILT_QA, CALL_NAME, COMPANY_NAME, "
           f"EARNINGS_CALL, ERROR, TRANSCRIPT_STATUS, UPLOAD_DT_UTC, VERSION_ID, "
           f"EVENT_DATETIME_UTC, PARSED_DATETIME_EASTERN_TZ, SENT_LABELS_FILT_MD, "
           f"SENT_LABELS_FILT_QA "
           f"FROM EDS_PROD.QUANT.PARTHA_FUND_CTS_STG_1_VIEW "
           f"WHERE DATE >= {mind} AND DATE < {maxd} " #mind
           f"ORDER BY PARSED_DATETIME_EASTERN_TZ DESC")

# tsQuery = (f"SELECT CALL_ID, ENTITY_ID, DATE, FILT_MD, FILT_QA, CALL_NAME, COMPANY_NAME, "
#            f"EARNINGS_CALL, ERROR, TRANSCRIPT_STATUS, UPLOAD_DT_UTC, VERSION_ID, "
#            f"EVENT_DATETIME_UTC, PARSED_DATETIME_EASTERN_TZ, SENT_LABELS_FILT_MD, "
#            f"SENT_LABELS_FILT_QA "
#            f"FROM EDS_PROD.QUANT.PARTHA_FUND_CTS_STG_1_VIEW "
#            f"WHERE DATE >= {mind} AND DATE < {maxd} AND " #mind
#            f"ENTITY_ID in ('00872M-E', '0NPXTB-E', '05HWJR-E', '001B9Z-E', '0C9L39-E', '0087HJ-E', '0NV64L-E', '0NPJ0S-E', '008JF4-E','05MT1B-E')"
#            f"ORDER BY PARSED_DATETIME_EASTERN_TZ DESC")

currdf_spark = new_sf.read_from_snowflake(tsQuery)  # Assuming newDBFS is correctly initialized

# currdf_spark = currdf_spark.limit(LIMIT)
# print(currdf_spark.count())


# COMMAND ----------

# # Step 6: Validate DataFrame
# if currdf_spark.head(1):
#     min_parsed_date = currdf_spark.agg({"PARSED_DATETIME_EASTERN_TZ": "min"}).collect()[0][0]
#     max_parsed_date = currdf_spark.agg({"PARSED_DATETIME_EASTERN_TZ": "max"}).collect()[0][0]
#     row_count = currdf_spark.count()
#     col_count = len(currdf_spark.columns)
#     print(f'The data spans from {min_parsed_date} to {max_parsed_date} and has {row_count} rows and {col_count} columns.')
# else:
#     print('No new transcripts to parse.')
#     dbutils.notebook.exit(1)
#     os._exit(1)

# COMMAND ----------

# Step 7: Data Transformation Using Spark DataFrame API

# Convert stringified lists to actual arrays
currdf_spark = (currdf_spark 
    .withColumn('FILT_MD', parse_json_udf(col('FILT_MD'))) 
    .withColumn('FILT_QA', parse_json_udf(col('FILT_QA'))) 
    .withColumn('SENT_LABELS_FILT_MD', parse_json_udf(col('SENT_LABELS_FILT_MD'))) 
    .withColumn('SENT_LABELS_FILT_QA', parse_json_udf(col('SENT_LABELS_FILT_QA'))) 
    .withColumn('LEN_FILT_MD', size(col('FILT_MD'))) 
    .withColumn('LEN_FILT_QA', size(col('FILT_QA'))) 
    .dropDuplicates(['ENTITY_ID', 'EVENT_DATETIME_UTC']) 
    .orderBy(col('UPLOAD_DT_UTC').asc()))

# currdf_spark = (currdf_spark 
#     .withColumn('SENT_LABELS_FILT_QA', expr("filter(SENT_LABELS_FILT_QA, (x, i) -> not FILTER_QA[i].endsWith('?'))")) 
#     .withColumn('FILT_QA', expr("filter(FILT_QA, x -> not x.endswith('?'))")) 
#     .withColumn('LEN_FILT_QA', size(col('FILT_QA'))))

# currdf_spark = (currdf_spark
#     .withColumn('SENT_LABELS_FILT_QA', F.expr("filter(SENT_LABELS_FILT_QA, (x, i) -> not FILTER_QA[i].endsWith('?'))"))
#     .withColumn('FILT_QA', 
#         F.filter(F.col('FILT_QA'), lambda x: not x.endswith('?'))
#     )
#     .withColumn('LEN_FILT_QA', F.size(F.col('FILT_QA')))
# )

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

# schema_summary = StructType([
#     StructField("result_dict", MapType(StringType(), ArrayType(StringType())), False),
#     StructField("total_dict", MapType(StringType(), ArrayType(IntegerType())), False)
# ])

# inference_schema = ArrayType(ArrayType(StringType()))

# Define the schema for the dictionary
# dict_schema = StructType([
#     StructField('label', StringType(), True),
#     StructField('score', FloatType(), True)
# ])

# Define the schema for the output of the UDF
inference_schema = ArrayType(ArrayType(dict_schema))

@pandas_udf(inference_schema, PandasUDFType.SCALAR_ITER)
def inference_udf(iterator):
    global nli_pipeline
    if nli_pipeline is None:
        nli_pipeline = initialize_nli_pipeline(enable_quantization=ENABLE_QUANTIZATION)
    
    for batch_num, batch in enumerate(iterator, start=1):
        logger.debug(f"Processing MD inference batch {batch_num} with {type(iterator)} {type(batch)} rows.")
        # try:
        # Flatten the list of text pairs in the batch
        flat_text_pairs = [pair for sublist in batch for pair in sublist]
        logger.debug(f"Batch {batch_num}: Total text pairs to infer: {len(flat_text_pairs)}")
        
        # Perform inference in batch
        results = nli_pipeline(
            flat_text_pairs,
            padding=True,
            top_k=None,
            batch_size=BATCH_SIZE,  # Adjusted batch size
            truncation=True,
            max_length=512
        )
        logger.debug(f"Batch {batch_num}: Inference completed with {len(results)} results.")

        
        # Split results back to original rows
        split_results = []
        idx = 0
        for pairs in batch:
            if len(pairs):
                split_results.append(results[idx:idx+len(pairs)])
                idx += len(pairs)
            else:
                split_results.append([])
        
        yield pd.Series(split_results)
        # except Exception as e:
        #     logger.error(f"Error in MD inference batch {batch_num}: {e}")
        #     # Yield empty results for this batch to continue processing
        #     yield pd.Series([[] for _ in batch])

@pandas_udf(inference_schema, PandasUDFType.SCALAR)
def inference_udf(texts: pd.Series) -> pd.Series:
    global nli_pipeline
    if nli_pipeline is None:
        nli_pipeline = initialize_nli_pipeline(enable_quantization=ENABLE_QUANTIZATION)

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
    # logger.debug(f"Batch: Inference completed with {len(flat_text_pairs)} results.")
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


logger.info(f"Repartitioning DataFrame to {N_PARTITION} partitions for GPU workers.")
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
# Define the UDF for extracting information
# def extract_inf(row, section, section_len, threshold):
#     count_col = {}
#     rel_col = {}
#     score_col = {}
#     total_col = {}

#     # set_trace()
#     for tp, score in row.items():
#         if section_len != 0:
#             score_binary = [1 if s > threshold else 0 for s in score]
#             total_col[f'{tp}_TOTAL_{section}'] = score_binary 
#             count_col[f'{tp}_COUNT_{section}'] = float(sum(score_binary))
#             rel_col[f'{tp}_REL_{section}'] = sum(score_binary) / section_len
#             score_col[f'{tp}_SCORE_{section}'] = [round(s, 4) for s in score]
#         else:
#             count_col[f'{tp}_COUNT_{section}'] = None
#             rel_col[f'{tp}_REL_{section}'] = None
#             total_col[f'{tp}_TOTAL_{section}'] = []
#             score_col[f'{tp}_SCORE_{section}'] = []
#     # print(count_col.keys())

#     return {**count_col, **rel_col, **score_col, **total_col}

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

columns_filt =(['ENTITY_ID', 'CALL_ID', 'VERSION_ID', 'DATE', 'CALL_NAME', 'COMPANY_NAME', 'LEN_FILT_MD', 'LEN_FILT_QA', 'FILT_MD', 'FILT_QA'] + 
           [col.replace('.', '') for col in md_final_score_extracted_cols] + 
          [col.replace('.', '') for col in qa_final_score_extracted_cols] )
currdf_spark = currdf_spark.select(*columns_filt)

# COMMAND ----------

import ast

float_type_cols = ["This text is about consumer weakness_COUNT_FILT_MD",
                    "This text is about consumer strength_COUNT_FILT_MD",
                    "This text is about reduced consumer's spending patterns_COUNT_FILT_MD",
                    "This text is about consumer weakness_REL_FILT_MD",
                    "This text is about consumer strength_REL_FILT_MD",
                    "This text is about reduced consumer's spending patterns_REL_FILT_MD",
                    "This text is about consumer weakness_COUNT_FILT_QA",
                    "This text is about consumer strength_COUNT_FILT_QA",
                    "This text is about reduced consumer's spending patterns_COUNT_FILT_QA",
                    "This text is about consumer weakness_REL_FILT_QA",
                    "This text is about consumer strength_REL_FILT_QA",
                    "This text is about reduced consumer's spending patterns_REL_FILT_QA"] 

array_type_cols = ["This text is about consumer weakness_SCORE_FILT_MD",
                  "This text is about consumer strength_SCORE_FILT_MD",
                  "This text is about consumer strength_TOTAL_FILT_MD",
                  "This text is about consumer weakness_TOTAL_FILT_MD",
                  "This text is about reduced consumer's spending patterns_TOTAL_FILT_MD",
                  "This text is about reduced consumer's spending patterns_SCORE_FILT_MD",
                  "This text is about consumer weakness_SCORE_FILT_QA",
                  "This text is about consumer strength_SCORE_FILT_QA",
                  "This text is about consumer strength_TOTAL_FILT_QA",
                  "This text is about consumer weakness_TOTAL_FILT_QA",
                  "This text is about reduced consumer's spending patterns_TOTAL_FILT_QA",
                  "This text is about reduced consumer's spending patterns_SCORE_FILT_QA"
                  ]                

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

# Define the mapping of original labels to new labels
LABELS_MAPPING = {
    "This text is about consumer strength": "CONSUMER_STRENGTH",
    "This text is about consumer weakness": "CONSUMER_WEAKNESS",
    "This text is about reduced consumer's spending patterns": "CONSUMER_SPENDING_PATTERNS"
}

# Create a new list for the updated column names
new_columns = []

# Iterate through the current DataFrame columns
for old_col in currdf_spark.columns:
    # Replace the original label with the new label if it exists in the mapping
    for original_label, new_label in LABELS_MAPPING.items():
        if original_label in old_col:
            new_col = old_col.replace(original_label, new_label)
            currdf_spark = currdf_spark.withColumnRenamed(old_col, new_col)
            break  # Exit the loop once a match is found
    
    # new_columns.append(col)

# COMMAND ----------

  columns_order = ['ENTITY_ID', 'CALL_ID', 'VERSION_ID', 'DATE', 'CALL_NAME',
       'COMPANY_NAME', 'LEN_FILT_MD', 'LEN_FILT_QA', 'FILT_MD', 'FILT_QA',
       'CONSUMER_STRENGTH_COUNT_FILT_MD', 'CONSUMER_STRENGTH_REL_FILT_MD',
       'CONSUMER_STRENGTH_SCORE_FILT_MD', 'CONSUMER_STRENGTH_TOTAL_FILT_MD',
       'CONSUMER_WEAKNESS_COUNT_FILT_MD', 'CONSUMER_WEAKNESS_REL_FILT_MD',
       'CONSUMER_WEAKNESS_SCORE_FILT_MD', 'CONSUMER_WEAKNESS_TOTAL_FILT_MD',
       'CONSUMER_SPENDING_PATTERNS_COUNT_FILT_MD',
       'CONSUMER_SPENDING_PATTERNS_REL_FILT_MD',
       'CONSUMER_SPENDING_PATTERNS_SCORE_FILT_MD',
       'CONSUMER_SPENDING_PATTERNS_TOTAL_FILT_MD',
       'CONSUMER_STRENGTH_COUNT_FILT_QA', 'CONSUMER_STRENGTH_REL_FILT_QA',
       'CONSUMER_STRENGTH_SCORE_FILT_QA', 'CONSUMER_STRENGTH_TOTAL_FILT_QA',
       'CONSUMER_WEAKNESS_COUNT_FILT_QA', 'CONSUMER_WEAKNESS_REL_FILT_QA',
       'CONSUMER_WEAKNESS_SCORE_FILT_QA', 'CONSUMER_WEAKNESS_TOTAL_FILT_QA',
       'CONSUMER_SPENDING_PATTERNS_COUNT_FILT_QA',
       'CONSUMER_SPENDING_PATTERNS_REL_FILT_QA',
       'CONSUMER_SPENDING_PATTERNS_SCORE_FILT_QA',
       'CONSUMER_SPENDING_PATTERNS_TOTAL_FILT_QA']


# spark_parsedDF = pandas_to_spark(currdf_all[currdf_sample.columns])
currdf_spark = currdf_spark.select(*columns_order)
currdf_spark = currdf_spark.replace(np.nan, None)
currdf_spark = currdf_spark.withColumn("DATE", F.to_timestamp(currdf_spark.DATE, 'yyyy-MM-dd'))

# COMMAND ----------

# pd_df = currdf_spark.toPandas()

# pd_df

# COMMAND ----------

tablename_curr = 'SANTHOSH_MASS_NLI_DEMAND_DEV_3'
result_curr = new_sf.write_to_snowflake_table(currdf_spark, tablename_curr)

# COMMAND ----------

currdf_spark.count()

# COMMAND ----------


