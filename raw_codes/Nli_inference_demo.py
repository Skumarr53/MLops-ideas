# Databricks notebook source
import os 
os.environ['DATABRICKS_HOST'] = 'adb-2762743938046900'
os.environ['DATABRICKS_TOKEN'] = dbutils.secrets.get(scope="id-secretscope-dbk-pr4707-prod-work", key="edp-access-pat-prod-key")

# COMMAND ----------

# MAGIC
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
# MAGIC %pip install "dask[dataframe,distributed]"==2023.9.3
# MAGIC %pip install torch==2.0.0
# MAGIC %pip install mlflow==2.18.0
# MAGIC %pip install mlflow-skinny==2.18.0
# MAGIC %pip install pydantic==1.10.6
# MAGIC %pip install cymem==2.0.8
# MAGIC %pip install scikit-learn==1.1.0
# MAGIC %pip install typer==0.7.0
# MAGIC %pip install accelerate==0.26.0
# MAGIC # %pip install databricks-runtime

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
from centralized_nlp_package.data_processing import initialize_spark_session
from centralized_nlp_package.nli_utils import initialize_nli_infer_pipeline
from topic_modelling_package.nli_process import create_text_pairs, inference_summary


# COMMAND ----------

# MAGIC %md
# MAGIC ### Create Spark Session

# COMMAND ----------

# MAGIC %md
# MAGIC #### Old

# COMMAND ----------

N_PARTITION = 5

spark_old = (SparkSession.builder.appName("Optimized_NLI_Inference")
         .config("spark.sql.shuffle.partitions",N_PARTITION)
         .config("spark.executor.resource.gpu.amount", "1")
         .config("spark.task.resource.gpu.amount", "1")
         .getOrCreate())

# COMMAND ----------

# MAGIC %md
# MAGIC #### Refactored

# COMMAND ----------

spark_ref = initialize_spark_session(app_name = "NLI_Inference", shuffle_partitions = N_PARTITION)


# COMMAND ----------

# MAGIC %md
# MAGIC ###  Define Helper Functions

# COMMAND ----------

# MAGIC %md
# MAGIC #### old

# COMMAND ----------

MODEL_FOLDER_PATH = "/dbfs/mnt/access_work/UC25/Libraries/HuggingFace/"
MODEL_NAME = "deberta-v3-large-zeroshot-v2"
LABELS = [
    "This text is about consumer strength",
    "This text is about consumer weakness",
    "This text is about reduced consumer's spending patterns"
]
labels_broadcast = spark_ref.sparkContext.broadcast(LABELS)

# Function to parse JSON strings to lists
def parse_json_list_old(s):
    try:
        return json.loads(s)
    except Exception as e:
        print(f"JSON parsing error: {e} for input: {s}")
        return []

# UDF to parse JSON lists
parse_json_udf_old = udf(parse_json_list_old, ArrayType(StringType()))

# Function to create text pairs for NLI
def create_text_pairs_old(filt):
    text_pairs = []
    for t in filt:
        for l in labels_broadcast.value:
            text_pairs.append(f"{t}</s></s>{l}.")
    return text_pairs

# UDF to create text pairs
create_text_pairs_udf_old = udf(create_text_pairs_old, ArrayType(StringType()))



# COMMAND ----------

# MAGIC %md
# MAGIC #### Refactored

# COMMAND ----------

from topic_modelling_package.nli_process import parse_json_list, create_text_pairs
parse_json_udf_ref = udf(parse_json_list, ArrayType(StringType()))
create_text_pairs_udf_ref = udf(create_text_pairs, ArrayType(StringType()))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Function to initialize the NLI pipeline on each executor

# COMMAND ----------

# MAGIC %md
# MAGIC #### old

# COMMAND ----------

from loguru import logger
def initialize_nli_pipeline_old(enable_quantization=False):
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

# COMMAND ----------

# MAGIC %md
# MAGIC ##### refactored

# COMMAND ----------

initialize_nli_infer_pipeline(model_path = MODEL_FOLDER_PATH + MODEL_NAME, enable_quantization = False)

# COMMAND ----------

# MAGIC %md
# MAGIC ### inference summary

# COMMAND ----------

# MAGIC %md
# MAGIC #### old

# COMMAND ----------

import ast
def inference_summary_old(texts, inference_result, threshold=0.8):
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

schema_summary = StructType([
    StructField("total_dict", MapType(StringType(), ArrayType(IntegerType())), False),
    StructField("score_dict", MapType(StringType(), ArrayType(FloatType())), False)
])

#EDITED: Define UDF for parsing string representations of lists
parse_udf = udf(lambda x: ast.literal_eval(x) if x else [], ArrayType(StringType()))
summary_udf_old = udf(lambda texts, inference_result: inference_summary_old(texts, inference_result), schema_summary)


# COMMAND ----------

# MAGIC %md
# MAGIC #### refactored

# COMMAND ----------

from topic_modelling_package.nli_process import inference_summary

summary_udf_ref = udf(lambda texts, inference_result: inference_summary(texts, inference_result), schema_summary)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Fetch Data

# COMMAND ----------

import pandas as pd
minDateNewQuery = pd.to_datetime("01-12-2021").strftime('%Y-%m-%d')
maxDateNewQuery = pd.to_datetime("01-01-2022").strftime('%Y-%m-%d')

mind = f"'{minDateNewQuery}'"
maxd = f"'{maxDateNewQuery}'"

tsQuery = (f"SELECT CALL_ID, ENTITY_ID, DATE, FILT_MD, FILT_QA, CALL_NAME, COMPANY_NAME, "
           f"EARNINGS_CALL, ERROR, TRANSCRIPT_STATUS, UPLOAD_DT_UTC, VERSION_ID, "
           f"EVENT_DATETIME_UTC, PARSED_DATETIME_EASTERN_TZ, SENT_LABELS_FILT_MD, "
           f"SENT_LABELS_FILT_QA "
           f"FROM EDS_PROD.QUANT.PARTHA_FUND_CTS_STG_1_VIEW "
           f"WHERE DATE >= {mind} AND DATE < {maxd} " #mind
           f"ORDER BY PARSED_DATETIME_EASTERN_TZ DESC LIMIT 10")

currdf_spark_old = read_from_snowflake(tsQuery)  # Assuming newDBFS is correctly initialized



# COMMAND ----------

currdf_spark_ref = currdf_spark_old.select("*")
currdf_ref = currdf_spark_ref.toPandas()


# COMMAND ----------

# MAGIC %md
# MAGIC ### this step applies series of functions on the spark columns

# COMMAND ----------

# MAGIC %md
# MAGIC #### old

# COMMAND ----------


currdf_spark_old = (currdf_spark_old 
    .withColumn('FILT_MD', parse_json_udf_old(col('FILT_MD'))) 
    .withColumn('FILT_QA', parse_json_udf_old(col('FILT_QA'))) 
    .withColumn('SENT_LABELS_FILT_MD', parse_json_udf_old(col('SENT_LABELS_FILT_MD'))) 
    .withColumn('SENT_LABELS_FILT_QA', parse_json_udf_old(col('SENT_LABELS_FILT_QA'))) 
    .withColumn('LEN_FILT_MD', size(col('FILT_MD'))) 
    .withColumn('LEN_FILT_QA', size(col('FILT_QA'))) 
    .dropDuplicates(['ENTITY_ID', 'EVENT_DATETIME_UTC']) 
    .orderBy(col('UPLOAD_DT_UTC').asc()))

currdf_spark_old = currdf_spark_old \
    .withColumn('TEXT_PAIRS_MD', create_text_pairs_udf_old(col('FILT_MD'))) \
    .withColumn('TEXT_PAIRS_QA', create_text_pairs_udf_old(col('FILT_QA')))



# COMMAND ----------

currdf_old = currdf_spark_old.toPandas()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Refactored

# COMMAND ----------

from topic_modelling_package.nli_process import parse_json_list
from centralized_nlp_package.data_processing import sparkdf_apply_transformations

parse_json_udf_ref = udf(parse_json_list, ArrayType(StringType()))


# COMMAND ----------

transformations1 = [
    ('FILT_MD', 'FILT_MD', parse_json_udf_ref),
    ('FILT_QA', 'FILT_QA', parse_json_udf_ref),
    ('SENT_LABELS_FILT_MD', 'SENT_LABELS_FILT_MD', parse_json_udf_ref),
    ('SENT_LABELS_FILT_QA', 'SENT_LABELS_FILT_QA', parse_json_udf_ref),
    ('LEN_FILT_MD', 'FILT_MD', size),
    ('LEN_FILT_QA', 'FILT_QA', size)
    ]

currdf_spark_ref = sparkdf_apply_transformations(currdf_spark_ref, transformations1)

currdf_spark_ref = (currdf_spark_ref
                    .dropDuplicates(['ENTITY_ID', 'EVENT_DATETIME_UTC'])
                    .orderBy(col('UPLOAD_DT_UTC').asc()))

transformations2 = [
    ('TEXT_PAIRS_MD', 'FILT_MD', create_text_pairs_udf_ref),
    ('TEXT_PAIRS_QA', 'FILT_QA', create_text_pairs_udf_ref)
]

currdf_spark_ref = sparkdf_apply_transformations(currdf_spark_ref, transformations2)


# COMMAND ----------

currdf_ref = currdf_spark_ref.toPandas()

# COMMAND ----------

assert currdf_old.equals(currdf_ref), "The DataFrames are not identical."

# COMMAND ----------

# MAGIC %md
# MAGIC ### Run Inference

# COMMAND ----------

# MAGIC %md
# MAGIC #### Old

# COMMAND ----------

BATCH_SIZE = 4
ENABLE_QUANTIZATION = False

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
        nli_pipeline = initialize_nli_pipeline_old(enable_quantization=ENABLE_QUANTIZATION)

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
# currdf_spark_old = currdf_spark_old.repartition(N_PARTITION)
# Step 9: Apply Inference UDFs to DataFrame

currdf_spark_old = (currdf_spark_old 
    .withColumn('MD_RESULT', inference_udf(col('TEXT_PAIRS_MD'))) 
    .withColumn('QA_RESULT', inference_udf(col('TEXT_PAIRS_QA'))))

currdf_spark_old.cache()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Refactored

# COMMAND ----------

## Define the UDF
inference_udf_init = partial(inference_udf, max_length=512, batch_size=32, enable_quantization=False) 

infernece_udf_func = pandas_udf(inference_udf_init, inference_schema, functionType=PandasUDFType.SCALAR_ITER)


spark_df = sparkdf_apply_transformations(
    currdf_spark_ref,
    [
        ("MD_RESULT", "TEXT_PAIRS_MD", infernece_udf_func),
        ("QA_RESULT", "TEXT_PAIRS_QA", infernece_udf_func)
    ])


# COMMAND ----------

assert currdf_old.equals(currdf_ref), "The DataFrames are not identical."

# COMMAND ----------

# MAGIC %md
# MAGIC ### Unpacking summary results into columns

# COMMAND ----------

# MAGIC %md
# MAGIC #### old

# COMMAND ----------

currdf_spark_old = currdf_spark_old \
    .withColumn("MD_SUMMARY", summary_udf_old(F.col("TEXT_PAIRS_MD"),F.col("MD_RESULT"))) \
    .withColumn("QA_SUMMARY", summary_udf_old(F.col("TEXT_PAIRS_QA"),F.col("QA_RESULT")))

# #EDITED: Extract summary fields
currdf_spark_old = (currdf_spark_old
    .withColumn("MD_FINAL_TOTAL", F.col("MD_SUMMARY.total_dict")) 
    .withColumn("MD_FINAL_SCORE", F.col("MD_SUMMARY.score_dict")) 
    .withColumn("QA_FINAL_TOTAL", F.col("QA_SUMMARY.total_dict")) 
    .withColumn("QA_FINAL_SCORE", F.col("QA_SUMMARY.score_dict")))

# COMMAND ----------

currdf_spark_old = currdf_spark_old.toPandas()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Refactored

# COMMAND ----------

transformations2 = [
    ("MD_SUMMARY", ("TEXT_PAIRS_MD", "MD_RESULT"), summary_udf_ref),
    ("QA_SUMMARY", ("TEXT_PAIRS_QA", "QA_RESULT"), summary_udf_ref),
    ("MD_FINAL_TOTAL", "MD_SUMMARY.total_dict", F.col),
    ("MD_FINAL_SCORE", "MD_SUMMARY.score_dict", F.col),
    ("QA_FINAL_TOTAL", "QA_SUMMARY.total_dict", F.col),
    ("QA_FINAL_SCORE", "QA_SUMMARY.score_dict", F.col)
]

currdf_spark_ref = sparkdf_apply_transformations(currdf_spark_ref, transformations2)

# COMMAND ----------

currdf_ref = currdf_spark_ref.toPandas()

# COMMAND ----------

assert currdf_old.equals(currdf_ref), "The DataFrames are not identical."

# COMMAND ----------

# MAGIC %md
# MAGIC ### Precessing nested columns

# COMMAND ----------

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
currdf_spark_old = currdf_spark_old.withColumn(
    'MD_FINAL_SCORE_EXTRACTED',
    extract_inf_udf(col('MD_FINAL_SCORE'), lit('FILT_MD'), col('LEN_FILT_MD'), lit(0.8))
).withColumn(
    'QA_FINAL_SCORE_EXTRACTED',
    extract_inf_udf(col('QA_FINAL_SCORE'), lit('FILT_QA'), col('LEN_FILT_QA'), lit(0.8))
)

# COMMAND ----------


# Extract the keys from the UDF output and create new columns
md_final_score_extracted_cols = currdf_spark_old.select('MD_FINAL_SCORE_EXTRACTED').first().asDict()['MD_FINAL_SCORE_EXTRACTED'].keys()
qa_final_score_extracted_cols = currdf_spark_old.select('QA_FINAL_SCORE_EXTRACTED').first().asDict()['QA_FINAL_SCORE_EXTRACTED'].keys()

for col_name in md_final_score_extracted_cols:
    currdf_spark_old = currdf_spark_old.withColumn(col_name, col('MD_FINAL_SCORE_EXTRACTED').getItem(col_name))

for col_name in qa_final_score_extracted_cols:
    currdf_spark_old = currdf_spark_old.withColumn(col_name, col('QA_FINAL_SCORE_EXTRACTED').getItem(col_name))



# COMMAND ----------

# Drop the intermediate columns
currdf_spark_old = currdf_spark_old.drop('MD_FINAL_SCORE_EXTRACTED', 'QA_FINAL_SCORE_EXTRACTED')

new_columns = [col.replace('.', '') for col in currdf_spark_old.columns]
for old_col, new_col in zip(currdf_spark_old.columns, new_columns):
    currdf_spark_old = currdf_spark_old.withColumnRenamed(old_col, new_col)

columns_filt =(['ENTITY_ID', 'CALL_ID', 'VERSION_ID', 'DATE', 'CALL_NAME', 'COMPANY_NAME', 'LEN_FILT_MD', 'LEN_FILT_QA', 'FILT_MD', 'FILT_QA'] + 
           [col.replace('.', '') for col in md_final_score_extracted_cols] + 
          [col.replace('.', '') for col in qa_final_score_extracted_cols] )
currdf_spark_old = currdf_spark_old.select(*columns_filt)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Refactored 

# COMMAND ----------



# COMMAND ----------

currdf_spark = processing_nested_columns(currdf_spark_ref,
                                         nested_columns = ['MD_FINAL_SCORE_EXTRACTED', 'QA_FINAL_SCORE_EXTRACTED'],
                                         fixed_columns = ['ENTITY_ID', 'CALL_ID', 'VERSION_ID', 'DATE', 'CALL_NAME', 'COMPANY_NAME', 'LEN_FILT_MD', 'LEN_FILT_QA', 'FILT_MD', 'FILT_QA']
                                         )

# COMMAND ----------

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

currdf_spark = convert_column_types(currdf_spark, 
                                    float_type_cols = float_type_cols, 
                                    float_type_cols = array_type_cols)

# COMMAND ----------



# COMMAND ----------

import time 

while 1>0:
  time.sleep(60)

# COMMAND ----------



# COMMAND ----------

