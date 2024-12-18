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
from topic_modelling_package.nli_process import create_text_pairs, inference_summary, initialize_nli_infer_pipeline

# COMMAND ----------
### create spark session

#### old
# # Step 2: Initialize Spark Session (if not already initialized)
N_PARTITION = 240

spark_old = (SparkSession.builder.appName("Optimized_NLI_Inference")
         .config("spark.sql.shuffle.partitions",N_PARTITION)
         .config("spark.executor.resource.gpu.amount", "1")
         .config("spark.task.resource.gpu.amount", "1")
         .getOrCreate())


#### Refactored
spark_ref = initialize_spark_session(app_name = "NLI_Inference", shuffle_partitions = N_PARTITION)


###  Define Helper Functions

#### old

MODEL_FOLDER_PATH = "/dbfs/mnt/access_work/UC25/Libraries/HuggingFace/"
MODEL_NAME = "deberta-v3-large-zeroshot-v2"
LABELS = [
    "This text is about consumer strength",
    "This text is about consumer weakness",
    "This text is about reduced consumer's spending patterns"
]
labels_broadcast = spark_ref.sparkContext.broadcast(LABELS)

# Function to parse JSON strings to lists
def parse_json_list(s):
    try:
        return json.loads(s)
    except Exception as e:
        print(f"JSON parsing error: {e} for input: {s}")
        return []

# UDF to parse JSON lists
parse_json_udf = udf(parse_json_list, ArrayType(StringType()))

# Function to create text pairs for NLI
def create_text_pairs_old(filt):
    text_pairs = []
    for t in filt:
        for l in labels_broadcast.value:
            text_pairs.append(f"{t}</s></s>{l}.")
    return text_pairs

# UDF to create text pairs
create_text_pairs_udf_old = udf(create_text_pairs_old, ArrayType(StringType()))


#### Refactored
create_text_pairs_udf_ref = udf(create_text_pairs, ArrayType(StringType()))


### Function to initialize the NLI pipeline on each executor

#### old
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
    
nli_pipeline = None

#### Refactored

initialize_nli_infer_pipeline(model_path = MODEL_FOLDER_PATH + MODEL_NAME, enable_quantization = False)


#### inference summary

#### old
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



### Refactored
summary_udf_ref = udf(lambda texts, inference_result: inference_summary(texts, inference_result), schema_summary)


### Read Data from snowflake
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
           f"ORDER BY PARSED_DATETIME_EASTERN_TZ DESC")

currdf_spark_old = read_from_snowflake(tsQuery)  # Assuming newDBFS is correctly initialized

currdf_spark_ref = currdf_spark_old.copy()

### this step applies series of functions on the spark columns

#### old
currdf_spark_old = (currdf_spark_old 
    .withColumn('FILT_MD', parse_json_udf(col('FILT_MD'))) 
    .withColumn('FILT_QA', parse_json_udf(col('FILT_QA'))) 
    .withColumn('SENT_LABELS_FILT_MD', parse_json_udf(col('SENT_LABELS_FILT_MD'))) 
    .withColumn('SENT_LABELS_FILT_QA', parse_json_udf(col('SENT_LABELS_FILT_QA'))) 
    .withColumn('LEN_FILT_MD', size(col('FILT_MD'))) 
    .withColumn('LEN_FILT_QA', size(col('FILT_QA'))) 
    .dropDuplicates(['ENTITY_ID', 'EVENT_DATETIME_UTC']) 
    .orderBy(col('UPLOAD_DT_UTC').asc()))

currdf_spark_old = currdf_spark_old \
    .withColumn('TEXT_PAIRS_MD', create_text_pairs_udf_old(col('FILT_MD'))) \
    .withColumn('TEXT_PAIRS_QA', create_text_pairs_udf_old(col('FILT_QA')))


#### refactored

transformations1 = [
    ('FILT_MD', 'FILT_MD', parse_json_udf),
    ('FILT_QA', 'FILT_QA', parse_json_udf),
    ('SENT_LABELS_FILT_MD', 'SENT_LABELS_FILT_MD', parse_json_udf),
    ('SENT_LABELS_FILT_QA', 'SENT_LABELS_FILT_QA', parse_json_udf),
    ('LEN_FILT_MD', 'FILT_MD', size)
    ('LEN_FILT_QA', 'FILT_QA', size)
    ]

currdf_spark_ref = sparkdf_apply_transformations(currdf_spark_ref, transformations1)

currdf_spark_ref = (currdf_spark_ref
                    .dropDuplicates(['ENTITY_ID', 'EVENT_DATETIME_UTC'])
                    .orderBy(col('UPLOAD_DT_UTC').asc()))

    .withColumn('TEXT_PAIRS_MD', create_text_pairs_udf(col('FILT_MD'))) \

transformations2 = [
    ('TEXT_PAIRS_MD', 'FILT_MD', create_text_pairs_udf),
    ('TEXT_PAIRS_QA', 'FILT_QA', create_text_pairs_udf),
]
currdf_spark_ref = sparkdf_apply_transformations(currdf_spark_ref, transformations2)
