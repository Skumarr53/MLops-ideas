# Step 1: Import Necessary Libraries
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col, size, expr
from pyspark.sql.types import ArrayType, StringType, FloatType, IntegerType, StructType, StructField, TimestampType
import json
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import numpy as np
from pyspark.sql import functions as F
from pyspark.sql.functions import pandas_udf, PandasUDFType
import logging
import sys

# Step 2: Initialize Logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger("NLI_Inference")

# Step 3: Initialize Spark Session with GPU Configuration
spark = SparkSession.builder \
    .appName("Optimized_NLI_Inference") \
    .config("spark.executor.resource.gpu.amount", "1") \
    .config("spark.task.resource.gpu.amount", "0.5") \
    .config("spark.executor.memory", "16g") \
    .config("spark.driver.memory", "16g") \
    .config("spark.executor.cores", "4") \
    .getOrCreate()

# Step 4: Define Constants and Broadcast Variables
MODEL_FOLDER_PATH = "/dbfs/mnt/access_work/UC25/Libraries/HuggingFace/"
MODEL_NAME = "deberta-v3-large-zeroshot-v2"
LABELS = [
    "This text is about consumer strength",
    "This text is about consumer weakness",
    "This text is about reduced consumer's spending patterns"
]
labels_broadcast = spark.sparkContext.broadcast(LABELS)

# Configuration Flags
ENABLE_QUANTIZATION = True  # Set to False to disable quantization
BATCH_SIZE = 32  # Adjust based on GPU memory and performance

# Step 5: Define Helper Functions

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

# Function to initialize the NLI pipeline on each executor
def initialize_nli_pipeline(enable_quantization=False):
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_FOLDER_PATH + MODEL_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_FOLDER_PATH + MODEL_NAME)
        
        if enable_quantization:
            model = torch.quantization.quantize_dynamic(
                model, {torch.nn.Linear}, dtype=torch.qint8
            )
            logger.info("Model quantization enabled.")
        else:
            logger.info("Model quantization disabled.")
        
        device = 0 if torch.cuda.is_available() else -1
        nli_pipeline = pipeline(
            task="text-classification",
            model=model,
            tokenizer=tokenizer,
            device=device
        )
        logger.info(f"NLI pipeline initialized on device: {'GPU' if device == 0 else 'CPU'}")
        return nli_pipeline
    except Exception as e:
        logger.error(f"Failed to initialize NLI pipeline: {e}")
        raise e

# Register the initialization function as a global variable to ensure it's loaded once per executor
nli_pipeline = None

# Step 6: Load Data from Snowflake Directly into Spark DataFrame
try:
    min_date_query = (pd.to_datetime(dbutils.widgets.get("Start Month") + "-01-" + dbutils.widgets.get("Start Year"))).strftime('%Y-%m-%d')
    max_date_query = (pd.to_datetime(dbutils.widgets.get("End Month") + "-01-" + dbutils.widgets.get("End Year"))).strftime('%Y-%m-%d')
    mind = f"'{min_date_query}'"
    maxd = f"'{max_date_query}'"
    logger.info(f'The next query spans {mind} to {maxd}')
    
    ts_query = f"""
    SELECT CALL_ID, ENTITY_ID, DATE, FILT_MD, FILT_QA, CALL_NAME, COMPANY_NAME,
           EARNINGS_CALL, ERROR, TRANSCRIPT_STATUS, UPLOAD_DT_UTC, VERSION_ID,
           EVENT_DATETIME_UTC, PARSED_DATETIME_EASTERN_TZ, SENT_LABELS_FILT_MD,
           SENT_LABELS_FILT_QA
    FROM EDS_PROD.QUANT.PARTHA_FUND_CTS_STG_1_VIEW
    WHERE DATE >= {mind} AND DATE < {maxd}
    ORDER BY PARSED_DATETIME_EASTERN_TZ DESC
    """
    
    spark_df = newSF.read_from_snowflake(ts_query)
    logger.info("Data loaded from Snowflake successfully.")
except Exception as e:
    logger.error(f"Data loading failed: {e}")
    dbutils.notebook.exit(1)
    os._exit(1)

# Step 7: Validate DataFrame
if spark_df.head(1):
    min_parsed_date = spark_df.agg({"PARSED_DATETIME_EASTERN_TZ": "min"}).collect()[0][0]
    max_parsed_date = spark_df.agg({"PARSED_DATETIME_EASTERN_TZ": "max"}).collect()[0][0]
    row_count = spark_df.count()
    col_count = len(spark_df.columns)
    logger.info(f'The data spans from {min_parsed_date} to {max_parsed_date} and has {row_count} rows and {col_count} columns.')
else:
    logger.warning('No new transcripts to parse.')
    dbutils.notebook.exit(1)
    os._exit(1)

# Step 8: Data Transformation Using Spark DataFrame API

# Convert stringified lists to actual arrays
spark_df = spark_df \
    .withColumn('FILT_MD', parse_json_udf(col('FILT_MD'))) \
    .withColumn('FILT_QA', parse_json_udf(col('FILT_QA'))) \
    .withColumn('SENT_LABELS_FILT_MD', parse_json_udf(col('SENT_LABELS_FILT_MD'))) \
    .withColumn('SENT_LABELS_FILT_QA', parse_json_udf(col('SENT_LABELS_FILT_QA'))) \
    .withColumn('LEN_FILT_MD', size(col('FILT_MD'))) \
    .withColumn('LEN_FILT_QA', size(col('FILT_QA'))) \
    .dropDuplicates(['ENTITY_ID', 'EVENT_DATETIME_UTC']) \
    .orderBy(col('UPLOAD_DT_UTC').asc())

# Filter out QA sentences ending with '?'
spark_df = spark_df \
    .withColumn('SENT_LABELS_FILT_QA', expr("filter(SENT_LABELS_FILT_QA, (x, i) -> not FILT_QA[i].endsWith('?'))")) \
    .withColumn('FILT_QA', expr("filter(FILT_QA, x -> not x.endswith('?'))")) \
    .withColumn('LEN_FILT_QA', size(col('FILT_QA')))

# Create text pairs for MD and QA
spark_df = spark_df \
    .withColumn('TEXT_PAIRS_MD', create_text_pairs_udf(col('FILT_MD'))) \
    .withColumn('TEXT_PAIRS_QA', create_text_pairs_udf(col('FILT_QA')))

# Step 9: Define Iterator-Based Pandas UDF for Batch Inference with Logging and Error Handling

from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import ArrayType, StructType, StructField, StringType, FloatType

# Define the schema for the inference results
inference_schema = ArrayType(StructType([
    StructField("label", StringType(), False),
    StructField("score", FloatType(), False)
]))

# Define the iterator-based Pandas UDF for MD inference
@pandas_udf(inference_schema, PandasUDFType.SCALAR_ITER)
def md_inference_udf(iterator):
    global nli_pipeline
    if nli_pipeline is None:
        nli_pipeline = initialize_nli_pipeline(enable_quantization=ENABLE_QUANTIZATION)
    
    for batch_num, batch in enumerate(iterator, start=1):
        logger.info(f"Processing MD inference batch {batch_num} with {len(batch)} rows.")
        try:
            # Flatten the list of text pairs in the batch
            flat_text_pairs = [pair for sublist in batch for pair in sublist]
            logger.debug(f"Batch {batch_num}: Total text pairs to infer: {len(flat_text_pairs)}")
            
            if flat_text_pairs:
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
            else:
                results = []
            
            # Split results back to original rows
            split_results = []
            idx = 0
            for pairs in batch:
                if pairs:
                    split_results.append(results[idx:idx+len(pairs)])
                    idx += len(pairs)
                else:
                    split_results.append([])
            
            yield pd.Series(split_results)
        except Exception as e:
            logger.error(f"Error in MD inference batch {batch_num}: {e}")
            # Yield empty results for this batch to continue processing
            yield pd.Series([[] for _ in batch])

# Define the iterator-based Pandas UDF for QA inference
@pandas_udf(inference_schema, PandasUDFType.SCALAR_ITER)
def qa_inference_udf(iterator):
    global nli_pipeline
    if nli_pipeline is None:
        nli_pipeline = initialize_nli_pipeline(enable_quantization=ENABLE_QUANTIZATION)
    
    for batch_num, batch in enumerate(iterator, start=1):
        logger.info(f"Processing QA inference batch {batch_num} with {len(batch)} rows.")
        try:
            # Flatten the list of text pairs in the batch
            flat_text_pairs = [pair for sublist in batch for pair in sublist]
            logger.debug(f"Batch {batch_num}: Total text pairs to infer: {len(flat_text_pairs)}")
            
            if flat_text_pairs:
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
            else:
                results = []
            
            # Split results back to original rows
            split_results = []
            idx = 0
            for pairs in batch:
                if pairs:
                    split_results.append(results[idx:idx+len(pairs)])
                    idx += len(pairs)
                else:
                    split_results.append([])
            
            yield pd.Series(split_results)
        except Exception as e:
            logger.error(f"Error in QA inference batch {batch_num}: {e}")
            # Yield empty results for this batch to continue processing
            yield pd.Series([[] for _ in batch])

# Step 10: Apply Inference UDFs to DataFrame with Repartitioning and Caching

# Determine the number of GPU workers dynamically
# This assumes that each GPU worker has one GPU. Adjust accordingly if multiple GPUs per worker.
num_gpu_workers = spark.sparkContext._conf.get("spark.executor.instances")
try:
    num_gpu_workers = int(num_gpu_workers)
except:
    num_gpu_workers = 4  # Default value; replace with actual number if known

logger.info(f"Repartitioning DataFrame to {num_gpu_workers} partitions for GPU workers.")
spark_df = spark_df.repartition(num_gpu_workers)

# Apply the MD Inference UDF
spark_df = spark_df.withColumn('MD_RESULT', md_inference_udf(col('TEXT_PAIRS_MD')))

# Apply the QA Inference UDF
spark_df = spark_df.withColumn('QA_RESULT', qa_inference_udf(col('TEXT_PAIRS_QA')))

# Step 11: Persist the DataFrame and Display Sample Results
spark_df.cache()
logger.info("DataFrame cached after inference.")

# Optional: Show a sample of the results
spark_df.select('CALL_ID', 'MD_RESULT', 'QA_RESULT').show(5, truncate=False)

# Step 12: Save or Further Process the Results
# Example: Write the results back to Snowflake or another storage system
# spark_df.write \
#     .format("snowflake") \
#     .options(**snowflake_options) \
#     .option("dbtable", "EDS_PROD.QUANT.PARTHA_FUND_CTS_RESULTS") \
#     .mode("overwrite") \
#     .save()

logger.info("Inference pipeline completed successfully.")
