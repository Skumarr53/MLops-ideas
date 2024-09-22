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

# Step 2: Initialize Spark Session (if not already initialized)
spark = SparkSession.builder.appName("Optimized_NLI_Inference").getOrCreate()

# Step 3: Define Constants and Broadcast Variables
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

# Step 4: Define Helper Functions

# Function to parse JSON strings to lists
def parse_json_list(s):
    try:
        return json.loads(s)
    except:
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
    tokenizer = AutoTokenizer.from_pretrained(MODEL_FOLDER_PATH + MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_FOLDER_PATH + MODEL_NAME)
    
    if enable_quantization:
        model = torch.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8
        )
        print("Model quantization enabled.")
    else:
        print("Model quantization disabled.")
    
    device = 0 if torch.cuda.is_available() else -1
    nli_pipeline = pipeline(
        task="text-classification",
        model=model,
        tokenizer=tokenizer,
        device=device
    )
    return nli_pipeline

# Register the initialization function as a global variable to ensure it's loaded once per executor
nli_pipeline = None

# Step 5: Load Data from Snowflake Directly into Spark DataFrame
min_date_query = (pd.to_datetime(dbutils.widgets.get("Start Month") + "-01-" + dbutils.widgets.get("Start Year"))).strftime('%Y-%m-%d')
max_date_query = (pd.to_datetime(dbutils.widgets.get("End Month") + "-01-" + dbutils.widgets.get("End Year"))).strftime('%Y-%m-%d')
mind = f"'{min_date_query}'"
maxd = f"'{max_date_query}'"
print(f'The next query spans {mind} to {maxd}')

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

# Step 6: Validate DataFrame
if spark_df.head(1):
    min_parsed_date = spark_df.agg({"PARSED_DATETIME_EASTERN_TZ": "min"}).collect()[0][0]
    max_parsed_date = spark_df.agg({"PARSED_DATETIME_EASTERN_TZ": "max"}).collect()[0][0]
    row_count = spark_df.count()
    col_count = len(spark_df.columns)
    print(f'The data spans from {min_parsed_date} to {max_parsed_date} and has {row_count} rows and {col_count} columns.')
else:
    print('No new transcripts to parse.')
    dbutils.notebook.exit(1)
    os._exit(1)

# Step 7: Data Transformation Using Spark DataFrame API

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
# Note: The original code applied `ast.literal_eval`, but here we've parsed JSON. Ensure the data is correctly parsed.
spark_df = spark_df \
    .withColumn('SENT_LABELS_FILT_QA', expr("filter(SENT_LABELS_FILT_QA, (x, i) -> not FILTER_QA[i].endsWith('?'))")) \
    .withColumn('FILT_QA', expr("filter(FILT_QA, x -> not x.endswith('?'))")) \
    .withColumn('LEN_FILT_QA', size(col('FILT_QA')))

# Create text pairs for MD and QA
spark_df = spark_df \
    .withColumn('TEXT_PAIRS_MD', create_text_pairs_udf(col('FILT_MD'))) \
    .withColumn('TEXT_PAIRS_QA', create_text_pairs_udf(col('FILT_QA')))

# Step 8: Define Pandas UDF for Batch Inference

from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import ArrayType, StructType, StructField, StringType, FloatType

# Define the schema for the inference results
inference_schema = ArrayType(StructType([
    StructField("label", StringType(), False),
    StructField("score", FloatType(), False)
]))

# Define the Pandas UDF for MD inference
@pandas_udf(inference_schema)
def md_inference_udf(text_pairs_series: pd.Series) -> pd.Series:
    global nli_pipeline
    if nli_pipeline is None:
        nli_pipeline = initialize_nli_pipeline(enable_quantization=ENABLE_QUANTIZATION)
    # Process each row's list of text pairs individually
    results = text_pairs_series.apply(lambda x: nli_pipeline(x, padding=True, top_k=None, batch_size=16, truncation=True, max_length=512) if x else [])
    return results

# Define the Pandas UDF for QA inference
@pandas_udf(inference_schema)
def qa_inference_udf(text_pairs_series: pd.Series) -> pd.Series:
    global nli_pipeline
    if nli_pipeline is None:
        nli_pipeline = initialize_nli_pipeline(enable_quantization=ENABLE_QUANTIZATION)
    # Process each row's list of text pairs individually
    results = text_pairs_series.apply(lambda x: nli_pipeline(x, padding=True, top_k=None, batch_size=16, truncation=True, max_length=512) if x else [])
    return results

# Step 9: Apply Inference UDFs to DataFrame

spark_df = spark_df \
    .withColumn('MD_RESULT', md_inference_udf(col('TEXT_PAIRS_MD'))) \
    .withColumn('QA_RESULT', qa_inference_udf(col('TEXT_PAIRS_QA')))

# Step 10: Post-processing of Inference Results

# Define a UDF to compute binary totals based on a threshold
def compute_binary_totals(results, threshold=0.8):
    return [1 if r['score'] > threshold else 0 for r in results]

binary_totals_udf = udf(lambda results: compute_binary_totals(results), ArrayType(IntegerType()))

# Apply the UDFs
spark_df = spark_df \
    .withColumn('MD_TOTAL', binary_totals_udf(col('MD_RESULT'))) \
    .withColumn('QA_TOTAL', binary_totals_udf(col('QA_RESULT')))

# Define UDFs to compute COUNT and REL
def compute_count(total_list):
    return float(sum(total_list)) if total_list else 0.0

def compute_rel(count, length):
    return count / length if length > 0 else None

count_udf = udf(compute_count, FloatType())
rel_udf = udf(compute_rel, FloatType())

# Apply COUNT and REL UDFs
spark_df = spark_df \
    .withColumn('MD_COUNT', count_udf(col('MD_TOTAL'))) \
    .withColumn('MD_REL', rel_udf(col('MD_COUNT'), col('LEN_FILT_MD'))) \
    .withColumn('QA_COUNT', count_udf(col('QA_TOTAL'))) \
    .withColumn('QA_REL', rel_udf(col('QA_COUNT'), col('LEN_FILT_QA')))

# Round the scores to 4 decimal places
spark_df = spark_df \
    .withColumn('MD_SCORES', expr("transform(MD_RESULT, x -> round(x.score, 4))")) \
    .withColumn('QA_SCORES', expr("transform(QA_RESULT, x -> round(x.score, 4))"))

# Step 11: Final Data Aggregation and Cleaning

# Select and rename necessary columns
final_columns = [
    'ENTITY_ID', 'DATE', 'CALL_NAME', 'COMPANY_NAME',
    'LEN_FILT_MD', 'LEN_FILT_QA',
    'MD_COUNT', 'MD_REL', 'MD_TOTAL', 'MD_SCORES',
    'QA_COUNT', 'QA_REL', 'QA_TOTAL', 'QA_SCORES'
]

final_df = spark_df.select(*final_columns)

# Step 12: Additional Transformations (e.g., Sentiment Scoring)
# Placeholder for sentiment scoring functions and transformations
def sentscore(text, labels, weight=False):
    # Placeholder implementation
    return 0.0

sentscore_udf = udf(sentscore, FloatType())

# Example sentiment scoring (to be customized based on actual logic)
# final_df = final_df.withColumn('TP1_SENT_NLI_MD', sentscore_udf(col('Some_Text'), col('Some_Labels')))
# Repeat for other sentiment columns as needed

# Step 13: Validation to Ensure Output Consistency

# Function to perform sample validation
def validate_output(original_df, new_df):
    # Compare row counts
    original_count = original_df.count()
    new_count = new_df.count()
    assert original_count == new_count, f"Row count mismatch: {original_count} vs {new_count}"
    
    # Compare sample data
    original_sample = original_df.sample(False, 0.01).collect()
    new_sample = new_df.sample(False, 0.01).collect()
    
    # Implement detailed comparison logic as needed
    # For simplicity, we'll just print counts
    print(f"Original DataFrame rows: {original_count}, New DataFrame rows: {new_count}")
    print("Sample validation completed.")

# Note: To perform validation, you should have the original output for comparison.
# Here, we assume you have access to it as `original_output_df`.
# Uncomment and modify the following lines as per your context.

# original_output_df = ...  # Load the original output DataFrame
# validate_output(original_output_df, final_df)

# Step 14: Write the Final DataFrame Back to Snowflake

# Define Snowflake connection options (Ensure credentials are securely managed)
sfOptions = {
    "sfURL" : "your_snowflake_url",
    "sfUser" : dbutils.secrets.get(scope="snowflake", key="username"),
    "sfPassword" : dbutils.secrets.get(scope="snowflake", key="password"),
    "sfDatabase" : "EDS_PROD",
    "sfSchema" : "QUANT",
    "sfWarehouse" : "your_warehouse"
}

# Define the target table name
tablename_curr = 'YUJING_MASS_NLI_DEMAND_DEV_3'

# Write to Snowflake
final_df.write \
    .format("snowflake") \
    .options(**sfOptions) \
    .option("dbtable", tablename_curr) \
    .mode("overwrite") \
    .save()

print("Data successfully written to Snowflake.")

# Step 15: Additional Optimizations for Standard_D16ds_v5 Cluster

# Optimize Spark configurations based on cluster specifications
spark.conf.set("spark.executor.instances", "5")  # One executor per worker node
spark.conf.set("spark.executor.cores", "16")     # Utilize all 16 vCPUs per node
spark.conf.set("spark.executor.memory", "60g")   # Allocate memory (out of 64 GB)
spark.conf.set("spark.executor.resource.gpu.amount", "1")
spark.conf.set("spark.task.resource.gpu.amount", "1")
spark.conf.set("spark.driver.memory", "16g")     # Allocate driver memory

# Enable GPU scheduling and isolation
spark.conf.set("spark.task.resource.gpu.discoveryScript", "/path/to/gpu-discovery-script.sh")  # Customize as needed
spark.conf.set("spark.executor.resource.gpu.discoveryScript", "/path/to/gpu-discovery-script.sh")  # Customize as needed
spark.conf.set("spark.task.resource.gpu.vendor", "nvidia")  # Assuming NVIDIA GPUs

# Note: Ensure that the GPU discovery script is properly configured and executable on your cluster.

# Optional: Cache intermediate DataFrames if reused multiple times
# final_df.cache()

# Optional: Persist DataFrames to speed up repeated accesses
# final_df.persist(StorageLevel.MEMORY_AND_DISK)

print("Spark configurations optimized for Standard_D16ds_v5 cluster.")
