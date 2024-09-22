# Step 1: Import Necessary Libraries
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col, size, explode, collect_list, when, expr
from pyspark.sql.types import ArrayType, StringType, FloatType, IntegerType, StructType, StructField, TimestampType
import json
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd

# Initialize Spark Session (if not already initialized)
spark = SparkSession.builder.appName("Optimized_NLI_Inference").getOrCreate()

# Step 2: Define Constants and Broadcast Variables
MODEL_FOLDER_PATH = "/dbfs/mnt/access_work/UC25/Libraries/HuggingFace/"
MODEL_NAME = "deberta-v3-large-zeroshot-v2"
LABELS = [
    "This text is about consumer strength",
    "This text is about consumer weakness",
    "This text is about reduced consumer's spending patterns"
]
labels_broadcast = spark.sparkContext.broadcast(LABELS)

# Step 3: Define Helper Functions

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
def initialize_nli_pipeline():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_FOLDER_PATH + MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_FOLDER_PATH + MODEL_NAME)
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

# Step 4: Load Data from Snowflake Directly into Spark DataFrame
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

spark_df = new_sf.read_from_snowflake(ts_query)

# Step 5: Validate DataFrame
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

# Step 6: Data Transformation Using Spark DataFrame API

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

# Create text pairs for MD and QA
spark_df = spark_df \
    .withColumn('TEXT_PAIRS_MD', create_text_pairs_udf(col('FILT_MD'))) \
    .withColumn('TEXT_PAIRS_QA', create_text_pairs_udf(col('FILT_QA')))

# Step 7: Define Pandas UDF for Batch Inference

from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import ArrayType, StructType, StructField, StringType, FloatType

# Define the schema for the inference results
inference_schema = ArrayType(StructType([
    StructField("label", StringType()),
    StructField("score", FloatType())
]))

# Define the Pandas UDF for MD inference
@pandas_udf(inference_schema)
def md_inference_udf(text_pairs_series: pd.Series) -> pd.Series:
    global nli_pipeline
    if nli_pipeline is None:
        nli_pipeline = initialize_nli_pipeline()
    results = nli_pipeline(text_pairs_series.tolist(), padding=True, top_k=None, batch_size=16, truncation=True, max_length=512)
    return pd.Series(results)

# Define the Pandas UDF for QA inference
@pandas_udf(inference_schema)
def qa_inference_udf(text_pairs_series: pd.Series) -> pd.Series:
    global nli_pipeline
    if nli_pipeline is None:
        nli_pipeline = initialize_nli_pipeline()
    results = nli_pipeline(text_pairs_series.tolist(), padding=True, top_k=None, batch_size=16, truncation=True, max_length=512)
    return pd.Series(results)

# Step 8: Apply Inference UDFs to DataFrame

spark_df = spark_df \
    .withColumn('MD_RESULT', md_inference_udf(col('TEXT_PAIRS_MD'))) \
    .withColumn('QA_RESULT', qa_inference_udf(col('TEXT_PAIRS_QA')))

# Step 9: Post-processing of Inference Results

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

# Step 10: Final Data Aggregation and Cleaning

# Select and rename necessary columns
final_columns = [
    'ENTITY_ID', 'DATE', 'CALL_NAME', 'COMPANY_NAME',
    'LEN_FILT_MD', 'LEN_FILT_QA',
    'MD_COUNT', 'MD_REL', 'MD_TOTAL', 'MD_SCORES',
    'QA_COUNT', 'QA_REL', 'QA_TOTAL', 'QA_SCORES'
]

final_df = spark_df.select(*final_columns)

# Additional transformations as per original code (e.g., sentiment scoring)
# Define UDFs or use Spark's built-in functions as needed
# Example placeholder for sentiment scoring
def sentscore(text, labels, weight=False):
    # Placeholder implementation
    return 0.0

sentscore_udf = udf(sentscore, FloatType())

# Apply sentiment scoring (example)
# final_df = final_df.withColumn('TP1_SENT_NLI_MD', sentscore_udf(col('Some_Text'), col('Some_Labels')))
# Repeat for other sentiment columns as needed

# Step 11: Write the Final DataFrame Back to Snowflake

# Define Snowflake connection options
sfOptions = {
    "sfURL" : "your_snowflake_url",
    "sfUser" : "your_username",
    "sfPassword" : "your_password",
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
