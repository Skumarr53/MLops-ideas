# Import necessary libraries
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf, explode, size, to_timestamp, lit, concat_ws, pandas_udf, PandasUDFType
from pyspark.sql.types import ArrayType, StringType, StructType, StructField, FloatType, IntegerType, TimestampType, MapType
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
import ast
import pandas as pd

# Initialize Spark Session with optimized GPU configurations
spark = (SparkSession.builder 
    .appName("OptimizedNLIProcessing") 
    .config("spark.executor.instances", "5")            # One executor per worker node
    .config("spark.executor.cores", "5")                # 5 cores per executor
    .config("spark.executor.memory", "90g")             # 90 GB RAM per executor
    .config("spark.executor.memoryOverhead", "10g")     # 10 GB overhead per executor
    # GPU Configuration
    .config("spark.executor.resource.gpu.amount", "1")   # 1 GPU per executor
    .config("spark.task.resource.gpu.amount", "1")       # 1 GPU per task
    # Driver Configuration
    .config("spark.driver.cores", "6")                   # 6 cores for driver
    .config("spark.driver.memory", "90g")                # 90 GB RAM for driver
    .config("spark.driver.memoryOverhead", "10g")        # 10 GB overhead for driver
    # Shuffle and Parallelism
    .config("spark.sql.shuffle.partitions", "100")       # 100 shuffle partitions
    # Disable Dynamic Allocation
    .config("spark.dynamicAllocation.enabled", "false")   # Fixed executors for GPU usage
    # Additional Configurations
    .config("spark.sql.legacy.setCommandRejectsSparkCoreConfs", "false")
    .config("spark.rpc.message.maxSize", "1024")
    .getOrCreate())

# Set additional Spark configurations if needed
spark.conf.set('spark.rpc.message.maxSize', '1024')

# Broadcast model configuration details instead of the pipeline object
model_info = {
    "model_folder_path": "/dbfs/mnt/access_work/UC25/Libraries/HuggingFace/",
    "model_name": "deberta-v3-large-zeroshot-v2",
    "task": "text-classification"
}
model_broadcast = spark.sparkContext.broadcast(model_info)

# Define a function to initialize the pipeline within each executor
def get_pipeline():
    if not hasattr(get_pipeline, "pl_inference"):
        # Retrieve model configuration from broadcast variable
        model_config = model_broadcast.value
        tokenizer = AutoTokenizer.from_pretrained(model_config["model_folder_path"] + model_config["model_name"])
        model = AutoModelForSequenceClassification.from_pretrained(model_config["model_folder_path"] + model_config["model_name"])
        device = 0 if torch.cuda.is_available() else -1
        get_pipeline.pl_inference = pipeline(task=model_config["task"], model=model, tokenizer=tokenizer, device=device)
    return get_pipeline.pl_inference

# Define the Pandas UDF for transformer inference
inference_schema = ArrayType(StructType([
    StructField("label", StringType(), False),
    StructField("score", FloatType(), False)
]))

@pandas_udf(inference_schema, PandasUDFType.SCALAR)
def transformer_inference_pandas_udf(texts: pd.Series) -> pd.Series:
    # Initialize the pipeline once per executor
    pl_inference = get_pipeline()
    
    # Prepare the texts for inference
    processed_texts = texts.apply(lambda texts_list: [f"{it[0]}</s></s>{it[1]}" for it in texts_list])
    
    # Perform inference
    results = processed_texts.apply(pl_inference)
    
    return results

# Define UDF for inference summary (assuming it's a standard UDF)
def inference_summary(texts, inference_result):
    topic_set = ["topic1", "topic2", "topic3"]  # Define your topic set
    result_dict = {tp + '.': [] for tp in topic_set}
    total_dict = {tp + '.': [] for tp in topic_set}
    
    for i, (text_pair, inference) in enumerate(zip(texts, inference_result)):
        text1, text2_label = text_pair
        for s in inference:
            if s['label'] == 'entailment' and s['score'] > 0.91:
                result_dict[text2_label].append(text1)
                total_dict[text2_label].append(1)
            elif s['label'] == 'entailment':
                total_dict[text2_label].append(0)
    
    return result_dict, total_dict

# Define the schema for the summary UDF
schema_summary = StructType([
    StructField("result_dict", MapType(StringType(), ArrayType(StringType())), False),
    StructField("total_dict", MapType(StringType(), ArrayType(IntegerType())), False)
])

# Register the summary UDF (assuming it's a standard UDF due to complex return types)
summary_udf = udf(inference_summary, schema_summary)

# Define UDF for parsing string representations of lists
parse_udf = udf(lambda x: ast.literal_eval(x) if x else [], ArrayType(StringType()))

# Define function to create text pairs
def create_text_pairs(transcripts, labels, inference_template):
    text1 = []
    text2 = []
    for t in transcripts:
        for l in labels:
            text1.append(t)
            text2.append(f"{inference_template}{l}.")
    return list(zip(text1, text2))

# Define the schema for text pairs
text_pairs_schema = ArrayType(StructType([
    StructField("text1", StringType(), False),
    StructField("text2", StringType(), False)
]))

# Define UDF for creating text pairs
create_text_pairs_udf = udf(lambda transcripts: create_text_pairs(
    transcripts, 
    ["This text is about consumer strength", 
     "This text is about consumer weakness", 
     "This text is about reduced consumer's spending patterns"], 
    ""), 
    text_pairs_schema
)

# Read parameters from widgets or define them directly
import pandas as pd

minDateNewQuery = pd.to_datetime("2021-01-01").strftime('%Y-%m-%d')
maxDateNewQuery = pd.to_datetime("2022-01-01").strftime('%Y-%m-%d')

mind = f"'{minDateNewQuery}'"
maxd = f"'{maxDateNewQuery}'"

print(f'The next query spans {mind} to {maxd}')

# Initialize DBFS helper and read data
# Assuming DBFShelper is defined elsewhere and correctly initialized
myDBFS = DBFShelper()
new_sf = pd.read_pickle(r'/dbfs/FileStore/Yujing/resources/mysf_prod_quant.pkl')

# Construct and execute the Snowflake query using Spark
tsQuery = (f"SELECT CALL_ID, ENTITY_ID, DATE, FILT_MD, FILT_QA, CALL_NAME, COMPANY_NAME, "
           f"EARNINGS_CALL, ERROR, TRANSCRIPT_STATUS, UPLOAD_DT_UTC, VERSION_ID, "
           f"EVENT_DATETIME_UTC, PARSED_DATETIME_EASTERN_TZ, SENT_LABELS_FILT_MD, "
           f"SENT_LABELS_FILT_QA "
           f"FROM EDS_PROD.QUANT.PARTHA_FUND_CTS_STG_1_VIEW "
           f"WHERE DATE >= {mind} AND DATE < {maxd} "
           f"ORDER BY PARSED_DATETIME_EASTERN_TZ DESC")

resultspkdf = new_sf.read_from_snowflake(tsQuery)  # Ensure new_sf is correctly initialized
currdf_spark = resultspkdf

# Convert to Spark DataFrame and parse necessary columns
currdf_spark = currdf_spark \
    .withColumn("FILT_MD", parse_udf(col("FILT_MD"))) \
    .withColumn("FILT_QA", parse_udf(col("FILT_QA"))) \
    .withColumn("SENT_LABELS_FILT_MD", parse_udf(col("SENT_LABELS_FILT_MD"))) \
    .withColumn("SENT_LABELS_FILT_QA", parse_udf(col("SENT_LABELS_FILT_QA")))

# Add length columns
currdf_spark = currdf_spark \
    .withColumn("LEN_FILT_MD", size(col("FILT_MD"))) \
    .withColumn("LEN_FILT_QA", size(col("FILT_QA")))

# Sort and drop duplicates within Spark
currdf_spark = currdf_spark.orderBy(col("UPLOAD_DT_UTC").asc()) \
    .dropDuplicates(["ENTITY_ID", "EVENT_DATETIME_UTC"])

# Filter out rows with no data and handle exit
if currdf_spark.count() > 0:
    date_min = currdf_spark.agg({"PARSED_DATETIME_EASTERN_TZ": "min"}).collect()[0][0]
    date_max = currdf_spark.agg({"PARSED_DATETIME_EASTERN_TZ": "max"}).collect()[0][0]
    row_count = currdf_spark.count()
    col_count = len(currdf_spark.columns)
    print(f'The data spans from {date_min} to {date_max} and has {row_count} rows and {col_count} columns.')
else:
    print('No new transcripts to parse.')
    dbutils.notebook.exit(1)
    os._exit(1)

# Create text pairs for MD and QA sections
currdf_spark = currdf_spark \
    .withColumn("TEXT_PAIRS_MD", create_text_pairs_udf(col("FILT_MD"))) \
    .withColumn("TEXT_PAIRS_QA", create_text_pairs_udf(col("FILT_QA")))

# Apply transformer inference using Pandas UDF
currdf_spark = currdf_spark \
    .withColumn("MD_RESULT", transformer_inference_pandas_udf(col("TEXT_PAIRS_MD"))) \
    .withColumn("QA_RESULT", transformer_inference_pandas_udf(col("TEXT_PAIRS_QA")))

# Optionally, apply summary UDF if necessary
# Assuming inference_summary requires texts and results
currdf_spark = currdf_spark \
    .withColumn("SUMMARY", summary_udf(col("TEXT_PAIRS_MD"), col("TEXT_PAIRS_QA"), col("MD_RESULT"), col("QA_RESULT")))

# Avoid collecting data to the driver
# Instead, write results to storage or proceed with further transformations
# Example: Write to Parquet
currdf_spark.write.mode("overwrite").parquet("/mnt/access_work/UC25/ProcessedResults/")
