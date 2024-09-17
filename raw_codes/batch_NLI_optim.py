#EDITED: Import necessary Spark and Transformer libraries for optimization
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf, explode, size, to_timestamp, lit, concat_ws
from pyspark.sql.types import ArrayType, StringType, StructType, StructField, FloatType, IntegerType, TimestampType, StructType, StructField
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
import ast


#EDITED: Initialize Spark Session with GPU configurations
spark = (SparkSession.builder 
    .appName("OptimizedNLIProcessing") 
    .config("spark.executor.resource.gpu.amount", "1")  
    .config("spark.task.resource.gpu.amount", "1")       
    .config("spark.sql.shuffle.partitions", "150")   
    .config("spark.executor.memoryOverhead", "10g")  
    .config("spark.dynamicAllocation.enabled", "true") 
    .config("spark.dynamicAllocation.minExecutors", "1") 
    .config("spark.dynamicAllocation.maxExecutors", "5") 
    .getOrCreate())

#EDITED: Set Spark configurations for optimized performance
spark.conf.set("spark.sql.legacy.setCommandRejectsSparkCoreConfs", False)
spark.conf.set('spark.rpc.message.maxSize', '1024')


#EDITED: Load and broadcast the transformer model to all executors
model_1_folder_name = "deberta-v3-large-zeroshot-v2"
model_folder_path = "/dbfs/mnt/access_work/UC25/Libraries/HuggingFace/"

tokenizer_1 = AutoTokenizer.from_pretrained(model_folder_path + model_1_folder_name)
model_1 = AutoModelForSequenceClassification.from_pretrained(model_folder_path + model_1_folder_name)

#EDITED: Initialize the pipeline with GPU if available and broadcast the model
device = 0 if torch.cuda.is_available() else -1
pl_inference1 = pipeline(task="text-classification", model=model_1, tokenizer=tokenizer_1, device=device)
model_broadcast = spark.sparkContext.broadcast(pl_inference1)


#EDITED: Define UDF for transformer inference
def transformer_inference(texts):
    texts = [f"{it[0]}</s></s>{it[1]}" for it in texts]
    model = model_broadcast.value
    return model(texts)

inference_udf = udf(transformer_inference, ArrayType(StructType([
    StructField("label", StringType(), False),
    StructField("score", FloatType(), False)
])))

def inference_summary(texts, inference_result):
    topic_set = ["topic1", "topic2", "topic3"]  # Define your topic set
    result_dict = {tp + '.': [] for tp in topic_set}
    total_dict = {tp + '.': [] for tp in topic_set}
    for i, text1, text2 in enumerate(texts):
        for s in inference_result[i]:
            if s['label'] == 'entailment':
                if s['score'] > 0.91:
                    result_dict[text2[i]].append(text1)
                    total_dict[text2[i]].append(1)
                else:
                    total_dict[text2[i]].append(0)
    return result_dict, total_dict

# Define the schema for the UDF
schema_summary = StructType([
    StructField("result_dict", MapType(StringType(), ArrayType(StringType())), False),
    StructField("total_dict", MapType(StringType(), ArrayType(IntegerType())), False)
])

#EDITED: Define UDF for parsing string representations of lists
parse_udf = udf(lambda x: ast.literal_eval(x) if x else [], ArrayType(StringType()))
summary_udf = udf(lambda text1, text2, inference_result: inference_summary1(text1, text2, inference_result), schema_summary)


#EDITED: Define function to create text pairs
def create_text_pairs(transcripts, labels, inference_template):
    text1 = []
    text2 = []
    for t in transcripts:
        for l in labels:
            text1.append(t)
            text2.append(f"{inference_template}{l}.")
    return list(zip(text1, text2))
  
schema = ArrayType(StructType([
    StructField("text1", StringType(), False),
    StructField("text2", StringType(), False)
]))

create_text_pairs_udf = udf(lambda transcripts: create_text_pairs(transcripts, 
                                                                    ["This text is about consumer strength", 
                                                                     "This text is about consumer weakness", 
                                                                     "This text is about reduced consumer's spending patterns"], 
                                                                    ""), 
                            ArrayType(StructType([
                                StructField("text1", StringType(), False),
                                StructField("text2", StringType(), False)
                            ])))


#EDITED: Read parameters from widgets
# minDateNewQuery = (pd.to_datetime(dbutils.widgets.get("Start Month") + "-01-" + dbutils.widgets.get("Start Year"))).strftime('%Y-%m-%d')
# maxDateNewQuery = (pd.to_datetime(dbutils.widgets.get("End Month") + "-01-" + dbutils.widgets.get("End Year"))).strftime('%Y-%m-%d')
import pandas as pd
minDateNewQuery = pd.to_datetime("01-01-2021").strftime('%Y-%m-%d')
maxDateNewQuery = pd.to_datetime("01-01-2022").strftime('%Y-%m-%d')

mind = f"'{minDateNewQuery}'"
maxd = f"'{maxDateNewQuery}'"

print(f'The next query spans {mind} to {maxd}')


myDBFS = DBFShelper()
# new_sf = pd.read_pickle(r'/dbfs' + myDBFS.iniPath + 'mysf_prod_quant.pkl')
new_sf = pd.read_pickle(r'/dbfs/FileStore/Yujing/resources/mysf_prod_quant.pkl')


#EDITED: Construct and execute the Snowflake query using Spark
tsQuery = (f"SELECT CALL_ID, ENTITY_ID, DATE, FILT_MD, FILT_QA, CALL_NAME, COMPANY_NAME, "
           f"EARNINGS_CALL, ERROR, TRANSCRIPT_STATUS, UPLOAD_DT_UTC, VERSION_ID, "
           f"EVENT_DATETIME_UTC, PARSED_DATETIME_EASTERN_TZ, SENT_LABELS_FILT_MD, "
           f"SENT_LABELS_FILT_QA "
           f"FROM EDS_PROD.QUANT.PARTHA_FUND_CTS_STG_1_VIEW "
           f"WHERE DATE >= {mind} AND DATE < {maxd} "
           f"ORDER BY PARSED_DATETIME_EASTERN_TZ DESC")

resultspkdf = new_sf.read_from_snowflake(tsQuery)  # Assuming newDBFS is correctly initialized
currdf_spark = resultspkdf

#EDITED: Convert to Spark DataFrame and parse necessary columns
currdf_spark = currdf_spark \
    .withColumn("FILT_MD", parse_udf(col("FILT_MD"))) \
    .withColumn("FILT_QA", parse_udf(col("FILT_QA"))) \
    .withColumn("SENT_LABELS_FILT_MD", parse_udf(col("SENT_LABELS_FILT_MD"))) \
    .withColumn("SENT_LABELS_FILT_QA", parse_udf(col("SENT_LABELS_FILT_QA")))

#EDITED: Add length columns
currdf_spark = currdf_spark \
    .withColumn("LEN_FILT_MD", size(col("FILT_MD"))) \
    .withColumn("LEN_FILT_QA", size(col("FILT_QA")))

#EDITED: Sort and drop duplicates within Spark
currdf_spark = currdf_spark.orderBy(col("UPLOAD_DT_UTC").asc()) \
    .dropDuplicates(["ENTITY_ID", "EVENT_DATETIME_UTC"])


#EDITED: Filter out rows with no data and handle exit
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


#EDITED: Create text pairs for MD and QA sections
labels = ["This text is about consumer strength", 
          "This text is about consumer weakness", 
          "This text is about reduced consumer's spending patterns"]

currdf_spark = currdf_spark \
    .withColumn("TEXT_PAIRS_MD", create_text_pairs_udf(col("FILT_MD"))) \
    .withColumn("TEXT_PAIRS_QA", create_text_pairs_udf(col("FILT_QA")))




#EDITED: Apply transformer inference using UDF
currdf_spark = currdf_spark \
    .withColumn("MD_RESULT", inference_udf(col("TEXT_PAIRS_MD"))) \
    .withColumn("QA_RESULT", inference_udf(col("TEXT_PAIRS_QA")))


pd_df = currdf_spark.toPandas()