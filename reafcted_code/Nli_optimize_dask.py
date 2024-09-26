import ast
import os
import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
import pickle
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
import dask
import dask.dataframe as dd
from dask.distributed import Client, LocalCUDACluster
from dask import delayed
from pyspark.sql.types import *
from pyspark.sql import functions as F

# Initialize Dask client with GPU support
# Ensure that each Dask worker has access to a GPU
cluster = LocalCUDACluster()  # Adjust parameters as needed for your Databricks environment
client = Client(cluster)

print("Dask cluster initialized with the following workers:")
print(client)

# Load model and tokenizer globally to ensure they are loaded once per worker
# Use Dask's worker initialization to load models per worker
from dask.distributed import get_worker

def load_model():
    worker = get_worker()
    if not hasattr(worker, 'model'):
        device = 0 if torch.cuda.is_available() else -1
        model_1_folder_name = "deberta-v3-large-zeroshot-v2"
        model_folder_path = "/dbfs/mnt/access_work/UC25/Libraries/HuggingFace/"
        tokenizer_1 = AutoTokenizer.from_pretrained(os.path.join(model_folder_path, model_1_folder_name))
        model_1 = AutoModelForSequenceClassification.from_pretrained(os.path.join(model_folder_path, model_1_folder_name))
        worker.pl_inference1 = pipeline(task="text-classification", model=model_1, tokenizer=tokenizer_1, device=device)
    return worker.pl_inference1

# Function to initialize the model on each worker
def initialize_worker():
    load_model()

# Register the worker initialization
client.run_on_scheduler(initialize_worker)

# Spark Configuration Settings
spark.conf.set("spark.sql.legacy.setCommandRejectsSparkCoreConfs", False)
spark.conf.set('spark.rpc.message.maxSize', '1024')  # Set the Spark RPC message max size

# Function to read from Snowflake
def read_from_snowflake(query):
    return new_sf.read_from_snowflake(query)

# Function to write to Snowflake
def write_to_snowflake(df, table_name):
    new_sf.db = 'EDS_PROD'
    new_sf.schema = 'QUANT'
    new_sf.write_to_snowflake_table(df, table_name)

# Load model paths and tokenizer
model_1_folder_name = "deberta-v3-large-zeroshot-v2"
model_folder_path = "/dbfs/mnt/access_work/UC25/Libraries/HuggingFace/"
tokenizer_1 = AutoTokenizer.from_pretrained(os.path.join(model_folder_path, model_1_folder_name))
model_1 = AutoModelForSequenceClassification.from_pretrained(os.path.join(model_folder_path, model_1_folder_name))

# Define class data path
class_data_path = "/dbfs/mnt/access_work/UC25/Topic Modeling/NLI Models/"

# Read start & end ranges from widgets
minDateNewQuery = (pd.to_datetime(dbutils.widgets.get("Start Month") + "-01-" + dbutils.widgets.get("Start Year"))).strftime('%Y-%m-%d')
maxDateNewQuery = (pd.to_datetime(dbutils.widgets.get("End Month") + "-01-" + dbutils.widgets.get("End Year"))).strftime('%Y-%m-%d')
mind = f"'{minDateNewQuery}'"
maxd = f"'{maxDateNewQuery}'"

print(f'The next query spans {mind} to {maxd}')

# Construct SQL query
tsQuery = (
    f"SELECT CALL_ID, ENTITY_ID, DATE, FILT_MD, FILT_QA, CALL_NAME, COMPANY_NAME, EARNINGS_CALL, "
    f"ERROR, TRANSCRIPT_STATUS, UPLOAD_DT_UTC, VERSION_ID, EVENT_DATETIME_UTC, "
    f"PARSED_DATETIME_EASTERN_TZ, SENT_LABELS_FILT_MD, SENT_LABELS_FILT_QA "
    f"FROM EDS_PROD.QUANT.PARTHA_FUND_CTS_STG_1_VIEW "
    f"WHERE DATE >= {mind} AND DATE < {maxd} "
    f"ORDER BY PARSED_DATETIME_EASTERN_TZ DESC;"
)

# Read data from Snowflake into Spark DataFrame
resultspkdf = read_from_snowflake(tsQuery)
currdf_spark = resultspkdf.toPandas()

if len(currdf_spark) > 0:
    print(f'The data spans from {currdf_spark["PARSED_DATETIME_EASTERN_TZ"].min()} to '
          f'{currdf_spark["PARSED_DATETIME_EASTERN_TZ"].max()} and has '
          f'{currdf_spark.shape[0]} rows and {currdf_spark.shape[1]} columns.')
else:
    print('No new transcripts to parse.')
    dbutils.notebook.exit(1)
    os._exit(1)

# Convert Spark DataFrame to Dask DataFrame
currdf_dd = dd.from_pandas(currdf_spark, npartitions=client.ncores())

# Apply necessary transformations using Dask

# Define functions for processing
def process_columns(df):
    # Apply ast.literal_eval safely
    df['FILT_MD'] = df['FILT_MD'].apply(ast.literal_eval, meta=('FILT_MD', 'object'))
    df['FILT_QA'] = df['FILT_QA'].apply(ast.literal_eval, meta=('FILT_QA', 'object'))
    df['SENT_LABELS_FILT_MD'] = df['SENT_LABELS_FILT_MD'].apply(ast.literal_eval, meta=('SENT_LABELS_FILT_MD', 'object'))
    df['SENT_LABELS_FILT_QA'] = df['SENT_LABELS_FILT_QA'].apply(ast.literal_eval, meta=('SENT_LABELS_FILT_QA', 'object'))
    df['LEN_FILT_MD'] = df['FILT_MD'].apply(len, meta=('LEN_FILT_MD', 'int'))
    df['LEN_FILT_QA'] = df['FILT_QA'].apply(len, meta=('LEN_FILT_QA', 'int'))
    return df

currdf_dd = process_columns(currdf_dd)

# Sort and drop duplicates
currdf_dd = currdf_dd.map_partitions(
    lambda df: df.sort_values(by='UPLOAD_DT_UTC').drop_duplicates(subset=['ENTITY_ID', 'EVENT_DATETIME_UTC'], keep='first')
)

# Define topic set
topic_set = [
    "This text is about consumer strength",
    "This text is about consumer weakness",
    "This text is about reduced consumer's spending patterns"
]

# Create text pairs
def create_text_pair(transcript, inference_template, labels):
    template = inference_template + "{label}."
    text1, text2 = [], []
    for t in transcript:
        for l in labels:
            text1.append(t)
            text2.append(template.format(label=l))
    return text1, text2

# Convert the create_text_pair function to work with Dask
def create_text_columns(df, column, new_text1, new_text2):
    df[new_text1], df[new_text2] = zip(*df[column].map(lambda x: create_text_pair(x, "", topic_set)))
    return df

currdf_dd = create_text_columns(currdf_dd, 'FILT_MD', 'TEXT1_MD', 'TEXT2_MD')
currdf_dd = create_text_columns(currdf_dd, 'FILT_QA', 'TEXT1_QA', 'TEXT2_QA')

# Define the inference summary function
def inference_summary1(text1, text2, results, threshold=0.8):
    total = 0
    scores = []
    for res in results:
        if res['score'] >= threshold:
            total += 1
            scores.append(res['score'])
    return total, scores

# Apply model inference using Dask's map_partitions and delayed functions
@dask.delayed
def perform_inference(text_pairs):
    pl_inference = load_model()
    combined_text = [f"{t1}</s></s>{t2}" for t1, t2 in text_pairs]
    results = pl_inference(combined_text, padding=True, top_k=None, batch_size=16, truncation=True, max_length=512)
    return results

def apply_inference(df, text1_col, text2_col, result_col):
    # Create list of text pairs
    text_pairs = list(zip(df[text1_col], df[text2_col]))
    # Perform inference
    results = perform_inference(text_pairs)
    df[result_col] = results
    return df

# Apply inference to MD and QA sections
currdf_dd = currdf_dd.map_partitions(lambda df: apply_inference(df, 'TEXT1_MD', 'TEXT2_MD', 'MD_RESULT'))
currdf_dd = currdf_dd.map_partitions(lambda df: apply_inference(df, 'TEXT1_QA', 'TEXT2_QA', 'QA_RESULT'))

# Compute the Dask DataFrame to get a pandas DataFrame
currdf_pandas = currdf_dd.compute()

# Apply the inference_summary1 function
currdf_pandas['MD_FINAL_TOTAL'], currdf_pandas['MD_FINAL_SCORE'] = zip(*currdf_pandas.apply(
    lambda x: inference_summary1(x['TEXT1_MD'], x['TEXT2_MD'], x['MD_RESULT'], 0.8), axis=1))
currdf_pandas['QA_FINAL_TOTAL'], currdf_pandas['QA_FINAL_SCORE'] = zip(*currdf_pandas.apply(
    lambda x: inference_summary1(x['TEXT1_QA'], x['TEXT2_QA'], x['QA_RESULT'], 0.8), axis=1))

# Define the extract_inf function
def extract_inf(row, section, section_len, threshold=0.8):
    count_col = {}
    rel_col = {}
    score_col = {}
    total_col = {}
    for tp, score in row.items():
        if section_len != 0:
            score_binary = [1 if s > threshold else 0 for s in score]
            total_col[f'{tp}_TOTAL_{section}'] = score_binary
            count_col[f'{tp}_COUNT_{section}'] = float(sum(score_binary))
            rel_col[f'{tp}_REL_{section}'] = sum(score_binary) / section_len
            score_col[f'{tp}_SCORE_{section}'] = [round(s, 4) for s in score]
        else:
            count_col[f'{tp}_COUNT_{section}'] = None
            rel_col[f'{tp}_REL_{section}'] = None
            total_col[f'{tp}_TOTAL_{section}'] = []  # Keep as empty list for Snowflake compatibility
            score_col[f'{tp}_SCORE_{section}'] = []
    return pd.Series({**count_col, **rel_col, **score_col, **total_col})

# Apply the extract_inf function
currdf_all = pd.concat([
    currdf_pandas[['ENTITY_ID', 'CALL_ID', 'VERSION_ID', 'DATE', 'CALL_NAME', 'COMPANY_NAME', 
                  'LEN_FILT_MD', 'LEN_FILT_QA', 'FILT_MD', 'FILT_QA']],
    currdf_pandas.apply(lambda x: extract_inf(x['MD_FINAL_SCORE'], 'FILT_MD', x['LEN_FILT_MD'], 0.8), axis=1),
    currdf_pandas.apply(lambda x: extract_inf(x['QA_FINAL_SCORE'], 'FILT_QA', x['LEN_FILT_QA'], 0.8), axis=1)
], axis=1)

# Convert DATE column to datetime
currdf_all['DATE'] = pd.to_datetime(currdf_all['DATE'])

# Define output path
output_path = (
    f"{class_data_path}NLI_Demand_{dbutils.widgets.get('Start Month')}_"
    f"{dbutils.widgets.get('Start Year')[2:]}_{dbutils.widgets.get('End Month')}_"
    f"{dbutils.widgets.get('End Year')[2:]}"
)

# Save to CSV
currdf_all.to_csv(output_path, index=False)

# Rename columns for clarity
new_columns = []
for col in currdf_all.columns:
    if "This text is about reduced consumer's spending patterns." in col:
        col = col.replace("This text is about reduced consumer's spending patterns.", "CONSUMER_SPENDING_PATTERNS")
    elif "This text is about consumer weakness." in col:
        col = col.replace("This text is about consumer weakness.", "CONSUMER_WEAKNESS")
    elif "This text is about consumer strength." in col:
        col = col.replace("This text is about consumer strength.", "CONSUMER_STRENGTH")
    new_columns.append(col)
currdf_all.columns = new_columns

# Sample query for verification
tsQuery_sample = "SELECT TOP 2 * FROM EDS_PROD.QUANT.YUJING_MASS_NLI_DEMAND_DEV_3;"
resultspkdf_sample = read_from_snowflake(tsQuery_sample)
currdf_sample = resultspkdf_sample.toPandas()

# Function to define Spark schema from pandas DataFrame
def equivalent_type(string, f):
    if f == 'datetime64[ns]':
        return TimestampType()
    elif f == 'int64':
        return LongType()
    elif f == 'int32':
        return IntegerType()
    elif f == 'float64':
        return FloatType()
    elif string in ['FILT_MD', 'FILT_QA']:
        return ArrayType(StringType())
    elif '_total_' in string.lower():
        return ArrayType(IntegerType())
    elif '_score_' in string.lower():
        return ArrayType(FloatType())
    else:
        return StringType()

def define_structure(string, format_type):
    typo = equivalent_type(string, format_type)
    return StructField(string, typo)

def pandas_to_spark(pandas_df):
    columns = list(pandas_df.columns)
    types = list(pandas_df.dtypes)
    struct_list = [define_structure(col, fmt) for col, fmt in zip(columns, types)]
    p_schema = StructType(struct_list)
    return spark.createDataFrame(pandas_df, p_schema)

# Convert pandas DataFrame to Spark DataFrame
spark_parsedDF = pandas_to_spark(currdf_all[currdf_sample.columns])
spark_parsedDF = spark_parsedDF.replace(np.nan, None)
spark_parsedDF = spark_parsedDF.withColumn("DATE", F.to_timestamp(spark_parsedDF.DATE, 'yyyy-MM-dd'))

# Write to Snowflake
write_to_snowflake(spark_parsedDF, 'YUJING_MASS_NLI_DEMAND_DEV_3')

# Additional Processing for Final CSV
def sentscore(main_text, sentiment_labels, weight=False):
    # Placeholder for your sentscore function
    # Implement the actual logic as per your requirements
    return 0  # Replace with actual computation

# Apply additional sentiment scoring
def apply_sentiment_scoring(df):
    df['TP1_SENT_NLI_MD'] = df['MD_FINAL_TOTAL'].apply(
        lambda x: sentscore(
            "This text is about the company's business or market demand, not accounting metrics, will have a strong looking-forward increase or acceleration in the future.",
            "This text is about positive sentiment.", 
            weight=False
        ), 
        meta=('TP1_SENT_NLI_MD', 'float')
    )
    # Repeat for other sentiment columns as needed
    # ...
    return df

currdf_dd = currdf_dd.map_partitions(apply_sentiment_scoring)
currdf_final = currdf_dd.compute()

# Final CSV Output
final_output_path = (
    f"/dbfs/mnt/access_work/UC25/Topic Modeling/NLI Models/SF_NLI_"
    f"{dbutils.widgets.get('Start Month')}_"
    f"{dbutils.widgets.get('Start Year')}_"
    f"{dbutils.widgets.get('End Month')}_"
    f"{dbutils.widgets.get('End Year')}_v2.csv"
)
currdf_final.to_csv(final_output_path, index=False)

# Prepare DataFrame for final Snowflake write
df_all = currdf_final.copy()
df_all["TP1_EXTRACT_FILT_MD"] = df_all["CONSUMER_WEAKNESS_TOTAL_FILT_MD"]  # Adjust based on actual column names
df_all["TP1_EXTRACT_FILT_QA"] = df_all["CONSUMER_WEAKNESS_TOTAL_FILT_QA"]
df_all["TP2_EXTRACT_FILT_MD"] = df_all["CONSUMER_STRENGTH_TOTAL_FILT_MD"]
df_all["TP2_EXTRACT_FILT_QA"] = df_all["CONSUMER_STRENGTH_TOTAL_FILT_QA"]

# Drop unnecessary columns
columns_to_drop = [
    "This text is about positive sentiment._COUNT_FILT_MD", 
    "This text is about positive sentiment._REL_FILT_MD", 
    "This text is about positive sentiment._COUNT_FILT_QA", 
    "This text is about positive sentiment._REL_FILT_QA", 
    "This text is about positive sentiment._EXTRACT_FILT_MD",
    "This text is about positive sentiment._EXTRACT_FILT_QA",
    "CONSUMER_WEAKNESS_EXTRACT_FILT_MD",
    "CONSUMER_WEAKNESS_EXTRACT_FILT_QA",
    "CONSUMER_STRENGTH_EXTRACT_FILT_MD",
    "CONSUMER_STRENGTH_EXTRACT_FILT_QA"
]
df_all = df_all.drop(columns=columns_to_drop)

# Rename columns appropriately
df_all.columns = [
    'ENTITY_ID', 'DATE', 'CALL_NAME', 'COMPANY_NAME', 'LEN_FILT_MD',
    'LEN_FILT_QA', 'TP1_SENT_NLI_MD', 'TP2_SENT_NLI_MD',
    'TP1_SENT_FINBERT_MD', 'TP2_SENT_FINBERT_MD', 'TP1_SENT_NLI_QA',
    'TP2_SENT_NLI_QA', 'TP1_SENT_FINBERT_QA', 'TP2_SENT_FINBERT_QA',
    "TP1_COUNT_FILT_MD", 'TP2_COUNT_FILT_MD',
    "TP1_REL_FILT_MD", 'TP2_REL_FILT_MD',
    "TP1_COUNT_FILT_QA", 'TP2_COUNT_FILT_QA',
    "TP1_REL_FILT_QA", 'TP2_REL_FILT_QA',
    'TP1_EXTRACT_FILT_MD', 'TP1_EXTRACT_FILT_QA', 
    'TP2_EXTRACT_FILT_MD', 'TP2_EXTRACT_FILT_QA'
]

# Handle empty lists by setting them to None
for col in ['TP1_EXTRACT_FILT_MD', 'TP1_EXTRACT_FILT_QA', 
            'TP2_EXTRACT_FILT_MD', 'TP2_EXTRACT_FILT_QA']:
    df_all[col] = df_all[col].apply(lambda x: x if x else None)

# Convert to Spark DataFrame
spark_parsedDF_final = pandas_to_spark(df_all)
spark_parsedDF_final = spark_parsedDF_final.replace(np.nan, None)
spark_parsedDF_final = spark_parsedDF_final.withColumn("DATE", F.to_timestamp(spark_parsedDF_final.DATE, 'yyyy-MM-dd'))

# Write final DataFrame to Snowflake
write_to_snowflake(spark_parsedDF_final, 'YUJING_SF_SUPER_STOCK_DEV1')

print("Data processing and inference pipeline completed successfully.")
