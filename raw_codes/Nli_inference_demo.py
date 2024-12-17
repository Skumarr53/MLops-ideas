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

from centralized_nlp_package.data_processing import initialize_spark_session
from topic_modelling_package.nli_process import create_text_pairs

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
create_text_pairs_udf = udf(create_text_pairs_old, ArrayType(StringType()))


#### Refactored

create_text_pairs()

### Define Helper Functions
