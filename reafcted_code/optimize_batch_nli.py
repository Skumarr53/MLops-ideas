from pyspark.sql.functions import col
from pyspark.sql.types import ArrayType, StructType, StructField, StringType, FloatType
from pyspark.sql import SparkSession
from pyspark.sql.functions import pandas_udf, PandasUDFType
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
import pandas as pd
import logging
import sys

# Constants (Ensure these are correctly defined)
MODEL_FOLDER_PATH = "~/.cache/huggingface/hub/"  # Update this path
MODEL_NAME = "models--MoritzLaurer--deberta-v3-large-zeroshot-v2.0/"
ENABLE_QUANTIZATION = False
BATCH_SIZE = 1  # Adjust based on your GPU capacity

# Step 2: Initialize Logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger("NLI_Inference")

# Initialize a global variable for the pipeline
nli_pipeline = None

def initialize_nli_pipeline(enable_quantization=False):
    global nli_pipeline
    try:
        # Load the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Load the model
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        
        if enable_quantization:
            model = torch.quantization.quantize_dynamic(
                model, {torch.nn.Linear}, dtype=torch.qint8
            )
            logger.info("Model quantization enabled.")
        else:
            logger.info("Model quantization disabled.")
        
        device = -1 if torch.cuda.is_available() else -1
        nli_pipeline = pipeline(
            task="zero-shot-classification",  # Changed task
            model=model,
            tokenizer=tokenizer,
            device=device
        )
        logger.info(f"NLI pipeline initialized on device: {'GPU' if device == 0 else 'CPU'}")
    except Exception as e:
        logger.error(f"Failed to initialize NLI pipeline: {e}")
        raise e

# Define the schema for the inference results
inference_schema = ArrayType(StructType([
    StructField("label", StringType(), False),
    StructField("score", FloatType(), False)
]))

@pandas_udf(inference_schema, PandasUDFType.SCALAR_ITER)
def md_inference_udf(iterator):
    global nli_pipeline
    if nli_pipeline is None:
        initialize_nli_pipeline(enable_quantization=ENABLE_QUANTIZATION)
    
    for batch_num, batch in enumerate(iterator, start=1):
        logger.info(f"Processing MD inference batch {batch_num} with {len(batch)} rows.")
        try:
            # Each 'batch' is a Pandas Series containing lists of text pairs
            # Flatten the list of text pairs in the batch
            flat_text_pairs = [pair for sublist in batch for pair in sublist]
            logger.debug(f"Batch {batch_num}: Total text pairs to infer: {len(flat_text_pairs)}")
            
            if len(flat_text_pairs):
                # Perform inference in batch using zero-shot-classification pipeline
                results = nli_pipeline(
                    sequences=flat_text_pairs,
                    candidate_labels=[
                        "This text is about consumer strength",
                        "This text is about consumer weakness",
                        "This text is about reduced consumer's spending patterns"
                    ],
                    multi_class=True,
                    batch_size=BATCH_SIZE
                )
                logger.debug(f"Batch {batch_num}: Inference completed with {len(results)} results.")
            else:
                results = []
            
            # Split results back to original rows
            split_results = []
            idx = 0
            for pairs in batch:
                if len(pairs):
                    # Extract the relevant slice of results
                    row_results = results[idx:idx+len(pairs)]
                    # Transform each result into a list of dicts with 'label' and 'score'
                    formatted_results = [
                        {"label": res['labels'][i], "score": res['scores'][i]}
                        for res in row_results
                        for i in range(len(res['labels']))
                    ]
                    split_results.append(formatted_results)
                    idx += len(pairs)
                else:
                    split_results.append([])
            
            yield pd.Series(split_results)
        except Exception as e:
            logger.error(f"Error in MD inference batch {batch_num}: {e}")
            # Yield empty results for this batch to continue processing
            yield pd.Series([[] for _ in batch])