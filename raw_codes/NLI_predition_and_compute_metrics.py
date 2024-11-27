# Databricks notebook source
# MAGIC %md
# MAGIC ## Install and import required libraries

# COMMAND ----------

!pip install accelerate
!pip install datasets
!pip install sentencepiece
!pip install scipy
!pip install scikit-learn
!pip install protobuf
!pip install torch==2.0.0
!pip install evaluate
!pip install git+https://github.com/huggingface/transformers@v4.46.1
!pip install git+https://github.com/huggingface/accelerate
!pip install accuracy



# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel
from transformers import TextClassificationPipeline
from transformers import ZeroShotClassificationPipeline
import torch
import evaluate

import pandas as pd
from sklearn.model_selection import train_test_split
import os


# COMMAND ----------

import os
os.chdir("/Workspace/Quant/Call_Transcript_Topic_Modeling/Development/NLI_Topic_Modeling/")

# COMMAND ----------

METRIC = evaluate.load("accuracy")
DEVICE = 0 if torch.cuda.is_available() else -1

MODEL_FOLDER = "deberta-v3-large-zeroshot-v2"

MODEL_DIR = "/dbfs/mnt/access_work/UC25/Libraries/HuggingFace/"
MAIN_DATASET_PATH = "/Workspace/Users/santhosh.kumar3@voya.com/ConsumerNLI_Topic/data/main.csv"
TRAIN_DATASET_PATH = "/Workspace/Users/santhosh.kumar3@voya.com/ConsumerNLI_Topic/data/train.csv"
TEST_DATASET_PATH = "/Workspace/Users/santhosh.kumar3@voya.com/ConsumerNLI_Topic/data/test.csv"

OUTPUT_PATH = "/dbfs/mnt/access_work/UC25/Topic Modeling/NLI Models/Fine-tune NLI models/trained_RD_deberta-v3-base-zeroshot-v2_Santhosh_test_{version}"

NLI_ENTAILMENT_SCORE_THRESOLDS = [0.70, 0.80, 0.90, 1.0]


HYPOTHESIS_TEMPLATE = "{}"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Function Definations

# COMMAND ----------



def preprocess_and_split_data(csv_path):
    '''Preprocess the data into train and test sets'''
    # Step 0: Read CSV from the provided path
    df = pd.read_csv(csv_path)
    
    # Step 1: Rename columns
    df.rename(columns={'Matched Sentence': 'sentence1', 'NLI Topic': 'sentence2'}, inplace=True)
    
    # Step 2: Create a new column named 'label'
    df['label'] = None
    df.loc[df['True Positive'] == 1, 'label'] = 'entailment'
    df.loc[df['False Positive'] == 1, 'label'] = 'not_entailment'
    
    # Step 3: Filter data to contain non-empty 'label' and keep only specified columns
    df = df[df['label'].notna()][['sentence1', 'sentence2', 'label']]
    # df['label_GT'] = df['label'].apply(lambda x: 1 if x == 'entailment' else 0)
    
    # Step 4: Shuffle data
    df = df.sample(frac=1).reset_index(drop=True)
    
    # Step 5: Create 3 sets and write them to the same directory as the input
    # Get the directory of the input file
    input_dir = os.path.dirname(csv_path)
    
    # Save the original data
    main_csv_path = os.path.join(input_dir, 'main.csv')
    df.to_csv(main_csv_path, index=False)
    
    # Create train and test split (80:20)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    
    # Save train and test data
    train_csv_path = os.path.join(input_dir, 'train.csv')
    test_csv_path = os.path.join(input_dir, 'test.csv')
    
    train_df.to_csv(train_csv_path, index=False)
    test_df.to_csv(test_csv_path, index=False)


def get_accuracy_at_threshold(df, thresholds):
  accuracy_dict = {}

  GT_labels = df['label'].apply(lambda x: 1 if x == 'entailment' else 0)

  for thres in thresholds:
    print(thres)
    thres_int = int(thres * 100)
    nli_labels = df['NLI_score'].apply(lambda x: 1 if x > thres else 0)
    accuracy_dict[f"accuracy_at_thres_{thres_int}"] = (nli_labels == GT_labels).mean()

  return accuracy_dict


def compute_nli_score(classifier,hypothesis_template, val_df):
    val_df['NLI_score'] = val_df.apply(lambda x: classifier(x['sentence1'] , x['sentence2'], hypothesis_template=hypothesis_template, multi_label=True)['scores'][0] , axis = 1)
    return val_df

# COMMAND ----------

# preprocess_and_split_data("/Workspace/Users/santhosh.kumar3@voya.com/ConsumerNLI_Topic/data/ConsumerNL_Fine-tuning_combined.csv")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Scenarios

# COMMAND ----------

# MAGIC %md
# MAGIC ### Scenario 2: Fine-Tuning the base model with the NLI dataset

# COMMAND ----------

# MAGIC %md
# MAGIC ### V1

# COMMAND ----------

import pandas as pd
dat = pd.read_csv("/dbfs/mnt/access_work/UC25/Topic Modeling/NLI Models/NLI_role_training_data/Role_FT_test_v5_11012024.csv")

# COMMAND ----------

dat.head(5)

# COMMAND ----------

pd.read_csv(TEST_DATASET_PATH)

# COMMAND ----------



# COMMAND ----------

# MAGIC %fs ls OUTPUT_PATH

# COMMAND ----------

  # --train_file "/Workspace/Users/santhosh.kumar3@voya.com/ConsumerNLI_Topic/data/train.csv" \
  # --validation_file "/Workspace/Users/santhosh.kumar3@voya.com/ConsumerNLI_Topic/data/test.csv" \

# COMMAND ----------

# %sh

# export TASK_NAME=mnli
# python run_glue.py \
#   --model_name_or_path "/dbfs/mnt/access_work/UC25/Libraries/HuggingFace/deberta-v3-large-zeroshot-v2" \
#   --output_dir "/dbfs/mnt/access_work/UC25/Topic Modeling/NLI Models/Fine-tune NLI models/trained_RD_deberta-v3-base-zeroshot-v2_Santhosh_test_v1" \
#   --train_file "/Workspace/Users/santhosh.kumar3@voya.com/ConsumerNLI_Topic/data/train.csv" \
#   --validation_file "/Workspace/Users/santhosh.kumar3@voya.com/ConsumerNLI_Topic/data/test.csv" \
#   --do_train \
#   --do_eval \
#   --num_train_epochs 3 \
#   --fp16 \
#   --report_to "none" \
#   --learning_rate 2e-5 \
#   --weight_decay 0.01 \
#   --per_device_train_batch_size 16 \
#   --per_device_eval_batch_size 16

# COMMAND ----------

# db_0_classifier_v1 = pipeline("zero-shot-classification", model = OUTPUT_PATH)

# COMMAND ----------

# val_df = pd.read_csv(TEST_DATASET_PATH)

# COMMAND ----------

# val_df = compute_nli_score(db_0_classifier_v1, HYPOTHESIS_TEMPLATE, val_df)
# # get_accuracy_at_threshold(val_df, NLI_ENTAILMENT_SCORE_THRESOLDS)

# COMMAND ----------

import subprocess


hyperparmaetrers = [{"n_epochs": 5}, {"n_epochs": 8}]
results = {}

val_df = pd.read_csv(TEST_DATASET_PATH)

for i,param_dict in enumerate(hyperparmaetrers):
  version = f"v{i+2}"
  print(f"\n****** Training model {version} with params:\n {param_dict}\n")

  out_path = OUTPUT_PATH.format(version=version)
  dbutils.fs.mkdirs(out_path)
  print(f"sucessfully created {out_path}")


  subprocess.run([
            "python", "run_glue.py",
              "--model_name_or_path", MODEL_DIR+MODEL_FOLDER,
              "--output_dir", out_path,
              "--train_file", TRAIN_DATASET_PATH,
              "--validation_file", TEST_DATASET_PATH,
            "--do_train",
            "--do_eval",
            "--num_train_epochs", str(param_dict.get("n_epochs", 3)),
            "--fp16",
            "--report_to", "none",
            "--learning_rate", str(param_dict.get("learning_rate", 2e-5)),
            "--weight_decay", str(param_dict.get("weight_decay", 0.01)),
            "--per_device_train_batch_size", str(param_dict.get("train_batch_size", 16)),
            "--per_device_eval_batch_size", str(param_dict.get("eval_batch_size", 16))
        ], check=True, capture_output=True, text=True)
  db_0_classifier = pipeline("zero-shot-classification", model = out_path)
  val_score_df = compute_nli_score(db_0_classifier, HYPOTHESIS_TEMPLATE, val_df)
  accuracy_dict = get_accuracy_at_threshold(val_score_df, NLI_ENTAILMENT_SCORE_THRESOLDS)
  print(f"v{version} accuracy:\n {accuracy_dict}")
  
  results[version] = accuracy_dict


# COMMAND ----------

# MAGIC %md
# MAGIC ### Scenario 1: Pretrained model checking its performance

# COMMAND ----------

val_df = pd.read_csv(MAIN_DATASET_PATH)
val_df.head(3)

# COMMAND ----------

val_df.head(10)

# COMMAND ----------

# MAGIC %md
# MAGIC #### load model

# COMMAND ----------

db_0_classifier3 = pipeline("zero-shot-classification", model = "/dbfs/mnt/access_work/UC25/Libraries/HuggingFace/deberta-v3-large-zeroshot-v2")

# COMMAND ----------

val_df = compute_nli_score(db_0_classifier3, HYPOTHESIS_TEMPLATE, val_df)

# COMMAND ----------

val_df

# COMMAND ----------

nli_entailment_score_thresolds = [0.7, 0.8, 0.9, 1.0]



# COMMAND ----------


