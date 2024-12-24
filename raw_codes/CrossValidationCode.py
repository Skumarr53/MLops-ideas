# Databricks notebook source
# MAGIC %md
# MAGIC ### Remember to use 12.2 GPU RUNTIME to fine-tune NLI model
# MAGIC Once you have the fine-tuned model, you can use 13.3 runtime to load the model. It's much faster than 12.2 runtime. 

# COMMAND ----------

!pip install accelerate
!pip install datasets
!pip install sentencepiece
!pip install scipy
!pip install scikit-learn
!pip install protobuf


# COMMAND ----------

!pip install torch==2.0.0
!pip install evaluate

# COMMAND ----------

!pip install transformers==4.28.0

# COMMAND ----------

pip install git+https://github.com/huggingface/transformers

# COMMAND ----------

pip install accuracy

# COMMAND ----------

pip install evaluate

# COMMAND ----------

pip install accelerate==0.27.2

# COMMAND ----------

pip install git+https://github.com/huggingface/accelerate

# COMMAND ----------

pip install thefuzz

# COMMAND ----------

# pip install huggingface_hub==0.23.2

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import os

os.chdir("/Workspace/Quant/Call_Transcript_Topic_Modeling/Development/NLI_Topic_Modeling/")

# COMMAND ----------

import evaluate
metric = evaluate.load("accuracy")

# COMMAND ----------

import torch
torch.cuda.empty_cache()

# COMMAND ----------

import pandas as pd
from collections import Counter

# COMMAND ----------

from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel
from transformers import TextClassificationPipeline
from transformers import ZeroShotClassificationPipeline
import torch

device = 0 if torch.cuda.is_available() else -1

model_1_folder_name = "deberta-v3-large-zeroshot-v2"

model_folder_path = "/dbfs/mnt/access_work/UC25/Libraries/HuggingFace/"

# tokenizer_1 = AutoTokenizer.from_pretrained(model_folder_path + model_1_folder_name)
# model_1 = AutoModelForSequenceClassification.from_pretrained(model_folder_path + model_1_folder_name)

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ### Fine-tune NLI Model

# COMMAND ----------

# MAGIC %sh
# MAGIC
# MAGIC export TASK_NAME=mnli
# MAGIC python run_glue.py \
# MAGIC   --model_name_or_path "/dbfs/mnt/access_work/UC25/Libraries/HuggingFace/deberta-v3-large-zeroshot-v2" \
# MAGIC   --output_dir "/dbfs/mnt/access_work/UC25/Topic Modeling/NLI Models/Fine-tune NLI models/trained_sentiment_deberta-v3-large-zeroshot-v2_C2_lc" \
# MAGIC   --train_file "/dbfs/mnt/access_work/UC25/Topic Modeling/NLI Models/MLFlow_FT_Sentiment/data/train_partition2_with2NE_v2_lc.csv" \
# MAGIC   --validation_file "/dbfs/mnt/access_work/UC25/Topic Modeling/NLI Models/MLFlow_FT_Sentiment/data/test_partition2_with2NE.csv" \
# MAGIC   --do_train \
# MAGIC   --do_eval \
# MAGIC   --num_train_epochs 5 \
# MAGIC   --fp16 \
# MAGIC   --report_to "none" \
# MAGIC   --learning_rate 2e-5 \
# MAGIC   --weight_decay 0.01 \
# MAGIC   --per_device_train_batch_size 16 \
# MAGIC   --per_device_eval_batch_size 16 

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ## Load the model and check the predicted label

# COMMAND ----------

FT_classifier_A1 = pipeline("zero-shot-classification", model = "/dbfs/mnt/access_work/UC25/Topic Modeling/NLI Models/Fine-tune NLI models/trained_sentiment_deberta-v3-large-zeroshot-v2_A1")
FT_classifier_B1 = pipeline("zero-shot-classification", model = "/dbfs/mnt/access_work/UC25/Topic Modeling/NLI Models/Fine-tune NLI models/trained_sentiment_deberta-v3-large-zeroshot-v2_B1")
FT_classifier_C1 = pipeline("zero-shot-classification", model = "/dbfs/mnt/access_work/UC25/Topic Modeling/NLI Models/Fine-tune NLI models/trained_sentiment_deberta-v3-large-zeroshot-v2_C1")



# COMMAND ----------

FT_classifier_A2 = pipeline("zero-shot-classification", model = "/dbfs/mnt/access_work/UC25/Topic Modeling/NLI Models/Fine-tune NLI models/trained_sentiment_deberta-v3-large-zeroshot-v2_A2")
FT_classifier_B2 = pipeline("zero-shot-classification", model = "/dbfs/mnt/access_work/UC25/Topic Modeling/NLI Models/Fine-tune NLI models/trained_sentiment_deberta-v3-large-zeroshot-v2_B2")
FT_classifier_C2 = pipeline("zero-shot-classification", model = "/dbfs/mnt/access_work/UC25/Topic Modeling/NLI Models/Fine-tune NLI models/trained_sentiment_deberta-v3-large-zeroshot-v2_C2")


# COMMAND ----------

FT_classifier_A3 = pipeline("zero-shot-classification", model = "/dbfs/mnt/access_work/UC25/Topic Modeling/NLI Models/Fine-tune NLI models/trained_sentiment_deberta-v3-large-zeroshot-v2_A3")
FT_classifier_B3 = pipeline("zero-shot-classification", model = "/dbfs/mnt/access_work/UC25/Topic Modeling/NLI Models/Fine-tune NLI models/trained_sentiment_deberta-v3-large-zeroshot-v2_B3")
FT_classifier_C3 = pipeline("zero-shot-classification", model = "/dbfs/mnt/access_work/UC25/Topic Modeling/NLI Models/Fine-tune NLI models/trained_sentiment_deberta-v3-large-zeroshot-v2_C3")


# COMMAND ----------

# MAGIC %md
# MAGIC ##### Note: Loading more than 9 NLI models may reach CUDA memory limit

# COMMAND ----------

FT_classifier_C1 = pipeline("zero-shot-classification", model = "/dbfs/mnt/access_work/UC25/Topic Modeling/NLI Models/Fine-tune NLI models/trained_sentiment_deberta-v3-large-zeroshot-v2_C1_st")
FT_classifier_C2 = pipeline("zero-shot-classification", model = "/dbfs/mnt/access_work/UC25/Topic Modeling/NLI Models/Fine-tune NLI models/trained_sentiment_deberta-v3-large-zeroshot-v2_C2_st")
FT_classifier_C3 = pipeline("zero-shot-classification", model = "/dbfs/mnt/access_work/UC25/Topic Modeling/NLI Models/Fine-tune NLI models/trained_sentiment_deberta-v3-large-zeroshot-v2_C3_st")

# COMMAND ----------

OOB_classifier = pipeline("zero-shot-classification", model = "/dbfs/mnt/access_work/UC25/Libraries/HuggingFace/deberta-v3-large-zeroshot-v2")

# COMMAND ----------

FT_classifier_A4 = pipeline("zero-shot-classification", model = "/dbfs/mnt/access_work/UC25/Topic Modeling/NLI Models/Fine-tune NLI models/trained_sentiment_deberta-v3-large-zeroshot-v2_A4")
FT_classifier_B4 = pipeline("zero-shot-classification", model = "/dbfs/mnt/access_work/UC25/Topic Modeling/NLI Models/Fine-tune NLI models/trained_sentiment_deberta-v3-large-zeroshot-v2_B4")
FT_classifier_C4 = pipeline("zero-shot-classification", model = "/dbfs/mnt/access_work/UC25/Topic Modeling/NLI Models/Fine-tune NLI models/trained_sentiment_deberta-v3-large-zeroshot-v2_C4")

# COMMAND ----------

FT_classifier_A5 = pipeline("zero-shot-classification", model = "/dbfs/mnt/access_work/UC25/Topic Modeling/NLI Models/Fine-tune NLI models/trained_sentiment_deberta-v3-large-zeroshot-v2_A5")
FT_classifier_B5 = pipeline("zero-shot-classification", model = "/dbfs/mnt/access_work/UC25/Topic Modeling/NLI Models/Fine-tune NLI models/trained_sentiment_deberta-v3-large-zeroshot-v2_B5")
FT_classifier_C5 = pipeline("zero-shot-classification", model = "/dbfs/mnt/access_work/UC25/Topic Modeling/NLI Models/Fine-tune NLI models/trained_sentiment_deberta-v3-large-zeroshot-v2_C5")

# COMMAND ----------



# COMMAND ----------

FT_classifier_A1("Overall, this leads us with good liquidity and significant headroom on our facilities, thereby providing security, flexibility and options for the future." , 'This text has the positive sentiment.', hypothesis_template="{}", multi_label=True)['scores'][0]

# COMMAND ----------



# COMMAND ----------

from sklearn.metrics import classification_report

# COMMAND ----------

def get_sent_score(sentence):
  if 'neutral' in sentence:
    return 0
  elif 'positive' in sentence:
    return 1
  else:
    return -1
def get_max_label(row, neutral_col, positive_col, negative_col):
    if row[neutral_col] >= row[positive_col] and row[neutral_col] >= row[negative_col]:
        return 0
    elif row[positive_col] >= row[neutral_col] and row[positive_col] >= row[negative_col]:
        return 1
    else:
        return -1


# COMMAND ----------

# You can change the file name and get the prediction for different folds
test1 = pd.read_csv("/dbfs/mnt/access_work/UC25/Topic Modeling/NLI Models/MLFlow_FT_Sentiment/data/test_partition1.csv")

# COMMAND ----------

test1.tail(2)

# COMMAND ----------

# Steve's 35 samples from last iteration
test1 = pd.read_csv("/Workspace/Users/yujing.sun@voya.com/(Clone) MLFlow_and_NLI_finetune/data/finetuning_sample_nli_moderate_aug_w_tpf.csv")

# COMMAND ----------

# Josh's 86 low confidence samples
test1 = pd.read_csv("/Workspace/Users/yujing.sun@voya.com/(Clone) MLFlow_and_NLI_finetune/data/labeled_sentiment_low_confidence_samples.csv")

# COMMAND ----------

test1['voya label'] = test1['FineTune Label']

# COMMAND ----------

test1 = test1[test1['label'] == 'entailment']

# COMMAND ----------

test1['voya label'] = test1['sentence2'].apply(lambda x: get_sent_score(x))

# COMMAND ----------

test1['OOB_positive_score'] = test1['sentence1'].apply(lambda x: OOB_classifier(x, 'This text has the positive sentiment.', hypothesis_template="{}", multi_label=True)['scores'][0])
test1['OOB_neutral_score'] = test1['sentence1'].apply(lambda x: OOB_classifier(x, 'This text has the neutral sentiment.', hypothesis_template="{}", multi_label=True)['scores'][0])
test1['OOB_negative_score'] = test1['sentence1'].apply(lambda x: OOB_classifier(x, 'This text has the negative sentiment.', hypothesis_template="{}", multi_label=True)['scores'][0])

# COMMAND ----------

test1['FT_A_positive_score'] = test1['sentence1'].apply(lambda x: FT_classifier_A1(x, 'This text has the positive sentiment.', hypothesis_template="{}", multi_label=True)['scores'][0])
test1['FT_A_neutral_score'] = test1['sentence1'].apply(lambda x: FT_classifier_A1(x, 'This text has the neutral sentiment.', hypothesis_template="{}", multi_label=True)['scores'][0])
test1['FT_A_negative_score'] = test1['sentence1'].apply(lambda x: FT_classifier_A1(x, 'This text has the negative sentiment.', hypothesis_template="{}", multi_label=True)['scores'][0])

# COMMAND ----------

test1['FT_B_positive_score'] = test1['sentence1'].apply(lambda x: FT_classifier_B1(x, 'This text has the positive sentiment.', hypothesis_template="{}", multi_label=True)['scores'][0])
test1['FT_B_neutral_score'] = test1['sentence1'].apply(lambda x: FT_classifier_B1(x, 'This text has the neutral sentiment.', hypothesis_template="{}", multi_label=True)['scores'][0])
test1['FT_B_negative_score'] = test1['sentence1'].apply(lambda x: FT_classifier_B1(x, 'This text has the negative sentiment.', hypothesis_template="{}", multi_label=True)['scores'][0])

# COMMAND ----------

test1['FT_C_positive_score'] = test1['sentence1'].apply(lambda x: FT_classifier_C1(x, 'This text has the positive sentiment.', hypothesis_template="{}", multi_label=True)['scores'][0])
test1['FT_C_neutral_score'] = test1['sentence1'].apply(lambda x: FT_classifier_C1(x, 'This text has the neutral sentiment.', hypothesis_template="{}", multi_label=True)['scores'][0])
test1['FT_C_negative_score'] = test1['sentence1'].apply(lambda x: FT_classifier_C1(x, 'This text has the negative sentiment.', hypothesis_template="{}", multi_label=True)['scores'][0])

# COMMAND ----------

test1

# COMMAND ----------

test1['OOB_label'] = test1.apply(lambda x: get_max_label(x, 'OOB_neutral_score', 'OOB_positive_score', 'OOB_negative_score'), axis=1)
test1['FT_A_label'] = test1.apply(lambda x: get_max_label(x, 'FT_A_neutral_score', 'FT_A_positive_score', 'FT_B_negative_score'), axis=1)
test1['FT_B_label'] = test1.apply(lambda x: get_max_label(x, 'FT_B_neutral_score', 'FT_B_positive_score', 'FT_B_negative_score'), axis=1)
test1['FT_C_label'] = test1.apply(lambda x: get_max_label(x, 'FT_C_neutral_score', 'FT_C_positive_score', 'FT_C_negative_score'), axis=1)

# COMMAND ----------

test1

# COMMAND ----------



# COMMAND ----------

accuracy_A1 = (test1['voya label'] == test1['FT_A_label']).mean()
accuracy_B1 = (test1['voya label'] == test1['FT_B_label']).mean()
accuracy_C1 = (test1['voya label'] == test1['FT_C_label']).mean()

# COMMAND ----------

accuracy_OOB = (test1['voya label'] == test1['OOB_label']).mean()

# COMMAND ----------

# test2_result
print(accuracy_A1, accuracy_B1, accuracy_C1)

# COMMAND ----------

report_A1 = classification_report(test1['voya label'], test1['FT_A_label'], target_names=['negative', 'neutral', 'positive'])
report_B1 = classification_report(test1['voya label'], test1['FT_B_label'], target_names=['negative', 'neutral', 'positive'])
report_C3_st = classification_report(test1['voya label'], test1['FT_C_label'], target_names=['negative', 'neutral', 'positive'])
report_OOB = classification_report(test1['voya label'], test1['OOB_label'], target_names=['negative', 'neutral', 'positive'])

# COMMAND ----------

print(report_OOB)

# COMMAND ----------

print(report_C3_st)

# COMMAND ----------

report_A1 = classification_report(test_1_5_result['voya label'], test_1_5_result['FT_A_label'], target_names=['negative', 'neutral', 'positive'])
report_B1 = classification_report(test_1_5_result['voya label'], test_1_5_result['FT_B_label'], target_names=['negative', 'neutral', 'positive'])
report_C1 = classification_report(test_1_5_result['voya label'], test_1_5_result['FT_C_label'], target_names=['negative', 'neutral', 'positive'])
report_OOB = classification_report(test_1_5_result['voya label'], test_1_5_result['OOB_label'], target_names=['negative', 'neutral', 'positive'])

# COMMAND ----------

# Steve's samples
print("Classification Report for model A")
print(report_A1)
print("Classification Report for model B")
print(report_B1)
print("Classification Report for model C")
print(report_C1)
print("Classification Report for OOB model")
print(report_OOB)

# COMMAND ----------

# all labeled data
print("Classification Report for model A")
print(report_A1)
print("Classification Report for model B")
print(report_B1)
print("Classification Report for model C")
print(report_C1)
print("Classification Report for OOB model")
print(report_OOB)

# COMMAND ----------

# Josh's low confidence samples
print("Classification Report for model A")
print(report_A1)
print("Classification Report for model B")
print(report_B1)
print("Classification Report for model C")
print(report_C1)
print("Classification Report for OOB model")
print(report_OOB)

# COMMAND ----------



# COMMAND ----------

test1_result = test1.copy()

# COMMAND ----------

test2_result = test1.copy()

# COMMAND ----------

test3_result = test1.copy()

# COMMAND ----------

test4_result = test1.copy()

# COMMAND ----------

test5_result = test1.copy()

# COMMAND ----------

test_steve_result = test1.copy()

# COMMAND ----------

test_lc_result = test1.copy()

# COMMAND ----------

test1_result['fold'] = 'test1'
test2_result['fold'] = 'test2'
test3_result['fold'] = 'test3'

# COMMAND ----------

test4_result['fold'] = 'test4'

# COMMAND ----------

test5_result['fold'] = 'test5'

# COMMAND ----------

test_steve_result['fold'] = '35 samples from Steve'

# COMMAND ----------

test_lc_result['fold'] = '86 low confidence samples from Josh'

# COMMAND ----------

test_1_3_result = pd.concat([test1_result, test2_result, test3_result])

# COMMAND ----------

test_1_5_result = pd.concat([test5_result, test_1_3_result, test4_result]).drop(columns = ['sentence2', 'label'])

# COMMAND ----------

test_1_5_result.to_csv('/dbfs/mnt/access_work/UC25/Topic Modeling/NLI Models/MLFlow_FT_Sentiment/data/FT_label_test_1_5.csv', index=False)

# COMMAND ----------

test_steve_result.to_csv('/dbfs/mnt/access_work/UC25/Topic Modeling/NLI Models/MLFlow_FT_Sentiment/data/FT_label_test_steve.csv', index=False)

# COMMAND ----------

test_lc_result.to_csv('/dbfs/mnt/access_work/UC25/Topic Modeling/NLI Models/MLFlow_FT_Sentiment/data/FT_label_test_lc_josh.csv', index=False)

# COMMAND ----------

test_1_3_result = test_1_3_result.drop(columns =['sentence2', 'label', 'voay label'])

# COMMAND ----------

test_1_3_result.to_csv('/dbfs/mnt/access_work/UC25/Topic Modeling/NLI Models/MLFlow_FT_Sentiment/data/FT_label_test_1_3.csv', index=False)

# COMMAND ----------

test_1_5_result.to_csv('/dbfs/mnt/access_work/UC25/Topic Modeling/NLI Models/MLFlow_FT_Sentiment/data/FT_label_test_1_5.csv', index=False)

# COMMAND ----------

test_1_3_result = test1.copy()

# COMMAND ----------

test_1_5_result = pd.read_csv('/dbfs/mnt/access_work/UC25/Topic Modeling/NLI Models/MLFlow_FT_Sentiment/data/FT_label_test_1_5.csv')

# COMMAND ----------

raw_df = pd.read_csv('/dbfs/mnt/access_work/UC25/Topic Modeling/NLI Models/MLFlow_FT_Sentiment/data/NLI_FineTuning_samples_for_Yujing_20241210.csv')

# COMMAND ----------

test_1_5_result = test_1_5_result.sort_values(by = 'sentence1')

# COMMAND ----------

test_1_5_result['Topic'] = raw_df.sort_values(by = 'Matched Sentences')['Topic Name'][:-1]

# COMMAND ----------

test_1_5_result

# COMMAND ----------

grouped = test_1_5_result.groupby('Topic')

for topic, group in grouped:
    print(f"Classification report for topic {topic}:")
    report = classification_report(group['voya label'], group['OOB_label'],target_names=['negative', 'neutral', 'positive'])
    print(report)
    # print("\n")

# COMMAND ----------

grouped = test_1_5_result.groupby('Topic')

for topic, group in grouped:
    print(f"Classification report for topic {topic}:")
    report = classification_report(group['voya label'], group['FT_C_label'],target_names=['negative', 'neutral', 'positive'])
    print(report)
    # print("\n")

# COMMAND ----------



# COMMAND ----------

## Refactored

import os
import pandas as pd
import torch
from transformers import pipeline
from sklearn.metrics import classification_report
from pathlib import Path
import yaml
import logging

# Setup Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Load Configurations
def load_config(config_path="config.yaml"):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

CONFIG = load_config()

# Dependency Management

def install_dependencies():
    dependencies = [
        "accelerate==0.27.2",
        "datasets",
        "sentencepiece",
        "scipy",
        "scikit-learn",
        "protobuf",
        "transformers==4.28.0",
        "evaluate",
        "thefuzz"
    ]
    for package in dependencies:
        os.system(f"pip install {package}")

# install_dependencies()

# Utility Functions

def init_pipeline(model_path, device):
    return pipeline("zero-shot-classification", model=model_path, device=device)

def load_data(file_path):
    try:
        return pd.read_csv(file_path)
    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")
        return None

def save_data(df, output_path):
    df.to_csv(output_path, index=False)
    logging.info(f"Saved results to {output_path}")

# Scoring and Labeling

def get_sent_score(sentence):
    if 'neutral' in sentence:
        return 0
    elif 'positive' in sentence:
        return 1
    else:
        return -1

def get_max_label(row, neutral_col, positive_col, negative_col):
    if row[neutral_col] >= row[positive_col] and row[neutral_col] >= row[negative_col]:
        return 0
    elif row[positive_col] >= row[neutral_col] and row[positive_col] >= row[negative_col]:
        return 1
    else:
        return -1

# Evaluation

def evaluate_model(pipeline, test_data, sentiment_template):
    for label in ["positive", "neutral", "negative"]:
        test_data[f"{label}_score"] = test_data["sentence1"].apply(
            lambda x: pipeline(x, f"This text has the {label} sentiment.", hypothesis_template=sentiment_template, multi_label=True)['scores'][0]
        )
    
    test_data['predicted_label'] = test_data.apply(
        lambda row: get_max_label(row, "neutral_score", "positive_score", "negative_score"), axis=1
    )
    return test_data

# Main Processing

def main():
    # Load test data
    test_data_path = CONFIG['data']['test_file']
    test_data = load_data(test_data_path)
    if test_data is None:
        return

    # Load model and pipeline
    model_path = CONFIG['model']['path']
    device = 0 if torch.cuda.is_available() else -1
    sentiment_pipeline = init_pipeline(model_path, device)

    # Evaluate model
    test_data = evaluate_model(sentiment_pipeline, test_data, CONFIG['templates']['sentiment'])

    # Save results
    output_path = CONFIG['data']['output_file']
    save_data(test_data, output_path)

    # Generate classification report
    target_names = ['negative', 'neutral', 'positive']
    report = classification_report(test_data['voya_label'], test_data['predicted_label'], target_names=target_names)
    logging.info("Classification Report:\n" + report)

if __name__ == "__main__":
    main()
