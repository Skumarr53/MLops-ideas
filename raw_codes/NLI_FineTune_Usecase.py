# Databricks notebook source
# MAGIC %md
# MAGIC ### Remember to use 12.2 GPU RUNTIME

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

db0_classifier = pipeline("zero-shot-classification", model= model_folder_path + model_1_folder_name)

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ### Load fine-tuning dataset

# COMMAND ----------

## You can preprocessing the data before passing it to the model and save it as training and validation set.I dropped the preprocessing code since it's based on another dataset.

# COMMAND ----------

new_test_df

# COMMAND ----------

Counter(new_test_df['sentence2'])

# COMMAND ----------

new_train_df.to_csv("/dbfs/mnt/access_work/UC25/Topic Modeling/NLI Models/NLI_role_training_data/Role_FT_train_v3.csv", index=False)
new_test_df.to_csv("/dbfs/mnt/access_work/UC25/Topic Modeling/NLI Models/NLI_role_training_data/Role_FT_test_v3.csv", index=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Fine-tune NLI Model

# COMMAND ----------

# MAGIC %sh
# MAGIC
# MAGIC export TASK_NAME=mnli
# MAGIC python run_glue.py \
# MAGIC   --model_name_or_path "/dbfs/mnt/access_work/UC25/Libraries/HuggingFace/deberta-v3-large-zeroshot-v2" \
# MAGIC   --output_dir "/dbfs/mnt/access_work/UC25/Topic Modeling/NLI Models/Fine-tune NLI models/trained_RD_deberta-v3-large-zeroshot-v2_v3" \
# MAGIC   --train_file "/dbfs/mnt/access_work/UC25/Topic Modeling/NLI Models/NLI_role_training_data/Role_FT_train_v3.csv" \
# MAGIC   --validation_file "/dbfs/mnt/access_work/UC25/Topic Modeling/NLI Models/NLI_role_training_data/Role_FT_test_v3.csv" \
# MAGIC   --do_train \
# MAGIC   --do_eval \
# MAGIC   --num_train_epochs 3 \
# MAGIC   --fp16 \
# MAGIC   --report_to "none" \
# MAGIC   --learning_rate 2e-5 \
# MAGIC   --weight_decay 0.01 \
# MAGIC   --per_device_train_batch_size 16 \
# MAGIC   --per_device_eval_batch_size 16 

# COMMAND ----------

db_RD_classifier3 = pipeline("zero-shot-classification", model = "/dbfs/mnt/access_work/UC25/Topic Modeling/NLI Models/Fine-tune NLI models/trained_RD_deberta-v3-large-zeroshot-v2_v3")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Model Result

# COMMAND ----------

val_df = pd.read_csv("/dbfs/mnt/access_work/UC25/Topic Modeling/NLI Models/NLI_role_training_data/Role_FT_test_v3.csv")

# COMMAND ----------

val_df.head(3)

# COMMAND ----------

hypothesis_template = "{}"

# COMMAND ----------

Counter(val_df['sentence2'])

# COMMAND ----------

val_df['NLI_score'] = val_df.apply(lambda x: db0_classifier(x['sentence1'] , x['sentence2'], hypothesis_template=hypothesis_template, multi_label=True)['scores'][0] , axis = 1)

# COMMAND ----------

val_df['FT3_NLI_score'] = val_df.apply(lambda x: db_RD_classifier3(x['sentence1'] , x['sentence2'], hypothesis_template=hypothesis_template, multi_label=True)['scores'][0] , axis = 1)

# COMMAND ----------

val_df['label_GT'] = val_df['label'].apply(lambda x: 1 if x == 'entailment' else 0)

# COMMAND ----------

val_df['NLI_label'] = val_df['NLI_score'].apply(lambda x: 1 if x > 0.8 else 0)
val_df['FT3_NLI_label'] = val_df['FT3_NLI_score'].apply(lambda x: 1 if x > 0.8 else 0)

# COMMAND ----------

(val_df['label_GT'] == val_df['NLI_label']).mean()

# COMMAND ----------

(val_df['label_GT'] == val_df['FT3_NLI_label']).mean()

# COMMAND ----------

from sklearn.metrics import confusion_matrix, classification_report

# COMMAND ----------

cm = confusion_matrix(val_df['label_GT'], val_df['NLI_label'])
print("Confusion Matrix:\n", cm)

# COMMAND ----------

report = classification_report(val_df['label_GT'], val_df['NLI_label'])
print("Classification Report:\n", report)

# COMMAND ----------

cm = confusion_matrix(val_df['label_GT'], val_df['FT3_NLI_label'])
print("Confusion Matrix:\n", cm)

# COMMAND ----------

report = classification_report(val_df['label_GT'], val_df['FT3_NLI_label'])
print("Classification Report:\n", report)

# COMMAND ----------


print('Total number of ground truth sentences: ', len(val_df))
print('Accuracy for original NLI model: ', (val_df['label_GT'] == val_df['NLI_label']).mean())
print('Accuracy for fine-tuned NLI model1: ', (val_df['label_GT'] == val_df['FT1_NLI_label']).mean())
print('Accuracy for fine-tuned NLI model2: ', (val_df['label_GT'] == val_df['FT2_NLI_label']).mean())

# COMMAND ----------

role_entail = val_df[(val_df['sentence2'] == 'This role description contains clear information on who the role is for') & (val_df['label'] == 'entailment')]
print('Total number of ground truth sentences that do entail ROLE: ', len(role_entail))
print('Accuracy for original NLI model: ', (role_entail['label_GT'] == role_entail['NLI_label']).mean())
print('Accuracy for fine-tuned NLI1 model: ', (role_entail['label_GT'] == role_entail['FT1_NLI_label']).mean())
print('Accuracy for fine-tuned NLI2 model: ', (role_entail['label_GT'] == role_entail['FT2_NLI_label']).mean())

# COMMAND ----------

role_entail = val_df[(val_df['sentence2'] == 'This role description contains clear information on who the role is for') & (val_df['label'] == 'not_entailment')]
print('Total number of ground truth sentences that do NOT entail ROLE: ', len(role_entail))
print('Accuracy for original NLI model: ', (role_entail['label_GT'] == role_entail['NLI_label']).mean())
print('Accuracy for fine-tuned NLI1 model: ', (role_entail['label_GT'] == role_entail['FT1_NLI_label']).mean())
print('Accuracy for fine-tuned NLI2 model: ', (role_entail['label_GT'] == role_entail['FT2_NLI_label']).mean())

# COMMAND ----------

role_entail = val_df[(val_df['sentence2'] == 'This role description contains clear information on what permissions are granted with this role') & (val_df['label'] == 'entailment')]
print('Total number of ground truth sentences that do entail PERMISSION: ', len(role_entail))
print('Accuracy for fine-tuned NLI1 model: ', (role_entail['label_GT'] == role_entail['FT1_NLI_label']).mean())
print('Accuracy for fine-tuned NLI2 model: ', (role_entail['label_GT'] == role_entail['FT2_NLI_label']).mean())

# COMMAND ----------

role_entail = val_df[(val_df['sentence2'] == 'This role description contains clear information on what permissions are granted with this role') & (val_df['label'] == 'not_entailment')]
print('Total number of ground truth sentences that do NOT entail PERMISSION: ', len(role_entail))
print('Accuracy for fine-tuned NLI1 model: ', (role_entail['label_GT'] == role_entail['FT1_NLI_label']).mean())
print('Accuracy for fine-tuned NLI2 model: ', (role_entail['label_GT'] == role_entail['FT2_NLI_label']).mean())

# COMMAND ----------

val_df

# COMMAND ----------


