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

pip install git+https://github.com/huggingface/transformers@v4.46.1

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

# db0_classifier = pipeline("zero-shot-classification", model= model_folder_path + model_1_folder_name)

# COMMAND ----------

# MAGIC %md
# MAGIC ## MLFLow Setup Testing

# COMMAND ----------

import mlflow
from transformers import AutoModel


# COMMAND ----------

# MAGIC %md
# MAGIC ### Organizing the Training/Development Environment
# MAGIC
# MAGIC This environment is where you'll fine-tune your model and log experiments.

# COMMAND ----------

# MAGIC %md
# MAGIC ### a. Structure Your Fine-Tuning Script with MLflow Logging
# MAGIC Integrate MLflow logging into your fine-tuning process to capture parameters, metrics, and models.

# COMMAND ----------

from datetime import datetime
import random


data_src = "CallTranscript"
run_date = datetime.today().strftime('%Y%m%d')
experiment_name = f"/Users/santhosh.kumar3@voya.com/Fine-tune_DeBERTa_v3_SF_euity_{data_src}_{run_date}"
mlflow.set_experiment(experiment_name) ## experiment name for specific for use case ex: SF equity

# Get today's date
base_run_name = "Fine-tune_DeBERTa_v3"

## run_name for current quarter
run_name = f"{base_run_name}_{run_date}"

dataset_partition = ["full", "sample"] ## comutizable
params = [{"n_epoches": 5}, {"n_epoches": 10}]


for dataset in dataset_partition:
    for i,param_set in enumerate(params):
        with mlflow.start_run(run_name=f"{run_name}_{dataset}_paramset_{i}") as run: # for current run it named "Fine-tune_DeBERTa_v3_test"
            mlflow.set_tag("run_date", run_date)

            # Log parameters
            mlflow.log_param("run_date", run_date) 
            mlflow.log_param("dataset", dataset)
            mlflow.log_param("base_model_name", "deberta-v3-large-zeroshot-v2")
            mlflow.log_param("num_train_epochs", param_set["n_epoches"])
            mlflow.log_param("learning_rate", 2e-5)
            mlflow.log_param("weight_decay", 0.01)
            mlflow.log_param("per_device_train_batch_size", 16)

            # Define device
            device = 0 if torch.cuda.is_available() else -1

            # # Initialize the pipeline
            # model_folder_path = "/dbfs/mnt/access_work/UC25/Libraries/HuggingFace/"
            # model_1_folder_name = "deberta-v3-large-zeroshot-v2"
            # db0_classifier = pipeline("zero-shot-classification", model=model_folder_path + model_1_folder_name)

            # import subprocess
            # subprocess.run([
            #     "python", "run_glue.py",
            #     "--model_name_or_path", f"{model_folder_path}{model_1_folder_name}",
            #     "--output_dir", "/dbfs/mnt/access_work/UC25/Topic Modeling/NLI Models/Fine-tune NLI models/trained_RD_deberta-v3-large-zeroshot-v2_v3",
            #     "--train_file", "/dbfs/mnt/access_work/UC25/Topic Modeling/NLI Models/NLI_role_training_data/Role_FT_train_v3.csv",
            #     "--validation_file", "/dbfs/mnt/access_work/UC25/Topic Modeling/NLI Models/NLI_role_training_data/Role_FT_test_v3.csv",
            #     "--do_train",
            #     "--do_eval",
            #     "--num_train_epochs", "3",
            #     "--fp16",
            #     "--report_to", "none",
            #     "--learning_rate", "2e-5",
            #     "--weight_decay", "0.01",
            #     "--per_device_train_batch_size", "16",
            #     "--per_device_eval_batch_size", "16"
            # ], check=True)

            # After training, load the fine-tuned model
            fine_tuned_model_path = "/dbfs/mnt/access_work/UC25/Libraries/HuggingFace/deberta-v3-large-zeroshot-v2"
            ft_model = AutoModel.from_pretrained(fine_tuned_model_path)
            # Log the fine-tuned model
            mlflow.pytorch.log_model(ft_model, "model")
            
            # Evaluate model (example: log accuracy)
            # Replace with your evaluation logic
            accuracy = random.uniform(0.9, 0.95)  # Placeholder
            mlflow.log_metric("accuracy", accuracy) 


# COMMAND ----------

# MAGIC %md
# MAGIC Note: If there mutilple models with each run name, the latest one will be used by default. based on the highest accuracy.

# COMMAND ----------

from mlflow.tracking import MlflowClient

client = MlflowClient()
experiment = client.get_experiment_by_name(experiment_name)
best_run = client.search_runs(experiment_ids=[experiment.experiment_id],
                          order_by=["metrics.accuracy DESC"],
                          max_results=1)
best_run_id = best_run[0].info.run_id

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setting Up the MLflow Model Registry
# MAGIC The Model Registry allows you to manage model versions and their lifecycle stages (e.g., Staging, Production).
# MAGIC
# MAGIC ### b. Register the Fine-Tuned Model
# MAGIC After logging the model in your training run, register it to the Model Registry.

# COMMAND ----------

# Get the run ID
# run_id = run.info.run_id
## function

# Register the model
model_name = "DeBERTa_v3_NLI_Model"
model_uri = f"runs:/{best_run_id}/model"

registered_model = mlflow.register_model(model_uri, model_name)

# COMMAND ----------

Transition two ways: manual on the interface and programatically

# COMMAND ----------

model_obj=mlflow.client.get_latest_versions(self.model_name,stages=[self.from_env])
model_version = model_obj[0].version
if (len(model_obj)>0 and (model_obj[0].version==self.get_latest_model_version_based_on_env(self.from_env))):
  self.version=model_obj[0].version
  self.client.transition_model_version_stage(
                name=self.model_name,
                version=self.version,
                stage=self.to_env
              )

# COMMAND ----------

# MAGIC %md
# MAGIC ### Loading model from production in the prediction pipiline

# COMMAND ----------


production_model = mlflow.pyfunc.load_model(
    model_uri=f"models:/{model_name}/Staging"
)

# COMMAND ----------

type(production_model)

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


