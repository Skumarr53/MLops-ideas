# Databricks notebook source
# MAGIC
# MAGIC %pip install loguru==0.7.2
# MAGIC %pip install hydra-core==1.3
# MAGIC %pip install python-dotenv==1.0.1
# MAGIC %pip install numpy==1.24.4
# MAGIC %pip install cryptography==43.0.1
# MAGIC %pip install gensim==4.3.3
# MAGIC %pip install cython==3.0.11
# MAGIC %pip install spacy==3.4.4 #3.0.4
# MAGIC %pip install thinc==8.1.7
# MAGIC %pip install pandas==2.0.0
# MAGIC %pip install snowflake-connector-python==3.12.2
# MAGIC %pip install transformers==4.46.1
# MAGIC %pip install pyarrow==16.0.0
# MAGIC %pip install datasets==3.1.0
# MAGIC %pip install evaluate==0.4.3
# MAGIC %pip install pyspark==3.5.3
# MAGIC %pip install "dask[dataframe,distributed]"==2023.9.3
# MAGIC %pip install torch==2.0.0
# MAGIC # %pip install mlflow
# MAGIC %pip install pydantic==1.10.6
# MAGIC %pip install cymem==2.0.8
# MAGIC %pip install scikit-learn==1.1.0
# MAGIC %pip install typer==0.7.0
# MAGIC %pip install accelerate==0.26.0

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# from centralized_nlp_package.data_access import *

# COMMAND ----------

# pip install git+https://github.com/huggingface/transformers@v4.46.1

# COMMAND ----------

# MAGIC %md
# MAGIC ## Testing Experimentation for single run

# COMMAND ----------

# MAGIC %md
# MAGIC ### original code

# COMMAND ----------

## original code is run through bash command below using single script run_glue.py

# python run_glue.py \
#   --model_name_or_path "/dbfs/mnt/access_work/UC25/Libraries/HuggingFace/deberta-v3-large-zeroshot-v2" \
#   --output_dir "/dbfs/mnt/access_work/UC25/Topic Modeling/NLI Models/Fine-tune NLI models/trained_RD_deberta-v3-large-zeroshot-v2_v3" \
#   --train_file "/dbfs/mnt/access_work/UC25/Topic Modeling/NLI Models/NLI_role_training_data/Role_FT_train_v3.csv" \
#   --validation_file "/dbfs/mnt/access_work/UC25/Topic Modeling/NLI Models/NLI_role_training_data/Role_FT_test_v3.csv" \
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

# MAGIC %md
# MAGIC ### refactored nli finetuning 

# COMMAND ----------

# from datetime import datetime
# # from centralized_nlp_package.nli_utils import arguments, data, metrics, nli_inference, nli_trainer
# from typing import List, Dict, Any, Tuple
# from centralized_nlp_package.nli_utils import ModelArguments, DataTrainingArguments, run_glue
# from transformers import AutoModel, pipeline, AutoModelForSequenceClassification, TrainingArguments

# import torch
# import transformers
# from loguru import logger


# def run_finetune(base_model: str,train_file: str, validation_file: str, param_dict: Dict[str, Any]) -> Tuple[AutoModelForSequenceClassification, Dict[str, float]]:
#         logger.info("Starting training for DeBERTa model")

#         # Prepare ModelArguments
#         model_args = ModelArguments(
#             model_name_or_path=base_model,
#             cache_dir=param_dict.get("cache_dir", None),
#             # learning_rate=param_dict.get("learning_rate", 2e-5),
#             # weight_decay=param_dict.get("weight_decay", 0.01),
#             # per_device_train_batch_size=param_dict.get("train_batch_size", 16),
#             # per_device_eval_batch_size=param_dict.get("eval_batch_size", 16)
#         )

#         # Prepare DataTrainingArguments
#         data_args = DataTrainingArguments(
#             task_name=param_dict.get("task_name", None),
#             train_file=train_file,
#             validation_file=validation_file,
#             max_seq_length=param_dict.get("max_seq_length", 128),
#             pad_to_max_length=param_dict.get("pad_to_max_length", True),
#             overwrite_cache=param_dict.get("overwrite_cache", False),
#             max_train_samples=param_dict.get("max_train_samples", None),
#             max_eval_samples=param_dict.get("max_eval_samples", None),
#             max_predict_samples=param_dict.get("max_predict_samples", None)
#         )

#         # Prepare TrainingArguments
#         training_args = TrainingArguments(
#             output_dir=param_dict.get("output_dir", "/dbfs/mnt/access_work/UC25/Topic Modeling/NLI Models/Fine-tune NLI models/trained_RD_deberta-v3-base-zeroshot-v2_Santhosh_test/"),
#             do_train=True,
#             do_eval=True,
#             num_train_epochs=param_dict.get("n_epochs", 3),
#             learning_rate=param_dict.get("learning_rate", 2e-5),
#             weight_decay=param_dict.get("weight_decay", 0.01),
#             per_device_train_batch_size=param_dict.get("train_batch_size", 16),
#             per_device_eval_batch_size=param_dict.get("eval_batch_size", 16),
#             fp16=param_dict.get("fp16", True),
#             report_to=None,
#             overwrite_output_dir=param_dict.get("overwrite_output_dir", True),
#             push_to_hub=param_dict.get("push_to_hub", False),
#             seed=param_dict.get("seed", 42)
#         )

#         # Call run_glue
#         trained_model, eval_metrics = run_glue(model_args, data_args, training_args)

#         logger.info("Training completed for DeBERTa model")
#         return trained_model, eval_metrics


# COMMAND ----------

# import torch
# from centralized_nlp_package.model_utils import get_model



# base_model = "/dbfs/mnt/access_work/UC25/Libraries/HuggingFace/deberta-v3-base-zeroshot-v2"
# base_model_name = base_model.split('/')[-1]

# # model = get_model(model_name=base_model_name,
# #   model_path=base_model,
# #                   device=0 if torch.cuda.is_available() else -1
# #                   )

# COMMAND ----------

# from centralized_nlp_package.nli_utils import run_finetune

# ## sample run for demo
# param_dict =  {"n_epochs": 5, "learning_rate": 2e-5, "weight_decay": 0.01, "train_batch_size": 16, "eval_batch_size": 16}
# out_dir = "/dbfs/mnt/access_work/UC25/Topic Modeling/NLI Models/trained_RD_deberta-v3-base-zeroshot-v2_Santhosh_mlflow_test/"
# train_file_path = "/dbfs/mnt/access_work/UC25/Topic Modeling/NLI Models/NLI_role_training_data/Role_FT_train_v5_11012024.csv"
# test_file_path = f"/dbfs/mnt/access_work/UC25/Topic Modeling/NLI Models/NLI_role_training_data/Role_FT_test_v5_11012024.csv"
# model, eval_metric = run_finetune(base_model, train_file_path, out_dir, test_file_path, param_dict)


# COMMAND ----------

# MAGIC %md
# MAGIC ## Run MLFlow Experiment
# MAGIC

# COMMAND ----------

## MLflow Experiment settings
## basic settings for the MLflow experiment, including the experiment name, data source, and paths for training and testing data.
from centralized_nlp_package.model_utils import ExperimentManager

base_exp_name = "Mass_ConsumerTopic_FineTune_DeBERTa_v3"
data_src = "CallTranscript"
train_file_path = "/Workspace/Users/santhosh.kumar3@voya.com/MLFlow_and_NLI_finetune/data/{data_version}"
test_file_path = "/Workspace/Users/santhosh.kumar3@voya.com/MLFlow_and_NLI_finetune/data/test.csv"
output_dir = "/dbfs/mnt/access_work/UC25/Topic Modeling/NLI Models/Fine-tune NLI models/trained_RD_deberta-v3-base-zeroshot-v2_Santhosh_test"

## Run Pramaters
# section specifies the models, datasets, and hyperparameters to be used in the experiment. It allows for flexibility in experimenting with different configurations.
# model_versions = ["/dbfs/mnt/access_work/UC25/Libraries/HuggingFace/deberta-v3-base-zeroshot-v2"]
model_versions = ["/dbfs/mnt/access_work/UC25/Libraries/HuggingFace/deberta-v3-large-zeroshot-v2"]
dataset_versions = ["main.csv","train.csv"]
hyperparameters = [
        {"n_epochs": 5, "learning_rate": 2e-5, "weight_decay": 0.01, "train_batch_size": 16, "eval_batch_size": 16},
        {"n_epochs": 8, "learning_rate": 3e-5, "weight_decay": 0.02, "train_batch_size": 24, "eval_batch_size": 24}
    ]


# ExperimentManager object is created with the provided settings, which manages the execution and tracking of experiments using MLflow.
experiment_manager = ExperimentManager(
    base_name=base_exp_name,
    data_src=data_src,
    dataset_versions=dataset_versions,
    hyperparameters=hyperparameters,
    base_model_versions=model_versions,
    output_dir=output_dir,
    train_file=train_file_path,
    validation_file=test_file_path,
    evalute_pretrained_model = True, # default = True
    
)



# COMMAND ----------

## This will excute iteratively run the experiments for the settings provided above.
experiment_manager.run_experiments()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Model Registertion 

# COMMAND ----------


from centralized_nlp_package.model_utils import ModelSelector

# COMMAND ----------

# Create an model selctor object for registering the best model
Model_Select_object = ModelSelector("/Users/santhosh.kumar3@voya.com/Mass_ConsumerTopic_FineTune_DeBERTa_v3_CallTranscript_20241205", 'accuracy')

# COMMAND ----------

## list all the runs availabe in the experiment
Model_Select_object.list_available_models()

# COMMAND ----------

## This method will print the best model from the experiment and return the run object
best_run = Model_Select_object.get_best_model()
print(best_run)

# COMMAND ----------

## Choose the best model from each group categorized by dataset version. This can also be done using other parameters, such as model architecture.
## comparing large and small base model 

models_by_basemodel_version = Model_Select_object.get_best_models_by_tag('base_model_name') # base model  

# COMMAND ----------

# MAGIC %md
# MAGIC ### Model registering

# COMMAND ----------

# dbutils.library.restartPython()
from centralized_nlp_package.model_utils import ModelTransition

# COMMAND ----------

use_case = "Topic_Modeling"
task_name = "Consumer_Topic"
model_name = f"{use_case}_{task_name}"

model_trans_obj = ModelTransition(model_name)

# COMMAND ----------

# register the best models by group in mlflow registry   

for run in models_by_basemodel_version:
  print(f"runs:/{run.info.run_id}/model")
  model_trans_obj._register_model(run)
print("Models registered successfully")
  

# COMMAND ----------

# MAGIC %md
# MAGIC ### Model Transitioning 

# COMMAND ----------

import mlflow
def _fetch_model_version(model_name, version: int):
    client = mlflow.tracking.MlflowClient()
    try:
        model_version = client.get_model_version(model_name, str(version))
        return model_version
    except mlflow.exceptions.RestException as e:
        print(f"Error fetching model version {version}: {e}")
        return None
    
def _update_stage(model_name, model_version, stage: str):
    client = mlflow.tracking.MlflowClient()
    client.transition_model_version_stage(
        name=model_name,
        version=model_version.version,
        stage=stage,
        archive_existing_versions=True
    )
    print(f"Model version {model_version.version} updated to stage {stage}")

# COMMAND ----------

print(model_name)
selcted_version = _fetch_model_version(model_name, 1)
_update_stage(model_name, selcted_version, "Production")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Load production model 

# COMMAND ----------

production_model = mlflow.pyfunc.load_model(
    model_uri=f"models:/{model_name}/Production"
)

# COMMAND ----------

dir(production_model)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Evaluate the FineTuned models

# COMMAND ----------

from centralized_nlp_package.nli_utils import evaluate_nli_models
  
model_paths = ["/dbfs/mnt/access_work/UC25/Topic Modeling/NLI Models/Fine-tune NLI models/trained_RD_deberta-v3-base-zeroshot-v2_Santhosh_test_v2", # n_epochs=5
               "/dbfs/mnt/access_work/UC25/Topic Modeling/NLI Models/Fine-tune NLI models/trained_RD_deberta-v3-base-zeroshot-v2_Santhosh_test_v3"]

test_file_path = "/Workspace/Users/santhosh.kumar3@voya.com/MLFlow_and_NLI_finetune/data/test.csv"
evaluate_nli_models(model_paths, test_file_path, entailment_threshold=0.7)

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------


