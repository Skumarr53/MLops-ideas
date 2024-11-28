# Databricks notebook source
# MAGIC %pip install accelerate
# MAGIC %pip install python-dotenv
# MAGIC %pip install loguru
# MAGIC %pip install hydra-core
# MAGIC %pip install evaluate

# COMMAND ----------

pip install git+https://github.com/huggingface/transformers@v4.46.1

# COMMAND ----------

from datetime import datetime

# COMMAND ----------

# # Load model directly
# from transformers import AutoTokenizer, AutoModelForSequenceClassification

# tokenizer = AutoTokenizer.from_pretrained("MoritzLaurer/deberta-v3-base-zeroshot-v2.0")
# model = AutoModelForSequenceClassification.from_pretrained("MoritzLaurer/deberta-v3-base-zeroshot-v2.0")

# COMMAND ----------

from datetime import datetime


# COMMAND ----------

from centralized_nlp_package.mlflow_utils import ExperimentManager

# COMMAND ----------

from datetime import datetime
datetime.today().strftime('%Y%m%d')

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

from datetime import datetime
# from centralized_nlp_package.nli_utils import arguments, data, metrics, nli_inference, nli_trainer
from typing import List, Dict, Any, Tuple
from centralized_nlp_package.nli_utils import ModelArguments, DataTrainingArguments, run_glue
from transformers import AutoModel, pipeline, AutoModelForSequenceClassification, TrainingArguments

import torch
import transformers
from loguru import logger


def run_finetune(base_model: str,train_file: str, validation_file: str, param_dict: Dict[str, Any]) -> Tuple[AutoModelForSequenceClassification, Dict[str, float]]:
        logger.info("Starting training for DeBERTa model")

        # Prepare ModelArguments
        model_args = ModelArguments(
            model_name_or_path=base_model,
            cache_dir=param_dict.get("cache_dir", None),
            # learning_rate=param_dict.get("learning_rate", 2e-5),
            # weight_decay=param_dict.get("weight_decay", 0.01),
            # per_device_train_batch_size=param_dict.get("train_batch_size", 16),
            # per_device_eval_batch_size=param_dict.get("eval_batch_size", 16)
        )

        # Prepare DataTrainingArguments
        data_args = DataTrainingArguments(
            task_name=param_dict.get("task_name", None),
            train_file=train_file,
            validation_file=validation_file,
            max_seq_length=param_dict.get("max_seq_length", 128),
            pad_to_max_length=param_dict.get("pad_to_max_length", True),
            overwrite_cache=param_dict.get("overwrite_cache", False),
            max_train_samples=param_dict.get("max_train_samples", None),
            max_eval_samples=param_dict.get("max_eval_samples", None),
            max_predict_samples=param_dict.get("max_predict_samples", None)
        )

        # Prepare TrainingArguments
        training_args = TrainingArguments(
            output_dir=param_dict.get("output_dir", "/dbfs/mnt/access_work/UC25/Topic Modeling/NLI Models/Fine-tune NLI models/trained_RD_deberta-v3-base-zeroshot-v2_Santhosh_test/"),
            do_train=True,
            do_eval=True,
            num_train_epochs=param_dict.get("n_epochs", 3),
            learning_rate=param_dict.get("learning_rate", 2e-5),
            weight_decay=param_dict.get("weight_decay", 0.01),
            per_device_train_batch_size=param_dict.get("train_batch_size", 16),
            per_device_eval_batch_size=param_dict.get("eval_batch_size", 16),
            fp16=param_dict.get("fp16", True),
            report_to=None,
            overwrite_output_dir=param_dict.get("overwrite_output_dir", True),
            push_to_hub=param_dict.get("push_to_hub", False),
            seed=param_dict.get("seed", 42)
        )

        # Call run_glue
        trained_model, eval_metrics = run_glue(model_args, data_args, training_args)

        logger.info("Training completed for DeBERTa model")
        return trained_model, eval_metrics


# COMMAND ----------

import torch
from centralized_nlp_package.mlflow_utils import get_model



base_model = "/dbfs/mnt/access_work/UC25/Libraries/HuggingFace/deberta-v3-base-zeroshot-v2"
base_model_name = base_model.split('/')[-1]

# model = get_model(model_name=base_model_name,
#   model_path=base_model,
#                   device=0 if torch.cuda.is_available() else -1
#                   )

# COMMAND ----------

## sample run for demo
param_dict =  {"n_epochs": 5, "learning_rate": 2e-5, "weight_decay": 0.01, "train_batch_size": 16, "eval_batch_size": 16}
train_file_path = "/dbfs/mnt/access_work/UC25/Topic Modeling/NLI Models/NLI_role_training_data/Role_FT_train_v5_11012024.csv"
test_file_path = f"/dbfs/mnt/access_work/UC25/Topic Modeling/NLI Models/NLI_role_training_data/Role_FT_test_v5_11012024.csv"
model, eval_metric = run_finetune( train_file_path, test_file_path, param_dict)


# COMMAND ----------

model

# COMMAND ----------

eval_metric

# COMMAND ----------

# MAGIC %md
# MAGIC ## Run MLFlow Experiment
# MAGIC

# COMMAND ----------

## MLflow Experiment settings
## basic settings for the MLflow experiment, including the experiment name, data source, and paths for training and testing data.

base_exp_name = "RoleDesc_FineTune_DeBERTa_v3"
data_src = "CallTranscript"
train_file_path = "/Workspace/Users/santhosh.kumar3@voya.com/MLFlow_and_NLI_finetune/data/{data_version}"
test_file_path = f"/dbfs/mnt/access_work/UC25/Topic Modeling/NLI Models/NLI_role_training_data/Role_FT_test_v5_11012024.csv"
output_dir = "/dbfs/mnt/access_work/UC25/Topic Modeling/NLI Models/Fine-tune NLI models/trained_RD_deberta-v3-base-zeroshot-v2_Santhosh_test"

## Run Pramaters
# section specifies the models, datasets, and hyperparameters to be used in the experiment. It allows for flexibility in experimenting with different configurations.
model_versions = ["/dbfs/mnt/access_work/UC25/Libraries/HuggingFace/deberta-v3-base-zeroshot-v2"]
# model_versions = ["/dbfs/mnt/access_work/UC25/Libraries/HuggingFace/deberta-v3-base-zeroshot-v2", "/dbfs/mnt/access_work/UC25/Libraries/HuggingFace/deberta-v3-large-zeroshot-v2"]
dataset_versions = ["Role_FT_train_v5_11012024_Full.csv","Role_FT_train_v5_11012024_v1.csv"]
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
    validation_file=test_file_path
)



# COMMAND ----------

## This will excute iteratively run the experiments for the settings provided above.
experiment_manager.run_experiments()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Model Registertion 

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------


from centralized_nlp_package.mlflow_utils import ModelSelector

# COMMAND ----------

# Create an model selctor object for registering the best model
Model_Select_object = ModelSelector("/Users/santhosh.kumar3@voya.com/RoleDesc_FineTune_DeBERTa_v3_CallTranscript_20241121", 'accuracy')

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

models_by_dat_version = Model_Select_object.get_best_models_by_param('base_model') # base model  

# COMMAND ----------

# MAGIC %md
# MAGIC ### Model registering

# COMMAND ----------

# dbutils.library.restartPython()
from centralized_nlp_package.mlflow_utils import ModelTransition

# COMMAND ----------

use_case = "Topic_Modeling"
task_name = "Nli-Fintuning"
model_name = f"{use_case}_{task_name}"
model_uri = f"runs:/{best_run.info.run_id}/model"

model_trans_obj = ModelTransition(model_name, model_uri)

# COMMAND ----------

# register the best models by group in mlflow registry   

for run in models_by_dat_version:
  model_trans_obj._register_model(run)
print("Models registered successfully")
  

# COMMAND ----------

# MAGIC %md
# MAGIC ### Model Transitioning 

# COMMAND ----------

## still being tested 
def transition_model(
    self,
    stage: str,
    experiment_name: str,
    metric: str = "accuracy",
    version: Optional[int] = None  ## We can select a custom model version based on human evaluation.
):
    selector = ModelSelector(experiment_name, metric)
    if version is None:
        best_run = selector.get_best_model()
        if not best_run:
            logger.error("No best model found to transition.")
            return
        version = self._register_model(best_run)
    else:
        # Fetch specific version
        model_version = self._fetch_model_version(version)
        if not model_version:
            logger.error(f"Model version {version} does not exist.")
            return
        self._update_stage(model_version, stage)
    logger.info(f"Model {self.model_name} transitioned to stage: {stage}")

# COMMAND ----------


