# Databricks notebook source
# pip install mlflow==2.18.0 accelerate==0.26.0 torch torchvision==0.14.1 transformers==4.46.1 optimum
!pip install transformers==4.38.1 optimum==1.17.1 torch==2.0

# COMMAND ----------

import torch

torch.__version__

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import mlflow

# COMMAND ----------

# MAGIC %md
# MAGIC ### Consumer Topic

# COMMAND ----------

reg_name = "TopicModelling_ConsumerTopic_Deberta"

# Load the1 model from the staging branch of the model registry
staging_model = mlflow.transformers.load_model(f"models:/{reg_name}/Staging")
# model_path = mlflow.artifacts.download_artifacts(f"models:/{reg_name}/Staging")


# Access the model and tokenizer
model = staging_model.model
tokenizer = staging_model.tokenizer


# COMMAND ----------

out_path = "/dbfs/mnt/access_work/UC25/Santhosh_MlFlow/ConsumerTopic"

try:
  dbutils.fs.rm(out_path, recurse=True)
except:
  pass
dbutils.fs.mkdirs(out_path)

mlflow.transformers.save_model(staging_model, out_path)

# COMMAND ----------

dbutils.fs.ls("/dbfs/mnt/access_work/UC25/Santhosh_MlFlow/ConsumerTopic")

# COMMAND ----------

# Load the model from the staging branch of the model registry
# load_model = mlflow.transformers.load_model(out_path)


# Access the model and tokenizer
model = load_model.model
tokenizer = load_model.tokenizer

# COMMAND ----------

from transformers import pipeline

pl_inference_pros = pipeline(task="text-classification", model = model, tokenizer = tokenizer, device = 0)
pl_inference_pros([{'text':'work life balance', 'text_pair':'this employee review is about flexibility'}])

# COMMAND ----------

# MAGIC %md
# MAGIC ### Finbert Model

# COMMAND ----------

stg2Ver_map = {'dev':8,'Staging':9,'Production':12}
out_path = "/dbfs/mnt/access_work/UC25/Santhosh_MlFlow/FinBert/{version}"

reg_name = "finbert-better-transformer-sentiment-classification_model"
def download_mlflow_models(reg_name,map_dict = stg2Ver_map):
  for stg, ver in map_dict.items():
    path = out_path.format(version=stg)
    print(path)
    print("models:/{0}/{1}".format(reg_name, ver))
    model = mlflow.pyfunc.load_model("models:/{0}/{1}".format(reg_name, ver))
    dbutils.fs.mkdirs(path.format(version=stg))
    mlflow.pyfunc.save_model(path, python_model=model)
    # print("Model saved to {0}".format(path))

    test_model = mlflow.pyfunc.load_model(path)
    print(test_model.predict(["this is a good product", "this is a bad product"]))

import os
def download_mlflow_models(reg_name, map_dict=stg2Ver_map):
    for stg, ver in map_dict.items():
        model_uri = f"models:/{reg_name}/{ver}"
        dst_path = out_path.format(version=stg)
        
        print(f"Downloading model from: {model_uri}")
        print(f"Saving to: {dst_path}")
        
        # Create target directory if not exists
        os.makedirs(dst_path, exist_ok=True)
        
        # Download artifacts to local path
        local_path = mlflow.artifacts.download_artifacts(artifact_uri=model_uri)
        shutil.copytree(local_path, dst_path, dirs_exist_ok=True)
 
        # (Optional) Test the model
        test_model = mlflow.pyfunc.load_model(dst_path)
        preds = test_model.predict(["this is a good product", "this is a bad product"])
        print(preds)

download_mlflow_models(reg_name,map_dict = stg2Ver_map)


# COMMAND ----------

model = mlflow.pyfunc.load_model("models:/finbert-better-transformer-sentiment-classification_model/8")
print(model.predict(["this is a good product", "this is a bad product"]))


# COMMAND ----------

### Zip code bases


import shutil
modelPath = "/Workspace/Users/santhosh.kumar3@voya.com/SanthoshMigrate"
zipPath = "/Workspace/Users/santhosh.kumar3@voya.com/SanthoshMigrate.zip"
shutil.make_archive(base_dir=modelPath, format='zip', base_name=zipPath)


# COMMAND ----------


import zipfile
with zipfile.ZipFile(zipPath, 'w', strict_timestamps=False) as zipf:
  zipf.write(modelPath)


# COMMAND ----------

