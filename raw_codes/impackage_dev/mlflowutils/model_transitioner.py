# Databricks notebook source
# MAGIC %run ./model_metadata

# COMMAND ----------

from mlflow.tracking import MlflowClient

# COMMAND ----------

"""
NAME : ModelTransitioner

DESCRIPTION:
This module serves the MLFlow model functionalities:
                MOVES MODEL FROM ONE ENVIRONMENT TO OTHER BASED ON THE DEV,STAGING OR PROD PIPELINE
"""


class ModelTransitioner(ModelMetaData):
  
  """ModelTransitioner CLASS"""
  def __init__(self,model_name,env,model_tag,from_env,to_env):
    self.client = MlflowClient()
    self.model_name=model_name
    self.env=env
    self.mlflow_model_tag_key=model_tag
    self.from_env=from_env
    self.to_env=to_env
    
    
  def perform_model_transition(self):
      """
      performed the model movement from one stage to another stage like staging to production.
    
      """
      model_obj=self.client.get_latest_versions(self.model_name,stages=[self.from_env])
      if (len(model_obj)>0 and (model_obj[0].version==self.get_latest_model_version_based_on_env(self.from_env))):
        self.version=model_obj[0].version
        self.client.transition_model_version_stage(
                      name=self.model_name,
                      version=self.version,
                      stage=self.to_env
                    )
        print("performed model transition from {0} to {1}".format(self.from_env,self.to_env))
      else:
        raise Exception("model is not present in {0} environment".format(self.from_env))
      return True

  
