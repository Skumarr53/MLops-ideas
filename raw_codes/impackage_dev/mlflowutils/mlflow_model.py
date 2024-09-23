# Databricks notebook source
# MAGIC %pip install transformers==4.38.1 optimum==1.17.1 torch==2.0

# COMMAND ----------

# MAGIC %run ./model_transitioner

# COMMAND ----------

import torch
import torch.nn as nn

# COMMAND ----------

"""
NAME : MLFlowModel

DESCRIPTION:
This module serves the MLFlow model load functionalities:
                HELPS IN LOADING THE MODEL FROM SPECIFIC ENVIRONMENT
"""


class MLFlowModel(ModelTransitioner):
  
  """MLFlowModel CLASS"""
  def __init__(self,model_name,env,skip_transition,model_tag="",from_env="",to_env=""):
    self.skip_transition=skip_transition
    super().__init__(model_name,env,model_tag,from_env,to_env)
    
  def load_model(self):
    """
    Loads the model from  different stages like None, staging or production
    
    Parameters:
    argument1 (str): notebook path
   
    Returns:
    model object
    
    """
    try:
      self.model = mlflow.pyfunc.load_model("models:/{0}/{1}".format(self.model_name, self.env))
    except Exception as ex:
      if self.skip_transition==False:
        transition_flag=self.perform_model_transition()
        if transition_flag and self.wait_until_model_is_ready(self.env):
          self.model = mlflow.pyfunc.load_model("models:/{0}/{1}".format(self.model_name, self.env))
        else:
          raise Exception("model is not loaded from {0}".format(self.env))
      else:
        raise Exception("model is not loaded from {0}".format(self.env))
    print("model loaded from {0}".format(self.env))
    return self.model
      
    
