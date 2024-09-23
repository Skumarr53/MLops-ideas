# Databricks notebook source
# MAGIC %run ./../utilities/config_utility

# COMMAND ----------

import time
import mlflow
from mlflow.entities.model_registry.model_version_status import ModelVersionStatus

# COMMAND ----------

from abc import ABC, abstractmethod 
  

# COMMAND ----------

"""
NAME : ModelMetaData

DESCRIPTION:
This module serves the MLFlow model metadata functionalities:
               GET LATEST MODEL VERSION 
               GET LATEST MODEL VERSION DETAILS 
               GET MODEL TAGS
               DELETE MODEL TAGS
               SET MODEL TAGS
               SEARCH MODELS BASED ON NAMES
               GET MODEL COUNT
               WAIT UNTIL THE MODEL IS READY STATE
"""


class ModelMetaData(ABC):
  
  """ModelMetaData CLASS"""
  @abstractmethod  
  def perform_model_transition(self):
      """
      performed the model movement from one stage to another stage like staging to production.
    
      """
      pass
    
  def get_latest_model_version_based_on_env(self,env):
      """
      get latest model version based on environment like None(development),staging,production
      
      Parameters:
      argument1 (str): env

      Returns:
      str: model_version
    
      """
      model_version = self.client.get_latest_versions(self.model_name,stages=[env])[0].version
      return model_version

  def get_latest_model_version_details_based_on_env(self,env):
      """
      get latest model version details based on environment like None(development),staging,production

      Parameters:
      argument1 (str): env

      Returns:
      str: model_version_details
    
      """
      model_version_details = self.client.get_latest_versions(self.model_name,stages=[env])[0]
      return model_version_details
  
  def get_latest_model_tags_based_on_env(self,env):
      """
      get latest model tag based on environment like None(development),staging,production

      Parameters:
      argument1 (str): env

      Returns:
      str: model_tags
    
      """
      model_tags = self.client.get_latest_versions(self.model_name,stages=[env])[0].tags
      return model_tags

  def delete_model_tags_based_on_env(self,env):
      """
      delete model tag based on environment like None(development),staging,production

      Parameters:
      argument1 (str): env

      Returns:
      bool: True
    
      """
      model_version=self.get_latest_model_version_based_on_env(env)
      self.client.delete_model_version_tag(name=self.model_name,version=model_version,key=self.mlflow_model_tag_key)
      return True
  
  def set_model_tag(self,env):
      """
      set model tag based on environment like None(development),staging,production
      
      Parameters:
      argument1 (str): env

      Returns:
      bool: True
    
      """

      model_version=self.get_latest_model_version_based_on_env(env)
      # list out keys and values separately
      git_stages_list = list(config.mlflow_stages_dict.keys())
      env_list = list(config.mlflow_stages_dict.values())
      # print key with val 100
      position = env_list.index(env)
      self.client.set_model_version_tag(self.model_name, model_version, self.mlflow_model_tag_key, "{0}_{1}".format(git_stages_list[position],model_version))
      return True

  def search_model_based_on_names(self):
      """
      search models in mlflow model registry based names

      Returns:
      model objects
      """

      models=self.client.search_model_versions("name = '{0}'".format(self.model_name))
      return models

  def get_models_count(self):
      """
      get the models count based on model names

      Returns:
      model objects
      """

      models=self.client.search_model_versions("name = '{0}'".format(self.model_name))
      return len(models)
  
  def wait_until_model_is_ready(self,env):
      """
      wait until the model is ready after deployment or stage transition

      Parameters:
      argument1 (str): env

      Returns:
      bool: True    
      
      """
      for _ in range(10):
        model_version_details = self.get_latest_model_version_details_based_on_env(env)
        status = ModelVersionStatus.from_string(model_version_details.status)
        print("Model status: %s" % ModelVersionStatus.to_string(status))
        if status == ModelVersionStatus.READY:
          return True
        time.sleep(1)
    

# COMMAND ----------


