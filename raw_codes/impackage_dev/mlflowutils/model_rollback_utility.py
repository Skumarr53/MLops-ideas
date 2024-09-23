# Databricks notebook source
# MAGIC %pip install transformers==4.38.1 optimum==1.17.1 torch==2.0

# COMMAND ----------

# MAGIC %run ./../utilities/config_utility

# COMMAND ----------

# Databricks notebook source
# MAGIC %pip install transformers==4.38.1 optimum==1.17.1 torch==2.0

# COMMAND ----------
# MAGIC %run ./../utilities/config_utility

# COMMAND ----------
import pandas as pd
import torch
import mlflow
from mlflow import MlflowClient
import logging
from typing import List, Dict
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelVersionManager:
  def __init__(self, config: Config):
    """
    Initialize the ModelVersionManager with the given configuration.
    Args:
        config (Config): Configuration object containing necessary settings.
    """
    self.client = MlflowClient()
    self.config = config
    

  def get_latest_version(self, stage: str) -> str:
    """
    Get the latest version of the model in the specified stage.
    Args:
      stage (str): The stage of the model (e.g., "Production", "Archived").
    Returns:
      str: The latest version of the model in the specified stage.
    """
    try:
      latest_version_info = self.client.get_latest_versions(self.config.mlflow_FINBERT_model_name, stages=[stage])
      latest_version = latest_version_info[0].version
      logger.info(f"Latest {stage} version: {latest_version}")
      return latest_version
    except Exception as e:
      logger.error(f"Error getting latest version for stage {stage}: {e}")
      raise

  def transition_model_version_stage(self, version: str, stage: str):
    """
    Transition the model version to the specified stage.

    Args:
      version (str): The version of the model to transition.
      stage (str): The target stage (e.g., "Staging", "Production").
    """
    try:
      self.client.transition_model_version_stage(
        name=self.config.mlflow_FINBERT_model_name,
        version=version,
        stage=stage
      )
      logger.info(f"Model version {version} transitioned to {stage} stage.")
    except Exception as e:
      logger.error(f"Error transitioning model version {version} to {stage} stage: {e}")
      raise

  def manage_model_version_transitions(self):
    """
    Main function to manage model version transitions.

    This function initializes the configuration and the ModelVersionManager,
    then transitions the latest production version to staging and the latest
    archived version to production.
    """

    # Transition latest production version to staging
    latest_production_version = manager.get_latest_version("Production")
    manager.transition_model_version_stage(latest_production_version, "Staging")

    # Transition latest archived version to production
    latest_archived_version = manager.get_latest_version("Archived")
    manager.transition_model_version_stage(latest_archived_version, "Production")

