# Databricks notebook source
# MAGIC %run ./register_BERT_model

# COMMAND ----------

# MAGIC %run ./../utilities/config_utility

# COMMAND ----------

import os
import pathlib
import logging
from typing import List, Dict
from mlflow.pyfunc import PythonModel, PythonModelContext
import mlflow

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelManager:
  """
  Initialize the ModelManager with the given configuration.
  Args:
    config: Configuration object containing MLflow settings and paths.
  """
  def __init__(self):
    self.config = config
    self.artifacts = self._load_artifacts()
    self.model = None

  def _load_artifacts(self) -> Dict[str, str]:
    """
    Load model artifacts from the specified directory.
    Returns:
        Dict[str, str]: A dictionary mapping artifact names to their file paths.
    """
    try:
      artifacts = {
        pathlib.Path(file).stem: os.path.join(self.config.mlflow_config.finbert_model_path, file)
        for file in os.listdir(self.config.mlflow_config.finbert_model_path)
        if not os.path.basename(file).startswith('.')
      }
      logger.info("Artifacts loaded successfully.")
      return artifacts
    except Exception as e:
      logger.error(f"Error loading artifacts: {e}")
      raise

  def register_model(self):
    """
    Register the model with MLflow.
    """
    try:
      mlflow.set_experiment(self.config.mlflow_config.finbert_experiment_name)
      with mlflow.start_run(run_name=self.config.mlflow_config.finbert_run_name):
        mlflow.pyfunc.log_model(
          'classifier',
          python_model=RegisterBERTModel(),
          artifacts=self.artifacts,
          registered_model_name=self.config.mlflow_config.finbert_registered_model_name
        )
      logger.info("Model registered successfully.")
    except Exception as e:
      logger.error(f"Error registering model: {e}")
      raise

  def load_model(self):
    """
    Load the registered model from MLflow.
    Returns:
      PythonModel: The loaded model.
    """
    try:
      self.model = mlflow.pyfunc.load_model(f'models:/{self.config.mlflow_config.finbert_registered_model_name}/latest')
      logger.info("Model loaded successfully.")
    except Exception as e:
      logger.error(f"Error loading model: {e}")
      raise

  def predict(self, texts: List[str]) -> List[str]:
    """
    Predict the sentiment of the given texts using the loaded model.
    Args:
      model (PythonModel): The loaded model.
      texts (List[str]): A list of texts to predict.
    Returns:
      List[str]: The prediction results.
    """
    try:
      predictions = self.model.predict(texts)
      logger.info("Predictions made successfully.")
      return predictions
    except Exception as e:
      logger.error(f"Error making predictions: {e}")
      raise
