# Import necessary libraries
import torch
import torch.nn as nn
import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient
from mlflow.entities.model_registry.model_version_status import ModelVersionStatus
from abc import ABC, abstractmethod
import time
import subprocess
import os
import pathlib
import logging
from typing import List, Dict
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# COMMAND ----------

"""
NAME : ModelMetaData

DESCRIPTION:
This module serves the MLFlow model metadata functionalities:
    - GET LATEST MODEL VERSION 
    - GET LATEST MODEL VERSION DETAILS 
    - GET MODEL TAGS
    - DELETE MODEL TAGS
    - SET MODEL TAGS
    - SEARCH MODELS BASED ON NAMES
    - GET MODEL COUNT
    - WAIT UNTIL THE MODEL IS READY STATE
"""
class ModelMetaData(ABC):
    
    """ModelMetaData CLASS"""
    def __init__(self, model_name: str):
        self.client = MlflowClient()
        self.model_name = model_name
    
    @abstractmethod  
    def perform_model_transition(self):
        """
        Abstract method to perform model transition between stages.
        """
        pass
    
    def get_latest_model_version_based_on_env(self, env: str) -> str:
        """
        Get latest model version based on environment like None(development), staging, production.
        """
        model_versions = self.client.get_latest_versions(self.model_name, stages=[env])
        if not model_versions:
            raise Exception(f"No models found in the {env} stage.")
        return model_versions[0].version

    def get_latest_model_version_details_based_on_env(self, env: str):
        """
        Get latest model version details based on environment.
        """
        model_versions = self.client.get_latest_versions(self.model_name, stages=[env])
        if not model_versions:
            raise Exception(f"No models found in the {env} stage.")
        return model_versions[0]
    
    def get_latest_model_tags_based_on_env(self, env: str) -> Dict[str, str]:
        """
        Get latest model tags based on environment.
        """
        model_versions = self.client.get_latest_versions(self.model_name, stages=[env])
        if not model_versions:
            raise Exception(f"No models found in the {env} stage.")
        return model_versions[0].tags
    
    def delete_model_tags_based_on_env(self, env: str, tag_key: str) -> bool:
        """
        Delete model tag based on environment.
        """
        model_version = self.get_latest_model_version_based_on_env(env)
        self.client.delete_model_version_tag(name=self.model_name, version=model_version, key=tag_key)
        return True
    
    def set_model_tag(self, env: str, tag_key: str, tag_value: str) -> bool:
        """
        Set model tag based on environment.
        """
        model_version = self.get_latest_model_version_based_on_env(env)
        self.client.set_model_version_tag(self.model_name, model_version, tag_key, tag_value)
        return True
    
    def search_model_based_on_names(self) -> List[Dict]:
        """
        Search models in MLflow Model Registry based on names.
        """
        models = self.client.search_model_versions(f"name = '{self.model_name}'")
        return models
    
    def get_models_count(self) -> int:
        """
        Get the count of models based on model names.
        """
        models = self.client.search_model_versions(f"name = '{self.model_name}'")
        return len(models)
    
    def wait_until_model_is_ready(self, env: str) -> bool:
        """
        Wait until the model is ready after deployment or stage transition.
        """
        for _ in range(10):
            model_version_details = self.get_latest_model_version_details_based_on_env(env)
            status = ModelVersionStatus.from_string(model_version_details.status)
            logger.info(f"Model status: {ModelVersionStatus.to_string(status)}")
            if status == ModelVersionStatus.READY:
                return True
            time.sleep(1)
        raise Exception(f"Model in {env} environment is not ready after waiting.")

# COMMAND ----------

"""
NAME : ModelTransitioner

DESCRIPTION:
This module handles the MLFlow model transition functionalities:
    - Moves model from one environment to another based on the pipeline stages.
"""
class ModelTransitioner(ModelMetaData):
    
    """ModelTransitioner CLASS"""
    def __init__(self, model_name: str, from_env: str, to_env: str, tag_key: str = ""):
        super().__init__(model_name)
        self.from_env = from_env
        self.to_env = to_env
        self.tag_key = tag_key
        
    def perform_model_transition(self) -> bool:
        """
        Perform the model movement from one stage to another stage like staging to production.
        """
        try:
            model_obj = self.client.get_latest_versions(self.model_name, stages=[self.from_env])
            if model_obj and model_obj[0].version:
                version = model_obj[0].version
                self.client.transition_model_version_stage(
                    name=self.model_name,
                    version=version,
                    stage=self.to_env,
                    archive_existing_versions=True
                )
                logger.info(f"Performed model transition from {self.from_env} to {self.to_env}")
                # Optionally set a tag
                if self.tag_key:
                    self.set_model_tag(self.to_env, self.tag_key, f"{self.to_env}_{version}")
                # Wait until the model is ready
                if self.wait_until_model_is_ready(self.to_env):
                    return True
            else:
                raise Exception(f"No models found in the {self.from_env} stage.")
        except Exception as e:
            logger.error(f"Error during model transition: {e}")
            return False
        return False

# COMMAND ----------

"""
NAME : MLFlowModel

DESCRIPTION:
This module handles loading models from specific environments using MLFlow.
"""
class MLFlowModel(ModelTransitioner):
    
    """MLFlowModel CLASS"""
    def __init__(self, model_name: str, env: str, skip_transition: bool, tag_key: str = "", from_env: str = "", to_env: str = ""):
        super().__init__(model_name, from_env, to_env, tag_key)
        self.env = env
        self.skip_transition = skip_transition
        self.model = None
        
    def load_model(self):
        """
        Loads the model from different stages like development, staging, or production.
        """
        try:
            self.model = mlflow.pyfunc.load_model(f"models:/{self.model_name}/{self.env}")
            logger.info(f"Model loaded from {self.env} environment.")
        except Exception as ex:
            logger.warning(f"Model not found in {self.env} environment: {ex}")
            if not self.skip_transition:
                logger.info("Attempting to transition the model to the desired environment.")
                transition_flag = self.perform_model_transition()
                if transition_flag and self.wait_until_model_is_ready(self.env):
                    self.model = mlflow.pyfunc.load_model(f"models:/{self.model_name}/{self.env}")
                    logger.info(f"Model loaded from {self.env} environment after transition.")
                else:
                    raise Exception(f"Model could not be loaded from {self.env} environment after transition.")
            else:
                raise Exception(f"Model could not be loaded from {self.env} environment and transitions are skipped.")
        return self.model

# COMMAND ----------

@dataclass
class Config:
    """
    Configuration dataclass for MLflow settings and paths.
    """
    mlflow_experiment_name: str
    mlflow_run_name: str
    mlflow_registered_model_name: str
    mlflow_model_path: str  # Path where the model artifacts are stored
    mlflow_tag_key: str = "stage"

# COMMAND ----------

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
            stage (str): The stage of the model (e.g., "Production", "Staging").
        Returns:
            str: The latest version of the model in the specified stage.
        """
        try:
            latest_version_info = self.client.get_latest_versions(self.config.mlflow_registered_model_name, stages=[stage])
            if not latest_version_info:
                raise Exception(f"No models found in the {stage} stage.")
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
                name=self.config.mlflow_registered_model_name,
                version=version,
                stage=stage,
                archive_existing_versions=True
            )
            logger.info(f"Model version {version} transitioned to {stage} stage.")
        except Exception as e:
            logger.error(f"Error transitioning model version {version} to {stage} stage: {e}")
            raise
    
    def manage_model_version_transitions(self):
        """
        Manage model version transitions based on performance comparison.
        """
        # Example logic: Promote to Staging if performance improves
        # This should be replaced with actual performance comparison logic
        latest_production_version = self.get_latest_version("Production")
        latest_development_version = self.get_latest_version("Development")
        
        # Placeholder for performance comparison
        # Assume a function compare_models returns True if dev > prod
        if self.compare_models(latest_development_version, latest_production_version):
            self.transition_model_version_stage(latest_development_version, "Staging")
        else:
            logger.info("No improvement in model performance. Transition skipped.")
    
    def compare_models(self, dev_version: str, prod_version: str) -> bool:
        """
        Compare development and production models based on their metrics.
        Args:
            dev_version (str): Development model version.
            prod_version (str): Production model version.
        Returns:
            bool: True if development model is better, else False.
        """
        try:
            dev_metrics = self.get_model_metrics(dev_version)
            prod_metrics = self.get_model_metrics(prod_version)
            # Example comparison based on accuracy
            dev_accuracy = dev_metrics.get("accuracy", 0)
            prod_accuracy = prod_metrics.get("accuracy", 0)
            logger.info(f"Development Model Accuracy: {dev_accuracy}")
            logger.info(f"Production Model Accuracy: {prod_accuracy}")
            return dev_accuracy > prod_accuracy
        except Exception as e:
            logger.error(f"Error comparing models: {e}")
            return False
    
    def get_model_metrics(self, version: str) -> Dict[str, float]:
        """
        Retrieve metrics for a specific model version.
        Args:
            version (str): Model version.
        Returns:
            Dict[str, float]: Metrics dictionary.
        """
        run = self.client.get_run(f"{self.config.mlflow_registered_model_name}/{version}")
        metrics = run.data.metrics
        return metrics

# COMMAND ----------

class ModelManager:
    """
    ModelManager handles model registration, loading, and prediction.
    """
    def __init__(self, config: Config):
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
                pathlib.Path(file).stem: os.path.join(self.config.mlflow_model_path, file)
                for file in os.listdir(self.config.mlflow_model_path)
                if not file.startswith('.')
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
            mlflow.set_experiment(self.config.mlflow_experiment_name)
            with mlflow.start_run(run_name=self.config.mlflow_run_name):
                mlflow.log_params(self.get_training_params())
                # Example: Log metrics after evaluation
                mlflow.log_metrics(self.evaluate_model())
                mlflow.pyfunc.log_model(
                    'model',
                    python_model=self,  # Assuming ModelManager inherits from mlflow.pyfunc.PythonModel if needed
                    artifacts=self.artifacts,
                    registered_model_name=self.config.mlflow_registered_model_name
                )
            logger.info("Model registered successfully.")
        except Exception as e:
            logger.error(f"Error registering model: {e}")
            raise

    def get_training_params(self) -> Dict[str, any]:
        """
        Retrieve training parameters.
        Returns:
            Dict[str, any]: Training parameters.
        """
        # Placeholder for actual training parameters
        return {
            "model_name": "deberta-v3-large-zeroshot-v2",
            "num_train_epochs": 3,
            "learning_rate": 2e-5,
            "weight_decay": 0.01,
            "per_device_train_batch_size": 16,
            "per_device_eval_batch_size": 16
        }

    def evaluate_model(self) -> Dict[str, float]:
        """
        Evaluate the model and return metrics.
        Returns:
            Dict[str, float]: Evaluation metrics.
        """
        # Placeholder for actual evaluation logic
        return {
            "accuracy": 0.95,
            "f1_score": 0.93
        }

    def load_model(self):
        """
        Load the registered model from MLflow.
        Returns:
            PythonModel: The loaded model.
        """
        try:
            self.model = mlflow.pyfunc.load_model(f"models:/{self.config.mlflow_registered_model_name}/Production")
            logger.info("Model loaded successfully from Production stage.")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

    def predict(self, inputs: List[str]) -> List[str]:
        """
        Predict using the loaded model.
        Args:
            inputs (List[str]): Input data for prediction.
        Returns:
            List[str]: Model predictions.
        """
        if not self.model:
            raise Exception("Model is not loaded. Call load_model() before prediction.")
        try:
            predictions = self.model.predict(inputs)
            logger.info("Predictions made successfully.")
            return predictions
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            raise

# COMMAND ----------

# Example Usage Scenario
if __name__ == "__main__":
    # Initialize configuration
    config = Config(
        mlflow_experiment_name="/Users/your_username/NLI_FineTuning_Experiment",
        mlflow_run_name="Fine-tune_DeBERTa_v3",
        mlflow_registered_model_name="DeBERTa_v3_NLI_Model",
        mlflow_model_path="/dbfs/mnt/access_work/UC25/Libraries/HuggingFace/deberta-v3-large-zeroshot-v2",
        mlflow_tag_key="stage"
    )
    
    # Initialize ModelManager
    model_manager = ModelManager(config)
    
    # Register the model
    model_manager.register_model()
    
    # Initialize ModelVersionManager
    version_manager = ModelVersionManager(config)
    
    # Manage model transitions based on performance
    version_manager.manage_model_version_transitions()
    
    # Load the production model for inference
    model_manager.load_model()
    
    # Example prediction
    sample_inputs = ["Example sentence for NLI inference."]
    predictions = model_manager.predict(sample_inputs)
    logger.info(f"Predictions: {predictions}")
