# Databricks notebook source
# MAGIC %run ./../utilities/config_utility

# COMMAND ----------

# MAGIC  %run ../mlflowutils/model_register_utility

# COMMAND ----------

# MAGIC %pip install nutter

# COMMAND ----------

import unittest
from unittest.mock import patch, MagicMock
from runtime.nutterfixture import NutterFixture
import mlflow

class ModelManagerFixture(NutterFixture):
    """
    This ModelManager fixture is used for unit testing all the methods that are used in the model_manager.py module 
    """
    def __init__(self):
        """
        Helps in initializing all the instance variables
        """
        self.config = MagicMock()
        self.config.mlflow_config = MagicMock()
        self.config.mlflow_config.finbert_model_path = '/dbfs/mnt/access_work/data-science-nlp-ml-common-code/models/finbert_mlflow/logged_model'
        self.config.mlflow_config.finbert_experiment_name = '/dbfs/mnt/access_work/data-science-nlp-ml-common-code/models/finbert_mlflow/run_experiment/'
        self.config.mlflow_config.finbert_run_name = 'finbert_better_transformer_run'
        self.config.mlflow_config.finbert_registered_model_name = 'finbert_better_transformer_model'
        
        self.model_manager = ModelManager()
        NutterFixture.__init__(self)
    
    def assertion_load_artifacts(self):
        """
        This method is used for unit testing ModelManager._load_artifacts method 
        """
        mock_files = ['model1.bin', 'model2.bin']
        with patch('os.listdir', return_value=mock_files):
            artifacts = self.model_manager._load_artifacts()
            expected_artifacts = {
                'model1': '/dbfs/mnt/access_work/data-science-nlp-ml-common-code/models/finbert_mlflow/logged_model/model1.bin',
                'model2': '/dbfs/mnt/access_work/data-science-nlp-ml-common-code/models/finbert_mlflow/logged_model/model2.bin'
            }
            assert artifacts == expected_artifacts
    
    def assertion_register_model(self):
        """
        This method is used for unit testing ModelManager.register_model method 
        """
        with patch('mlflow.set_experiment') as mock_set_experiment, \
             patch('mlflow.start_run') as mock_start_run, \
             patch('mlflow.pyfunc.log_model') as mock_log_model:
            self.model_manager.register_model()
            mock_set_experiment.assert_called_once_with('/dbfs/mnt/access_work/data-science-nlp-ml-common-code/models/finbert_mlflow/run_experiment/')
            mock_start_run.assert_called_once_with(run_name='finbert_better_transformer_run')
            mock_log_model.assert_called_once()
    
    def assertion_load_model(self):
        """
        This method is used for unit testing ModelManager.load_model method 
        """
        mock_model = MagicMock()
        with patch('mlflow.pyfunc.load_model', return_value=mock_model):
            self.model_manager.load_model()
            assert self.model_manager.model == mock_model
    
    def assertion_predict(self):
        """
        This method is used for unit testing ModelManager.predict method 
        """
        # Mock the model's predict method
        mock_model = MagicMock()
        mock_model.predict.return_value = ['positive', 'negative']
        
        # Assign the mock model to the model_manager
        self.model_manager.model = mock_model
        
        texts = ['This is a good day.', 'This is a bad day.']
        predictions = self.model_manager.predict(texts)
        
        # Assert predictions
        assert predictions == ['positive', 'negative']

result = ModelManagerFixture().execute_tests()
print(result.to_string())
