# MLFlow Utilities for Databricks

## Overview

This project provides a set of utilities for managing machine learning models using MLFlow within a Databricks environment. The utilities facilitate model registration, loading, transitioning between different stages (development, staging, production), and making predictions using BERT-based models. 
## Project Structure

The project consists of the following main modules:

- **MLFlowModel**:  `impackage_dev/mlflowutils/mlflow_model.py` 
- **ModelMetaData**:  `impackage_dev/mlflowutils/model_metadata.py` 
- **ModelManager**:  `impackage_dev/mlflowutils/model_register_utility.py` 
- **ModelVersionManager**:  `impackage_dev/mlflowutils/model_rollback_utility.py` 
- **ModelTransitioner**:  `impackage_dev/mlflowutils/model_transitioner.py` 
- **RegisterBERTModel**:  `impackage_dev/mlflowutils/register_BERT_model.py` 

## MLFlowModel

### File Path

 `impackage_dev/mlflowutils/mlflow_model.py` 

### Description

The  `MLFlowModel`  class provides functionalities to load models from different environments (development, staging, production) using MLFlow.

### Key Methods

1. `__init__(model_name, env, skip_transition, model_tag="", from_env="", to_env="")`
   - Initializes the  `MLFlowModel`  with the model name, environment, and transition settings.

2. `load_model()`
   - Loads the model from the specified environment.
   - **Returns**: The loaded model object.
   - **Raises**: Exception if the model cannot be loaded.

## ModelMetaData

### File Path

 `impackage_dev/mlflowutils/model_metadata.py` 

### Description

The  `ModelMetaData`  class provides functionalities to manage MLFlow model metadata, including retrieving model versions, tags, and managing transitions between stages.

### Key Methods

1. `get_latest_model_version_based_on_env(env)`
   - Retrieves the latest model version for the specified environment.
   - **Returns**: The latest model version.

2. `get_latest_model_tags_based_on_env(env)`
   - Retrieves the latest model tags for the specified environment.
   - **Returns**: Model tags.

3. `set_model_tag(env)`
   - Sets a model tag based on the specified environment.
   - **Returns**: True if successful.

4. `wait_until_model_is_ready(env)`
   - Waits until the model is ready after deployment or stage transition.
   - **Returns**: True when the model is ready.

## ModelManager

### File Path

 `impackage_dev/mlflowutils/model_register_utility.py` 

### Description

The  `ModelManager`  class provides functionalities for registering and loading models in MLFlow, as well as making predictions using the loaded model.

### Key Methods

1. `register_model()`
   - Registers the model with MLFlow.
   - **Raises**: Exception if registration fails.

2. `load_model()`
   - Loads the registered model from MLFlow.
   - **Returns**: The loaded model.

3. `predict(texts)`
   - Predicts the sentiment of the given texts using the loaded model.
   - **Returns**: A list of prediction results.

## ModelVersionManager

### File Path

 `impackage_dev/mlflowutils/model_rollback_utility.py` 

### Description

The  `ModelVersionManager`  class manages the versioning of models in MLFlow, including transitioning models between stages.

### Key Methods

1. `get_latest_version(stage)`
   - Retrieves the latest version of the model in the specified stage.
   - **Returns**: The latest version.

2. `transition_model_version_stage(version, stage)`
   - Transitions the specified model version to the target stage.
   - **Raises**: Exception if the transition fails.

## ModelTransitioner

### File Path

 `impackage_dev/mlflowutils/model_transitioner.py` 

### Description

The  `ModelTransitioner`  class handles the transition of models between different stages (e.g., from staging to production).

### Key Methods

1. `perform_model_transition()`
   - Performs the model transition from one stage to another.
   - **Returns**: True if successful.
   - **Raises**: Exception if the model is not present in the source environment.

## RegisterBERTModel

### File Path

 `impackage_dev/mlflowutils/register_BERT_model.py` 

### Description

The  `RegisterBERTModel`  class defines a BERT model for sentiment analysis and provides methods for loading and predicting with the model.

### Key Methods

1. `load_context(context)`
   - Loads the model and tokenizer from the specified context.
   
2. `predict(context, doc)`
   - Makes predictions on the provided documents.
   - **Returns**: A list of predictions.
