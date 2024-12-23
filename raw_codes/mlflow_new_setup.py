import mlflow
from mlflow.tracking import MlflowClient
import subprocess
import pandas as pd
import os
import logging
from typing import Dict, List

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration Dataclass
class Config:
    """
    Configuration settings for MLflow and model management.
    """
    def __init__(self,
                 experiment_name: str,
                 registered_model_name: str,
                 training_script_path: str,
                 development_data_paths: List[str],
                 staging_validation_data_path: str,
                 model_name_or_path: str,
                 output_dir: str,
                 num_train_epochs: int = 3,
                 learning_rate: float = 2e-5,
                 weight_decay: float = 0.01,
                 batch_size: int = 16):
        self.experiment_name = experiment_name
        self.registered_model_name = registered_model_name
        self.training_script_path = training_script_path
        self.development_data_paths = development_data_paths  # List of paths for different development runs
        self.staging_validation_data_path = staging_validation_data_path
        self.model_name_or_path = model_name_or_path
        self.output_dir = output_dir
        self.num_train_epochs = num_train_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size

# Function to initialize MLflow experiment
def initialize_experiment(config: Config):
    """
    Initialize MLflow experiment.
    """
    mlflow.set_experiment(config.experiment_name)
    logger.info(f"MLflow experiment set to '{config.experiment_name}'.")

# Function to run training script and log to MLflow
def train_model(config: Config, train_data_path: str, run_name: str):
    """
    Train the model using the provided training script and log the run to MLflow.
    """
    logger.info(f"Starting training run: {run_name} with training data at {train_data_path}")
    with mlflow.start_run(run_name=run_name):
        # Log training parameters
        mlflow.log_param("model_name_or_path", config.model_name_or_path)
        mlflow.log_param("num_train_epochs", config.num_train_epochs)
        mlflow.log_param("learning_rate", config.learning_rate)
        mlflow.log_param("weight_decay", config.weight_decay)
        mlflow.log_param("batch_size", config.batch_size)
        mlflow.log_param("train_data_path", train_data_path)
        
        # Execute the training script
        try:
            subprocess.run([
                "python", config.training_script_path,
                "--model_name_or_path", config.model_name_or_path,
                "--output_dir", config.output_dir,
                "--train_file", train_data_path,
                "--validation_file", config.staging_validation_data_path,
                "--do_train",
                "--do_eval",
                "--num_train_epochs", str(config.num_train_epochs),
                "--fp16",
                "--report_to", "none",
                "--learning_rate", str(config.learning_rate),
                "--weight_decay", str(config.weight_decay),
                "--per_device_train_batch_size", str(config.batch_size),
                "--per_device_eval_batch_size", str(config.batch_size)
            ], check=True)
            logger.info(f"Training script executed successfully for run: {run_name}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Training script failed for run: {run_name} with error: {e}")
            raise
        
        # Assume that the training script logs metrics to MLflow
        # Alternatively, you can parse and log metrics here if needed

# Function to select the best run based on accuracy
def select_best_run(config: Config) -> str:
    """
    Select the best run from the development experiment based on the highest accuracy.
    Returns the run ID of the best run.
    """
    client = MlflowClient()
    experiment = client.get_experiment_by_name(config.experiment_name)
    runs = client.search_runs(experiment_ids=[experiment.experiment_id],
                              order_by=["metrics.accuracy DESC"],
                              max_results=1)
    if not runs:
        logger.warning("No runs found in the experiment.")
        return None
    best_run = runs[0]
    best_run_id = best_run.info.run_id
    best_accuracy = best_run.data.metrics.get("accuracy", 0)
    logger.info(f"Best run ID: {best_run_id} with accuracy: {best_accuracy}")
    return best_run_id

# Function to transition model stage based on metric threshold
def transition_model(config: Config, run_id: str, target_stage: str):
    """
    Register the model from a specific run and transition it to the target stage if accuracy > 0.9.
    """
    client = MlflowClient()
    run = client.get_run(run_id)
    accuracy = run.data.metrics.get("accuracy", 0)
    logger.info(f"Run ID: {run_id} has accuracy: {accuracy}")
    
    if accuracy > 0.9:
        # Register the model
        model_uri = f"runs:/{run_id}/model"
        logger.info(f"Registering model from run ID: {run_id} to '{config.registered_model_name}'")
        registered_model = mlflow.register_model(model_uri, config.registered_model_name)
        
        # Wait for the model to be registered
        client.wait_for_model_version_creation(config.registered_model_name, registered_model.version)
        
        # Transition the model to the target stage
        client.transition_model_version_stage(
            name=config.registered_model_name,
            version=registered_model.version,
            stage=target_stage,
            archive_existing_versions=True
        )
        logger.info(f"Model version {registered_model.version} transitioned to stage: {target_stage}")
    else:
        logger.info(f"Model accuracy {accuracy} does not exceed threshold. Transition to {target_stage} skipped.")

# Function to validate model in staging
def validate_model_in_staging(config: Config, model_version: str) -> bool:
    """
    Validate the model in staging by loading it and checking its performance on validation data.
    Returns True if validation metric > 0.9, else False.
    """
    model_uri = f"models:/{config.registered_model_name}/{model_version}"
    logger.info(f"Loading model version {model_version} from staging for validation.")
    model = mlflow.pyfunc.load_model(model_uri)
    
    # Load validation data
    validation_data = pd.read_csv(config.staging_validation_data_path)
    # Assuming the validation data has 'text' and 'label' columns
    texts = validation_data['text'].tolist()
    true_labels = validation_data['label'].tolist()
    
    # Perform predictions
    predictions = model.predict(texts)
    
    # Calculate accuracy
    correct = sum([pred == true for pred, true in zip(predictions, true_labels)])
    accuracy = correct / len(true_labels)
    logger.info(f"Validation accuracy for model version {model_version}: {accuracy}")
    
    return accuracy > 0.9

# Main orchestration function
def manage_mlflow_workflow(config: Config):
    """
    Manage the MLflow workflow from development to production.
    """
    initialize_experiment(config)
    
    # Development Phase: Train multiple runs
    for idx, train_data_path in enumerate(config.development_data_paths, start=1):
        run_name = f"development_run_{idx}"
        train_model(config, train_data_path, run_name)
    
    # Select the best run from development
    best_run_id = select_best_run(config)
    if not best_run_id:
        logger.error("No suitable runs found in development phase.")
        return
    
    # Transition the best run to Staging if accuracy > 0.9
    transition_model(config, best_run_id, target_stage="Staging")
    
    # Get the latest version in Staging
    client = MlflowClient()
    staging_versions = client.get_latest_versions(config.registered_model_name, stages=["Staging"])
    if not staging_versions:
        logger.error("No models found in Staging stage.")
        return
    staging_model_version = staging_versions[0].version
    
    # Validate the model in Staging
    if validate_model_in_staging(config, staging_model_version):
        # Transition to Production if validation passes
        client.transition_model_version_stage(
            name=config.registered_model_name,
            version=staging_model_version,
            stage="Production",
            archive_existing_versions=True
        )
        logger.info(f"Model version {staging_model_version} transitioned to Production.")
    else:
        logger.warning(f"Model version {staging_model_version} failed validation. Not transitioning to Production.")

# COMMAND ----------

### **Usage Example**

Below is an example of how to utilize the `manage_mlflow_workflow` function with your DeBERTa-v3-large-zeroshot-v2 NLI model. This example assumes that you have multiple training datasets for development and a fixed validation dataset for staging.

```python
# Import necessary libraries
import os

# Define the configuration
config = Config(
    experiment_name="/Users/your_username/NLI_FineTuning_Experiment",
    registered_model_name="DeBERTa_v3_NLI_Model",
    training_script_path="/dbfs/mnt/access_work/UC25/Topic Modeling/NLI Models/train_nli_model.py",  # Replace with your actual training script path
    development_data_paths=[
        "/dbfs/mnt/access_work/UC25/Topic Modeling/NLI Models/NLI_role_training_data/Role_FT_train_v1.csv",
        "/dbfs/mnt/access_work/UC25/Topic Modeling/NLI Models/NLI_role_training_data/Role_FT_train_v2.csv",
        "/dbfs/mnt/access_work/UC25/Topic Modeling/NLI Models/NLI_role_training_data/Role_FT_train_v3.csv"
    ],
    staging_validation_data_path="/dbfs/mnt/access_work/UC25/Topic Modeling/NLI Models/NLI_role_training_data/Role_FT_validation.csv",  # Fixed path for staging validation
    model_name_or_path="/dbfs/mnt/access_work/UC25/Libraries/HuggingFace/deberta-v3-large-zeroshot-v2",
    output_dir="/dbfs/mnt/access_work/UC25/Topic Modeling/NLI Models/Fine-tune_NLI_models/trained_RD_deberta-v3-large-zeroshot-v2",
    num_train_epochs=3,
    learning_rate=2e-5,
    weight_decay=0.01,
    batch_size=16
)

# Execute the MLflow workflow
manage_mlflow_workflow(config)


def list_available_models(experiment_name, metric = "accuracy"):
    experiment = mlflow.get_experiment_by_name(experiment_name)
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=[f"metrics.{metric} DESC"]
    )
    models = []
    for _, row in runs.iterrows():
        models.append({
            "run_id": row["run_id"],
            "run_name": row["tags.mlflow.runName"],
            "metrics": {key.replace("metrics.", ""): value for key, value in row.items() if key.startswith("metrics.")}
        })
    logger.info(f"Total models found: {len(models)}")
    return models

run_names = [run['run_name'] for run in runs_list]

match_df_negate = match_df_v0_ref[~match_df_v0_ref['Negation'].isna()][['Subtopic', 'Negation']]#.apply(lambda x: ast.literal_eval(x['Negation']), axis=1)#.explode(column = 'Negation')
match_df_negate = df_apply_transformations(match_df_negate, [('Negation', 'Negation', ast.literal_eval)])
match_df_negate = match_df_negate.explode(column = 'Negation')
match_df_negate['negate'] = True
match_df_negate = match_df_negate.rename(columns = {'Subtopic': 'label', 'Negation': 'match'})
match_df_ref['negate'] = False
match_df_ref = match_df_ref.rename(columns={'Subtopic':'label', 'Refined Keywords':'match'})
match_df_ref = pd.concat([match_df_ref, match_df_negate])