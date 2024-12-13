from centralized_nlp_package.model_utils import ExperimentManager

base_exp_name = "Mass_ConsumerTopic_FineTune_DeBERTa_v3"
data_src = "CallTranscript"
train_file_path = "/Workspace/Users/santhosh.kumar3@voya.com/MLFlow_and_NLI_finetune/data/{data_version}"
test_file_path = "/Workspace/Users/santhosh.kumar3@voya.com/MLFlow_and_NLI_finetune/data/test.csv"
output_dir = "/dbfs/mnt/access_work/UC25/Topic Modeling/NLI Models/Fine-tune NLI models/trained_RD_deberta-v3-base-zeroshot-v2_Santhosh_test"




model_versions = ["/dbfs/mnt/access_work/UC25/Libraries/HuggingFace/deberta-v3-large-zeroshot-v2"]
dataset_versions = ["main.csv","train.csv"]
hyperparameters = [
        {"n_epochs": 5, "learning_rate": 2e-5, "weight_decay": 0.01, "train_batch_size": 16, "eval_batch_size": 16},
        {"n_epochs": 8, "learning_rate": 3e-5, "weight_decay": 0.02, "train_batch_size": 24, "eval_batch_size": 24}
    ]



experiment_manager = ExperimentManager(
    base_name=base_exp_name,
    data_src=data_src,
    dataset_versions=dataset_versions,
    hyperparameters=hyperparameters,
    base_model_versions=model_versions,
    output_dir=output_dir,
    train_file=train_file_path,
    validation_file=test_file_path,
    evalute_pretrained_model = True, # default = True
    
)


experiment_manager.run_experiments()



from centralized_nlp_package.model_utils import ModelSelector



Model_Select_object = ModelSelector("/Users/santhosh.kumar3@voya.com/Mass_ConsumerTopic_FineTune_DeBERTa_v3_CallTranscript_20241205", 'accuracy')



Model_Select_object.list_available_models()



best_run = Model_Select_object.get_best_model()
print(best_run)


models_by_basemodel_version = Model_Select_object.get_best_models_by_tag('base_model_name') # base model  



from centralized_nlp_package.model_utils import ModelTransition



use_case = "Topic_Modeling"
task_name = "Consumer_Topic"
model_name = f"{use_case}_{task_name}"

model_trans_obj = ModelTransition(model_name)

for run in models_by_basemodel_version:
  print(f"runs:/{run.info.run_id}/model")
  model_trans_obj._register_model(run)
print("Models registered successfully")
  



import mlflow
def _fetch_model_version(model_name, version: int):
    client = mlflow.tracking.MlflowClient()
    try:
        model_version = client.get_model_version(model_name, str(version))
        return model_version
    except mlflow.exceptions.RestException as e:
        print(f"Error fetching model version {version}: {e}")
        return None
    
def _update_stage(model_name, model_version, stage: str):
    client = mlflow.tracking.MlflowClient()
    client.transition_model_version_stage(
        name=model_name,
        version=model_version.version,
        stage=stage,
        archive_existing_versions=True
    )
    print(f"Model version {model_version.version} updated to stage {stage}")



print(model_name)
selcted_version = _fetch_model_version(model_name, 1)
_update_stage(model_name, selcted_version, "Production")



production_model = mlflow.pyfunc.load_model(
    model_uri=f"models:/{model_name}/Production"
)












