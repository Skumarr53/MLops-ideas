from hydra import initialize, compose

initialize(config_path="config", job_name="app")
cfg = compose(config_name="config.yaml")

# Example usage in code
MODEL_FOLDER_PATH = cfg.paths.model_folder