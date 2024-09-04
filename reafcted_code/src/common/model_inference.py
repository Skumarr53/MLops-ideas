# common/model_inference.py

from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import logging
from joblib import Parallel, delayed

class InferenceModel:
    def __init__(self, model_folder_name: str, model_folder_path: str, device: int):
        self.model_folder_name = model_folder_name
        self.model_folder_path = model_folder_path
        self.device = device
        self.pipeline = self._load_model()

    def _load_model(self):
        """Load the model and tokenizer from the specified folder."""
        try:
            tokenizer = AutoTokenizer.from_pretrained(self.model_folder_path + self.model_folder_name)
            model = AutoModelForSequenceClassification.from_pretrained(self.model_folder_path + self.model_folder_name)
            logging.info(f"Model {self.model_folder_name} loaded successfully.")
            return pipeline(task="text-classification", model=model, tokenizer=tokenizer, device=self.device)
        except Exception as e:
            logging.error(f"Failed to load model: {e}")
            raise

    def run(self, df):
        """Run inference on the provided DataFrame."""
        try:
            # Assuming TEXT1 and TEXT2 columns are present in the DataFrame for inference
            results = Parallel(n_jobs=-1)(
                delayed(self.pipeline)(f"{t1}</s></s>{t2}", padding=True, top_k=None, batch_size=16, truncation=True, max_length=512)
                for t1, t2 in zip(df['TEXT1_MD'], df['TEXT2_MD'])
            )
            logging.info("Inference run successfully in parallel.")
            return results
        except Exception as e:
            logging.error(f"Error running inference: {e}")
            raise
