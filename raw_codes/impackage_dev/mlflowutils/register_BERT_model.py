# Databricks notebook source
!pip install optimum[onnxruntime]==1.18.0
!pip install onnxruntime-gpu==1.17.1
!pip install transformers==4.39.2
!pip install xformers==0.0.25.post1

# COMMAND ----------

from mlflow.pyfunc import PythonModel, PythonModelContext
from typing import Dict
import numpy as np
import os, pathlib

# COMMAND ----------

class RegisterBERTModel(PythonModel):

    def load_context(self, context: PythonModelContext):
        from transformers import TextClassificationPipeline,BertTokenizer, BertForSequenceClassification,BertConfig
        from optimum.bettertransformer import BetterTransformer
        config_file = os.path.dirname(context.artifacts["config"])
        self.config = BertConfig.from_pretrained(config_file)
        self.tokenizer = BertTokenizer.from_pretrained(config_file)
        self.better_model = BetterTransformer.transform(BertForSequenceClassification.from_pretrained(config_file, config=self.config))
        self.pipe = TextClassificationPipeline(model = self.better_model, tokenizer = self.tokenizer, device = 0)
        self.label_map={'LABEL_0': 1, 'LABEL_1': -1, 'LABEL_2': 0}
         
        _ = self.model.eval()

    def predict(self, context: PythonModelContext, doc):
        if self.label_map:
            return [(self.label_map[dict['label']], dict['score']) for dict in self.pipe(doc, batch_size = 16, truncation = True, max_length = 512)]
        else:
            return [(dict['label'], dict['score']) for dict in self.pipe(doc, batch_size = 16, truncation = True, max_length = 512)]
      
