- [x] Update the existing word_tokenize function to exclude words specified by a parameter. below functions tokenize text
   - *centralized_nlp_package\preprocessing\text_preprocessing.py\tokenize_and_lemmatize_text* (mainly focusing on customizing token.pos_ not in ['NUM', 'SYM'] and token.ent_type_ not in ['PERSON', 'GPE', 'LOC', 'ORG'])
   - *centralized_nlp_package\text_processing\text_utils.py\word_tokenizer* (This is for psycholinguistic code)
- [x] Filter a DataFrame by eliminating rows where the specified column includes any of the given keywords. 
       *centralized_nlp_package/utils/helpers.py/df_remove_rows_with_keywords.*
- [x] Add a function to apply pandas row transofmration group of functions, 
- [ ] add a parameter as list as to what columns need to be computedfor topic report generation


### updated
- [ ] Implement functionality to evaluate fine-tuned models and record run parameters and metrics for comparison.
- [ ] Update the existing NLI module to operate without evaluation if a validation dataset is not provided.
- [ ] Adjust the MLflow module to log the pretrained model and its run metrics to establish a benchmark.
- [ ] testing NLI finetuning on cluster runtime 13.3
- [ ] snowflake configartion fix 


https://github.com/Skumarr53/Topic-modelling-Loc




2024-12-03 13:20:31 | ERROR   | experiment_manager.py:103 | Failed to log model: Cannot pickle a prepared model with automatic mixed precision, please unwrap the model with `Accelerator.unwrap_model(model)` before pickling it.



trained_model is of type AutoModelForSequenceClassification


code:
unwrapped_model = self.accelerator.unwrap_model(trained_model)
if hasattr(unwrapped_model, 'half'):
    unwrapped_model = unwrapped_model.float()

try:
    mlflow.pytorch.log_model(unwrapped_model, "model")
    logger.info(f"Model logged successfully")
except Exception as e:
    logger.error(f"Failed to log model: {e}")