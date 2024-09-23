# Databricks notebook source
# MAGIC %run "./test_config_utility"

# COMMAND ----------

# MAGIC %run "./test_logging"

# COMMAND ----------

# MAGIC %run "./test_text_preprocessor"

# COMMAND ----------

# MAGIC %run "./test_topicx_preprocessor"

# COMMAND ----------

# MAGIC %run "./test_finbert_preprocessor.py"

# COMMAND ----------

# MAGIC %run "./test_dictionary_model_preprocessor"

# COMMAND ----------

# MAGIC %run "./test_dataframe_utility"

# COMMAND ----------

from runtime.runner import NutterFixtureParallelRunner


# COMMAND ----------

parallel_runner = NutterFixtureParallelRunner(num_of_workers=6)
parallel_runner.add_test_fixture(ConfigFixture())
parallel_runner.add_test_fixture(LoggingFixture())
parallel_runner.add_test_fixture(TextPreprocessorFixture())
parallel_runner.add_test_fixture(TopicXPreprocessorFixture())
parallel_runner.add_test_fixture(FINBERTPreprocessorFixture())
parallel_runner.add_test_fixture(DictionaryModelPreprocessorFixture())
parallel_runner.add_test_fixture(DFUtilityFixture())


# COMMAND ----------

result = parallel_runner.execute()
print(result.to_string())
