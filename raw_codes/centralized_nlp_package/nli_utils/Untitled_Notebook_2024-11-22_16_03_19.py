# Databricks notebook source
# MAGIC %pip install evaluate

# COMMAND ----------

import evaluate


# COMMAND ----------

evaluate.load('./accuracy.py')

# COMMAND ----------

