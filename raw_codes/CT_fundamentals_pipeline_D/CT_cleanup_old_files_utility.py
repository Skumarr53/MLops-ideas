# Databricks notebook source
# MAGIC %run ./../../data-science-nlp-ml-common-code/impackage/utilities/config_utility

# COMMAND ----------

import pandas as pd
from datetime import timedelta
import datetime
import pytz
from datetime import datetime
import os
import glob
import shutil

# COMMAND ----------

current_date=(datetime.now(pytz.timezone("America/New_York"))).strftime("%Y%m%d")

# COMMAND ----------

current_date

# COMMAND ----------

ct_fundamental_filepath_archive = config.CT_parquet_file_path+'archive/'
ct_legacy_filepath_archive = config.CT_legacy_parquet_file_path+'archive/'

os.environ['ct_fundamental_filepath_archive'] = ct_fundamental_filepath_archive
os.environ['ct_legacy_filepath_archive'] = ct_legacy_filepath_archive

# COMMAND ----------

# MAGIC %sh
# MAGIC echo "This is fundamental archive path :" ${ct_legacy_filepath_archive}
# MAGIC echo "This is legacy archive path :" ${ct_fundamental_filepath_archive}

# COMMAND ----------

# MAGIC %sh
# MAGIC find  ${ct_legacy_filepath_archive} -type d -ctime +30 -exec rm -rf {} \;
# MAGIC find  ${ct_fundamental_filepath_archive} -type d -ctime +30 -exec rm -rf {} \;

# COMMAND ----------

# fundamentals_path = config.CT_parquet_file_path+current_date
# # Check whether the specified path exists or not
# isExist = os.path.exists(fundamentals_path)
# allfiles = [os.path.basename(file_path) for file_path in glob.glob(os.path.join(config.CT_parquet_file_path, '*.parquet'))]
# for file in allfiles:
#    if not isExist:
#    # Create a new directory because it does not exist
#      shutil.move(config.CT_parquet_file_path, os.path.join(os.makedirs(fundamentals_path),file))
#    else:
#      shutil.move(config.CT_parquet_file_path, os.path.join(fundamentals_path,file)) 


# COMMAND ----------

# MAGIC %md
# MAGIC allfiles = glob.glob(os.path.join(source, '*_A_*'))
# MAGIC print("Files to move", allfiles)
# MAGIC  
# MAGIC # iterate on all files to move them to destination folder
# MAGIC for file_path in allfiles:
# MAGIC     dst_path = os.path.join(destination, os.path.basename(file_path))
# MAGIC     shutil.move(file_path, dst_path)

# COMMAND ----------

# MAGIC %md
# MAGIC legacy_path = config.CT_legacy_parquet_file_path+current_date
# MAGIC # Check whether the specified path exists or not
# MAGIC isExist = os.path.exists(legacy_path)
# MAGIC if not isExist:
# MAGIC
# MAGIC    # Create a new directory because it does not exist
# MAGIC    os.makedirs(legacy_path)

# COMMAND ----------

# MAGIC %md
# MAGIC for file in file_list:
# MAGIC   try:
# MAGIC       file=file.format(current_date)
# MAGIC       dbutils.fs.ls(file_path+file)
# MAGIC       dbutils.fs.rm((file_path+file))
# MAGIC       print("{0} file deleted".format(file_path+file))
# MAGIC   except:
# MAGIC       print("{0} is not present".format(file_path+file))
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC for file in legacy_file_list:
# MAGIC   try:
# MAGIC       file=file.format(current_date)
# MAGIC       dbutils.fs.ls(legacy_file_path+file)
# MAGIC       dbutils.fs.rm((legacy_file_path+file))
# MAGIC       print("{0} file deleted".format(legacy_file_path+file))
# MAGIC   except:
# MAGIC       print("{0} is not present".format(legacy_file_path+file))
# MAGIC
