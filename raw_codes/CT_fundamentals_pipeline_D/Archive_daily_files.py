# Databricks notebook source
# MAGIC %md
# MAGIC <br> Move all fundamentals and legacy scores yesterday's parquet files into archive.
# MAGIC <br>__Author__: Ajaya Babu devalla
# MAGIC <br>__ScriptName__: Archive_daily_files
# MAGIC <br>__Input Files__: /dbfs/mnt/access_work/CallTranscriptPipelines/fundamentals/Quant/CT_data_staging/*  &  /dbfs/mnt/access_work/CallTranscriptPipelines/fundamentals/Quant/CT_legacy_data_staging/*
# MAGIC <br>__Output Tables__: None
# MAGIC <br>__Output Files__: Files are moved to archive folder by date fundamentals: /dbfs/mnt/access_work/CallTranscriptPipelines/fundamentals/Quant/CT_data_staging/archive/$(date '+%Y_%m_%d')
# MAGIC <br>                                                           legacy:  /dbfs/mnt/access_work/CallTranscriptPipelines/fundamentals/Quant/CT_legacy_data_staging/archive/$(date '+%Y_%m_%d')
# MAGIC <br>__Contact__: ajaya.devalla@voya.com
# MAGIC <br>__Version__: 1
# MAGIC <br>__Revision History__: 
# MAGIC <br>__Rerun Steps__: This script can be rerun from the top.

# COMMAND ----------

import os

# COMMAND ----------

notebook_path=dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()

# COMMAND ----------

os.environ['path']=notebook_path

# COMMAND ----------

# MAGIC %sh
# MAGIC today="$(date +%a)"
# MAGIC dt=""
# MAGIC if [ "$today" == "Mon" ]; then dt=$(date +"%Y%m%d" -d "-3 day"); else  dt=$(date +"%Y%m%d" -d "-1 day"); fi
# MAGIC if [[ $path == *"@voya.com"* ]];
# MAGIC then
# MAGIC     cd /dbfs/mnt/access_work/CallTranscriptPipelines/fundamentals/Quant/CT_data_staging                  
# MAGIC     for file in CT*df.parquet                
# MAGIC     do                           
# MAGIC         base=${file%.*}
# MAGIC         folder_name=${base: -10}
# MAGIC         mkdir -p /dbfs/mnt/access_work/CallTranscriptPipelines/fundamentals/Quant/CT_data_staging/archive/f_${dt} && cp ${file} $_
# MAGIC     done
# MAGIC elif [[ $path == *"Quant_Stg"* ]];
# MAGIC then
# MAGIC     cd /dbfs/mnt/access_work/CallTranscriptPipelines/fundamentals/Quant_Stg/CT_data_staging                  
# MAGIC     for file in CT*df.parquet                
# MAGIC     do                           
# MAGIC         base=${file%.*}
# MAGIC         folder_name=${base: -10}
# MAGIC         mkdir -p /dbfs/mnt/access_work/CallTranscriptPipelines/fundamentals/Quant_Stg/CT_data_staging/archive/f_${dt} && cp ${file} $_
# MAGIC     done
# MAGIC elif [[ $path == *"Quant_Live"* ]];
# MAGIC then
# MAGIC     cd /dbfs/mnt/access_work/CallTranscriptPipelines/fundamentals/Quant_Live/CT_data_staging                  
# MAGIC     for file in CT*df.parquet                
# MAGIC     do                           
# MAGIC         base=${file%.*}
# MAGIC         folder_name=${base: -10}
# MAGIC         mkdir -p /dbfs/mnt/access_work/CallTranscriptPipelines/fundamentals/Quant_Live/CT_data_staging/archive/f_${dt} && cp ${file} $_
# MAGIC     done
# MAGIC elif [[ $path == *"Quant"* ]];
# MAGIC then
# MAGIC     cd /dbfs/mnt/access_work/CallTranscriptPipelines/fundamentals/Quant/CT_data_staging                  
# MAGIC     for file in CT*df.parquet                
# MAGIC     do                           
# MAGIC         base=${file%.*}
# MAGIC         folder_name=${base: -10}
# MAGIC         mkdir -p /dbfs/mnt/access_work/CallTranscriptPipelines/fundamentals/Quant/CT_data_staging/archive/f_${dt} && cp ${file} $_
# MAGIC     done
# MAGIC fi

# COMMAND ----------

# MAGIC %sh
# MAGIC today="$(date +%a)"
# MAGIC dt=""
# MAGIC if [ "$today" == "Mon" ]; then dt=$(date +"%Y%m%d" -d "-3 day"); else  dt=$(date +"%Y%m%d" -d "-1 day"); fi
# MAGIC if [[ $path == *"@voya.com"* ]];
# MAGIC then
# MAGIC     cd /dbfs/mnt/access_work/CallTranscriptPipelines/fundamentals/Quant/CT_legacy_data_staging                 
# MAGIC     for file in *df.parquet                
# MAGIC     do                           
# MAGIC         base=${file%.*}
# MAGIC         folder_name=${base: -10}
# MAGIC         mkdir -p /dbfs/mnt/access_work/CallTranscriptPipelines/fundamentals/Quant/CT_legacy_data_staging/archive/f_${dt} && cp ${file} $_
# MAGIC     done
# MAGIC elif [[ $path == *"Quant_Stg"* ]];
# MAGIC then
# MAGIC     cd /dbfs/mnt/access_work/CallTranscriptPipelines/fundamentals/Quant_Stg/CT_legacy_data_staging                 
# MAGIC     for file in *df.parquet                
# MAGIC     do                           
# MAGIC         base=${file%.*}
# MAGIC         folder_name=${base: -10}
# MAGIC         mkdir -p /dbfs/mnt/access_work/CallTranscriptPipelines/fundamentals/Quant_Stg/CT_legacy_data_staging/archive/f_${dt} && cp ${file} $_
# MAGIC     done
# MAGIC elif [[ $path == *"Quant_Live"* ]];
# MAGIC then
# MAGIC     cd /dbfs/mnt/access_work/CallTranscriptPipelines/fundamentals/Quant_Live/CT_legacy_data_staging                  
# MAGIC     for file in *df.parquet                
# MAGIC     do                           
# MAGIC         base=${file%.*}
# MAGIC         folder_name=${base: -10}
# MAGIC         mkdir -p /dbfs/mnt/access_work/CallTranscriptPipelines/fundamentals/Quant_Live/CT_legacy_data_staging/archive/f_${dt} && cp ${file} $_
# MAGIC     done
# MAGIC elif [[ $path == *"Quant"* ]];
# MAGIC then
# MAGIC     cd /dbfs/mnt/access_work/CallTranscriptPipelines/fundamentals/Quant/CT_legacy_data_staging                  
# MAGIC     for file in *df.parquet                
# MAGIC     do                           
# MAGIC         base=${file%.*}
# MAGIC         folder_name=${base: -10}
# MAGIC         mkdir -p /dbfs/mnt/access_work/CallTranscriptPipelines/fundamentals/Quant/CT_legacy_data_staging/archive/f_${dt} && cp ${file} $_
# MAGIC     done
# MAGIC fi
# MAGIC
# MAGIC

# COMMAND ----------

# DBTITLE 1,Script is complete
print("Script to archive data for fundamentals and legacy is complete")

# COMMAND ----------

# MAGIC %sh
# MAGIC
# MAGIC ls -ltr /dbfs/mnt/access_work/CallTranscriptPipelines/fundamentals/Quant/CT_data_staging/archive
