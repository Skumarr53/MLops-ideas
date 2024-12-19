# Databricks notebook source
# MAGIC %md
# MAGIC ## Instruction
# MAGIC CONTEXT DESCRIPTION : Calculate and visualize the coverage rate and relevance score for each topic.  
# MAGIC - Time series line plot  
# MAGIC This part shows that line plot of the 120-day rolling avg coverage rate and relevance score during specific time range. Within 120 days rolling back window, if one company has several transcripts, we only focus on the latest transcript for each company.
# MAGIC
# MAGIC Reads from: QUANT.YUJING_MASS_FT_NLI_DEMAND_DEV_1
# MAGIC
# MAGIC Writes to: QUANT.YUJING_MASS_FT_DEMAND_TS_DEV_1
# MAGIC
# MAGIC Recommended cluster: Any Standard D series cluster with 32gb RAM and 8 cores. (14.3 LTS runtime)
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load packages

# COMMAND ----------

# MAGIC %run ./../../../data-science-nlp-ml-common-code/impackage/utilities/config_utility

# COMMAND ----------

# MAGIC %run ./../../../data-science-nlp-ml-common-code/impackage/database/snowflake_dbutility

# COMMAND ----------

new_sf = SnowFlakeDBUtility(config.schema, config.eds_db_prod)

# COMMAND ----------

import pandas as pd
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta
from ast import literal_eval
from collections import Counter
from pyspark.sql.types import *
import warnings
warnings.filterwarnings("ignore")


# COMMAND ----------

# MAGIC %md
# MAGIC ## Connect with SQL database
# MAGIC It has monthly company market cap data

# COMMAND ----------

# MAGIC %run "./../../package_loader/Cloud_DB_module_Azure_SQL_dev_2_Yujing_git.py"

# COMMAND ----------

myDBFS_sql = DBFShelper_sql()
myDBFS_sql.get_DBFSdir_content(myDBFS_sql.iniPath)

# COMMAND ----------

azSQL_LinkUp = pd.read_pickle(r'/dbfs/' + myDBFS_sql.iniPath + 'my_azSQL_LinkUp.pkl')
azSQL_LinkUp.databaseName = 'QNT'
remote_table = azSQL_LinkUp.read_from_azure_SQL("qnt.p_coe.earnings_calls_mapping_table_mcap")
market_cap  = remote_table.toPandas()

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Read Data from Snowflake

# COMMAND ----------

# Read start & end ranges. Note that the range does is NOT inclusive of end month; i.e the range ends at the beginning of the end month
minDateNewQuery = (pd.to_datetime(dbutils.widgets.get("Start Date"))).strftime('%Y-%m-%d')
maxDateNewQuery = (pd.to_datetime(dbutils.widgets.get("End Date"))).strftime('%Y-%m-%d')

mind = "'" + minDateNewQuery + "'"
maxd = "'" + maxDateNewQuery + "'"

print('The next query spans ' + mind + ' to ' + maxd)

# COMMAND ----------

# Query all parsed transcripts parsed after the last known parsed date.
tsQuery= ("SELECT * "
    "FROM  EDS_PROD.QUANT.YUJING_MASS_FT_NLI_DEMAND_DEV_1 "
          
   "WHERE DATE >= " + mind  + " AND DATE <= " + maxd  + " ;")

resultspkdf = new_sf.read_from_snowflake(tsQuery)

currdf = resultspkdf.toPandas()

if len(currdf)>0:
    print('The data spans from ' + str(currdf['DATE'].min()) + ' to ' + str(currdf['DATE'].max()) + 'and has ' + str(currdf.shape[0]) + ' rows and ' + str(currdf.shape[1]) + ' columns.')
else:
    print('No new transcripts to parse.')
    dbutils.notebook.exit(1)
    os._exit(1)

# COMMAND ----------

currdf

# COMMAND ----------

del resultspkdf

# COMMAND ----------

# Keep the earliest version of transcript from the same date.
currdf = currdf.sort_values(['ENTITY_ID', 'DATE','VERSION_ID']).drop_duplicates(['ENTITY_ID', 'DATE'], keep = 'first' )

# COMMAND ----------

currdf['DATE'] = currdf['DATE'].dt.date

# COMMAND ----------

for col in currdf.filter(like = 'COUNT'):
  currdf[col[:-13] + 'COVERAGE' + col[-8:]] = currdf[col].apply(lambda x: 1 if x>=1 else 0)

# COMMAND ----------

currdf

# COMMAND ----------

# MAGIC %md
# MAGIC ## Preprocessing Market Cap data

# COMMAND ----------

market_cap = market_cap.sort_values(by=['factset_entity_id','date']).drop_duplicates(['factset_entity_id', 'YearMonth'], keep='last')

# COMMAND ----------

market_cap['YEAR_MONTH'] = pd.to_datetime(market_cap['date'], format='%Y-%m-%d').apply(lambda x: str(x)[:7])
market_cap['MCAP_RANK'] = market_cap.groupby('YEAR_MONTH')['MCAP'].rank(ascending=False, method='first')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Helper functions

# COMMAND ----------

## Functions to rank companies by different market cap groups
def classify_rank(rank):
  if rank <= 100:
    return '1-Mega firms (top 100)'
  elif rank <= 1000:
    return 'Large firms (top 100 - 1000)'
  else:
    return 'Small firms (1000+)'
  
def classify_rank_C2(rank):
  if rank <= 1000:
    return 'top 1-1000'
  else:
    return 'top 1001-3000'
  
def classify_rank_C3(rank):
  if rank <= 500:
    return 'top 1-500'
  elif rank <= 900:
    return 'top 501-900'
  elif rank <= 1500:
    return 'top 901-1500'
  else:
    return 'top 1501-3000'
  
def create_plot_df(df_raw, start_date, end_date):
  '''
  Create three data frames that contains the relevance scores for each theme during different time period.
  
  Input:
    df_raw: the backtest dataframe that contains the sentiment score and topics details for each transcript
    start_date: the start date of the period
    end_date: the end date of the period

  Output:
    df_mean: a dataframe that contains sentiment score for each theme/section during different time period

  '''
  
  df = df_raw.copy()
  # Select the data within time range(including the 120 days look-back period before the start date)
  start_date_120days = str(datetime.strptime(start_date,'%Y-%m-%d').date() - relativedelta(days =+ 120))
  startdate = datetime.strptime(start_date_120days, "%Y-%m-%d").date()
  enddate = datetime.strptime(end_date, "%Y-%m-%d").date() 
  time_range = (df['DATE'] > startdate) & (df['DATE'] <= enddate)
  df = df.loc[time_range]

  pre_columns = ['CALL_ID','ENTITY_ID','DATE'] 

  # Select Relevance columns.
  col_type = 'REL'
  df = df[pre_columns + list(df.filter(like = 'REL_FILT_').columns)]
  
  df['DATE'] = df['DATE'].apply(lambda x: str(x))
  
  # Calculate the average of MD & QA
  for col in set([i[:-2] for i in df.filter(like = col_type).columns]):
    df[col + 'AVERAGE'] = df.filter(regex = col, axis=1).mean(axis=1)

  df_mean = pd.DataFrame()
  # Calculate the 120-day rolling average of latest relevance score for each company                            
  unique_date = pd.period_range(start = start_date, end = end_date, freq='D').map(str)  
  for d in unique_date:
      tem_range = pd.period_range(end=d, freq='D', periods=120)
      df_rolling = df.loc[df['DATE'].isin(tem_range.map(str))].copy()
      df_rolling = df_rolling.drop_duplicates(subset='ENTITY_ID', keep='last')
      series = pd.Series([d],index=['DATE_ROLLING']).append(df_rolling.mean(axis = 0).drop(labels=['CALL_ID']))
      df_series = pd.DataFrame(series).T
      df_mean = df_mean.append(df_series)
  df_mean = df_mean.dropna(axis=1,how='all').set_index('DATE_ROLLING')
  return df_mean            

def create_plot_df_coverage_rate(df_raw, start_date, end_date):
  '''
  Create three data frames that contains the coverage rate for each theme/section during different time period.
  
  Input:
    df_raw: the backtest dataframe that contains the sentiment score and topics details for each transcript
    start_date: the start date of the period
    end_date: the end date of the period

  Output:
    df_mean: a dataframe that contains sentiment score for each theme/section during different time period

  '''
  
  df = df_raw.copy()
  # Select the data within time range(including the 120 days look-back period before the start date)
  start_date_120days = str(datetime.strptime(start_date,'%Y-%m-%d').date() - relativedelta(days =+ 120))
  startdate = datetime.strptime(start_date_120days, "%Y-%m-%d").date()
  enddate = datetime.strptime(end_date, "%Y-%m-%d").date() 
  time_range = (df['DATE'] > startdate) & (df['DATE'] <= enddate)
  df = df.loc[time_range]
  
  pre_columns = ['CALL_ID','ENTITY_ID','DATE'] 

  # Select coverage rate columns.
  col_type = 'COVERAGE'
  df = df[pre_columns +  list(df.filter(like = '_' + col_type + '_FILT_').columns)]
  
  df['DATE'] = df['DATE'].apply(lambda x: str(x))
  
  # Calculate the average of MD & QA
  for col in set([i[:-2] for i in df.filter(like = col_type).columns]):
     df[col + 'AVERAGE'] = df.filter(regex = col, axis=1).mean(axis=1)
     df[col + 'AVERAGE'] = df[col + 'AVERAGE'].apply(lambda x: 1 if x > 0  else 0)

  df_mean = pd.DataFrame()

  # Calculate the 120-day rolling average of latest coverage rate for each company                            
  unique_date = pd.period_range(start = start_date, end = end_date, freq='D').map(str)  
  for d in unique_date:
      tem_range = pd.period_range(end=d, freq='D', periods=120)
      df_rolling = df.loc[df['DATE'].isin(tem_range.map(str))].copy()
      df_rolling = df_rolling.drop_duplicates(subset='ENTITY_ID', keep='last')
      df_rolling = df_rolling.drop(columns=df_rolling.filter(like='REL').columns)
      df_rolling['COMPANY_COUNT'] = 1
      series = pd.Series([d],index=['DATE_ROLLING']).append(df_rolling.sum(axis = 0).drop(labels=['CALL_ID','ENTITY_ID','DATE']))
      df_series = pd.DataFrame(series).T
      df_mean = df_mean.append(df_series)
  df_mean = df_mean.dropna(axis=1,how='all').set_index('DATE_ROLLING')
  for col in df_mean.filter(like = 'COVERAGE').columns:
    if '_AVERAGE' in col:
      df_mean[col[:-21] + 'COVERRATE' + col[-13:]] = df_mean.apply(lambda x: x[col] / x['COMPANY_COUNT'] if x['COMPANY_COUNT'] != 0 else None, axis = 1)
    else:
      df_mean[col[:-16] + 'COVERRATE' + col[-8:]] = df_mean.apply(lambda x: x[col] / x['COMPANY_COUNT'] if x['COMPANY_COUNT'] != 0 else None, axis = 1)
  df_mean = df_mean.drop(columns = [col for col in df_mean.columns if '_COVERAGE_FILT_' in col])
  return df_mean      

# Auxiliar functions
def equivalent_type(string, f):
    print(string, f)
    
    if f == 'datetime64[ns]': return TimestampType()
    elif string == "COMPANY_COUNT": return FloatType()
    elif f == 'int64': return LongType()
    elif f == 'int32': return IntegerType()
    elif f == 'float64': return FloatType()
    elif '_REL_FILT_' in string: return FloatType()
    elif '_COVERRATE_FILT_' in string: return FloatType()
    else: return StringType()

def define_structure(string, format_type):
    typo = equivalent_type(string, format_type)
    print(typo)
    return StructField(string, typo)

# Given pandas dataframe, it will return a spark's dataframe.
def pandas_to_spark(pandas_df):
    columns = list(pandas_df.columns)
    types = list(pandas_df.dtypes)
    struct_list = []
    for column, typo in zip(columns, types): 
      struct_list.append(define_structure(column, typo))
    p_schema = StructType(struct_list)
    return sqlContext.createDataFrame(pandas_df, p_schema)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Time Series Line Plot

# COMMAND ----------

# MAGIC %md
# MAGIC #### Market Cap Group 1

# COMMAND ----------

market_cap['MCAP_GROUP'] = market_cap['MCAP_RANK'].apply(lambda x: classify_rank(x))
currdf['YEAR_MONTH'] = currdf['DATE'].apply(lambda x: str(x)[:7])
currdf_merge = pd.merge(market_cap[['YEAR_MONTH', 'MCAP_GROUP', 'factset_entity_id','MCAP', 'biz_group']], currdf,  how='left', left_on=['factset_entity_id','YEAR_MONTH'], right_on = ['ENTITY_ID','YEAR_MONTH'])
currdf_R3K = currdf_merge[~currdf_merge.CALL_ID.isna()]
currdf_R3K_mega = currdf_R3K[currdf_R3K['MCAP_GROUP'] == '1-Mega firms (top 100)' ]
currdf_R3K_large = currdf_R3K[currdf_R3K['MCAP_GROUP'] == 'Large firms (top 100 - 1000)' ]
currdf_R3K_small = currdf_R3K[currdf_R3K['MCAP_GROUP'] == 'Small firms (1000+)' ]

# COMMAND ----------

R3K_dfs = [currdf_R3K, currdf_R3K_mega, currdf_R3K_large, currdf_R3K_small ]
R3K_categories = ['top 1-3000', 'top 1-100', 'top 101-1000', 'top 1001-3000']

df_rel_cover_C1 = pd.DataFrame([])
for i, df in enumerate(R3K_dfs):
  df_cover = create_plot_df_coverage_rate(df, 
                start_date = minDateNewQuery, 
                end_date = maxDateNewQuery)
  df_rel = create_plot_df(df, 
                start_date = minDateNewQuery, 
                end_date = maxDateNewQuery)
  df_cover.reset_index(inplace=True)
  df_rel.reset_index(inplace=True)
  df_rel_cover = pd.merge(df_cover, df_rel, left_on='DATE_ROLLING', right_on='DATE_ROLLING')
  df_rel_cover['CATEGORY'] = R3K_categories[i]
  df_rel_cover_C1 = pd.concat([df_rel_cover_C1 , df_rel_cover])

# COMMAND ----------

# MAGIC %md
# MAGIC #### Market Cap Group 2

# COMMAND ----------

market_cap['MCAP_GROUP'] = market_cap['MCAP_RANK'].apply(lambda x: classify_rank_C2(x))
currdf_merge = pd.merge(market_cap[['YEAR_MONTH', 'MCAP_GROUP', 'factset_entity_id','MCAP', 'biz_group']], currdf,  how='left', left_on=['factset_entity_id','YEAR_MONTH'], right_on = ['ENTITY_ID','YEAR_MONTH'])
currdf_R3K = currdf_merge[~currdf_merge.CALL_ID.isna()]
print(currdf_R3K['MCAP_GROUP'].unique())
currdf_R3K_C2_1 = currdf_R3K[currdf_R3K['MCAP_GROUP'] == 'top 1-1000' ]

# COMMAND ----------

R3K_dfs = [currdf_R3K_C2_1]
R3K_categories = ['top 1-1000']

df_rel_cover_C2 = pd.DataFrame([])
for i, df in enumerate(R3K_dfs):
  df_cover = create_plot_df_coverage_rate(df, 
                start_date = minDateNewQuery, 
                end_date = maxDateNewQuery)
  df_rel = create_plot_df(df, 
                start_date = minDateNewQuery, 
                end_date = maxDateNewQuery)
  df_cover.reset_index(inplace=True)
  df_rel.reset_index(inplace=True)
  df_rel_cover = pd.merge(df_cover, df_rel, left_on='DATE_ROLLING', right_on='DATE_ROLLING')
  df_rel_cover['CATEGORY'] = R3K_categories[i]
  df_rel_cover_C2 = pd.concat([df_rel_cover_C2 , df_rel_cover])

# COMMAND ----------

# MAGIC %md
# MAGIC #### Market Cap Group 3

# COMMAND ----------

market_cap['MCAP_GROUP'] = market_cap['MCAP_RANK'].apply(lambda x: classify_rank_C3(x))
currdf_merge = pd.merge(market_cap[['YEAR_MONTH', 'MCAP_GROUP', 'factset_entity_id','MCAP', 'biz_group']], currdf,  how='left', left_on=['factset_entity_id','YEAR_MONTH'], right_on = ['ENTITY_ID','YEAR_MONTH'])
currdf_R3K = currdf_merge[~currdf_merge.CALL_ID.isna()]
print(currdf_R3K['MCAP_GROUP'].unique())
currdf_R3K_C3_1 = currdf_R3K[currdf_R3K['MCAP_GROUP'] == 'top 1-500' ]
currdf_R3K_C3_2 = currdf_R3K[currdf_R3K['MCAP_GROUP'] == 'top 501-900' ]
currdf_R3K_C3_3 = currdf_R3K[currdf_R3K['MCAP_GROUP'] == 'top 901-1500' ]
currdf_R3K_C3 = currdf_R3K[(currdf_R3K['MCAP_GROUP'] == 'top 1-500')| (currdf_R3K['MCAP_GROUP'] == 'top 501-900')| (currdf_R3K['MCAP_GROUP'] == 'top 901-1500') ]

# COMMAND ----------

R3K_dfs = [currdf_R3K_C3, currdf_R3K_C3_1, currdf_R3K_C3_2, currdf_R3K_C3_3]
R3K_categories = ['top 1-1500', 'top 1-500', 'top 501-900', 'top 901-1500']

df_rel_cover_C3 = pd.DataFrame([])
for i, df in enumerate(R3K_dfs):
  df_cover = create_plot_df_coverage_rate(df, 
                start_date = minDateNewQuery, 
                end_date = maxDateNewQuery)
  df_rel = create_plot_df(df, 
                start_date = minDateNewQuery, 
                end_date = maxDateNewQuery)
  df_cover.reset_index(inplace=True)
  df_rel.reset_index(inplace=True)
  df_rel_cover = pd.merge(df_cover, df_rel, left_on='DATE_ROLLING', right_on='DATE_ROLLING')
  df_rel_cover['CATEGORY'] = R3K_categories[i]
  df_rel_cover_C3 = pd.concat([df_rel_cover_C3 , df_rel_cover])

# COMMAND ----------

# MAGIC %md
# MAGIC #### Industry Group

# COMMAND ----------

df_all_gp = pd.DataFrame([])
for gp in currdf_R3K['biz_group'].unique():
  df_cover_R3K_gp = create_plot_df_coverage_rate(currdf_R3K[currdf_R3K['biz_group'] == gp], 
              start_date = minDateNewQuery, 
              end_date = maxDateNewQuery)
  df_rel_R3K_gp = create_plot_df(currdf_R3K[currdf_R3K['biz_group'] == gp], 
              start_date = minDateNewQuery, 
              end_date = maxDateNewQuery)
  df_cover_R3K_gp.reset_index(inplace=True)
  df_rel_R3K_gp.reset_index(inplace=True)
  df_rel_cover_R3K_gp = pd.merge(df_cover_R3K_gp, df_rel_R3K_gp, left_on='DATE_ROLLING', right_on='DATE_ROLLING')
  df_rel_cover_R3K_gp['CATEGORY'] = gp
  df_all_gp = pd.concat([df_all_gp, df_rel_cover_R3K_gp])

# COMMAND ----------

# MAGIC %md
# MAGIC #### Combine all the results

# COMMAND ----------

df_rel_cover_all = pd.concat([df_rel_cover_C1, df_rel_cover_C2, df_rel_cover_C3, df_all_gp])

# COMMAND ----------

df_rel_cover_all

# COMMAND ----------

df_rel_cover_all['DATE_ROLLING'] = pd.to_datetime(df_rel_cover_all['DATE_ROLLING'])

# COMMAND ----------

df_rel_cover_all.columns

# COMMAND ----------

df_rel_cover_all = df_rel_cover_all[['DATE_ROLLING', 'COMPANY_COUNT', 'CONSUMER_STRENGTH_COVERRATE_FILT_MD',
       'CONSUMER_WEAKNESS_COVERRATE_FILT_MD',
       'CONSUMER_REDUCED_SPENDING_PATTERNS_COVERRATE_FILT_MD',
       'CONSUMER_STRENGTH_COVERRATE_FILT_QA',
       'CONSUMER_WEAKNESS_COVERRATE_FILT_QA',
       'CONSUMER_REDUCED_SPENDING_PATTERNS_COVERRATE_FILT_QA',
       'CONSUMER_REDUCED_SPENDING_PATTERNS_COVERRATE_FILT_AVERAGE',
       'CONSUMER_STRENGTH_COVERRATE_FILT_AVERAGE',
       'CONSUMER_WEAKNESS_COVERRATE_FILT_AVERAGE',
       'CONSUMER_STRENGTH_REL_FILT_MD', 'CONSUMER_WEAKNESS_REL_FILT_MD',
       'CONSUMER_REDUCED_SPENDING_PATTERNS_REL_FILT_MD',
       'CONSUMER_STRENGTH_REL_FILT_QA', 'CONSUMER_WEAKNESS_REL_FILT_QA',
       'CONSUMER_REDUCED_SPENDING_PATTERNS_REL_FILT_QA',
       'CONSUMER_WEAKNESS_REL_FILT_AVERAGE',
       'CONSUMER_REDUCED_SPENDING_PATTERNS_REL_FILT_AVERAGE',
       'CONSUMER_STRENGTH_REL_FILT_AVERAGE', 'CATEGORY']]

# COMMAND ----------

"/dbfs/mnt/access_work/UC25/Topic Modeling/NLI Models/MASS_Demand/MASS_FT_Demand_rel_coverage_git_" + maxDateNewQuery +".csv"

# COMMAND ----------

df_rel_cover_all.to_csv("/dbfs/mnt/access_work/UC25/Topic Modeling/NLI Models/MASS_Demand/MASS_FT_Demand_rel_coverage_git_" + maxDateNewQuery +".csv", index=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Store time series into Snowflake

# COMMAND ----------

spark_parsedDF = pandas_to_spark(df_rel_cover_all)
spark_parsedDF = spark_parsedDF.replace(np.nan, None)
spark_parsedDF = spark_parsedDF.withColumn("DATE_ROLLING", F.to_timestamp(spark_parsedDF.DATE_ROLLING, 'yyyy-MM-dd'))                                                                      
new_sf.db = 'EDS_PROD'
new_sf.schema = 'QUANT'
 
tablename_curr = 'YUJING_MASS_FT_DEMAND_TS_DEV_1'
result_curr = new_sf.write_to_snowflake_table(spark_parsedDF, tablename_curr)

# COMMAND ----------

