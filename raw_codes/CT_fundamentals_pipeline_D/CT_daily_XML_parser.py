# Databricks notebook source
# MAGIC %md
# MAGIC # ETL OF FACTSET CALL TRANSCRIPT XML DATA FROM SNOWFLAKE  
# MAGIC   
# MAGIC PRIMARY AUTHOR : Bea Yu (primary author of classes composing NLP methods, other classes called in this processing and "glue code")
# MAGIC
# MAGIC SECONDARY AUTHORS: Partha Kadmbi (primary author of most NLP methods)
# MAGIC
# MAGIC DATE : 04.03.2023
# MAGIC
# MAGIC CONTEXT DESCRIPTION : First notebook in the call transcript NLP processing pipeline to execute ETL from Snowflake nonproduction databases
# MAGIC
# MAGIC CONTEXT NOTEBOOKS :
# MAGIC
# MAGIC - "xml_parser.py" including classes to parse the Call Transcript XML.
# MAGIC - "dbfs_utility.py" including classes to interact with the dbfs file system.
# MAGIC - "snowflake_dbutility.py" including classes to interact with the snowflake database.
# MAGIC - "logging.py" including classes for custom logging.
# MAGIC - After then notebook has executed:
# MAGIC
# MAGIC   - Run CT_fundamentals_preprocessor_utility.py to generate the FILT values and saved in AJAY_CTS_PREPROCESSED_D table in snowflake.
# MAGIC   - Run CT_fundamentals_TOPICX_utility.py and CT_fundamentals_FINBERT_Scores_Utility.py to generate the topicx and FINBERT scores.
# MAGIC         And saves in topicx scores in AJAY_CTS_TOPICX_SCORES_D table and finbert scores in AJAY_CTS_FINBERT_SCORES_D table of snowflake
# MAGIC   - Run CT_fundamentals_sentiment_scores_utility.py to generate the relevance scores and saved in AJAY_CTS_COMBINED_SCORES_D table in snowflake
# MAGIC
# MAGIC
# MAGIC CONTACT INFO : bea.yu@voya.com, partha.kadambi@voya.com, ajaya.devalla@voya.com

# COMMAND ----------

# MAGIC %run ./../../data-science-nlp-ml-common-code/impackage/database/snowflake_dbutility

# COMMAND ----------

# MAGIC %run ./../../data-science-nlp-ml-common-code/impackage/ETL/XML_daily_parser

# COMMAND ----------

import pandas as pd
import datetime
import pytz
from datetime import timedelta
from datetime import datetime


# COMMAND ----------

# DBTITLE 1,Read SnowFlake Credentials.
myDBFS = DBFSUtility()
print(myDBFS.INIpath)

mysf_quant = pd.read_pickle(r'/dbfs' + myDBFS.INIpath + config.snowflake_cred_pkl_file)
mysf_work = pd.read_pickle(r'/dbfs' + myDBFS.INIpath + 'mysf_prod_work.pkl')


# COMMAND ----------

eastern_tzinfo = pytz.timezone("America/New_York")
load_date_time =(datetime.now()).replace(tzinfo=timezone.utc).astimezone(tz=eastern_tzinfo)
today_date_time = datetime.strptime(load_date_time.strftime("%Y-%m-%d %H:%M:%S"), "%Y-%m-%d %H:%M:%S")


# COMMAND ----------

# MAGIC %md
# MAGIC (today_date_time).strftime('%Y-%m-%d %H:%M:%S')

# COMMAND ----------

# DBTITLE 1,Query to read DATETIME Columns  from Historical table.
hist_df_query=("SELECT PARSED_DATETIME_EASTERN_TZ, UPLOAD_DT_UTC, DATE, VERSION_ID FROM {0}.{1}.{2} WHERE PARSED_DATETIME_EASTERN_TZ=(SELECT MAX(PARSED_DATETIME_EASTERN_TZ) as parsed_datetime FROM {0}.{1}.{2})").format(config.eds_db_prod,config.schema,config.CT_sentiment_historical_table)

# COMMAND ----------

# MAGIC %md
# MAGIC hist_df_query=("SELECT * FROM {0}.{1}.{2} WHERE PARSED_DATETIME_EASTERN_TZ=(SELECT MAX(PARSED_DATETIME_EASTERN_TZ) as parsed_datetime FROM {0}.{1}.{2})").format(config.eds_db_prod,config.schema,config.CT_sentiment_historical_table)

# COMMAND ----------

# DBTITLE 1,Execute the query in snowflake.
hist_spark_df = mysf_quant.read_from_snowflake(hist_df_query)

# COMMAND ----------

# DBTITLE 1,Convert Pyspark to pandas.
hist_df=hist_spark_df.toPandas()


# COMMAND ----------

hist_df.head(2)

# COMMAND ----------

# MAGIC %md
# MAGIC hist_df.columns

# COMMAND ----------

# MAGIC %md
# MAGIC print(type(hist_df["PARSED_DATETIME_EASTERN_TZ"][0]))

# COMMAND ----------

# MAGIC %md
# MAGIC hist_df["PARSED_DATETIME_EASTERN_TZ"].max()

# COMMAND ----------

# DBTITLE 1,Get Max of PARSED_DATETIME_EASTERN_TZ
last_parsed_datetime=hist_df["PARSED_DATETIME_EASTERN_TZ"].max()

# COMMAND ----------

# MAGIC %md
# MAGIC last_parsed_datetime.strftime('%Y-%m-%d')

# COMMAND ----------

# DBTITLE 1,Get max of last_upload_dt
last_upload_dt=hist_df["UPLOAD_DT_UTC"].max()
last_upload_dt.strftime('%Y-%m-%d %H:%M:%S')
# last_parsed_obj.last_upload_dt = pd.to_datetime('2023-09-12 07:34:15')

# COMMAND ----------

# DBTITLE 1,Get Min and Max of Date Column
min_last_ts_date=hist_df["DATE"].min()
max_last_ts_date=hist_df["DATE"].max()

# COMMAND ----------

# DBTITLE 1,Verify if new data in factset database.
# MAGIC %md
# MAGIC if hist_df.shape[0] == 0:
# MAGIC    raise Exception("No New Records found")
# MAGIC else:
# MAGIC   print(f"New Recods with shape {hist_df.shape} are available to be parsed.")

# COMMAND ----------

# MAGIC %md
# MAGIC #Process to get data from FactSet Database
# MAGIC <br>__step1__: Read data from factset database based on Lastupload date time from hist_df.
# MAGIC <br>__step2__: Convert the sparkdataframe to pandas.
# MAGIC <br>__step3__: Convert VersionId and Id to integer type
# MAGIC <br>__step4__: filter transcripts with upload date more than 30 days after the event date.
# MAGIC <br>__step5__: select the latest uploaded data if there are multiple corrected versions per id'
# MAGIC <br>__step6__: get max and min date from queried transcripts
# MAGIC <br>__step7__: check whether there are transcripts remaing to be parsed after removing duplicates and process and save them if so      
# MAGIC <br>__step8__: Instantiate nlp object which will parse trascripts in initializatio
# MAGIC <br>__step9__: By entity id if there are multiple transcripts with different ids but the same entity id, in case the call is split into analyst qs transcripts and executive answers transcripts
# MAGIC <br>__step10__: convert call id list object to string to ensure proper datatype conversion when saving outputs to snowflake
# MAGIC <br>__step11__: Write data to snowflake daily and historical table.
# MAGIC <br>__step12__: Write data to parquet file.

# COMMAND ----------

#CHECK TO SEE IF DATAFRAME INCLUDING TRANSCRIPT XML STRINGS FROM LAST PROCESSING EVENT EXISTS
if not hist_df.shape[0] == 0:

    latest_date_last_parsed = last_parsed_datetime.strftime('%Y-%m-%d')
    last_upload_DT = (last_upload_dt.strftime('%Y-%m-%d %H:%M:%S'))

    print('The last parsing date was ' + latest_date_last_parsed + ' and the queried table had ' + str(hist_df.shape[0]) + ' rows and ' + str(hist_df.shape[1]) + ' columns.')
    print('')
    
    lastULDT = "'" + last_upload_DT + "'"

    #QUERY SNOWFLAKE FOR ALL TRANSCRIPTES UPLOADED ON OR AFTER THE LATEST UPLOAD DATE FROM THE MOST RECENTLY PARSED DATA
    tsQuery= ("SELECT DATE, ID, FACTSET_ENTITY_ID, RAW_XML, EDS_FACTSET_PROD.EVT_V1.CE_TRANSCRIPT_VERSIONS.VERSION_ID,"
"EDS_FACTSET_PROD.EVT_V1.CE_TRANSCRIPT_VERSIONS.UPLOAD_DATETIME,"
"EDS_DLY_FACTSET_PROD.TRANSCRIPTS.TRANSCRIPTS.TRANSCRIPT_TYPE, EVENT_TYPE, EVENT_DATETIME_UTC, TITLE "  
"FROM EDS_DLY_FACTSET_PROD.TRANSCRIPTS.TRANSCRIPTS "
"INNER jOIN EDS_FACTSET_PROD.EVT_V1.CE_TRANSCRIPT_VERSIONS ON (EDS_FACTSET_PROD.EVT_V1.CE_TRANSCRIPT_VERSIONS.VERSION_ID = " 
"EDS_DLY_FACTSET_PROD.TRANSCRIPTS.TRANSCRIPTS.VERSION_ID)"
"INNER JOIN EDS_FACTSET_PROD.EVT_V1.CE_REPORTS ON EDS_DLY_FACTSET_PROD.TRANSCRIPTS.TRANSCRIPTS.ID = EDS_FACTSET_PROD.EVT_V1.CE_REPORTS.REPORT_ID "
"INNER JOIN EDS_FACTSET_PROD.EVT_V1.CE_EVENTS ON EDS_FACTSET_PROD.EVT_V1.CE_REPORTS.EVENT_ID = EDS_FACTSET_PROD.EVT_V1.CE_EVENTS.EVENT_ID "
"WHERE UPLOAD_DATETIME > " + lastULDT + " AND (EVENT_TYPE = 'E') AND (EDS_DLY_FACTSET_PROD.TRANSCRIPTS.TRANSCRIPTS.TRANSCRIPT_TYPE = 'CorrectedTranscript');")
    
    resultspkdf = mysf_work.read_from_snowflake(tsQuery)
    if resultspkdf.count() == 0: raise Exception("No New Records found")
    print(f"New Recods with rows {resultspkdf.count()} with {len(resultspkdf.columns)} columns are available to be parsed.")
    resultspkdf = resultspkdf.withColumn("ID",resultspkdf["ID"].cast(IntegerType()))
    resultspkdf = resultspkdf.withColumn("VERSION_ID",resultspkdf["VERSION_ID"].cast(IntegerType()))

    #CONVERT SPARK DF TO PANDAS DF AND REMOVE ANY DUPLICATED TRANSCRIPTS FROM THE MOST RECENTLY PARSED SET
    currdf = resultspkdf.toPandas()
    print('The latest data with duplicates from last processing session spans ' + str(currdf['DATE'].min()) + ' to ' + str(currdf['DATE'].max()) + ' and has ' + str(currdf.shape[0]) + ' rows and ' + str(currdf.shape[1]) + ' columns. ')
    print('')

    #FILTER TRANSCRIPTS WITH UPLOAD DATE MORE THAN 30 DAYS AFTER THE EVENT DATE
    currdfNoDups = currdf[currdf['UPLOAD_DATETIME'] <= currdf['EVENT_DATETIME_UTC'] + timedelta(days=30)]
    #SELECT THE LATEST UPLOADED DATA IF THERE ARE MULTIPLE CORRECTED VERSIONS PER ID 
    currdfNoDups = currdfNoDups.sort_values('UPLOAD_DATETIME').groupby('ID').last().reset_index()
    currdfNoDups = currdfNoDups[~currdfNoDups['VERSION_ID'].isin(hist_df['VERSION_ID'])]
    currdfNoDups = currdfNoDups.drop_duplicates()

    #GET MAX AND MIN DATE FROM QUERIED TRANSCRIPTS
    maxDateNew = currdfNoDups['DATE'].max()
    minDateNew = currdfNoDups['DATE'].min()
    newDataDims = currdfNoDups.shape
    
    #CHECK WHETHER THERE ARE TRANSCRIPTS REMAING TO BE PARSED AFTER REMOVING DUPLICATES AND PROCESS AND SAVE THEM IF SO
    if not currdfNoDups.empty:

      print('The latest data without duplicates from the last query spans ' + str(currdfNoDups['DATE'].min()) + ' to ' + str(currdfNoDups['DATE'].max()) + ' and has ' + str(currdfNoDups.shape[0]) + ' rows and ' + str(currdfNoDups.shape[1]) + ' columns.The last upload datetime is ' + str(currdfNoDups['UPLOAD_DATETIME'].min().strftime('%Y-%m-%d, %H:%M:%S')))

      print('')
      #INSTANTIATE NLP OBJECT WHICH WILL PARSE TRASCRIPTS IN INITIALIZATION
      new_parsed_obj = XMLParser(minDateNew,maxDateNew,newDataDims,currdfNoDups,hist_df)

      assert not new_parsed_obj.parsed_df.empty, 'None of the transcripts that were queried were parsed.  Inspect hist_df.\n'

      new_parsed_obj.not_parsed_df = new_parsed_obj.parsed_df[new_parsed_obj.parsed_df['ERROR'] == 'yes']
      new_parsed_obj.parsed_df = new_parsed_obj.parsed_df[new_parsed_obj.parsed_df['ERROR'] != 'yes']

      #GROUP BY ENTITY ID IF THERE ARE MULTIPLE TRANSCRIPTS WITH DIFFERENT IDS BUT THE SAME ENTITY ID, IN CASE THE CALL IS SPLIT INTO ANALYST QS TRANSCRIPTS AND EXECUTIVE ANSWERS TRANSCRIPTS
      new_parsed_obj.parsed_df = new_parsed_obj.parsed_df.groupby(['ENTITY_ID','DATE']).agg({'ANALYST_QS': 'sum', 
                                                   'CEO_ANS' : 'sum',
                                                   'CEO_DISCUSSION' : 'sum',
                                                   'EXEC_ANS' : 'sum',
                                                   'EXEC_DISCUSSION' : 'sum',
                                                   'MGNT_DISCUSSION' : 'sum',
                                                   'QA_SECTION' : 'sum',
                                                   'CALL_NAME' : lambda x : list(x),
                                                   'COMPANY_NAME': lambda x : x.iloc[0],
                                                   'EARNINGS_CALL' : 'sum',
                                                   'ERROR' : lambda x : x.iloc[0],
                                                   'TRANSCRIPT_STATUS' : lambda x : x.iloc[0],
                                                   'SF_DATE' : lambda x : x.iloc[0],
                                                   'SF_ID' : lambda x : x.iloc[0],
                                                   'CALL_ID' : lambda x : x.iloc[0],
                                                   'UPLOAD_DT_UTC': lambda x: x.iloc[0],
                                                   'VERSION_ID': lambda x: x.iloc[0],                                      
                                                   'EVENT_DATETIME_UTC': lambda x : x.iloc[0]
                                                  }).reset_index()

      #CONVERT CALL ID LIST OBJECT TO STRING TO ENSURE PROPER DATATYPE CONVERSION WHEN SAVING OUTPUTS TO SNOWFLAKE
      new_parsed_obj.parsed_df['CALL_NAME'] = new_parsed_obj.parsed_df['CALL_NAME'].apply(lambda x : ' , '.join(x))  
      new_parsed_obj.parsed_df['ANALYST_QS'] = new_parsed_obj.parsed_df['ANALYST_QS'].apply(lambda x : '  '.join(x))
      new_parsed_obj.parsed_df['CEO_ANS'] = new_parsed_obj.parsed_df['CEO_ANS'].apply(lambda x : '  '.join(x))
      new_parsed_obj.parsed_df['CEO_DISCUSSION'] = new_parsed_obj.parsed_df['CEO_DISCUSSION'].apply(lambda x : '  '.join(x))
      new_parsed_obj.parsed_df['EXEC_ANS'] = new_parsed_obj.parsed_df['EXEC_ANS'].apply(lambda x : '  '.join(x))
      new_parsed_obj.parsed_df['EXEC_DISCUSSION'] = new_parsed_obj.parsed_df['EXEC_DISCUSSION'].apply(lambda x : '  '.join(x))
      new_parsed_obj.parsed_df['MGNT_DISCUSSION'] = new_parsed_obj.parsed_df['MGNT_DISCUSSION'].apply(lambda x : '  '.join(x))
      new_parsed_obj.parsed_df['QA_SECTION'] = new_parsed_obj.parsed_df['QA_SECTION'].apply(lambda x : '  '.join(x))
      new_parsed_obj.parsed_df['PARSED_DATETIME_EASTERN_TZ'] = (today_date_time).strftime('%Y-%m-%d %H:%M:%S')
      new_parsed_obj.parsed_df['BACKFILL'] = 0

      #REORDER COLUMNS TO FIT SCHEMA IN SNOWFLAKE
      new_parsed_obj.parsed_df = new_parsed_obj.parsed_df[['ENTITY_ID', 'DATE', 'ANALYST_QS', 'CEO_ANS', 'CEO_DISCUSSION', 'EXEC_ANS', 'EXEC_DISCUSSION', 'MGNT_DISCUSSION', 'QA_SECTION', 'CALL_NAME', 'COMPANY_NAME', 'EARNINGS_CALL', 'ERROR','TRANSCRIPT_STATUS', 'SF_DATE', 'SF_ID', 'CALL_ID', 'UPLOAD_DT_UTC', 'VERSION_ID', 'EVENT_DATETIME_UTC', 'PARSED_DATETIME_EASTERN_TZ','BACKFILL']]

      #CONVERT TO SPARK DATAFRAME AND APPEND TO HISTORICAL PARSED TRANSCRIPT TABLE IN SNOWFLAKE
      df_final = spark.createDataFrame(new_parsed_obj.parsed_df) 
      df_final = df_final.withColumn('ENTITY_ID', df_final.ENTITY_ID.cast('string'))\
                       .withColumn('ANALYST_QS', df_final.ANALYST_QS.cast('string'))\
                       .withColumn('CEO_ANS', df_final.CEO_ANS.cast('string'))\
                       .withColumn('CEO_DISCUSSION', df_final.CEO_DISCUSSION.cast('string'))\
                       .withColumn('EXEC_ANS', df_final.EXEC_ANS.cast('string'))\
                       .withColumn('EXEC_DISCUSSION', df_final.EXEC_DISCUSSION.cast('string'))\
                       .withColumn('MGNT_DISCUSSION', df_final.MGNT_DISCUSSION.cast('string'))\
                       .withColumn('CALL_NAME', df_final.CALL_NAME.cast('string'))\
                       .withColumn('COMPANY_NAME', df_final.COMPANY_NAME.cast('string'))\
                       .withColumn('ERROR', df_final.ERROR.cast('string'))\
                       .withColumn('TRANSCRIPT_STATUS', df_final.TRANSCRIPT_STATUS.cast('string'))\
                       .withColumn('EARNINGS_CALL', df_final.EARNINGS_CALL.cast('float'))\
                       .withColumn('DATE', df_final.DATE.cast('timestamp'))\
                       .withColumn('SF_DATE', df_final.SF_DATE.cast('timestamp'))\
                       .withColumn('EVENT_DATETIME_UTC', df_final.EVENT_DATETIME_UTC.cast('timestamp'))\
                       .withColumn('PARSED_DATETIME_EASTERN_TZ', df_final.PARSED_DATETIME_EASTERN_TZ.cast('timestamp'))\
                       .withColumn('SF_ID', df_final.SF_ID.cast('integer'))\
                       .withColumn('UPLOAD_DT_UTC', df_final.UPLOAD_DT_UTC.cast('timestamp'))\
                       .withColumn('VERSION_ID', df_final.VERSION_ID.cast('integer'))\
                       .withColumn('CALL_ID', df_final.CALL_ID.cast('integer'))\
                       .withColumn('BACKFILL', df_final.BACKFILL.cast('integer'))\

      mysf_quant.db = config.eds_db_prod
      truncate_query="TRUNCATE TABLE {0}.{1}.{2}".format(config.eds_db_prod,config.schema,config.CT_parsed_daily_table)
      truncate_flag=mysf_quant.truncate_or_merge_table(truncate_query)
      result_curr = mysf_quant.write_to_snowflake_table(df_final, config.CT_parsed_daily_table)

      merge_query="""MERGE INTO {0}.{1} A \
  USING {2}.{3} B \
  ON A.ENTITY_ID = B.ENTITY_ID AND \
   A.VERSION_ID = B.VERSION_ID AND \
   A.UPLOAD_DT_UTC = B.UPLOAD_DT_UTC
  WHEN NOT MATCHED THEN \
     INSERT ( ENTITY_ID,DATE,ANALYST_QS,CEO_ANS,CEO_DISCUSSION,EXEC_ANS,EXEC_DISCUSSION,MGNT_DISCUSSION,QA_SECTION,CALL_NAME,COMPANY_NAME,EARNINGS_CALL,ERROR,TRANSCRIPT_STATUS,SF_DATE,SF_ID,CALL_ID,UPLOAD_DT_UTC,VERSION_ID,EVENT_DATETIME_UTC,PARSED_DATETIME_EASTERN_TZ,BACKFILL) \
     VALUES ( B.ENTITY_ID,B.DATE,B.ANALYST_QS,B.CEO_ANS,B.CEO_DISCUSSION,B.EXEC_ANS,B.EXEC_DISCUSSION,B.MGNT_DISCUSSION,B.QA_SECTION,B.CALL_NAME,B.COMPANY_NAME,B.EARNINGS_CALL,B.ERROR,B.TRANSCRIPT_STATUS,B.SF_DATE,B.SF_ID,B.CALL_ID,B.UPLOAD_DT_UTC,B.VERSION_ID,B.EVENT_DATETIME_UTC,B.PARSED_DATETIME_EASTERN_TZ,B.BACKFILL)""".format(config.schema, 
      config.CT_parsed_historical_table,config.schema,config.CT_parsed_daily_table)
      merge_flag=mysf_quant.truncate_or_merge_table(merge_query)

      print('writting to file.')
      print(len(new_parsed_obj.parsed_df))
      #SAVE UPDATED NLP OBJECT FOR FUTURE REFERENCE
      CT_parsed_file = open(config.CT_parsed_parquet_file, "wb")
      CT_parsed_file.close()
      new_parsed_obj.parsed_df.to_parquet(config.CT_parsed_parquet_file,engine = 'pyarrow',compression = 'gzip')
    #IF THERE ARE NO NEW TRANSCRIPTS, EXIT SCRIPT
    else:

      print('No new transcripts.\n')
      new_parsed_obj = XMLParser(min_last_ts_date, max_last_ts_date, (0,0), pd.DataFrame(), hist_df)
      CT_parsed_file = open(config.CT_parsed_parquet_file, "wb")
      CT_parsed_file.close()
      new_parsed_obj.parsed_df.to_parquet(config.CT_parsed_parquet_file,engine = 'pyarrow', compression = 'gzip')
      print(len(new_parsed_obj.parsed_df))
  #IF THE LAST PARSED TABLE IS EMPTY, QUERY LATEST DATE USING THE MAX DATE INFORMATION IN THE LOG FILE
else:

    print('No table from last parsing session exists for reference.\n')  

    latest_date_last_parsed = last_parsed_datetime.strftime('%Y-%m-%d')
    last_upload_DT = (last_upload_dt.strftime('%Y-%m-%d %H:%M:%S'))


    print('The last parsing date was ' + latest_date_last_parsed + ' and the queried table has ' + str(hist_df.shape[0]) + ' rows and ' + str(hist_df.shape[1]) + ' columns.')
    print('')

    lastULDT = "'" + last_upload_DT + "'" 

    #QUERY SNOWFLAKE FOR ALL TRANSCRIPTES UPLOADED ON OR AFTER THE LATEST UPLOAD DATE FROM THE MOST RECENTLY PARSED DATA
    tsQuery= ("SELECT DATE, ID, FACTSET_ENTITY_ID, RAW_XML, EDS_FACTSET_PROD.EVT_V1.CE_TRANSCRIPT_VERSIONS.VERSION_ID,"
"EDS_FACTSET_PROD.EVT_V1.CE_TRANSCRIPT_VERSIONS.UPLOAD_DATETIME,"
"EDS_DLY_FACTSET_PROD.TRANSCRIPTS.TRANSCRIPTS.TRANSCRIPT_TYPE, EVENT_TYPE, EVENT_DATETIME_UTC, TITLE "  
"FROM EDS_DLY_FACTSET_PROD.TRANSCRIPTS.TRANSCRIPTS "
"INNER jOIN EDS_FACTSET_PROD.EVT_V1.CE_TRANSCRIPT_VERSIONS ON (EDS_FACTSET_PROD.EVT_V1.CE_TRANSCRIPT_VERSIONS.VERSION_ID = " 
"EDS_DLY_FACTSET_PROD.TRANSCRIPTS.TRANSCRIPTS.VERSION_ID)"
"INNER JOIN EDS_FACTSET_PROD.EVT_V1.CE_REPORTS ON EDS_DLY_FACTSET_PROD.TRANSCRIPTS.TRANSCRIPTS.ID = EDS_FACTSET_PROD.EVT_V1.CE_REPORTS.REPORT_ID "
"INNER JOIN EDS_FACTSET_PROD.EVT_V1.CE_EVENTS ON EDS_FACTSET_PROD.EVT_V1.CE_REPORTS.EVENT_ID = EDS_FACTSET_PROD.EVT_V1.CE_EVENTS.EVENT_ID "
"WHERE UPLOAD_DATETIME > " + lastULDT + " AND (EVENT_TYPE = 'E') AND (EDS_DLY_FACTSET_PROD.TRANSCRIPTS.TRANSCRIPTS.TRANSCRIPT_TYPE = 'CorrectedTranscript');")

    resultspkdf = mysf_work.read_from_snowflake(tsQuery)
    if resultspkdf.count() == 0: raise Exception("No New Records found")
    print(f"New Recods with rows {resultspkdf.count()} with {len(resultspkdf.columns)} columns are available to be parsed.")

    resultspkdf = resultspkdf.withColumn("ID",resultspkdf["ID"].cast(IntegerType()))
    resultspkdf = resultspkdf.withColumn("VERSION_ID",resultspkdf["VERSION_ID"].cast(IntegerType()))

    #CONVERT SPARK DF TO PANDAS DF AND REMOVE ANY DUPLICATED TRANSCRIPTS FROM THE MOST RECENTLY PARSED SET
    currdf = resultspkdf.toPandas()

    print('The latest data spans ' + str(currdf['DATE'].min()) + ' to ' + str(currdf['DATE'].max()) + ' and has ' + str(currdf.shape[0]) + ' rows and ' + str(currdf.shape[1]) + ' columns.')
    print('')
    #FILTER TRANSCRIPTS WITH UPLOAD DATE MORE THAN 30 DAYS AFTER THE EVENT DATE
    currdfNoDups = currdf[currdf['UPLOAD_DATETIME'] <= currdf['EVENT_DATETIME_UTC'] + timedelta(days=30)]
    #SELECT THE LATEST UPLOADED DATA IF THERE ARE MULTIPLE CORRECTED VERSIONS PER ID 
    currdfNoDups = currdfNoDups.sort_values('UPLOAD_DATETIME').groupby('ID').last().reset_index()
    currdfNoDups = currdfNoDups[~currdfNoDups['VERSION_ID'].isin(hist_df['VERSION_ID'])]
    currdfNoDups = currdfNoDups.drop_duplicates()

    #GET MAX AND MIN DATE FROM QUERIED TRANSCRIPTS
    maxDateNew = currdfNoDups['DATE'].max()
    minDateNew = currdfNoDups['DATE'].min()
    newDataDims = currdfNoDups.shape

    #CHECK WHETHER THERE ARE TRANSCRIPTS REMAING TO BE PARSED AFTER REMOVING DUPLICATES AND PROCESS AND SAVE THEM IF SO
    if not currdfNoDups.empty:

      print('The latest data without duplicates from last query spans ' + str(currdfNoDups['DATE'].min()) + ' to ' + str(currdfNoDups['DATE'].max()) + ' and has ' + str(currdfNoDups.shape[0]) + ' rows and ' + str(currdfNoDups.shape[1]) + ' columns.  The last upload datetime is ' + str(currdfNoDups['UPLOAD_DATETIME'].min().strftime('%Y-%m-%d, %H:%M:%S')))
      print('')
      
      #INSTANTIATE NLP OBJECT WHICH WILL PARSE TRASCRIPTS IN INITIALIZATION
      new_parsed_obj = XMLParser(minDateNew, maxDateNew, newDataDims, currdfNoDups, hist_df)
      
      assert not new_parsed_obj.parsed_df.empty, 'None of the transcripts that were queried were parsed.  Inspect hist_df.\n'
      new_parsed_obj.not_parsed_df = new_parsed_obj.parsed_df[new_parsed_obj.parsed_df['ERROR'] == 'yes']
      new_parsed_obj.parsed_df = new_parsed_obj.parsed_df[new_parsed_obj.parsed_df['ERROR'] != 'yes']

      #SELECT THE LATEST UPLOADED DATA IF THERE ARE MULTIPLE CORRECTED VERSIONS PER ID 
      new_parsed_obj.parsed_df = new_parsed_obj.parsed_df.sort_values('UPLOAD_DT_UTC').groupby('CALL_ID').last().reset_index()

      #GROUP BY ENTITY ID IF THERE ARE MULTIPLE TRANSCRIPTS WITH DIFFERENT IDS BUT THE SAME ENTITY ID, IN CASE THE CALL IS SPLIT INTO ANALYST QS TRANSCRIPTS AND EXECUTIVE ANSWERS TRANSCRIPTS
      new_parsed_obj.parsed_df = new_parsed_obj.parsed_df.groupby(['ENTITY_ID','DATE']).agg({'ANALYST_QS': 'sum', 
                                                   'CEO_ANS' : 'sum',
                                                   'CEO_DISCUSSION' : 'sum',
                                                   'EXEC_ANS' : 'sum',
                                                   'EXEC_DISCUSSION' : 'sum',
                                                   'MGNT_DISCUSSION' : 'sum',
                                                   'QA_SECTION' : 'sum',
                                                   'CALL_NAME' : lambda x : list(x),
                                                   'COMPANY_NAME': lambda x : x.iloc[0],
                                                   'EARNINGS_CALL' : 'sum',
                                                   'ERROR' : lambda x : x.iloc[0],
                                                   'TRANSCRIPT_STATUS' : lambda x : x.iloc[0],
                                                   'SF_DATE' : lambda x : x.iloc[0],
                                                   'SF_ID' : lambda x : x.iloc[0],
                                                   'CALL_ID' : lambda x : x.iloc[0],
                                                   'UPLOAD_DT_UTC': lambda x: x.iloc[0],
                                                   'VERSION_ID': lambda x: x.iloc[0],                                      
                                                   'EVENT_DATETIME_UTC': lambda x : x.iloc[0]
                                                    }).reset_index()

      #CONVERT CALL ID LIST OBJECT TO STRING TO ENSURE PROPER DATATYPE CONVERSION WHEN SAVING OUTPUTS TO SNOWFLAKE
      new_parsed_obj.parsed_df['CALL_NAME'] = new_parsed_obj.parsed_df['CALL_NAME'].apply(lambda x : ' , '.join(x))  
      new_parsed_obj.parsed_df['ANALYST_QS'] = new_parsed_obj.parsed_df['ANALYST_QS'].apply(lambda x : '  '.join(x))
      new_parsed_obj.parsed_df['CEO_ANS'] = new_parsed_obj.parsed_df['CEO_ANS'].apply(lambda x : '  '.join(x))
      new_parsed_obj.parsed_df['CEO_DISCUSSION'] = new_parsed_obj.parsed_df['CEO_DISCUSSION'].apply(lambda x : '  '.join(x))
      new_parsed_obj.parsed_df['EXEC_ANS'] = new_parsed_obj.parsed_df['EXEC_ANS'].apply(lambda x : '  '.join(x))
      new_parsed_obj.parsed_df['EXEC_DISCUSSION'] = new_parsed_obj.parsed_df['EXEC_DISCUSSION'].apply(lambda x : '  '.join(x))
      new_parsed_obj.parsed_df['MGNT_DISCUSSION'] = new_parsed_obj.parsed_df['MGNT_DISCUSSION'].apply(lambda x : '  '.join(x))
      new_parsed_obj.parsed_df['QA_SECTION'] = new_parsed_obj.parsed_df['QA_SECTION'].apply(lambda x : '  '.join(x))
      new_parsed_obj.parsed_df['PARSED_DATETIME_EASTERN_TZ'] = (today_date_time).strftime('%Y-%m-%d %H:%M:%S')
      new_parsed_obj.parsed_df['BACKFILL'] = 0

      #REORDER COLUMNS TO FIT SCHEMA IN SNOWFLAKE
      new_parsed_obj.parsed_df = new_parsed_obj.parsed_df[['ENTITY_ID', 'DATE', 'ANALYST_QS', 'CEO_ANS', 'CEO_DISCUSSION', 'EXEC_ANS', 'EXEC_DISCUSSION', 'MGNT_DISCUSSION', 'QA_SECTION', 'CALL_NAME', 'COMPANY_NAME', 'EARNINGS_CALL', 'ERROR','TRANSCRIPT_STATUS', 'SF_DATE', 'SF_ID', 'CALL_ID', 'UPLOAD_DT_UTC', 'VERSION_ID', 'EVENT_DATETIME_UTC', 'PARSED_DATETIME_EASTERN_TZ','BACKFILL']]
      

      #CONVERT TO SPARK DATAFRAME AND APPEND TO HISTORICAL PARSED TRANSCRIPT TABLE IN SNOWFLAKE
      df_final = spark.createDataFrame(new_parsed_obj.parsed_df) 
      df_final = df_final.withColumn('ENTITY_ID', df_final.ENTITY_ID.cast('string'))\
                       .withColumn('ANALYST_QS', df_final.ANALYST_QS.cast('string'))\
                       .withColumn('CEO_ANS', df_final.CEO_ANS.cast('string'))\
                       .withColumn('CEO_DISCUSSION', df_final.CEO_DISCUSSION.cast('string'))\
                       .withColumn('EXEC_ANS', df_final.EXEC_ANS.cast('string'))\
                       .withColumn('EXEC_DISCUSSION', df_final.EXEC_DISCUSSION.cast('string'))\
                       .withColumn('MGNT_DISCUSSION', df_final.MGNT_DISCUSSION.cast('string'))\
                       .withColumn('CALL_NAME', df_final.CALL_NAME.cast('string'))\
                       .withColumn('COMPANY_NAME', df_final.COMPANY_NAME.cast('string'))\
                       .withColumn('ERROR', df_final.ERROR.cast('string'))\
                       .withColumn('TRANSCRIPT_STATUS', df_final.TRANSCRIPT_STATUS.cast('string'))\
                       .withColumn('EARNINGS_CALL', df_final.EARNINGS_CALL.cast('float'))\
                       .withColumn('DATE', df_final.DATE.cast('timestamp'))\
                       .withColumn('SF_DATE', df_final.SF_DATE.cast('timestamp'))\
                       .withColumn('EVENT_DATETIME_UTC', df_final.EVENT_DATETIME_UTC.cast('timestamp'))\
                       .withColumn('PARSED_DATETIME_EASTERN_TZ', df_final.PARSED_DATETIME_EASTERN_TZ.cast('timestamp'))\
                       .withColumn('SF_ID', df_final.SF_ID.cast('integer'))\
                       .withColumn('UPLOAD_DT_UTC', df_final.UPLOAD_DT_UTC.cast('timestamp'))\
                       .withColumn('VERSION_ID', df_final.VERSION_ID.cast('integer'))\
                       .withColumn('CALL_ID', df_final.CALL_ID.cast('integer'))\
                       .withColumn('BACKFILL', df_final.BACKFILL.cast('integer'))\

      mysf_quant.db = config.eds_db_prod
      truncate_query="TRUNCATE TABLE {0}.{1}.{2}".format(config.eds_db_prod,config.schema,config.CT_parsed_daily_table)
      truncate_flag=mysf_quant.truncate_or_merge_table(truncate_query)
      result_curr = mysf_quant.write_to_snowflake_table(df_final, config.CT_parsed_daily_table)
      merge_query="""MERGE INTO {0}.{1} A \
  USING {2}.{3} B \
  ON A.ENTITY_ID = B.ENTITY_ID AND \
   A.VERSION_ID = B.VERSION_ID AND \
   A.UPLOAD_DT_UTC = B.UPLOAD_DT_UTC
  WHEN NOT MATCHED THEN \
     INSERT ( ENTITY_ID,DATE,ANALYST_QS,CEO_ANS,CEO_DISCUSSION,EXEC_ANS,EXEC_DISCUSSION,MGNT_DISCUSSION,QA_SECTION,CALL_NAME,COMPANY_NAME,EARNINGS_CALL,ERROR,TRANSCRIPT_STATUS,SF_DATE,SF_ID,CALL_ID,UPLOAD_DT_UTC,VERSION_ID,EVENT_DATETIME_UTC,PARSED_DATETIME_EASTERN_TZ,BACKFILL) \
     VALUES ( B.ENTITY_ID,B.DATE,B.ANALYST_QS,B.CEO_ANS,B.CEO_DISCUSSION,B.EXEC_ANS,B.EXEC_DISCUSSION,B.MGNT_DISCUSSION,B.QA_SECTION,B.CALL_NAME,B.COMPANY_NAME,B.EARNINGS_CALL,B.ERROR,B.TRANSCRIPT_STATUS,B.SF_DATE,B.SF_ID,B.CALL_ID,B.UPLOAD_DT_UTC,B.VERSION_ID,B.EVENT_DATETIME_UTC,B.PARSED_DATETIME_EASTERN_TZ,B.BACKFILL)""".format(config.schema,config.CT_parsed_historical_table,config.schema,config.CT_parsed_daily_table)
      merge_flag=mysf_quant.truncate_or_merge_table(merge_query)
      print('writing to file.')
      print(len(new_parsed_obj.parsed_df))
      #SAVE UPDATED NLP OBJECT FOR FUTURE REFERENCE
      CT_parsed_file = open(config.CT_parsed_parquet_file, "wb")
      CT_parsed_file.close()
      new_parsed_obj.parsed_df.to_parquet(config.CT_parsed_parquet_file,engine = 'pyarrow', compression = 'gzip')
    #IF THERE ARE NO NEW TRANSCRIPTS, EXIT SCRIPT
    else:
      print('No new transcripts.\n')
      new_parsed_obj = XMLParser(min_last_ts_date, max_last_ts_date, (0,0), pd.DataFrame(), hist_df)

      #SAVE UPDATED NLP OBJECT FOR FUTURE REFERENCE
      CT_parsed_file = open(config.CT_parsed_parquet_file, "wb")
      CT_parsed_file.close()
      print(len(new_parsed_obj.parsed_df))
      new_parsed_obj.parsed_df.to_parquet(config.CT_parsed_parquet_file,engine = 'pyarrow', compression = 'gzip')
      


# COMMAND ----------

# MAGIC %md
# MAGIC last_upload_dt.strftime('%Y-%m-%d %H:%M:%S')

# COMMAND ----------

# MAGIC %md
# MAGIC hist_df

# COMMAND ----------

# MAGIC %md
# MAGIC  new_parsed_obj.parsed_df.UPLOAD_DT_UTC.min()

# COMMAND ----------

# MAGIC %md
# MAGIC new_parsed_obj.parsed_df.shape
