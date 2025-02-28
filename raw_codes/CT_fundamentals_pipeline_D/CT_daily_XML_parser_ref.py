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




from logging import config

from Config.queries import HIST_DF_QUERY, TS_QUERY


class DataProcessor:
    def __init__(self, config: Any, mysf_quant: Any, mysf_work: Any):
        self.config = config
        self.mysf_quant = mysf_quant
        self.mysf_work = mysf_work
        self.eastern_tzinfo = pytz.timezone("America/New_York")

    def get_current_eastern_time(self) -> datetime:
        """Get the current time in Eastern timezone."""
        utc_time = datetime.now().replace(tzinfo=timezone.utc)
        return utc_time.astimezone(tz=self.eastern_tzinfo)

    def fetch_historical_data(self) -> pd.DataFrame:
        """Fetch historical data from Snowflake and return as a pandas DataFrame."""
        query = HIST_DF_QUERY.format(self.config.eds_db_prod, self.config.schema, self.config.CT_sentiment_historical_table)
        
        hist_spark_df = self.mysf_quant.read_from_snowflake(query)
        return hist_spark_df.toPandas()

    def log_historical_data_info(self, hist_df: pd.DataFrame) -> None:
        """Log information about the historical data."""
        if hist_df.empty:
            logger.info("No historical data found.")
            return

        last_parsed_datetime = hist_df["PARSED_DATETIME_EASTERN_TZ"].max()
        last_upload_dt = hist_df["UPLOAD_DT_UTC"].max()

        latest_date_last_parsed = last_parsed_datetime.strftime('%Y-%m-%d')
        last_upload_dt_str = last_upload_dt.strftime('%Y-%m-%d %H:%M:%S')

        logger.info(
            f"The last parsing date was {latest_date_last_parsed} and the queried table had "
            f"{hist_df.shape[0]} rows and {hist_df.shape[1]} columns."
        )
        return last_upload_dt_str

    def fetch_new_transcripts(self, last_upload_dt: str) -> DataFrame:
        """Fetch new transcripts from Snowflake."""
        query = TS_QUERY.format(last_upload_dt=last_upload_dt)

        result_spark_df = self.mysf_work.read_from_snowflake(query)
        if result_spark_df.count() == 0:
            raise Exception("No New Records found")

        logger.info(
            f"New Records with {result_spark_df.count()} rows and {len(result_spark_df.columns)} columns are available to be parsed."
        )

        result_spark_df = result_spark_df.withColumn("ID", result_spark_df["ID"].cast(IntegerType()))
        result_spark_df = result_spark_df.withColumn("VERSION_ID", result_spark_df["VERSION_ID"].cast(IntegerType()))

        curr_df = result_spark_df.toPandas()
        print(f'The latest data with duplicates from last processing session spans {currdf["DATE"].min()} to {currdf["DATE"].max()} and has {currdf.shape[0]} rows and {currdf.shape[1]} columns.')
        return curr_df

    def process_data(self) -> pd.DataFrame:
        """Main method to process data."""
        load_date_time = self.get_current_eastern_time()
        logger.info(f"Loaded date and time: {load_date_time}")

        hist_df = self.fetch_historical_data()
        last_upload_dt_str = self.log_historical_data_info(hist_df)

        if hist_df.empty:
            return pd.DataFrame()

        result_spark_df = self.fetch_new_transcripts(last_upload_dt_str)
        return result_spark_df.toPandas()
