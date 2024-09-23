# Databricks notebook source
# MAGIC %run ../filesystem/dbfs_utility
# MAGIC

# COMMAND ----------

# MAGIC %run ../database/snowflake_dbutility
# MAGIC

# COMMAND ----------

# MAGIC %run ../utilities/config_utility

# COMMAND ----------

import pickle
import pandas as pd
import logging
from typing import Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class SnowflakeManager:
    def __init__(self):
        self.myDBFS = MyDBFS()
        self.mysf_quant = SnowFlakeDBUtility('prod', 'ROLE_EDS_PROD_DDLADMIN_QUANT', 'QUANT')
        self.pickle_path = self.get_pickle_path() 
        # self.mysf_quant_stg = SnowFlakeDBUtility('prod', 'ROLE_EDS_PROD_DDLADMIN_ALL', 'QUANT_STG')
        # self.mysf_quant_live = SnowFlakeDBUtility('prod', 'ROLE_EDS_PROD_DDLADMIN_ALL', 'QUANT_LIVE')

    def get_pickle_path():
        return f'/dbfs/{self.myDBFS.INIpath}{Config.snowflake_config.snowflake_cred_pkl_file}'


    def save_credentials(self) -> None:
        """
        Save Snowflake credentials to a pickle file.
        """
        self.myDBFS.write_to_pickle(self.pickle_path, self.mysf_quant)

    def load_credentials(self) -> SnowFlakeDBUtility:
        """
        Load Snowflake credentials from a pickle file.

        Returns:
            SnowFlakeDBUtility: The loaded SnowFlakeDBUtility object.
        """
        return self.myDBFS.read_from_pickle(self.pickle_path)

    def execute_query(self, query: str) -> Any:
        """
        Execute a query using the SnowFlakeDBUtility.

        Args:
            query (str): The SQL query to execute.

        Returns:
            Any: The result of the query.
        """
        return self.mysf_quant.read_from_snowflake(query)
    
    def validate_credentials(self) -> None:
        """
        Validate credentials by loading them from a pickle file and execute a predefined query.
        """
        # Load credentials from pickle file
        mysf_quant = self.load_credentials()

        # Execute query
        query_ids = 'SELECT * FROM EDS_PROD.QUANT_STG.EMPLOYEE_SUNIL'
        result_ids = mysf_quant.read_from_snowflake(query_ids)
        logging.info(f"Query result: {result_ids}")

    def main(self):
        # Save credentials to pickle file
        self.save_credentials()
        self.validate_credentials()


if __name__ == "__main__":
    obj = SnowflakeCredManager()
    obj.main()
