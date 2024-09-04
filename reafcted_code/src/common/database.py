# common/database.py

import pandas as pd
import logging
from typing import Optional

class DBFSHelper:
    def __init__(self, ini_path: str):
        self.ini_path = ini_path

    def get_full_path(self, filename: str) -> str:
        return os.path.join('/dbfs', self.ini_path, filename)

class SnowflakeLoader:
    def __init__(self, dbfs_helper: DBFSHelper):
        self.dbfs_helper = dbfs_helper

    def load_data(self, query: str) -> pd.DataFrame:
        """Load data from Snowflake into a Pandas DataFrame."""
        try:
            new_sf = pd.read_pickle(self.dbfs_helper.get_full_path('mysf_prod_quant.pkl'))
            result_df = new_sf.read_from_snowflake(query)
            logging.info("Data loaded from Snowflake successfully.")
            return result_df.toPandas()
        except Exception as e:
            logging.error(f"Error loading data from Snowflake: {e}")
            raise
