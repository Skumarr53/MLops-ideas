# data_loader.py
import pandas as pd
import snowflake.connector
import hydra
from omegaconf import DictConfig

class DataLoader:
    def __init__(self, cfg: DictConfig):
        # Use Hydra config for Snowflake and queries
        self.cfg = cfg

    def snowflake_connect(self):
        """Establish a connection to Snowflake using the config provided by Hydra."""
        try:
            connection = snowflake.connector.connect(
                user=self.cfg.schema.snowflake.user,
                password=self.cfg.schema.snowflake.password,
                account=self.cfg.schema.snowflake.account,
                warehouse=self.cfg.schema.snowflake.warehouse,
                database=self.cfg.schema.snowflake.database,
                schema=self.cfg.schema.snowflake.schema
            )
            return connection
        except Exception as e:
            print(f"Error connecting to Snowflake: {e}")
            return None

    def run_query(self, query: str) -> pd.DataFrame:
        """Run a query on Snowflake and return the result as a pandas DataFrame."""
        conn = self.snowflake_connect()
        if not conn:
            raise ConnectionError("Could not establish connection to Snowflake")
        
        try:
            df = pd.read_sql(query, conn)
        finally:
            conn.close()
        return df

@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    # Instantiate DataLoader with the loaded Hydra config
    data_loader = DataLoader(cfg)

    # Use a dynamic query from the loaded config
    start_date = '2023-01-01'
    end_date = '2023-02-01'

    # You can access multiple queries from the query YAML files based on pipeline
    query = cfg.queries.earnings_call_1.format(start_date=start_date, end_date=end_date)
    
    df = data_loader.run_query(query)
    print(df.head())  # For demonstration

if __name__ == "__main__":
    main()
