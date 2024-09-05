# test_data_loader.py
import unittest
import hydra
from omegaconf import DictConfig
from data.data_loader import DataLoader

@hydra.main(version_base=None, config_path="../config", config_name="config")
class TestDataLoader(unittest.TestCase):
    def setUp(self, cfg: DictConfig):
        # Initialize DataLoader with Hydra config
        self.data_loader = DataLoader(cfg.schema.snowflake)
    
    def test_snowflake_connect(self):
        conn = self.data_loader.snowflake_connect()
        self.assertIsNotNone(conn)

    def test_run_query(self):
        query = "SELECT 1"
        df = self.data_loader.run_query(query)
        self.assertEqual(len(df), 1)

if __name__ == '__main__':
    unittest.main()
