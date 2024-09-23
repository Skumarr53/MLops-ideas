# Databricks notebook source
# MAGIC %run "./../utilities/config_utility"

# COMMAND ----------

!pip install nutter

# COMMAND ----------

# Import necessary libraries
import os
from runtime.nutterfixture import NutterFixture, tag
from unittest.mock import patch, mock_open, MagicMock
import yaml
import csv

# Define the test class
class TestConfigUtilityFixture(NutterFixture):
   def __init__(self):
      """
      Helps in initializing all the instance variables
      """
      self.config_utility = ConfigUtility(CONFIG_MAPPING_PATH)
      NutterFixture.__init__(self)


   def assertion_get_env(self):
      env = self.config_utility.get_env()
      assert env == 'quant'

   @patch('__main__.open', new_callable=mock_open, read_data="partition_value: 1000\nduplication_threshold: 15\n")
   def assertion_load_yml(self, mock_file):
      file_path = 'quant/key_config.yml'
      expected_output = {
         'partition_value': 1000,
         'duplication_threshold': 15
      }
      output = self.config_utility.load_yml(file_path)
      assert output == expected_output
      mock_file.assert_called_once_with(file_path, 'r')

   @patch('__main__.open', new_callable=mock_open, read_data="kvConfig: 'quant/key_config.yml'\n")
   def assertion_load_config_mapping(self, mock_file):
      expected_output = {
         'kvConfig': 'quant/key_config.yml'
      }
      output = self.config_utility.load_config_mapping(CONFIG_MAPPING_PATH)
      assert output == expected_output
      mock_file.assert_called_once_with(CONFIG_MAPPING_PATH, 'r')

   def assertion_load_config(self):
      config_class = kvConfig
      config_instance = self.config_utility.load_config(config_class)
      assert config_instance.partition_value == 1000
      assert config_instance.duplication_threshold == 15

   @patch('__main__.open', new_callable=mock_open, read_data="MGNT_DISCUSSION\nQA_SECTION\n")
   def assertion_load_csv_as_list(self, mock_file):
      file_path = 'quant/table_columns_names/CT_legacy_aggregate_text_list.csv'
      expected_output = ['MGNT_DISCUSSION', 'QA_SECTION']
      output = self.config_utility.load_csv_as_list(file_path)
      assert output == expected_output
      mock_file.assert_called_once_with(file_path, mode='r', newline='', encoding='utf-8')

# Create the test fixture
test_fixture = TestConfigUtilityFixture()

# Run the tests
result = test_fixture.execute_tests()

# Display the results
print(result.to_string())
