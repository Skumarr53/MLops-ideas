# Databricks notebook source
# MAGIC %run ../dataclasses/dataclasses

# COMMAND ----------

import os
import re
import yaml
import csv
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Type

# COMMAND ----------

def get_repo_dir():

    pattern = r"^(.*?)(data[-_]science)"

    # Search for the pattern in the path
    match = re.search(pattern, NB_PATH)
    repo_dir = match.group(1)

    return repo_dir

# COMMAND ----------

NB_PATH = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
NB_PATH = os.getcwd()
REPO_DIR = get_repo_dir()
CONFIG_DIR =  REPO_DIR+"data-science-nlp-ml-common-code/impackage_dev/utilities/env_config"

CONFIG_MAPPING_PATH = f"{CONFIG_DIR}/config_mapping.yml"

# COMMAND ----------

class ConfigUtility:
    def __init__(self, config_mapping_path: Dict[Type, str]):
        """
        Initialize the ConfigUtility class.
        Args:
            config_mapping (Dict[Type, str]): A dictionary mapping configuration classes to their respective YAML file paths.
        """
        self.env = self.get_env()
        self.config_mapping = self.load_config_mapping(config_mapping_path)
        self.kv_config = self.load_config(kvConfig)
        self.snowflake_config = self.load_config(SnowFlakeConfig)
        self.mlflow_config = self.load_config(MLflowConfig)
        self.file_path_config = self.load_config(FilePathConfig)
        self.table_name_config = self.load_config(TableNameConfig)
        self.table_col_name_config = self.load_config(TableColNameConfig)
    
    @staticmethod
    def get_env():
        """
        Determine the environment based on the notebook path.
        Returns:
            str: The environment name. Possible values are 'quant', 'quant_stg', 'quant_live'.
        """
        nb_path = NB_PATH
        if "@voya.com" in nb_path:
            return 'quant'
        elif "Quant_Stg" in nb_path:
            return 'quant_stg'
        elif "Quant_Live" in nb_path:
            return 'quant_live'
        elif "Quant" in nb_path:
            return 'quant'
        return 'quant' 
    
    def load_yml(self, file_path):
        """
        Load a YAML file and return its contents.
        Args:
            file_path (str): The path to the YAML file.
        Returns:
            dict: The contents of the YAML file.
        """
        with open(file_path, 'r') as file:
            yml_out = yaml.safe_load(file)
        return yml_out
    
    def load_config_mapping(self, config_mapping_path: str) -> Dict[Type, str]:
        """
        Load and format the configuration mapping from a YAML file.
        Args:
            config_mapping_path (str): The path to the YAML file containing the configuration mapping.
        Returns:
            Dict[Type, str]: A dictionary mapping configuration classes to their respective formatted YAML file paths.
        """
        config_mapping = self.load_yml(config_mapping_path)
        for key in config_mapping:
            config_mapping[key] = config_mapping[key].format(env = self.env)
        return config_mapping
    
    def load_config(self, config_class: Type) -> object:
        """
        Load the configuration for a given class from its corresponding YAML file.
        Args:
            config_class (Type): The configuration class to load.
        Returns:
            object: An instance of the configuration class populated with data from the YAML file.
        """
        yaml_path = self.config_mapping[config_class.__name__]
        config_data = self.load_yml(f"{CONFIG_DIR}/{yaml_path}")
        
        if config_class == TableColNameConfig:
            for key, csv_file in config_data.items():
                config_data[key] = self.load_csv_as_list(f"{CONFIG_DIR}/{csv_file}")
        
        return config_class(**config_data)

    @staticmethod
    def load_csv_as_list(file_path: str) -> List[str]:
        """
        Loads the content of a single-column CSV file (without headers) into a list.
        Args:
            file_path (str): The path to the CSV file.
        Returns:
            List[str]: A list containing the values from the single column.
        """
        data_list = []
        try:
            with open(file_path, mode='r', newline='', encoding='utf-8') as csvfile:
                reader = csv.reader(csvfile)
                data_list = [row[0] for row in reader if row]  # Ensure row is not empty
        except FileNotFoundError:
            print(f"The file at path '{file_path}' was not found.")
        except Exception as e:
            print(f"An error occurred while reading the CSV file: {e}")
        return data_list

# COMMAND ----------

config = ConfigUtility(CONFIG_MAPPING_PATH)
