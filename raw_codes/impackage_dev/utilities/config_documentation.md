# NLP Project Configuration Documentation

## Overview

This project is designed for Natural Language Processing (NLP) and utilizes Databricks for its implementation. The configuration files are organized into different groups based on their purpose, allowing for easy management and access to various settings required for the NLP workflows.

## Table of Contents

- [NLP Project Configuration Documentation](#nlp-project-configuration-documentation)
  - [Overview](#overview)
  - [Table of Contents](#table-of-contents)
  - [Directory Structure](#directory-structure)
  - [Configuration Files](#configuration-files)
    - [Config Mapping](#config-mapping)
    - [Environment-Specific Configurations](#environment-specific-configurations)
  - [Usage](#usage)


## Directory Structure

The project has a well-defined directory structure that organizes configuration files based on their functionality. Below is a high-level view of the directory layout:
```
impackage_dev/
└── utilities/
    └── env_config/
        ├── config_mapping.yml
        ├── quant/
        │   ├── key_config.yml
        │   ├── model_config.yml
        │   ├── path_config.yml
        │   ├── snowflake_config.yml
        │   ├── table_col_name_config.yml
        │   └── table_name_config.yml
        ├── quant_stg/
        └── quant_live/
```
## Configuration Files

### Config Mapping

The  `config_mapping.yml`  file serves as a central point for defining the paths to various configuration files based on the environment. It maps different configuration categories to their respective YAML files.

### Environment-Specific Configurations

Each environment (e.g.,  `quant` ,  `quant_stg` ,  `quant_live` ) has its own set of configuration files, which include:

- **Key Configuration ( `key_config.yml` )**: Contains parameters related to system performance and operational settings, such as partition values and GPU counts.

- **Model Configuration ( `model_config.yml` )**: Defines settings related to MLflow, including model stages and paths for saved models.

- **Path Configuration ( `path_config.yml` )**: Specifies file paths used throughout the project, including paths for input data and model artifacts.

- **Snowflake Configuration ( `snowflake_config.yml` )**: Stores credentials and connection details for Snowflake databases.

- **Table Column Name Configuration ( `table_col_name_config.yml` )**: Maps table column names to their respective CSV files for easy access.

- **Table Name Configuration ( `table_name_config.yml` )**: Contains the names of the tables used in the database for easy reference.

## Usage

To use the configuration utility in your code, you can import the  `ConfigUtility`  class and instantiate it with the path to the  `config_mapping.yml`  file. This will load the appropriate configurations based on the current environment.
from utilities.config_utility import ConfigUtility

config = ConfigUtility("path/to/config_mapping.yml")
