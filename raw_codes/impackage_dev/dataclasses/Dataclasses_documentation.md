# NLP Project Documentation

## Overview

This project implements various functionalities for Natural Language Processing (NLP) using Databricks. The main focus is on processing and analyzing financial transcripts, utilizing machine learning models, and integrating with Snowflake databases.

## Directory Structure

The main configuration files are organized into various data classes that encapsulate the settings and parameters required for different components of the project.

## Configuration Classes

### 1. `kvConfig`
This class holds key-value configurations for the NLP process.

- **Attributes:**
  - `partition_value`: Integer representing the partition value.
  - `duplication_threshold`: Integer for the duplication threshold.
  - `num_labels`: Integer for the number of labels.
  - `no_of_days_files_to_retain`: Integer for the number of days to retain files.
  - `gpu_count`: Integer for the number of GPUs to use.
  - `xlmns`: String for the XML namespace.
  - `timezone`: String representing the timezone.

### 2. `SnowFlakeConfig`
This class manages configurations related to Snowflake databases.

- **Attributes:**
  - `eds_db_prod`: Production database name.
  - `eds_dly_factset_prod`: Daily FactSet production database name.
  - `schema`: Database schema name.
  - `cts_schema`: CTS schema name.
  - `snowflake_cred_pkl_file`: Path to the Snowflake credentials file.
  - `voyatest_snowflake_nonprod`: Non-production Snowflake database name.
  - `voyatest_snowflake_prod`: Production Snowflake database name.
  - `credentials_file`: Path to the credentials file.
  - `FILT_labels`: Dictionary of filter labels.

### 3. `MLflowConfig`
This class stores configurations for MLflow, a platform for managing the ML lifecycle.

- **Attributes:**
  - `mlflow_stages_dict`: Dictionary mapping stages to optional strings.
  - `mlflow_transition_dict`: Dictionary for stage transitions.
  - `mlflow_model_tag_key`: String for the model tag key.
  - `mlflow_FINBERT_model_name`: String for the FINBERT model name.
  - `finbert_model_path`: Path to the FINBERT model.
  - `finbert_experiment_name`: Name of the FINBERT experiment.
  - `finbert_run_name`: Name of the FINBERT run.
  - `finbert_registered_model_name`: Registered name of the FINBERT model.

### 4. `FilePathConfig`
This class defines file paths used throughout the project.

- **Attributes:**
  - `last_parsed_obj_H_path`: Path to the last parsed object.
  - `CT_fundamentals_path`: Path to the fundamentals data.
  - `CT_parquet_file_path`: Path to the parquet file.
  - `CT_legacy_parquet_file_path`: Path to the legacy parquet file.
  - `CT_parsed_parquet_file`: Path to the parsed parquet file.
  - `CT_preprocessed_parquet_file`: Path to the preprocessed parquet file.
  - `CT_TOPICX_parquet_file`: Path to the TOPICX parquet file.
  - `CT_FINBERT_parquet_file`: Path to the FINBERT parquet file.
  - `CT_sentiment_parquet_file`: Path to the sentiment parquet file.
  - `dbfs_resource_path`: DBFS resource path.
  - `dbfs_prefix`: DBFS prefix.
  - `model_artifacts_path`: Path to model artifacts.
  - `model_address`: Address of the model.
  - `CT_legacy_last_parsed_obj_path`: Path to the legacy last parsed object.
  - `preprocessing_words_path`: Path to preprocessing words.
  - `folder_name`: Name of the folder.

### 5. `TableNameConfig`
This class contains names of tables used in the database.

- **Attributes:**
  - `CT_parsed_daily_table`: Name of the parsed daily table.
  - `CT_legacy_daily_table`: Name of the legacy daily table.
  - `CT_sentiment_daily_table`: Name of the sentiment daily table.
  - `CT_parsed_historical_table`: Name of the parsed historical table.
  - `CT_legacy_historical_table`: Name of the legacy historical table.
  - `CT_sentiment_historical_table`: Name of the sentiment historical table.

### 6. `TableColNameConfig`
This class defines column names for various tables.

- **Attributes:**
  - `CT_FINBERT_label_columns`: List of FINBERT label columns.
  - `CT_legacy_aggregate_text_list`: List of legacy aggregate text columns.
  - `CT_legacy_drop_columns_list`: List of columns to drop in legacy data.
  - `CT_legacy_final_CT_output_columns`: List of final output columns for legacy data.
  - `CT_legacy_FINBERT_DF_merge_on_columns`: Columns for merging legacy FINBERT DataFrames.
  - `CT_legacy_FINBERT_drop_columns_list`: Columns to drop in legacy FINBERT data.
  - `CT_legacy_NLP_DF_merge_on_columns`: Columns for merging legacy NLP DataFrames.
  - `CTS_combined_scores_columns`: Columns for combined scores.
  - `CTS_FINBERT_labels_columns`: Columns for FINBERT labels.
  - `CTS_FINBERT_labels_columns_drop`: Columns to drop for FINBERT labels.
  - `CTS_FINBERT_specific_scores_columns`: Columns for specific FINBERT scores.
  - `discussion_labels`: List of discussion labels.
  - `FILT_sections`: List of filter sections.

### 7. `BlobFilenameConfig`
This class defines filenames for various blob resources.

- **Attributes:**
  - `litigious_flnm`: Filename for litigious words.
  - `complex_flnm`: Filename for complexity words.
  - `uncertianity_flnm`: Filename for uncertainty words.
  - `syllable_flnm`: Filename for syllable counts.
  - `vocab_pos_flnm`: Filename for positive vocabulary.
  - `vocab_neg_flnm`: Filename for negative vocabulary.
  - `contraction_flnm`: Filename for contraction words.
  - `stop_words_flnm`: Filename for stop words.
  - `negate_words_flnm`: Filename for negated words.

### 8. `TranscriptVarsData`
This class holds variables related to transcript data.

- **Attributes:**
  - `corp_representative_ids`: List of corporate representative IDs.
  - `analyst_ids`: List of analyst IDs.
  - `company_name`: List of company names.
  - `company_id`: List of company IDs.
  - `ceo_speech`: List of CEO speeches.
  - `executive_speech`: List of executive speeches.
  - `ceo_answers`: List of CEO answers.
  - `executive_answers`: List of executive answers.
  - `analysts_questions`: List of analyst questions.
  - `ceo_id`: List of CEO IDs.
  - `ceo_unique_id`: List of unique CEO IDs.
  - `ceo_matches`: Integer for CEO matches.
  - `number_of_analysts`: Integer for the number of analysts.
  - `earnings_call`: Integer indicating earnings call status.

## Getting Started

Update the configuration files with your specific settings, if needed. The relevant configuration values are located in the  `utilities/env_config`  folder.
