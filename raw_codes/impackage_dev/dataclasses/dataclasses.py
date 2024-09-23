# Databricks notebook source
import csv
from dataclasses import dataclass, field
from typing import Dict, Optional, List



@dataclass
class kvConfig:
    partition_value: int
    duplication_threshold: int
    num_labels: int
    no_of_days_files_to_retain: int
    gpu_count: int
    xlmns: str
    timezone: str

@dataclass
class SnowFlakeConfig:
    eds_db_prod: str
    eds_dly_factset_prod: str
    schema: str
    cts_schema: str
    snowflake_cred_pkl_file: str
    voyatest_snowflake_nonprod: str
    voyatest_snowflake_prod: str
    credentials_file: str
    FILT_labels: Dict[str, str] = field(default_factory=dict)

@dataclass
class MLflowConfig:
    mlflow_stages_dict: Dict[str, Optional[str]] = field(default_factory=dict)
    mlflow_transition_dict: Dict[str, Optional[str]] = field(default_factory=dict)
    mlflow_model_tag_key: str = ""
    mlflow_FINBERT_model_name: str = ""
    finbert_model_path: str = ""
    finbert_experiment_name: str = ""
    finbert_run_name: str = ""
    finbert_registered_model_name: str = ""

@dataclass
class FilePathConfig:
    last_parsed_obj_H_path: str
    CT_fundamentals_path: str
    CT_parquet_file_path: str
    CT_legacy_parquet_file_path: str
    CT_parsed_parquet_file: str
    CT_preprocessed_parquet_file: str
    CT_TOPICX_parquet_file: str
    CT_FINBERT_parquet_file: str
    CT_sentiment_parquet_file: str
    dbfs_resource_path: str
    dbfs_prefix: str
    model_artifacts_path: str
    model_address: str
    CT_legacy_last_parsed_obj_path: str
    preprocessing_words_path: str
    folder_name: str


@dataclass
class TableNameConfig:
    CT_parsed_daily_table: str
    CT_legacy_daily_table: str
    CT_sentiment_daily_table: str
    CT_parsed_historical_table: str
    CT_legacy_historical_table: str
    CT_sentiment_historical_table: str

@dataclass
class TableColNameConfig:
    CT_FINBERT_label_columns: List[str] = field(default_factory=list)
    CT_legacy_aggregate_text_list: List[str] = field(default_factory=list)
    CT_legacy_drop_columns_list: List[str] = field(default_factory=list)
    CT_legacy_final_CT_output_columns: List[str] = field(default_factory=list)
    CT_legacy_FINBERT_DF_merge_on_columns: List[str] = field(default_factory=list)
    CT_legacy_FINBERT_drop_columns_list: List[str] = field(default_factory=list)
    CT_legacy_NLP_DF_merge_on_columns: List[str] = field(default_factory=list)
    CTS_combined_scores_columns: List[str] = field(default_factory=list)
    CTS_FINBERT_labels_columns: List[str] = field(default_factory=list)
    CTS_FINBERT_labels_columns_drop: List[str] = field(default_factory=list)
    CTS_FINBERT_specific_scores_columns: List[str] = field(default_factory=list)
    discussion_labels: List[str] = field(default_factory=list)
    FILT_sections: List[str] = field(default_factory=list)

@dataclass(frozen=True)
class BlobFilenameConfig:
  litigious_flnm: str =  "litigious_words.txt"
  complex_flnm: str =  "complexity_words.txt"
  uncertianity_flnm: str =  "uncertainty_words.txt"
  syllable_flnm: str = "syllable_count.txt"
  vocab_pos_flnm: str = "vocab_pos.txt"
  vocab_neg_flnm: str = "vocab_neg.txt"
  contraction_flnm: str =  "contraction_words.txt"
  stop_words_flnm: str = "StopWords_Full_list.txt"
  negate_words_flnm: str = "negated_words.txt"

@dataclass
class TranscriptVarsData:
    corp_representative_ids: List[str] = field(default_factory=list)
    analyst_ids: List[str] = field(default_factory=list)
    company_name: List[str] = field(default_factory=list)
    company_id: List[str] = field(default_factory=list)
    ceo_speech: List[str] = field(default_factory=list)
    executive_speech: List[str] = field(default_factory=list)
    ceo_answers: List[str] = field(default_factory=list)
    executive_answers: List[str] = field(default_factory=list)
    analysts_questions: List[str] = field(default_factory=list)
    ceo_id: List[str] = field(default_factory=list)
    ceo_unique_id: List[str] = field(default_factory=list)
    ceo_matches: int = 0
    number_of_analysts: int = 0
    earnings_call: int = 1
