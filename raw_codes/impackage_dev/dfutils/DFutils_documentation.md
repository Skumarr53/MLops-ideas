# DF utils Documentation

## Overview

This NLP project is designed to perform various operations on dataframes, focusing on identifying and managing duplicate text within sentence lists. It utilizes both Pandas and PySpark for data manipulation. The project includes utility classes that provide essential functions for data processing, making it easier to handle and analyze textual data.

## Project Structure

The project consists of the following key modules:

- **DFUtils**: A utility class for operations on Pandas DataFrames.
- **SparkDFUtils**: A utility class for operations on Spark DataFrames.

## Modules

### DFUtils

**File Path**:  `impackage_dev/dfutils/dataframe_utils.py` 

#### Description
The  `DFUtils`  class provides functionalities for manipulating Pandas DataFrames, specifically focusing on duplicate text management in lists of sentences.

#### Key Methods

1. `df_get_percentage_of_duplicate_text_in_sentence_list(currdf)`
   - **Purpose**: Calculates the percentage of duplicate text present in a list of sentences in a DataFrame column.
   - **Parameters**: 
     -  `currdf` : A Pandas DataFrame containing the data.
   - **Returns**: A DataFrame with new columns indicating the percentage of significant duplicates.

2. `df_remove_duplicate_text_from_sentence_list(currdf)`
   - **Purpose**: Removes duplicate text present in a list of sentences in a DataFrame column if the percentage of occurrence exceeds a specified threshold.
   - **Parameters**: 
     -  `currdf` : A Pandas DataFrame containing the data.
   - **Returns**: A DataFrame with duplicates removed based on the defined threshold.

3. `df_column_convert_to_list(currdf, filt_sections_list)`
   - **Purpose**: Converts DataFrame column type from NumPy ndarray to a Python list.
   - **Parameters**: 
     -  `currdf` : A Pandas DataFrame containing the data.
     -  `filt_sections_list` : A list of DataFrame column names to convert.
   - **Returns**: A DataFrame with specified columns converted to lists.

### SparkDFUtils

**File Path**:  `impackage_dev/dfutils/spark_df_utils.py` 

#### Description
The  `SparkDFUtils`  class provides functionalities for manipulating Spark DataFrames, focusing on data type conversions.

#### Key Methods

1. `convert_column_to_date_timestamp_type(spark_df, column, format)`
   - **Purpose**: Converts a Spark DataFrame column to a date-time data type.
   - **Parameters**: 
     -  `spark_df` : A Spark DataFrame containing the data.
     -  `column` : The name of the column to convert.
     -  `format` : The date or date-time format to use for conversion.
   - **Returns**: A Spark DataFrame with the specified column converted to date-time type.

2. `convert_column_to_date_type(spark_df, column, format)`
   - **Purpose**: Converts a Spark DataFrame column to a date data type.
   - **Parameters**: 
     -  `spark_df` : A Spark DataFrame containing the data.
     -  `column` : The name of the column to convert.
     -  `format` : The date or date-time format to use for conversion.
   - **Returns**: A Spark DataFrame with the specified column converted to date type.

3. `cleanup(df_obj)`
   - **Purpose**: Cleans up an object by deleting it and running garbage collection.
   - **Parameters**: 
     -  `df_obj` : The DataFrame object to be cleaned up.
   - **Returns**: None.

## Configuration

Configuration values are stored in the  `utilities/config_utility`  module. This includes settings such as:
- Thresholds for duplicate text detection.
- Sections of the DataFrame to be processed.
