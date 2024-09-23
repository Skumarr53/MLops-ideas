# DataFrame Transformation Utility

## Overview

The DataFrame Transformation Utility is a module designed for efficient handling and transformation of data using Spark DataFrames in a Databricks environment. This utility provides functionalities to convert Pandas DataFrames to Spark DataFrames, define data types, and perform various DataFrame operations. It aims to facilitate data preprocessing and ensure compatibility between Pandas and Spark DataFrames.

## Table of Contents

- [DataFrame Transformation Utility](#dataframe-transformation-utility)
  - [Overview](#overview)
  - [Table of Contents](#table-of-contents)
  - [Module Functions](#module-functions)
    - [DFTransformUtility](#dftransformutility)
      - [Preprocessing Spark DataFrames](#preprocessing-spark-dataframes)
      - [Type Conversion](#type-conversion)
      - [Creating Empty DataFrames](#creating-empty-dataframes)
      - [Filtering Dictionaries](#filtering-dictionaries)


## Module Functions

### DFTransformUtility

The  `DFTransformUtility`  class contains several methods for DataFrame operations.

#### Preprocessing Spark DataFrames

**Method**:  `preprocess_spark_df(currdf, columns_list)` 

- **Parameters**:
  -  `currdf` : The current DataFrame (Pandas DataFrame).
  -  `columns_list` : List of columns to be processed.
- **Returns**: A Spark DataFrame with the specified columns.

This method converts a Pandas DataFrame to a Spark DataFrame, replaces NaN values with None, and performs any necessary preprocessing.

#### Type Conversion

**Method**:  `equivalent_type_fundamentals(string, type)` 

- **Parameters**:
  -  `string` : The column name.
  -  `type` : The data type of the column.
- **Returns**: The equivalent Spark SQL data type.

This method maps Pandas data types to their corresponding Spark SQL types, ensuring compatibility when converting DataFrames.

**Method**:  `define_structure(string, format_type)` 

- **Parameters**:
  -  `string` : The column name.
  -  `format_type` : The data type of the column.
- **Returns**: A Spark SQL StructField object with the column name and type.

This method defines the structure of a Spark DataFrame column based on its name and data type.

#### Creating Empty DataFrames

**Method**:  `create_empty_spark_DF()` 

- **Returns**: An empty Spark DataFrame.

This method creates an empty Spark DataFrame with predefined column names.

#### Filtering Dictionaries

**Method**:  `filter_stats_dictionary(stats_dict)` 

- **Parameters**:
  -  `stats_dict` : A dictionary to be filtered.
- **Returns**: A filtered dictionary with non-None values.

This method filters a dictionary, removing any key-value pairs where the value is None.
