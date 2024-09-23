# Blob and DBFS Utility for Databricks

## Overview

This project provides utility classes for interacting with Azure Blob Storage and the Databricks File System (DBFS). The  `BlobStorageUtility`  class facilitates reading content from text files stored in Azure Blob Storage, while the  `DBFSUtility`  class offers methods for managing files and directories within the Databricks File System. These utilities simplify file operations and enhance the efficiency of data processing workflows in Databricks.

## Project Structure

The project consists of two main utility modules:

- **Blob Storage Utility**:  `impackage_dev/filesystem/blob_storage_utility.py` 
- **DBFS Utility**:  `impackage_dev/filesystem/dbfs_utility.py` 

## BlobStorageUtility

### File Path

 `impackage_dev/filesystem/blob_storage_utility.py` 

### Description

The  `BlobStorageUtility`  class provides functionalities to read content from text files in Azure Blob Storage.

### Key Methods

1. `load_content_from_txt(file_path)`
   - **Purpose**: Reads the entire content of a text file from the specified file path.
   - **Parameters**:
     -  `file_path` : The path to the text file.
   - **Returns**: The content of the text file as a string.
   - **Raises**:  `FilesNotLoadedException`  if the file is not found.

2. `load_list_from_txt(file_path, is_lower=True)`
   - **Purpose**: Reads a text file and returns its content as a set of lines.
   - **Parameters**:
     -  `file_path` : The path to the text file.
     -  `is_lower` : If True, converts the content to lowercase (default is True).
   - **Returns**: A set of lines from the text file.
   - **Raises**: Exception if there is an error reading the file.

3. `read_syllable_count(file_path)`
   - **Purpose**: Reads a file containing words and their syllable counts, returning a dictionary.
   - **Parameters**:
     -  `file_path` : The path to the text file.
   - **Returns**: A dictionary where keys are words and values are their syllable counts.
   - **Raises**:  `FilesNotLoadedException`  if the file is not found.

## DBFSUtility

### File Path

 `impackage_dev/filesystem/dbfs_utility.py` 

### Description

The  `DBFSUtility`  class provides functionalities for managing files and directories in the Databricks File System (DBFS).

### Key Methods

1. `__init__()`
   - **Purpose**: Initializes the DBFSUtility class and sets up the resource path.

2. `get_directory_content(folder_path)`
   - **Purpose**: Returns a list of all subfolders and files within a specified folder.
   - **Parameters**:
     -  `folder_path` : The path of the folder to inspect.
   - **Returns**: A list of paths for all files and subdirectories.

3. `add_directory(file_path, new_dir)`
   - **Purpose**: Creates a new directory in DBFS.
   - **Parameters**:
     -  `file_path` : The path where the new directory should be created.
     -  `new_dir` : The name of the new directory.

4. `remove_file(file_path, file_name)`
   - **Purpose**: Removes a specified file from DBFS.
   - **Parameters**:
     -  `file_path` : The path from where the file should be removed.
     -  `file_name` : The name of the file to be removed.

5. `read_file(file_path, file_name)`
   - **Purpose**: Reads the content of a specified file from DBFS.
   - **Parameters**:
     -  `file_path` : The path of the file.
     -  `file_name` : The name of the file.
   - **Returns**: The content of the file as a string.

6. `write_file(file_path, file_name, content)`
   - **Purpose**: Writes content to a specified file in DBFS.
   - **Parameters**:
     -  `file_path` : The path where the file should be created.
     -  `file_name` : The name of the file.
     -  `content` : The content to write to the file.

7. `load_from_yaml(file_path: str) -> dict`
   - **Purpose**: Loads configuration from a YAML file.
   - **Parameters**:
     -  `file_path` : The file path to the YAML file.
   - **Returns**: A dictionary containing the configuration loaded from the YAML file.

8. `get_name()`
   - **Purpose**: Retrieves the name from the text widget.
   - **Returns**: The name as a string.

9. `__str__()`
   - **Purpose**: Returns a string description of the class.
   - **Returns**: A string describing the class functionality.

