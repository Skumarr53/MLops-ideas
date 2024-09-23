# XML Parser for NLP Project

## Overview

This project includes an XML parser designed to extract and process call transcript data from XML files. The parser is built using Python and is intended for use within a Databricks environment. The  `XMLParser`  class provides functionalities to handle the parsing of XML data, manage the extracted information, and integrate with existing dataframes for further analysis.

## Project Structure

The primary module for XML parsing is located in:

- **XML Parser**:  `impackage_dev/ETL/XML_parser.py` 

### Key Features

- **Date Filtering**: The parser can filter data based on specified minimum and maximum dates.
- **Data Extraction**: It extracts metadata, participant information, and discussions from XML data.
- **Error Handling**: The parser includes mechanisms to handle XML parsing errors gracefully.
- **Timezone Management**: It handles timezones to ensure accurate date and time representation.

## Class: XMLParser

### File Path

 `impackage_dev/ETL/XML_parser.py` 

### Description

The  `XMLParser`  class is responsible for parsing XML data and extracting relevant information related to call transcripts. It initializes with a set of parameters, including date ranges, current and historical data frames, and configuration settings.

### Key Methods

1. ** `__init__(self, min_date: datetime, max_date: datetime, new_data_dims: Tuple[int, int], curr_df: pd.DataFrame, hist_df: pd.DataFrame, config, run_mode: str = 'daily')` **
   - **Purpose**: Initializes the XMLParser instance with the required parameters.
   - **Parameters**:
     -  `min_date` : Minimum date for filtering data.
     -  `max_date` : Maximum date for filtering data.
     -  `new_data_dims` : Dimensions of the new data.
     -  `curr_df` : Current DataFrame containing the latest data.
     -  `hist_df` : Historical DataFrame containing past data.
     -  `config` : Configuration object containing various settings.
     -  `run_mode` : Mode of operation, either 'daily' or 'historical'.

2. ** `get_last_upload_dt(self)` **
   - **Purpose**: Retrieves the last upload datetime from the current or historical DataFrame.
   - **Returns**: The last upload datetime as a  `datetime`  object.

3. ** `get_parsed_df(self)` **
   - **Purpose**: Parses the current DataFrame into a new DataFrame with extracted XML data.
   - **Returns**: A DataFrame containing parsed XML data.

4. ** `get_last_pasred_time(self)` **
   - **Purpose**: Gets the last parsed time in the local timezone.
   - **Returns**: The last parsed time as a  `datetime`  object.

5. ** `utc_to_local(self, utc_dt: datetime, tz_var: pytz.timezone) -> datetime` **
   - **Purpose**: Converts a UTC datetime to the local timezone.
   - **Parameters**:
     -  `utc_dt` : UTC datetime to convert.
     -  `tz_var` : Timezone variable to convert to.
   - **Returns**: Local datetime.

6. ** `extract_metadata(self, root)` **
   - **Purpose**: Extracts metadata from the XML root element.
   - **Parameters**:
     -  `root` : The root element of the XML tree.
   - **Returns**: A tuple containing status, title, ID, and date.

7. ** `parse_participants(self, root)` **
   - **Purpose**: Parses participant information from the XML root element.
   - **Parameters**:
     -  `root` : The root element of the XML tree.

8. ** `parse_sections(self, root)` **
   - **Purpose**: Parses sections from the XML root element.
   - **Parameters**:
     -  `root` : The root element of the XML tree.

9. ** `parse_company_id(self, root)` **
   - **Purpose**: Parses the company ID from the XML root element.
   - **Parameters**:
     -  `root` : The root element of the XML tree.

10. ** `parse_ct_xml(self, xml_list_row: List[Any])` **
    - **Purpose**: Parses XML strings stored in Snowflake.
    - **Parameters**:
      -  `xml_list_row` : Call transcript XML row data.
    - **Returns**: A dictionary object with call transcript discussion details.

### Error Handling

The  `XMLParser`  class includes error handling for XML parsing. If a parsing error occurs, it logs the error and returns an empty transcript dictionary.

## Configuration

The parser relies on a configuration object that contains various settings, including:
- Timezone information.
- XML namespace configurations.