# WorKFlow:
Certainly! Let's thoroughly review your data pipeline by addressing each of your specified points. This comprehensive analysis will help you understand the workflow's objectives, design, step-by-step logic, simplify the process for non-technical stakeholders, and highlight the utility of the final results.

---

## 1. **Problem Statement and Purpose/Aim of the Workflow**

### **Problem Statement**
In the realm of financial analysis, understanding the dynamics of various companies over time is crucial. Specifically, analyzing the **coverage rate** (how extensively a topic is discussed) and **relevance score** (how pertinent a topic is) across different companies and market capitalization groups can provide valuable insights. However, processing large volumes of transcripts from earnings calls to compute these metrics manually is time-consuming and error-prone.

### **Purpose/Aim**
The primary aim of this workflow is to **automate the calculation and visualization of coverage rates and relevance scores** for various topics extracted from company transcripts over time. By doing so, it enables analysts to:

- **Monitor trends**: Observe how the discussion of specific topics evolves over a defined period.
- **Compare across segments**: Analyze differences in topic relevance and coverage among companies of varying market capitalizations.
- **Support decision-making**: Provide data-driven insights that can inform investment strategies, market analysis, and strategic planning.

---

## 2. **Design Flow of the Project**

The workflow is designed as a **Databricks notebook** that performs the following high-level operations:

1. **Setup and Configuration**:
   - Load necessary packages and utilities.
   - Establish connections to Snowflake and Azure SQL databases.

2. **Data Extraction**:
   - Retrieve market capitalization data from Azure SQL.
   - Fetch transcripts data from Snowflake within a specified date range.

3. **Data Preprocessing**:
   - Clean and deduplicate transcript data.
   - Process market capitalization data to categorize companies.

4. **Metric Computation**:
   - Calculate coverage rates and relevance scores using rolling averages.
   - Segment data based on market capitalization groups and industries.

5. **Data Visualization**:
   - Generate time series line plots to visualize trends in coverage and relevance.

6. **Data Storage**:
   - Save the computed metrics back to Snowflake for downstream consumption.

---

## 3. **Step-by-Step Logic with Code Snippets and Descriptions**

Let's break down the workflow into discrete steps, providing code snippets and detailed explanations for each.

### **Step 1: Setup and Configuration**

#### **1.1. Load Required Packages and Utilities**

```python
# Load custom utilities
# MAGIC %run ./../../../data-science-nlp-ml-common-code/impackage/utilities/config_utility
# MAGIC %run ./../../../data-science-nlp-ml-common-code/impackage/database/snowflake_dbutility
```

**Description**:
- **Purpose**: Incorporate custom utility modules that handle configuration settings and Snowflake database interactions.
- **Elaboration**: 
  - `config_utility`: Likely contains configuration parameters such as database credentials, schema names, etc.
  - `snowflake_dbutility`: Contains functions to connect and interact with Snowflake databases.

#### **1.2. Initialize Snowflake Utility**

```python
new_sf = SnowFlakeDBUtility(config.schema, config.eds_db_prod)
```

**Description**:
- **Purpose**: Instantiate a Snowflake utility object with specific schema and production database configurations.
- **Elaboration**: This object (`new_sf`) will be used for subsequent database operations like reading and writing data.

#### **1.3. Import Standard Libraries**

```python
import pandas as pd
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta
from ast import literal_eval
from collections import Counter
from pyspark.sql.types import *
import warnings
warnings.filterwarnings("ignore")
```

**Description**:
- **Purpose**: Import essential Python libraries for data manipulation (`pandas`, `numpy`), date operations, and Spark data types.
- **Elaboration**: Suppress warnings to ensure cleaner notebook outputs.

---

### **Step 2: Connect with SQL Database and Retrieve Market Cap Data**

#### **2.1. Load Azure SQL Module**

```python
# MAGIC %run "./../../package_loader/Cloud_DB_module_Azure_SQL_dev_2_Yujing_git.py"
```

**Description**:
- **Purpose**: Incorporate a module that facilitates connections to Azure SQL databases.
- **Elaboration**: This module likely contains classes or functions to authenticate and execute queries against Azure SQL.

#### **2.2. Initialize Azure SQL Helper and Load Configuration**

```python
myDBFS_sql = DBFShelper_sql()
myDBFS_sql.get_DBFSdir_content(myDBFS_sql.iniPath)
```

**Description**:
- **Purpose**: Instantiate a helper object to interact with Azure SQL and retrieve configuration files from DBFS (Databricks File System).
- **Elaboration**: `iniPath` likely points to configuration files containing database connection details.

#### **2.3. Load and Read Market Cap Data**

```python
azSQL_LinkUp = pd.read_pickle(r'/dbfs/' + myDBFS_sql.iniPath + 'my_azSQL_LinkUp.pkl')
azSQL_LinkUp.databaseName = 'QNT'
remote_table = azSQL_LinkUp.read_from_azure_SQL("qnt.p_coe.earnings_calls_mapping_table_mcap")
market_cap  = remote_table.toPandas()
```

**Description**:
- **Purpose**: 
  - Load a pickled Azure SQL connection object.
  - Set the target database (`'QNT'`).
  - Execute a SQL query to fetch market capitalization data.
- **Elaboration**: 
  - The `read_from_azure_SQL` function retrieves data from the specified table.
  - Convert the Spark DataFrame (`remote_table`) to a pandas DataFrame (`market_cap`) for easier manipulation.

---

### **Step 3: Read Data from Snowflake**

#### **3.1. Define Date Range for Data Extraction**

```python
# Read start & end ranges. Note that the range does is NOT inclusive of end month; i.e the range ends at the beginning of the end month
minDateNewQuery = (pd.to_datetime(dbutils.widgets.get("Start Date"))).strftime('%Y-%m-%d')
maxDateNewQuery = (pd.to_datetime(dbutils.widgets.get("End Date"))).strftime('%Y-%m-%d')

mind = "'" + minDateNewQuery + "'"
maxd = "'" + maxDateNewQuery + "'"

print('The next query spans ' + mind + ' to ' + maxd)
```

**Description**:
- **Purpose**: 
  - Retrieve start and end dates from notebook widgets (user inputs).
  - Format dates to `YYYY-MM-DD` for SQL querying.
- **Elaboration**: 
  - The date range is used to filter transcripts data within the specified period.

#### **3.2. Query Transcripts Data from Snowflake**

```python
tsQuery= ("SELECT * "
    "FROM  EDS_PROD.QUANT.YUJING_MASS_FT_NLI_DEMAND_DEV_1 "
          
   "WHERE DATE >= " + mind  + " AND DATE <= " + maxd  + " ;")

resultspkdf = new_sf.read_from_snowflake(tsQuery)

currdf = resultspkdf.toPandas()
```

**Description**:
- **Purpose**: 
  - Construct and execute a SQL query to fetch all records from the specified Snowflake table within the date range.
  - Convert the resulting Spark DataFrame (`resultspkdf`) to a pandas DataFrame (`currdf`).
- **Elaboration**: The table `YUJING_MASS_FT_NLI_DEMAND_DEV_1` likely contains processed transcripts with relevant metrics.

#### **3.3. Validate Retrieved Data**

```python
if len(currdf)>0:
    print('The data spans from ' + str(currdf['DATE'].min()) + ' to ' + str(currdf['DATE'].max()) + 'and has ' + str(currdf.shape[0]) + ' rows and ' + str(currdf.shape[1]) + ' columns.')
else:
    print('No new transcripts to parse.')
    dbutils.notebook.exit(1)
    os._exit(1)
```

**Description**:
- **Purpose**: 
  - Check if any data was retrieved.
  - If no data is found, terminate the notebook execution gracefully.
- **Elaboration**: Ensures that subsequent steps have data to process, preventing errors.

---

### **Step 4: Data Preprocessing**

#### **4.1. Deduplicate Transcript Data**

```python
# Keep the earliest version of transcript from the same date.
currdf = currdf.sort_values(['ENTITY_ID', 'DATE','VERSION_ID']).drop_duplicates(['ENTITY_ID', 'DATE'], keep = 'first' )
```

**Description**:
- **Purpose**: Remove duplicate transcripts for the same company (`ENTITY_ID`) and date, retaining only the earliest version (`VERSION_ID`).
- **Elaboration**: Ensures data quality by avoiding redundant entries.

#### **4.2. Convert DATE Column to Date Object**

```python
currdf['DATE'] = currdf['DATE'].dt.date
```

**Description**:
- **Purpose**: 
  - Convert the `DATE` column from a datetime object to a date object.
- **Elaboration**: Simplifies date handling and comparisons in subsequent steps.

#### **4.3. Create Coverage Columns**

```python
for col in currdf.filter(like = 'COUNT'):
  currdf[col[:-13] + 'COVERAGE' + col[-8:]] = currdf[col].apply(lambda x: 1 if x>=1 else 0)
```

**Description**:
- **Purpose**: 
  - For each column containing the substring `'COUNT'`, create a corresponding `'COVERAGE'` column.
  - The new column is binary: `1` if the count is >=1, else `0`.
- **Elaboration**: Transforms raw counts into binary indicators to represent the presence or absence of coverage.

---

### **Step 5: Preprocessing Market Cap Data**

#### **5.1. Clean Market Cap Data**

```python
market_cap = market_cap.sort_values(by=['factset_entity_id','date']).drop_duplicates(['factset_entity_id', 'YearMonth'], keep='last')
```

**Description**:
- **Purpose**: 
  - Sort the market cap data by `factset_entity_id` and `date`.
  - Remove duplicate entries for the same company and month (`YearMonth`), keeping the latest record.
- **Elaboration**: Ensures that each company has only one market cap entry per month.

#### **5.2. Create YEAR_MONTH and MCAP_RANK Columns**

```python
market_cap['YEAR_MONTH'] = pd.to_datetime(market_cap['date'], format='%Y-%m-%d').apply(lambda x: str(x)[:7])
market_cap['MCAP_RANK'] = market_cap.groupby('YEAR_MONTH')['MCAP'].rank(ascending=False, method='first')
```

**Description**:
- **Purpose**: 
  - Extract the year and month from the `date` column.
  - Rank companies within each `YEAR_MONTH` based on their market capitalization (`MCAP`), with rank `1` being the highest.
- **Elaboration**: Facilitates segmentation of companies into different market cap groups for analysis.

---

### **Step 6: Define Helper Functions**

#### **6.1. Classification Functions**

```python
def classify_rank(rank):
  if rank <= 100:
    return '1-Mega firms (top 100)'
  elif rank <= 1000:
    return 'Large firms (top 100 - 1000)'
  else:
    return 'Small firms (1000+)'

def classify_rank_C2(rank):
  if rank <= 1000:
    return 'top 1-1000'
  else:
    return 'top 1001-3000'

def classify_rank_C3(rank):
  if rank <= 500:
    return 'top 1-500'
  elif rank <= 900:
    return 'top 501-900'
  elif rank <= 1500:
    return 'top 901-1500'
  else:
    return 'top 1501-3000'
```

**Description**:
- **Purpose**: 
  - Categorize companies into different market cap groups based on their `MCAP_RANK`.
- **Elaboration**: Different classification schemes (`C1`, `C2`, `C3`) are used to create varied segmentation levels for in-depth analysis.

#### **6.2. DataFrame Creation Functions**

```python
def create_plot_df(df_raw, start_date, end_date):
    # Function logic...
    return df_mean

def create_plot_df_coverage_rate(df_raw, start_date, end_date):
    # Function logic...
    return df_mean
```

**Description**:
- **Purpose**: 
  - Compute the rolling average of relevance scores and coverage rates over a 120-day window.
- **Elaboration**: 
  - **Relevance Score**: Indicates how pertinent a topic is in the transcripts.
  - **Coverage Rate**: Represents the extent to which a topic is covered.

#### **6.3. Auxiliary Functions for Data Conversion**

```python
def equivalent_type(string, f):
    # Function logic...
    return appropriate_Spark_type

def define_structure(string, format_type):
    typo = equivalent_type(string, format_type)
    print(typo)
    return StructField(string, typo)

def pandas_to_spark(pandas_df):
    # Function logic...
    return spark_dataframe
```

**Description**:
- **Purpose**: 
  - Convert pandas DataFrames to Spark DataFrames with appropriate data types.
- **Elaboration**: Facilitates the storage of processed data back into Snowflake, which may require Spark DataFrame format.

---

### **Step 7: Time Series Line Plot Computation**

The workflow computes time series metrics across various market capitalization groups and industries. This involves segmenting the data, calculating coverage and relevance, and aggregating the results.

#### **7.1. Market Cap Group 1: Top 1-3000**

```python
market_cap['MCAP_GROUP'] = market_cap['MCAP_RANK'].apply(lambda x: classify_rank(x))
currdf['YEAR_MONTH'] = currdf['DATE'].apply(lambda x: str(x)[:7])
currdf_merge = pd.merge(market_cap[['YEAR_MONTH', 'MCAP_GROUP', 'factset_entity_id','MCAP', 'biz_group']], currdf,  how='left', left_on=['factset_entity_id','YEAR_MONTH'], right_on = ['ENTITY_ID','YEAR_MONTH'])
currdf_R3K = currdf_merge[~currdf_merge.CALL_ID.isna()]
currdf_R3K_mega = currdf_R3K[currdf_R3K['MCAP_GROUP'] == '1-Mega firms (top 100)' ]
currdf_R3K_large = currdf_R3K[currdf_R3K['MCAP_GROUP'] == 'Large firms (top 100 - 1000)' ]
currdf_R3K_small = currdf_R3K[currdf_R3K['MCAP_GROUP'] == 'Small firms (1000+)' ]

R3K_dfs = [currdf_R3K, currdf_R3K_mega, currdf_R3K_large, currdf_R3K_small ]
R3K_categories = ['top 1-3000', 'top 1-100', 'top 101-1000', 'top 1001-3000']

df_rel_cover_C1 = pd.DataFrame([])
for i, df in enumerate(R3K_dfs):
  df_cover = create_plot_df_coverage_rate(df, 
                start_date = minDateNewQuery, 
                end_date = maxDateNewQuery)
  df_rel = create_plot_df(df, 
                start_date = minDateNewQuery, 
                end_date = maxDateNewQuery)
  df_cover.reset_index(inplace=True)
  df_rel.reset_index(inplace=True)
  df_rel_cover = pd.merge(df_cover, df_rel, left_on='DATE_ROLLING', right_on='DATE_ROLLING')
  df_rel_cover['CATEGORY'] = R3K_categories[i]
  df_rel_cover_C1 = pd.concat([df_rel_cover_C1 , df_rel_cover])
```

**Description**:
- **Purpose**: 
  - Categorize data into `Mega`, `Large`, and `Small` firms based on market cap ranks.
  - Compute coverage rates and relevance scores for each category over the specified date range.
- **Elaboration**:
  - **Segmentation**: 
    - `currdf_R3K_mega`: Top 100 companies.
    - `currdf_R3K_large`: Companies ranked 101-1000.
    - `currdf_R3K_small`: Companies ranked 1001-3000.
  - **Computation**:
    - For each segment, calculate the rolling average of coverage rates and relevance scores.
    - Merge these metrics into a consolidated DataFrame (`df_rel_cover_C1`).

#### **7.2. Market Cap Group 2: Top 1-1000**

```python
market_cap['MCAP_GROUP'] = market_cap['MCAP_RANK'].apply(lambda x: classify_rank_C2(x))
currdf_merge = pd.merge(market_cap[['YEAR_MONTH', 'MCAP_GROUP', 'factset_entity_id','MCAP', 'biz_group']], currdf,  how='left', left_on=['factset_entity_id','YEAR_MONTH'], right_on = ['ENTITY_ID','YEAR_MONTH'])
currdf_R3K = currdf_merge[~currdf_merge.CALL_ID.isna()]
print(currdf_R3K['MCAP_GROUP'].unique())
currdf_R3K_C2_1 = currdf_R3K[currdf_R3K['MCAP_GROUP'] == 'top 1-1000' ]

R3K_dfs = [currdf_R3K_C2_1]
R3K_categories = ['top 1-1000']

df_rel_cover_C2 = pd.DataFrame([])
for i, df in enumerate(R3K_dfs):
  df_cover = create_plot_df_coverage_rate(df, 
                start_date = minDateNewQuery, 
                end_date = maxDateNewQuery)
  df_rel = create_plot_df(df, 
                start_date = minDateNewQuery, 
                end_date = maxDateNewQuery)
  df_cover.reset_index(inplace=True)
  df_rel.reset_index(inplace=True)
  df_rel_cover = pd.merge(df_cover, df_rel, left_on='DATE_ROLLING', right_on='DATE_ROLLING')
  df_rel_cover['CATEGORY'] = R3K_categories[i]
  df_rel_cover_C2 = pd.concat([df_rel_cover_C2 , df_rel_cover])
```

**Description**:
- **Purpose**: 
  - Re-categorize companies into broader groups (`top 1-1000`).
  - Compute corresponding coverage and relevance metrics.
- **Elaboration**:
  - Simplifies the analysis by focusing on the top 1000 companies.
  - Aggregates metrics into `df_rel_cover_C2`.

#### **7.3. Market Cap Group 3: Top 1-1500**

```python
market_cap['MCAP_GROUP'] = market_cap['MCAP_RANK'].apply(lambda x: classify_rank_C3(x))
currdf_merge = pd.merge(market_cap[['YEAR_MONTH', 'MCAP_GROUP', 'factset_entity_id','MCAP', 'biz_group']], currdf,  how='left', left_on=['factset_entity_id','YEAR_MONTH'], right_on = ['ENTITY_ID','YEAR_MONTH'])
currdf_R3K = currdf_merge[~currdf_merge.CALL_ID.isna()]
print(currdf_R3K['MCAP_GROUP'].unique())
currdf_R3K_C3_1 = currdf_R3K[currdf_R3K['MCAP_GROUP'] == 'top 1-500' ]
currdf_R3K_C3_2 = currdf_R3K[currdf_R3K['MCAP_GROUP'] == 'top 501-900' ]
currdf_R3K_C3_3 = currdf_R3K[currdf_R3K['MCAP_GROUP'] == 'top 901-1500' ]

R3K_dfs = [currdf_R3K_C3, currdf_R3K_C3_1, currdf_R3K_C3_2, currdf_R3K_C3_3]
R3K_categories = ['top 1-1500', 'top 1-500', 'top 501-900', 'top 901-1500']

df_rel_cover_C3 = pd.DataFrame([])
for i, df in enumerate(R3K_dfs):
  df_cover = create_plot_df_coverage_rate(df, 
                start_date = minDateNewQuery, 
                end_date = maxDateNewQuery)
  df_rel = create_plot_df(df, 
                start_date = minDateNewQuery, 
                end_date = maxDateNewQuery)
  df_cover.reset_index(inplace=True)
  df_rel.reset_index(inplace=True)
  df_rel_cover = pd.merge(df_cover, df_rel, left_on='DATE_ROLLING', right_on='DATE_ROLLING')
  df_rel_cover['CATEGORY'] = R3K_categories[i]
  df_rel_cover_C3 = pd.concat([df_rel_cover_C3 , df_rel_cover])
```

**Description**:
- **Purpose**: 
  - Further segment companies into smaller groups (`top 1-500`, `top 501-900`, `top 901-1500`).
  - Compute corresponding coverage and relevance metrics.
- **Elaboration**:
  - Provides a granular view of different company sizes.
  - Aggregates metrics into `df_rel_cover_C3`.

#### **7.4. Industry Group Segmentation**

```python
df_all_gp = pd.DataFrame([])
for gp in currdf_R3K['biz_group'].unique():
  df_cover_R3K_gp = create_plot_df_coverage_rate(currdf_R3K[currdf_R3K['biz_group'] == gp], 
              start_date = minDateNewQuery, 
              end_date = maxDateNewQuery)
  df_rel_R3K_gp = create_plot_df(currdf_R3K[currdf_R3K['biz_group'] == gp], 
              start_date = minDateNewQuery, 
              end_date = maxDateNewQuery)
  df_cover_R3K_gp.reset_index(inplace=True)
  df_rel_R3K_gp.reset_index(inplace=True)
  df_rel_cover_R3K_gp = pd.merge(df_cover_R3K_gp, df_rel_R3K_gp, left_on='DATE_ROLLING', right_on='DATE_ROLLING')
  df_rel_cover_R3K_gp['CATEGORY'] = gp
  df_all_gp = pd.concat([df_all_gp, df_rel_cover_R3K_gp])
```

**Description**:
- **Purpose**: 
  - Segment data based on industry groups (`biz_group`).
  - Compute coverage and relevance metrics for each industry.
- **Elaboration**: Allows for industry-specific trend analysis, offering insights into how different sectors engage with various topics.

#### **7.5. Combine All Results**

```python
df_rel_cover_all = pd.concat([df_rel_cover_C1, df_rel_cover_C2, df_rel_cover_C3, df_all_gp])
```

**Description**:
- **Purpose**: 
  - Consolidate all computed metrics across different market cap groups and industries into a single DataFrame (`df_rel_cover_all`).
- **Elaboration**: Facilitates unified visualization and storage of all relevant metrics.

#### **7.6. Final Formatting and Storage Path Definition**

```python
df_rel_cover_all['DATE_ROLLING'] = pd.to_datetime(df_rel_cover_all['DATE_ROLLING'])
df_rel_cover_all = df_rel_cover_all[['DATE_ROLLING', 'COMPANY_COUNT', 'CONSUMER_STRENGTH_COVERRATE_FILT_MD',
       'CONSUMER_WEAKNESS_COVERRATE_FILT_MD',
       'CONSUMER_REDUCED_SPENDING_PATTERNS_COVERRATE_FILT_MD',
       'CONSUMER_STRENGTH_COVERRATE_FILT_QA',
       'CONSUMER_WEAKNESS_COVERRATE_FILT_QA',
       'CONSUMER_REDUCED_SPENDING_PATTERNS_COVERRATE_FILT_QA',
       'CONSUMER_REDUCED_SPENDING_PATTERNS_COVERRATE_FILT_AVERAGE',
       'CONSUMER_STRENGTH_COVERRATE_FILT_AVERAGE',
       'CONSUMER_WEAKNESS_COVERRATE_FILT_AVERAGE',
       'CONSUMER_STRENGTH_REL_FILT_MD', 'CONSUMER_WEAKNESS_REL_FILT_MD',
       'CONSUMER_REDUCED_SPENDING_PATTERNS_REL_FILT_MD',
       'CONSUMER_STRENGTH_REL_FILT_QA', 'CONSUMER_WEAKNESS_REL_FILT_QA',
       'CONSUMER_REDUCED_SPENDING_PATTERNS_REL_FILT_QA',
       'CONSUMER_WEAKNESS_REL_FILT_AVERAGE',
       'CONSUMER_REDUCED_SPENDING_PATTERNS_REL_FILT_AVERAGE',
       'CONSUMER_STRENGTH_REL_FILT_AVERAGE', 'CATEGORY']]

output_path = "/dbfs/mnt/access_work/UC25/Topic Modeling/NLI Models/MASS_Demand/MASS_FT_Demand_rel_coverage_git_" + maxDateNewQuery +".csv"
df_rel_cover_all.to_csv(output_path, index=False)
```

**Description**:
- **Purpose**: 
  - Ensure the `DATE_ROLLING` column is in datetime format.
  - Select and order relevant columns for the final output.
  - Define the output file path and save the consolidated DataFrame as a CSV.
- **Elaboration**: Prepares the data for storage and further visualization or analysis.

---

### **Step 8: Store Time Series Data into Snowflake**

#### **8.1. Convert Pandas DataFrame to Spark DataFrame**

```python
spark_parsedDF = pandas_to_spark(df_rel_cover_all)
spark_parsedDF = spark_parsedDF.replace(np.nan, None)
spark_parsedDF = spark_parsedDF.withColumn("DATE_ROLLING", F.to_timestamp(spark_parsedDF.DATE_ROLLING, 'yyyy-MM-dd'))                                                                      
```

**Description**:
- **Purpose**: 
  - Convert the final pandas DataFrame to a Spark DataFrame.
  - Replace `NaN` values with `None` to handle missing data.
  - Ensure the `DATE_ROLLING` column is in timestamp format.
- **Elaboration**: Spark DataFrames are compatible with Snowflake for efficient data loading.

#### **8.2. Write Data to Snowflake Table**

```python
new_sf.db = 'EDS_PROD'
new_sf.schema = 'QUANT'
     
tablename_curr = 'YUJING_MASS_FT_DEMAND_TS_DEV_1'
result_curr = new_sf.write_to_snowflake_table(spark_parsedDF, tablename_curr)
```

**Description**:
- **Purpose**: 
  - Configure the target database and schema in Snowflake.
  - Write the processed time series data into the specified Snowflake table.
- **Elaboration**: Enables downstream applications and analysts to access the computed metrics directly from Snowflake.

---

## 4. **Explain the Process to a Non-Technical Person**

Imagine you're an analyst at a financial firm tasked with understanding how different topics are discussed in company earnings calls over time. To do this manually would mean listening to countless transcripts, taking notes, and trying to quantify how much each topic is covered and how important it is. This is not only tedious but also prone to mistakes.

### **What This Workflow Does**

1. **Gathering Information**:
   - **Companies' Financial Data**: It first collects data about companies, specifically their market size (market capitalization), from a secure database.
   - **Transcripts of Earnings Calls**: Next, it retrieves transcripts of these companies' earnings calls over a specified period.

2. **Cleaning the Data**:
   - **Removing Duplicates**: If a company has multiple transcripts on the same day, it keeps only the first one to avoid repetition.
   - **Preparing Data for Analysis**: It organizes the data to make sure dates are correctly formatted and sets up indicators to show whether a topic was discussed.

3. **Analyzing Topics**:
   - **Grouping Companies**: Companies are categorized based on their size (e.g., top 100, 100-1000, etc.).
   - **Calculating Metrics**:
     - **Coverage Rate**: How frequently a topic is discussed.
     - **Relevance Score**: How important or pertinent a topic is during discussions.
   - **Rolling Averages**: It looks at these metrics over a 120-day period to identify trends.

4. **Visualizing Trends**:
   - The workflow compiles all this information into charts that show how discussions around various topics change over time for different groups of companies.

5. **Storing Results**:
   - Finally, all the processed data and visualizations are saved in a central database, making it easy for analysts to access and interpret the results.

### **Why This Matters**

By automating this process, analysts can quickly and accurately see which topics are gaining or losing attention, compare how different sized companies discuss these topics, and make informed decisions based on these insights. This could influence investment strategies, identify emerging trends, and provide a competitive edge in understanding market dynamics.

---

## 5. **Utility of Final Results for End Users**

The final results, which include calculated coverage rates and relevance scores across various segments and over time, offer several valuable benefits to end users:

### **1. Strategic Decision-Making**

- **Investment Strategies**: Investors can identify which topics are gaining traction in the market and adjust their portfolios accordingly.
- **Market Analysis**: Understanding which sectors or topics are becoming more relevant can inform market entry or exit decisions.

### **2. Trend Identification**

- **Emerging Topics**: Detect rising themes in earnings calls, signaling shifts in company focus or industry trends.
- **Declining Interest**: Spotting decreasing coverage of certain topics can indicate waning interest or relevance.

### **3. Competitive Benchmarking**

- **Company Performance**: Compare how different companies discuss similar topics, offering insights into their strategic priorities.
- **Industry Standards**: Assess whether companies within the same industry are aligning on key topics or diverging in their focus.

### **4. Enhanced Reporting and Visualization**

- **Time Series Plots**: Visual representations make it easier to grasp complex data, enabling quicker comprehension of trends.
- **Customized Dashboards**: Stakeholders can interact with visualizations tailored to their specific needs, facilitating better communication of findings.

### **5. Operational Efficiency**

- **Automated Insights**: Reduces the time and resources required to manually analyze large volumes of transcript data.
- **Data Consistency**: Ensures that analyses are performed uniformly, enhancing the reliability of insights.

### **6. Risk Management**

- **Identifying Red Flags**: Sudden changes in topic relevance or coverage can signal potential issues, allowing for proactive risk management.
- **Market Sentiment Analysis**: Gauging how certain topics are perceived over time can aid in assessing overall market sentiment.

---

## **Conclusion**

Your data pipeline effectively automates the extraction, processing, and analysis of large volumes of transcripts to compute meaningful metrics like coverage rates and relevance scores. By segmenting data based on market capitalization and industry groups, it provides nuanced insights that are invaluable for strategic decision-making, trend analysis, and competitive benchmarking. The final visualizations and stored metrics empower end users to make informed, data-driven decisions with enhanced efficiency and accuracy.

Implementing robust memory management and optimization strategies, as discussed earlier, will ensure that this workflow remains efficient and scalable as data volumes grow.



## Questions

Taking over ownership of an existing codebase is a significant responsibility that requires a comprehensive understanding of the system's architecture, functionality, dependencies, and operational procedures. To ensure a smooth transition and effective maintenance of the pipeline you've provided, it's essential to ask the right questions to the original developer or team. Below is a categorized list of questions that will help you gain a deeper understanding and manage the workflow efficiently.

* * *

**1\. General Project Overview**
--------------------------------

### **a. Project Purpose and Goals**

*   **What is the primary objective of this pipeline?**
*   **What key problems does this workflow aim to solve?**
*   **Who are the main stakeholders or end-users of the outputs generated by this pipeline?**

### **b. Project History and Evolution**

*   **How long has this pipeline been in development and production?**
*   **What major changes or updates have been made since its inception?**
*   **Are there any historical issues or recurring problems that I should be aware of?**

* * *

**2\. Codebase Understanding**
------------------------------

### **a. Code Structure and Organization**

*   **Can you provide an overview of the codebase structure?**
*   **What are the main modules or components, and how do they interact?**
*   **Are there any design patterns or architectural principles followed in the code?**

### **b. Key Functions and Classes**

*   **Which functions or classes are central to the pipeline's operation?**
*   **Are there any custom utilities or helper functions that I should pay special attention to?**

### **c. Technical Jargon and Terminology**

*   **Are there any domain-specific terms or abbreviations used in the code that I should understand?**
*   **Can you explain any complex algorithms or processing steps implemented in the pipeline?**

* * *

**3\. Dependencies and Environment**
------------------------------------

### **a. Software and Libraries**

*   **What are the primary libraries and frameworks used (e.g., Pandas, PySpark, Snowflake connectors)?**
*   **Are there any specific versions of these libraries that the pipeline depends on?**
*   **How are dependencies managed (e.g., `requirements.txt`, Conda environments)?**

### **b. Development and Production Environments**

*   **What is the current development and production environment setup (e.g., Databricks clusters, hardware specifications)?**
*   **Are there any environment-specific configurations or settings?**

### **c. Configuration Management**

*   **How are configuration parameters (e.g., database credentials, file paths) managed and stored?**
*   **Are there environment variables or configuration files that need to be maintained?**

* * *

**4\. Data Sources and Data Flow**
----------------------------------

### **a. Data Ingestion**

*   **What are the primary data sources (e.g., Snowflake, Azure SQL)?**
*   **How frequently is data ingested, and what triggers data extraction?**

### **b. Data Processing**

*   **Can you explain the data preprocessing steps (e.g., deduplication, date formatting)?**
*   **Are there any data validation or cleaning procedures in place?**

### **c. Data Storage and Output**

*   **Where are the processed data and final outputs stored (e.g., Snowflake tables, CSV files on DBFS)?**
*   **Is there a data retention policy or archival process?**

* * *

**5\. Workflow Execution and Scheduling**
-----------------------------------------

### **a. Execution Flow**

*   **How is the pipeline executed (e.g., manually via notebooks, automated scheduling)?**
*   **Are there dependencies or order of execution among different pipeline stages?**

### **b. Scheduling and Automation**

*   **Is the pipeline scheduled to run at specific intervals (e.g., daily, monthly)?**
*   **What tools or services are used for scheduling (e.g., Databricks Jobs, Airflow)?**

### **c. Trigger Mechanisms**

*   **What events or conditions trigger the pipeline execution (e.g., new data availability)?**
*   **Are there any manual intervention points in the workflow?**

* * *

**6\. Error Handling and Logging**
----------------------------------

### **a. Error Detection and Management**

*   **How does the pipeline handle errors or exceptions during execution?**
*   **Are there retry mechanisms or fallback procedures for failed operations?**

### **b. Logging and Monitoring**

*   **What logging mechanisms are in place (e.g., Loguru, Databricks logging)?**
*   **Where can I access logs, and what information do they typically contain?**
*   **Are there monitoring tools or dashboards set up to track pipeline performance and health?**

### **c. Alerts and Notifications**

*   **Are there any alerting systems in place to notify stakeholders of pipeline failures or anomalies?**
*   **How are critical issues communicated to the team?**

* * *

**7\. Performance and Optimization**
------------------------------------

### **a. Performance Metrics**

*   **What are the key performance indicators (KPIs) for the pipeline?**
*   **How is performance measured and tracked over time?**

### **b. Bottlenecks and Optimization**

*   **Are there known performance bottlenecks in the current pipeline?**
*   **What optimization strategies have been implemented (e.g., caching, parallel processing)?**

### **c. Scalability**

*   **Can the pipeline handle increased data volumes or more frequent executions?**
*   **What scalability measures are in place or planned for future growth?**

* * *

**8\. Maintenance and Updates**
-------------------------------

### **a. Regular Maintenance Tasks**

*   **What routine maintenance tasks are necessary to keep the pipeline running smoothly?**
*   **Are there scheduled downtimes or maintenance windows?**

### **b. Updating Dependencies**

*   **How are library or framework updates managed to ensure compatibility?**
*   **Is there a process for testing updates before deploying them to production?**

### **c. Documentation and Knowledge Transfer**

*   **Is there comprehensive documentation available for the pipeline?**
*   **Are there any onboarding materials or guides for new maintainers?**

* * *

**9\. Security and Compliance**
-------------------------------

### **a. Data Security**

*   **How is sensitive data (e.g., database credentials) secured within the pipeline?**
*   **Are there encryption mechanisms in place for data at rest and in transit?**

### **b. Access Controls**

*   **Who has access to the pipeline's codebase, data sources, and outputs?**
*   **How are permissions managed and audited?**

### **c. Compliance Requirements**

*   **Are there any regulatory or compliance standards that the pipeline must adhere to (e.g., GDPR, HIPAA)?**
*   **How are compliance requirements implemented within the workflow?**

* * *

**10\. Version Control and Collaboration**
------------------------------------------

### **a. Source Code Management**

*   **Which version control system is used (e.g., Git, SVN)?**
*   **Where is the code repository hosted (e.g., GitHub, Bitbucket)?**

### **b. Branching and Merging Strategies**

*   **What branching strategy is followed (e.g., GitFlow, feature branches)?**
*   **How are code reviews and pull requests managed?**

### **c. Collaboration Practices**

*   **Are there any collaboration tools or practices in place (e.g., Slack channels, JIRA tickets)?**
*   **How are tasks and issues tracked within the team?**

* * *

**11\. Testing and Validation**
-------------------------------

### **a. Testing Procedures**

*   **Are there any automated tests (unit tests, integration tests) implemented for the pipeline?**
*   **How are tests executed and maintained?**

### **b. Data Validation**

*   **What data validation checks are performed to ensure data integrity?**
*   **Are there any sanity checks or data quality assessments in place?**

### **c. Deployment Validation**

*   **How is the pipeline validated before and after deployment to production?**
*   **Are there rollback procedures in case of deployment failures?**

* * *

**12\. Future Enhancements and Roadmap**
----------------------------------------

### **a. Planned Features**

*   **Are there any upcoming features or improvements planned for the pipeline?**
*   **What is the roadmap for future development?**

### **b. Known Issues and Technical Debt**

*   **Are there any known bugs or technical debts that need to be addressed?**
*   **What priorities should be set for resolving these issues?**

### **c. Feedback and Iteration**

*   **How is feedback from end-users collected and incorporated into the pipeline?**
*   **Are there regular reviews or retrospectives to assess pipeline performance and user satisfaction?**

* * *

**13\. Access and Credentials**
-------------------------------

### **a. System Access**

*   **What access rights do I need to manage and maintain the pipeline (e.g., Databricks, Snowflake, Azure SQL)?**
*   **How can I obtain these access rights?**

### **b. Credential Management**

*   **How are credentials managed and rotated within the pipeline?**
*   **Are there any secrets management tools in use (e.g., Azure Key Vault, AWS Secrets Manager)?**

### **c. Onboarding Procedures**

*   **What are the steps to onboard new team members to the project?**
*   **Are there any access provisioning scripts or documentation available?**

* * *

**14\. Documentation and Resources**
------------------------------------

### **a. Existing Documentation**

*   **Is there any existing documentation for the pipeline (e.g., README files, wikis)?**
*   **Where can I find architectural diagrams or data flow diagrams?**

### **b. Knowledge Base**

*   **Are there any knowledge base articles, FAQs, or troubleshooting guides available?**
*   **Who can provide additional insights or clarifications on complex aspects of the pipeline?**

### **c. Code Comments and Annotations**

*   **Is the codebase well-commented to explain non-trivial logic?**
*   **Are there docstrings or inline comments that I should pay attention to?**

* * *

**15\. Contact Points and Support**
-----------------------------------

### **a. Original Developer or Team**

*   **Who is the primary contact for questions or issues related to the pipeline?**
*   **Is there a point of contact for escalations or critical issues?**

### **b. Support Channels**

*   **What support channels are available (e.g., email, Slack, ticketing system)?**
*   **Are there any designated times for support availability?**

### **c. External Dependencies**

*   **Are there any external vendors or third-party services involved?**
*   **Who manages these relationships, and how are issues communicated?**

* * *

**Sample Questions Based on Provided Pipeline Code**
----------------------------------------------------

To make the transition even smoother, here are some **specific questions** tailored to the pipeline code you provided:

### **a. Data Extraction and Storage**

*   **How often is the data in `EDS_PROD.QUANT.YUJING_MASS_FT_NLI_DEMAND_DEV_1` updated?**
*   **What is the schema of the Snowflake table `YUJING_MASS_FT_NLI_DEMAND_DEV_1`?**
*   **Are there any data quality checks performed after extracting data from Snowflake?**

### **b. Data Preprocessing**

*   **What criteria determine which transcript version to keep when duplicates are found?**
*   **Can you explain the logic behind the coverage rate columns transformation?**

### **c. Market Cap Data Processing**

*   **What is the significance of the `MCAP_RANK` and how is it used in analysis?**
*   **How are `MCAP_GROUP` classifications (`C1`, `C2`, `C3`) determined and utilized in the pipeline?**

### **d. Helper Functions and Metrics Calculation**

*   **Can you provide more details on how `create_plot_df` and `create_plot_df_coverage_rate` functions compute rolling averages?**
*   **What is the rationale behind choosing a 120-day rolling window?**

### **e. Data Visualization and Storage**

*   **What types of visualizations are generated from the final DataFrame `df_rel_cover_all`?**
*   **How is the output CSV used downstream, and who are the consumers of this data?**

### **f. Error Handling and Data Validation**

*   **What happens if the Snowflake query returns no data? How is this scenario handled?**
*   **Are there any safeguards to ensure data integrity when merging and processing datasets?**

### **g. Performance Considerations**

*   **Have you encountered any performance issues with the current implementation, especially with large datasets?**
*   **Are there any optimizations in place for the Spark DataFrame operations or pandas transformations?**

### **h. Deployment and Scheduling**

*   **How is this notebook scheduled to run (e.g., via Databricks Jobs)?**
*   **Are there any dependencies on other notebooks or workflows within Databricks?**

### **i. Testing and Validation**

*   **Are there test cases or sample data used to validate the pipeline's accuracy?**
*   **How do you ensure that changes to the code do not break existing functionality?**

### **j. Documentation and Knowledge Transfer**

*   **Is there a flowchart or diagram that visually represents the pipeline's workflow?**
*   **Where can I find examples of input data and expected output for reference?**

* * *

**Conclusion**
--------------

Transitioning ownership of a complex data pipeline requires meticulous inquiry and thorough understanding of various components, from data sources and preprocessing steps to error handling and performance optimizations. By systematically addressing the categories and questions outlined above, you can ensure a smooth takeover, maintain the pipeline effectively, and continue to derive valuable insights from the data it processes. Additionally, establishing clear communication channels with the original developer or team will facilitate ongoing support and knowledge sharing, further enhancing your ability to manage and improve the workflow.

Remember to document any new insights or changes you make during this transition to aid future maintenance and onboarding of new team members.