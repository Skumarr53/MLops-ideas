
def create_speaker_identifier_with_fuzzy(row, threshold=80):
   ""
    Create speaker identifiers with fuzzy matching for unmatched cases
    Parameters:
    -----------
    row : pandas Series
        Input row containing filtered text columns
    threshold : int, default=80
        Fuzzy matching threshold (0-100)
    Returns:
    --------
    tuple : (speaker_identifiers, na_indices)
        speaker_identifiers : list of identified speakers
        na_indices : list of indices where original matching failed
   ""
    speaker_identifier = []
    na_indices = []
    filt_all_cleaned = [clean_text(sentence) for sentence in row['FILT_ALL']]
    ceo_md_cleaned = [clean_text(sentence) for sentence in row['FILT_CEO_MD']]
    exec_md_cleaned = [clean_text(sentence) for sentence in row['FILT_EXEC_MD']]
    ceo_qa_cleaned = [clean_text(sentence) for sentence in row['FILT_CEO_QA']]
    exec_qa_cleaned = [clean_text(sentence) for sentence in row['FILT_EXEC_QA']]
    anl_qa_cleaned = [clean_text(sentence) for sentence in row['FILT_ANL_QA']]
    
    last_speaker = None  # Variable to hold the last matched speaker
    for idx, sentence in enumerate(filt_all_cleaned):
        if sentence in ceo_md_cleaned:
            speaker_identifier.append('CEO')
            last_speaker = 'CEO'
        elif sentence in exec_md_cleaned:
            speaker_identifier.append('EXEC')
            last_speaker = 'EXEC'
        elif sentence in ceo_qa_cleaned:
            speaker_identifier.append('CEO')
            last_speaker = 'CEO'
        elif sentence in exec_qa_cleaned:
            speaker_identifier.append('EXEC')
            last_speaker = 'EXEC'
        elif sentence in anl_qa_cleaned:
            speaker_identifier.append('ANL')
            last_speaker = 'ANL'
        else:
            # If no exact match, use the last known speaker
            na_indices.append(idx)
            if last_speaker is not None:
                speaker_identifier.append(last_speaker)
            else:
                speaker_identifier.append('NA')  # If no previous speaker exists
    return speaker_identifier, na_indices


ceo_md_cleaned + exec_md_cleaned + ceo_qa_cleaned + exec_qa_cleaned + anl_qa_cleaned


------

import pandas as pd
def clean_dataframe(df):
    # Create a mask to identify indices where FILT_ALL elements are not in FILT_ALL_YUJ
    mask = df.apply(lambda row: [item not in row['FILT_ALL_YUJ'] for item in row['FILT_ALL']], axis=1)
    
    # Create a list of indices to drop
    indices_to_drop = []
    for i, row_mask in enumerate(mask):
        indices_to_drop.extend([i for i, val in enumerate(row_mask) if val])
    
    # Drop the identified indices from FILT_ALL, SECT_IDENTIFIER, and SPEAKER_IDENTIFIER
    df['FILT_ALL'] = df.apply(lambda row: [item for i, item in enumerate(row['FILT_ALL']) if i not in indices_to_drop], axis=1)
    df['SECT_IDENTIFIER'] = df.apply(lambda row: [item for i, item in enumerate(row['SECT_IDENTIFIER']) if i not in indices_to_drop], axis=1)
    df['SPEAKER_IDENTIFIER'] = df.apply(lambda row: [item for i, item in enumerate(row['SPEAKER_IDENTIFIER']) if i not in indices_to_drop], axis=1)
    return df


-----

import pandas as pd
def clean_dataframe(df):
    # Initialize lists to hold the cleaned values
    cleaned_filt_all = []
    cleaned_sect_identifier = []
    cleaned_speaker_identifier = []
    # Iterate through each row of the DataFrame
    for _, row in df.iterrows():
        # Get the current row's values
        filt_all = row['FILT_ALL']
        filt_all_yuj = row['FILT_ALL_YUJ']
        sect_identifier = row['SECT_IDENTIFIER']
        speaker_identifier = row['SPEAKER_IDENTIFIER']
        
        # Create a mask for indices to keep
        indices_to_keep = [i for i, item in enumerate(filt_all) if item in filt_all_yuj]
        
        # Filter the lists based on the indices to keep
        cleaned_filt_all.append([filt_all[i] for i in indices_to_keep])
        cleaned_sect_identifier.append([sect_identifier[i] for i in indices_to_keep])
        cleaned_speaker_identifier.append([speaker_identifier[i] for i in indices_to_keep])
    # Assign the cleaned lists back to the DataFrame
    df['FILT_ALL'] = cleaned_filt_all
    df['SECT_IDENTIFIER'] = cleaned_sect_identifier
    df['SPEAKER_IDENTIFIER'] = cleaned_speaker_identifier
    return df
# Sample DataFrame for demonstration
data = {
    'FILT_ALL': [['Hello', 'world', 'foo'], ['bar', 'baz'], ['test', 'example']],
    'FILT_ALL_YUJ': [['Hello', 'world'], ['bar'], ['test']],
    'SECT_IDENTIFIER': [['sec1', 'sec2', 'sec3'], ['sec4', 'sec5'], ['sec6', 'sec7']],
    'SPEAKER_IDENTIFIER': [['speaker1', 'speaker2', 'speaker3'], ['speaker4', 'speaker5'], ['speaker6', 'speaker7']]
}
df = pd.DataFrame(data)
# Clean the DataFrame
cleaned_df = clean_dataframe(df)
print(cleaned_df)







What is important is to let Josh know that we didn't do a full study on LoRA in the results we present and that there may be promise in the approach that we didn't find with default parameters.  The current fine-tuning results with the regular alg look really good so we don't really need to dig further for this small use case.



Here’s a clearer and more concise version of your message:

---

In the current iteration, we have explored several approaches. First, we added two new sample sets to the existing fine-tuning data: 700 non-entailment samples that we hand-labeled, which we previously requested you to review, and another sample set selected from the range of 0.4-0.6 entailment scores. These samples are ones that the Out-of-Bag (OOB) model struggles with, and we believe adding them to training set could enhance overall performance.

Additionally, we tested the LoRA fine-tuning approach on these datasets to compare it with the full fine-tuning approach. Since LoRA fine-tuning focuses on specific parts of the neural network, we wanted to assess how it performs when fine-tuned on a dataset rich in entailment samples versus one that includes both entailment and the other two sets.

Finally, for each approach, we experimented with several learning rates to evaluate their impact on model performance.

--- 

This version maintains the original meaning while improving clarity and readability.


---


data = {
    'YEAR_MONTH': ['2023-01', '2023-01', '2023-02', '2023-02', '2023-03'],
    'AVERAGE_SALARY': [5000, None, 6000, 7000, None],
    'COUNTS': [10, 5, 15, 10, 20]
}
df = pd.DataFrame(data)
# Group by YEAR_MONTH and aggregate
result = df.groupby('YEAR_MONTH').agg(
    non_missing_count=('AVERAGE_SALARY', lambda x: x.notnull().sum()),
    total_counts=('COUNTS', 'sum')
).reset_index()


I have df with month and salary columns.  I want to get count agg of salary in each month that are not na and percent of non missing salaries for each month. give me pyhton code


Hi Richard, 

I am writing to inform you that we fixed the issue and written updated data into below new table. We validated new data from out end and attaching below validation plots taht  compares inconsistencies in the old data with the updated data after the fix? we appriciate if you could validation as well and provide feedback.
 




Bea, just a quick update on the issue. After fixing the Reminton code, we backfilled the Linkup table using the updated code. Since the data in the Linkup table has been updated, we should also perform a historical backfill for mass labor income, as the source data has changed. It seems that either something was missing during the backfilling of this table, or it was backfilled but accidentally overwritten with the old data.



Hi Josh, To provide further feedback on the new model's performance, I will share time series aggregated data from 2020 to 2024, focusing on the companies in discretionary and staples industries falling in top Russell 3000 and 3000. These metrics are the same as I provided last time. Is there anything else you need from me? I just want to ensure I don’t miss anything.



Hi Richard,

I hope this message finds you well. I wanted to let you know that we have resolved the issue and updated the data in the new table below. We have validated the new data on our end and are attaching validation plots that compare the inconsistencies in the old data with the updated data after the fix. 

We would appreciate it if you could review the data as well and provide your feedback.

Thank you!

--------

Subject: Request for Query History and Investigation on Table Updates

Hello Snowflake Support Team,

We’ve encountered a data inconsistency issue with our MY_DB.MY_SCHEMA.MY_TABLE. Some updates seem to have been overwritten or never applied, resulting in incorrect query results. We’d like to investigate the specific queries that modified this table over the past week, but we currently do not have access to SNOWFLAKE.ACCOUNT_USAGE.QUERY_HISTORY.

Could you please Provide a log of all queries (INSERT, UPDATE, DELETE, MERGE, COPY, etc.) against MY_DB.MY_SCHEMA.MY_TABLE from 2024-12-01 to 2025-01-10.


We appreciate your prompt assistance and any additional guidance on diagnosing this issue. Feel free to let us know if further details are needed.

Thank you,


-----------


Hi Josh,

1. The aggregated results for companies in the discretionary and staples industries within the top Russell 3000 and 3000 are ready. Since you want to compare the new model's results with similar metrics from the previous model, please let me know your preferred way to share the results:
   - Present the current iteration results in a separate table
   - Combine the results in the existing tables, with the new aggregated results columns suffixed with _new and the previous aggregated results columns as _old

Which option would be easier for you to read and compare?

2. Regarding the aggregated results for all companies, would you like to compare the new model's results with similar metrics from the previous model, or would you prefer to analyze them individually? In the last iteration, we did not share the aggregated results for all companies. If you also need the aggregated results from the previous model, we will need to generate those metrics.




Hi Bea, 

This is Santhosh for voya india

I wanted to let you know that 10 min after meeting my voya account was revoked again and lost access to my work. Hopefully, access will be restored by monday. if issue still persist I will inform you.



------------

import mlflow
import mlflow.pyfunc
from transformers import AutoModelForSequenceClassification, AutoTokenizer
# Set the MLflow tracking URI if needed
# mlflow.set_tracking_uri("http://your_mlflow_server:5000")
# Specify the experiment ID or run ID from which you want to load the model
experiment_id =your_experiment_id"
run_id =your_run_id"
# Load the model as a PyFunc model
model_uri = runs:/{run_id}/model"  # Adjust the path if your model is saved under a different name
loaded_model = mlflow.pyfunc.load_model(model_uri)
# If your model is a transformer model, you might need to load the tokenizer and model separately
# Assuming the model is saved as a Hugging Face transformer model
model_name =your_model_name"  # Replace with the actual model name or path
tokenizer = AutoTokenizer.from_pretrained(model_name)
transformer_model = AutoModelForSequenceClassification.from_pretrained(model_name)
# Example usage of the loaded model for inference
def predict(input_text):
    inputs = tokenizer(input_text, return_tensors="pt")
    outputs = transformer_model(**inputs)
    return outputs
# Example prediction
input_text =This is a test input."
prediction = predict(input_text)
print(prediction)

----


from pyspark.sql import SparkSession
from pyspark.sql.functions import col
# Initialize Spark session
spark = SparkSession.builder \
    .appName("Merge Old and New Results") \
    .getOrCreate()
# Sample DataFrames (replace these with your actual DataFrames)
# old_results = spark.read.format("your_format").load("path_to_old_results")
# new_results = spark.read.format("your_format").load("path_to_new_results")
# Example DataFrames for demonstration
old_data = [(1,A", 100), (2,B", 200)]
new_data = [(1,A", 150), (2,B", 250), (3,C", 300)]
# Assuming primary keys are (primary_key1, primary_key2)
old_results = spark.createDataFrame(old_data, ["primary_key1",primary_key2",result"])
new_results = spark.createDataFrame(new_data, ["primary_key1",primary_key2",result"])
# Define primary key columns
primary_keys = ["primary_key1",primary_key2"]
# Merge DataFrames on primary keys
merged_df = old_results.join(new_results, on=primary_keys, how="outer")
# Rename result columns
for column in old_results.columns:
    if column not in primary_keys:
        merged_df = merged_df.withColumnRenamed(column, {column}_old")
for column in new_results.columns:
    if column not in primary_keys:
        merged_df = merged_df.withColumnRenamed(column, {column}_new")
# Show the result
merged_df.show()
# Stop Spark session
spark.stop()
# Stop Spark session
spark.stop()

-----


'INCREASED_CONSUMPTION_COVERRATE_FILT_AVERAGE_NEW',
'REDUCED_CONSUMPTION_COVERRATE_FILT_AVERAGE_NEW',
'INCREASED_CONSUMPTION_REL_FILT_AVERAGE_NEW',
'REDUCED_CONSUMPTION_REL_FILT_AVERAGE_NEW',
'INCREASED_CONSUMPTION_COVERRATE_FILT_AVERAGE_OLD',
'REDUCED_CONSUMPTION_COVERRATE_FILT_AVERAGE_OLD',
'INCREASED_CONSUMPTION_REL_FILT_AVERAGE_OLD',
'REDUCED_CONSUMPTION_REL_FILT_AVERAGE_OLD'


-------------


I hope this message finds you well.

I wanted to take a moment to explain the format we are using for merging the old and new results data. The merge process involves combining two datasets based on multiple primary key columns. Each dataset retains its original structure, and the result columns from the old dataset are suffixed with_old," while those from the new dataset are suffixed with_new." This approach allows us to clearly distinguish between the old and new results while maintaining the integrity of the primary keys.

For example, if we have primary keys such asprimary_key1" andprimary_key2," the merged dataset will include these keys along with the results from both datasets, clearly labeled to avoid confusion.

If you have any questions or need further clarification, please feel free to reach out.

Best regards,

[Your Name]
[Your Position]
[Your Contact Information]

-----------------

import pandas as pd
import matplotlib.pyplot as plt
# Sample DataFrame creation for demonstration (replace this with your actual DataFrame)
# df = pd.read_csv('your_data.csv')  # Load your DataFrame from a CSV or other source
# Example DataFrame for demonstration
data = {
    'DATE_ROLLING': pd.date_range(start='2023-01-01', periods=10, freq='D'),
    'INCREASED_CONSUMPTION_COVERRATE_FILT_AVERAGE_NEW': [10, 15, 20, 25, 30, 35, 40, 45, 50, 55],
    'REDUCED_CONSUMPTION_COVERRATE_FILT_AVERAGE_NEW': [5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
    'INCREASED_CONSUMPTION_REL_FILT_AVERAGE_NEW': [2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
    'REDUCED_CONSUMPTION_REL_FILT_AVERAGE_NEW': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'INCREASED_CONSUMPTION_COVERRATE_FILT_AVERAGE_OLD': [8, 12, 18, 22, 28, 33, 39, 44, 48, 52],
    'REDUCED_CONSUMPTION_COVERRATE_FILT_AVERAGE_OLD': [4, 9, 14, 19, 24, 29, 34, 39, 44, 49],
    'INCREASED_CONSUMPTION_REL_FILT_AVERAGE_OLD': [1, 3, 5, 7, 9, 11, 13, 15, 17, 19],
    'REDUCED_CONSUMPTION_REL_FILT_AVERAGE_OLD': [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]
}
df = pd.DataFrame(data)
# List of metrics
metrics = [
    'INCREASED_CONSUMPTION_COVERRATE_FILT_AVERAGE',
    'REDUCED_CONSUMPTION_COVERRATE_FILT_AVERAGE',
    'INCREASED_CONSUMPTION_REL_FILT_AVERAGE',
    'REDUCED_CONSUMPTION_REL_FILT_AVERAGE'
]
# Create subplots
fig, axs = plt.subplots(len(metrics), 1, figsize=(10, 15), sharex=True)
# Plot each metric
for i, metric in enumerate(metrics):
    new_col = f'{metric}_NEW'
    old_col = f'{metric}_OLD'
    
    axs[i].plot(df['DATE_ROLLING'], df[new_col], label='New', color='blue', marker='o')
    axs[i].plot(df['DATE_ROLLING'], df[old_col], label='Old', color='orange', marker='x')
    
    axs[i].set_title(metric.replace('_', ' ').title())
    axs[i].set_ylabel('Value')
    axs[i].legend()
    axs[i].grid()
# Set x-axis label
axs[-1].set_xlabel('Date')
# Adjust layout
plt.tight_layout()
plt.show()

-----------------


Hi Josh, 


The tables below presents time series data generated by the FT model, covering the period from 2020 to 2024. It focuses on the top 3000 stocks and 3000 for both all sectors and slecetd sectors i.e Discretionary and Staples. Also, I consolidated all scenarios (sector filters and top companies filter) with an additional column named CATEGORY indicating the scenario type, and and as well as old and new model aggreated result into a single view with column name model version consituining two i.e categories 'OLD_MODEL' and 'NEW_MODEL' 

List of scenario identifiers:
ALL_TOP3000
ALL_TOP3000
DISCRETIONARY_TOP3000
DISCRETIONARY_TOP3000
STAPLES_TOP3000
STAPLES_TOP3000

Table: EDS_PROD.QUANT.SANTHOSH_MASS_FT_NLI_DEMAND_TS_DEV_20250319_COMBINED_VIEW

Example:
SELECT * FROM EDS_PROD.QUANT.SANTHOSH_MASS_FT_NLI_DEMAND_TS_DEV_20250319_COMBINED_VIEW WHERE CATEGORY = ALL_TOP3000 AND MODEL_VERSION = 'OLD_MODEL'

wold fetch old model aggrgated  results from companies of all sectors

Obersertion:
 - Overall follows similar trend but bit low. This could be attributed to new model doing better job in excluding false positives that was missed from prvious model that was traned only on entailment samples.


 I would appreciate your thoughts and feedback on the results.




create TABLEEDS_PROD"."QUANT"."YUJING_ECALL_NLI_SENTIMENT_SCORE_PY_DEV_2" AS
select ENTITY_ID, CALL_ID, VERSION_ID, DATE, CALL_NAME, COMPANY_NAME, 
        FILT_ALL, FIN_SENT_LABELS_FILT_ALL, NLI_SENT_LABELS_FILT_ALL, 
        POS_SCORE_FILT_ALL, NEU_SCORE_FILT_ALL, NEG_SCORE_FILT_ALL 
        fromEDS_PROD"."QUANT"."YUJING_ECALL_NLI_SENTIMENT_SCORE_PY_DEV_1"
union all
select ENTITY_ID, CALL_ID, VERSION_ID, DATE, CALL_NAME, COMPANY_NAME, 
        FILT_ALL, FIN_SENT_LABELS_FILT_ALL, NLI_SENT_LABELS_FILT_ALL, 
        POS_SCORE_FILT_ALL, NEU_SCORE_FILT_ALL, NEG_SCORE_FILT_ALL 
        fromEDS_PROD"."QUANT"."YUJING_ECALL_NLI_SENTIMENT_SCORE_DEV_1"where DATE >= '2023-01-01';


DELETE FROM EDS_PROD.QUANT.SANTHOSH_SENTIMENT_CORE_TABLE
WHERE VERSION_ID IN (
    SELECT DISTINCT VERSION_ID 
    FROM EDS_PROD.QUANT.YUJING_ECALL_NLI_SENTIMENT_SCORE_PY_DEV_2 
    WHERE DATE >= '2024-01-01'
);


- save corrected sec and spkr identier in on table load only required columns
- load both py 2 table content and compute net sentiment score
- the combine both net sentiment and  sec and spkr identier and write into table along with required columns
- then merge above table with main py2 creationg new table to finish backfilling


I have table1 with cols A, B, C, D, T, U and table2 with cols A, B, C, E, F, G. Now I want to join both tables uysing A, B as identifer and Keep only C from tableA final table should contain only these columns A, B, C, D, T, E, F. Give me sql query to achieve this  

Ans:
SELECT t1.A, t1.B, t1.C, t1.D, t1.T, t2.E, t2.F
FROM table1 t1
JOIN table2 t2 ON t1.A = t2.A AND t1.B = t2.B;


I have 2 tables A and B with identical columns. I want to union all B with A and store cobined data in A. provide query for that

Ans:
INSERT INTO A
SELECT * FROM B;

------
Below code is not working I think it has to do with the way pool.map is set up. Can you fix

def  compute_net_sentiment_score(pos, neg, neu):
  pos, neg, neu = np.array(pos), np.array(neg), np.array(neu)
  net_sentiment_score = (pos - neg)/(pos + neg + neu)
  return list(net_sentiment_score)

with mp.Pool(min(mp.cpu_count(), 16)) as pool:
  main_sentiment_df['NET_SENTIMENT_SCORE'] = pool.map(compute_net_sentiment_score, main_sentiment_df['POS_SCORE_FILT_ALL'], 
                                                                                  main_sentiment_df['NEG_SCORE_FILT_ALL'], 
                                                                                  main_sentiment_df['NEU_SCORE_FILT_ALL'])


['ENTITY_ID', 'VERSION_ID', 'SECTION_IDENTIFIER', 'SPEAKER_IDENTIFIER']
['VERSION_ID', 'POS_SCORE_FILT_ALL', 'NEU_SCORE_FILT_ALL',
       'NEG_SCORE_FILT_ALL', 'NET_SENTIMENT_SCORE']


merged_df = pd.merge(df1, df2, on='VERSION_ID', how='inner')

-------

I have 3 tables A, B, and C. I want to pull version ids from VERSION_ID columns present in table A but missing in column B. then for resulting version ids I want to pull records from table C matcing these ids. Provide sql query   


SELECT C.*
FROM C
WHERE C.VERSION_ID IN (
    SELECT A.VERSION_ID
    FROM A
    WHERE A.VERSION_ID NOT IN (SELECT B.VERSION_ID FROM B)
);





dask code:
for col in main_sentiment_df.columns:
 main_sentiment_df[col] = main_sentiment_df[col].apply(ast.literal_eval)


equivalent multiprocessing code:

- when function takes single column input

n_workers=16
import multiprocessing as mp
with mp.Pool(min(mp.cpu_count(), n_workers)) as pool:
  for col in main_sentiment_df.columns:
      main_sentiment_df[col] = pool.map(ast.literal_eval, main_sentiment_df[col])



- when fuction takes multiple columns as input
import numpy as np  
import multiprocessing as mp


def  compute_net_sentiment_score(data):
  pos, neg, neu = data
  pos, neg, neu = np.array(pos), np.array(neg), np.array(neu)
  net_sentiment_score = (pos - neg)/(pos + neg + neu)
  return [float(val) for val in net_sentiment_score]


n_workers=16
with mp.Pool(min(mp.cpu_count(), n_workers)) as pool: 
  main_sentiment_df['NET_SENTIMENT_SCORE'] = pool.map(compute_net_sentiment_score, zip(main_sentiment_df['POS_SCORE_FILT_ALL'], 
                                                                                       main_sentiment_df['NEG_SCORE_FILT_ALL'], 
                                                                                       main_sentiment_df['NEU_SCORE_FILT_ALL']))



---------------------------

def clean_dataframe(df):
    # Initialize lists to hold the cleaned values
    cleaned_filt_all = []
    cleaned_sect_identifier = []
    cleaned_speaker_identifier = []
    # Iterate through each row of the DataFrame
    for _, row in tqdm(df.iterrows()):
        # Get the current row's values
        filt_all = row['FILT_ALL']
        filt_all_yuj = row['FILT_ALL_YUJ']
        sect_identifier = row['SECTION_IDENTIFIER']
        speaker_identifier = row['SPEAKER_IDENTIFIER']
        
        # Create a mask for indices to keep
        indices_to_keep = [i for i, item in enumerate(filt_all) if item in filt_all_yuj]
        
        # Filter the lists based on the indices to keep
        cleaned_filt_all.append([filt_all[i] for i in indices_to_keep])
        cleaned_sect_identifier.append([sect_identifier[i] for i in indices_to_keep])
        cleaned_speaker_identifier.append([speaker_identifier[i] for i in indices_to_keep])
    # Assign the cleaned lists back to the DataFrame
    df['FILT_ALL'] = cleaned_filt_all
    df['SECTION_IDENTIFIER'] = cleaned_sect_identifier
    df['SPEAKER_IDENTIFIER'] = cleaned_speaker_identifier
    return df


----

from tqdm import tqdm

def clean_dataframe(df):
    # Lists to hold the final cleaned values for the entire DataFrame
    cleaned_filt_all_final = []
    cleaned_sect_identifier_final = []
    cleaned_speaker_identifier_final = []
    
    # Iterate through each row of the DataFrame with a progress bar
    for _, row in tqdm(df.iterrows(), total=len(df)):
        filt_all = row['FILT_ALL']
        filt_all_yuj = row['FILT_ALL_YUJ']
        sect_identifier = row['SECTION_IDENTIFIER']
        speaker_identifier = row['SPEAKER_IDENTIFIER']
        
        # Initial filtering: keep only those sentences in filt_all that are in filt_all_yuj
        indices_to_keep = [i for i, item in enumerate(filt_all) if item in filt_all_yuj]
        cleaned_filt = [filt_all[i] for i in indices_to_keep]
        cleaned_sect = [sect_identifier[i] for i in indices_to_keep]
        cleaned_speaker = [speaker_identifier[i] for i in indices_to_keep]
        
        # Merge filtered data with filt_all_yuj to insert any missing sentences
        final_filt = []
        final_sect = []
        final_speaker = []
        p = 0  # pointer for cleaned_filt
        
        for sent in filt_all_yuj:
            if p < len(cleaned_filt) and cleaned_filt[p] == sent:
                # Sentence exists in filtered list; add it and its identifiers
                final_filt.append(sent)
                final_sect.append(cleaned_sect[p])
                final_speaker.append(cleaned_speaker[p])
                p += 1
            else:
                # Sentence is missing; insert it with identifiers from the previous element
                final_filt.append(sent)
                if final_sect:
                    final_sect.append(final_sect[-1])
                    final_speaker.append(final_speaker[-1])
                else:
                    # Fallback for the first element if missing
                    default_sect = cleaned_sect[0] if cleaned_sect else"
                    default_speaker = cleaned_speaker[0] if cleaned_speaker else"
                    final_sect.append(default_sect)
                    final_speaker.append(default_speaker)
        
        # Append the final merged lists for this row
        cleaned_filt_all_final.append(final_filt)
        cleaned_sect_identifier_final.append(final_sect)
        cleaned_speaker_identifier_final.append(final_speaker)
    
    # Update DataFrame with the final cleaned columns
    df['FILT_ALL'] = cleaned_filt_all_final
    df['SECTION_IDENTIFIER'] = cleaned_sect_identifier_final
    df['SPEAKER_IDENTIFIER'] = cleaned_speaker_identifier_final
    return df


--------------------------------

def inference_run(
    iterator: Iterator[List[str]],
    nli_pipeline,
    labels: List[str],
    max_length: int = 512,
    batch_size: int = 32,
    threshold: float = 0.8
) -> Iterator[pd.Series]:
    for batch_num, batch in enumerate(iterator, start=1):
        # logger.info(f"Processing inference batch {batch_num} with {len(batch)} rows.")
        # try:
        score_dict: Dict[str, List[float]] = {label: [] for label in labels}
        total_dict: Dict[str, List[int]] = {label: [] for label in labels}

        pairs = batch.tolist()
        # pairs = [pair for sublist in batch for pair in sublist]
        pair_list = list(chain.from_iterable([[x] * len(labels) for x in pairs]))
        labels_list = labels * len(pairs)
        flat_text_pairs = [
                                {'text': t, 'text_pair': f"{l}."} 
                                for t, l in zip(pair_list, labels_list)
                            ]
        
        # logger.debug(f"Batch {batch_num}: Total text pairs to infer: {len(flat_text_pairs)}")
        split_results = []
        if flat_text_pairs:
            # Perform inference in batch

            results = nli_pipeline(
                flat_text_pairs,
                padding=True,
                top_k=None,
                batch_size=batch_size,
                truncation=True,
                max_length=max_length
            )
            # logger.debug(f"Batch {batch_num}: Inference completed with {len(results)} results.")
            
            # get scores and labels
            for lab, result in zip(labels_list, results):
                for res in result:
                    if res['label'] == 'entailment':
                        score = res['score']
                        total_dict[lab].append(int(score > threshold))
                        score_dict[lab].append(score)                
            # Append results for the current row
            split_results.append([total_dict, score_dict])
        # else:
        #     split_results.append([])
        #     # logger.warning(f"Batch {batch_num}: No text pairs to infer for current row.")
        yield pd.Series(split_results)
        
        # except Exception as e:
        #     # logger.error(f"Error in inference batch {batch_num}: {e}")
        #     raise Exception(f"Error in inference batch {batch_num}: {e}")

-------

from itertools import chain
n_repeat = 4
original_list = [1, 2, 3, 4]
# Using itertools.chain to repeat each element
repeated_list = list(chain.from_iterable([[x] * n_repeat for x in original_list]))
print(repeated_list)


----------------

SELECT * FROM A
WHERE VERSION_ID NOT IN (SELECT VERSION_ID FROM C)
UNION ALL
SELECT * FROM B
WHERE VERSION_ID NOT IN (SELECT VERSION_ID FROM C);

SELECT CALL_ID, ENTITY_ID, DATE, FILT_MD, FILT_QA, CALL_NAME, COMPANY_NAME,
EARNINGS_CALL, ERROR, TRANSCRIPT_STATUS, UPLOAD_DT_UTC, VERSION_ID,
EVENT_DATETIME_UTC, PARSED_DATETIME_EASTERN_TZ, SENT_LABELS_FILT_MD,
SENT_LABELS_FILT_QA



SELECT *
FROM (
    SELECT CALL_ID, ENTITY_ID, DATE, FILT_MD, FILT_QA, CALL_NAME, COMPANY_NAME, EARNINGS_CALL, ERROR, TRANSCRIPT_STATUS, 
           UPLOAD_DT_UTC, VERSION_ID, EVENT_DATETIME_UTC, PARSED_DATETIME_EASTERN_TZ, SENT_LABELS_FILT_MD, SENT_LABELS_FILT_QA 
    FROM EDS_PROD.QUANT.PARTHA_FUND_CTS_STG_1_VIEW
    UNION ALL
    SELECT CAST(CALL_ID AS STRING) AS CALL_ID, ENTITY_ID, DATE, FILT_MD, FILT_QA, CALL_NAME, COMPANY_NAME, EARNINGS_CALL, ERROR, TRANSCRIPT_STATUS, 
           UPLOAD_DT_UTC, VERSION_ID, EVENT_DATETIME_UTC, PARSED_DATETIME_EASTERN_TZ, SENT_LABELS_FILT_MD, SENT_LABELS_FILT_QA 
    FROM EDS_PROD.QUANT_LIVE.CTS_FUND_COMBINED_SCORES_H
) AS CombinedRecords
WHERE VERSION_ID NOT IN (
    SELECT VERSION_ID 
    FROM EDS_PROD.QUANT.SANTHOSH_MASS_FT_NLI_DEMAND_DEV_202503_BACKFILL
);



------------

from pyspark.sql import SparkSession
from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import StructType, StructField, MapType, ArrayType, FloatType, StringType
import pandas as pd
from typing import List, Dict
from itertools import chain
# Define the schema for the output
output_schema = StructType([
    StructField("total_dict", MapType(StringType(), ArrayType(FloatType())), True),
    StructField("score_dict", MapType(StringType(), ArrayType(FloatType())), True)
])
@pandas_udf(output_schema)
def inference_run(
    pdf: pd.DataFrame,
    nli_pipeline,
    labels: List[str],
    max_length: int = 512,
    batch_size: int = 32,
    threshold: float = 0.8
) -> pd.DataFrame:
    # Prepare output dictionaries
    score_dict: Dict[str, List[float]] = {label: [] for label in labels}
    total_dict: Dict[str, List[int]] = {label: [] for label in labels}
    
    # Extract pairs from the DataFrame
    pairs = pdf['text_column'].tolist()  # Replace 'text_column' with the actual column name containing text
    pair_list = list(chain.from_iterable([[x] * len(labels) for x in pairs]))
    labels_list = labels * len(pairs)
    
    # Prepare flat text pairs
    flat_text_pairs = [{'text': t, 'text_pair': f"{l}."} for t, l in zip(pair_list, labels_list)]
    
    if flat_text_pairs:
        # Perform inference in batch
        results = nli_pipeline(
            flat_text_pairs,
            padding=True,
            top_k=None,
            batch_size=batch_size,
            truncation=True,
            max_length=max_length
        )
        
        # Get scores and labels
        for lab, result in zip(labels_list, results):
            for res in result:
                if res['label'] == 'entailment':
                    score = res['score']
                    total_dict[lab].append(int(score > threshold))
                    score_dict[lab].append(score)
    # Return results as a DataFrame
    return pd.DataFrame([{
        "total_dict": total_dict,
        "score_dict": score_dict
    }])
# Example of how to use this UDF in a Spark DataFrame
if __name__ == "__main__":
    # Create Spark session
    spark = SparkSession.builder \
        .appName("Pandas UDF Inference Example") \
        .getOrCreate()
    # Sample data
    data = [("text1",), ("text2",), ("text3",)]
    columns = ["text_column"]
    df = spark.createDataFrame(data, columns)
    # Define your NLI pipeline here
    nli_pipeline = None  # Replace with your actual NLI pipeline
    # Define labels
    labels = ["label1", "label2"]  # Replace with your actual labels
    # Apply the Pandas UDF to the DataFrame
    result_df = df.groupBy().apply(inference_run, nli_pipeline, labels)
    result_df.show(truncate=False)
    # Stop the Spark session
    spark.stop()

----------

WITH CombinedRecords AS (
    SELECT CALL_ID, ENTITY_ID, DATE, FILT_MD, FILT_QA, CALL_NAME, COMPANY_NAME, EARNINGS_CALL, ERROR, TRANSCRIPT_STATUS, 
           UPLOAD_DT_UTC, VERSION_ID, EVENT_DATETIME_UTC, PARSED_DATETIME_EASTERN_TZ, SENT_LABELS_FILT_MD, SENT_LABELS_FILT_QA 
    FROM EDS_PROD.QUANT.PARTHA_FUND_CTS_STG_1_VIEW
    UNION ALL
    SELECT CAST(CALL_ID AS STRING) AS CALL_ID, ENTITY_ID, DATE, FILT_MD, FILT_QA, CALL_NAME, COMPANY_NAME, EARNINGS_CALL, ERROR, TRANSCRIPT_STATUS, 
           UPLOAD_DT_UTC, VERSION_ID, EVENT_DATETIME_UTC, PARSED_DATETIME_EASTERN_TZ, SENT_LABELS_FILT_MD, SENT_LABELS_FILT_QA 
    FROM EDS_PROD.QUANT_LIVE.CTS_FUND_COMBINED_SCORES_H
),
RankedRecords AS (
    SELECT *,
           ROW_NUMBER() OVER (PARTITION BY CALL_ID ORDER BY DATE DESC) AS rn
    FROM CombinedRecords
)
SELECT *
FROM RankedRecords
WHERE rn = 1
AND VERSION_ID NOT IN (
    SELECT VERSION_ID 
    FROM EDS_PROD.QUANT.SANTHOSH_MASS_FT_NLI_DEMAND_DEV_202503_BACKFILL
);