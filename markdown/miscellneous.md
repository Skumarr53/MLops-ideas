
def create_speaker_identifier_with_fuzzy(row, threshold=80):
    """
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
    """
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

df_filtered = df[df['salary'].notna()]
# Count of non-NA salaries per month
count_per_month = df_filtered.groupby('month')['salary'].count().reset_index()
# Total count of non-NA salaries
total_count = df_filtered['salary'].count()
# Calculate percentage
count_per_month['percentage'] = (count_per_month['salary'] / total_count) * 100
# Rename columns for clarity
count_per_month.rename(columns={'salary': 'count'}, inplace=True)
# Display the results
print(count_per_month)