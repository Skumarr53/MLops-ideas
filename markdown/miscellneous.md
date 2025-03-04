
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