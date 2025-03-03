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

    for idx, sentence in enumerate(filt_all_cleaned):
        if sentence in ceo_md_cleaned:
            speaker_identifier.append('CEO')
        elif sentence in exec_md_cleaned:
            speaker_identifier.append('EXEC')
        elif sentence in ceo_qa_cleaned:
            speaker_identifier.append('CEO')
        elif sentence in exec_qa_cleaned:
            speaker_identifier.append('EXEC')
        elif sentence in anl_qa_cleaned:
            speaker_identifier.append('ANL')
        else:
            # If no exact match, try fuzzy matching
            na_indices.append(idx)
            
            # Store all possible matches and their scores
            fuzzy_scores = {
                'CEO_MD': max([fuzz.ratio(sentence, s) for s in ceo_md_cleaned]) if len(ceo_md_cleaned) > 0 else 0,
                'EXEC_MD': max([fuzz.ratio(sentence, s) for s in exec_md_cleaned]) if len(exec_md_cleaned) > 0 else 0,
                'CEO_QA': max([fuzz.ratio(sentence, s) for s in ceo_qa_cleaned]) if len(ceo_qa_cleaned) > 0 else 0,
                'EXEC_QA': max([fuzz.ratio(sentence, s) for s in exec_qa_cleaned]) if len(exec_qa_cleaned) > 0 else 0,
                'ANL_QA': max([fuzz.ratio(sentence, s) for s in anl_qa_cleaned]) if len(anl_qa_cleaned) > 0 else 0
            }
            
            # Get the best match
            best_match = max(fuzzy_scores.items(), key=lambda x: x[1])
            
            # If the best match score is above threshold, use it
            if best_match[1] >= threshold:
                speaker_identifier.append(best_match[0])
            else:
                speaker_identifier.append('NA')
    
    return speaker_identifier