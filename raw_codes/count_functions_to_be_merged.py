def count_matches_in_single_sentence(
    texts: str,
    match_sets: Dict[str, Dict[str, Any]],
    nlp: spacy.Language,
    phrases: bool = True,
    suppress: Optional[Dict[str, List[str]]] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Counts occurrences of match patterns within a single sentence.
    
    Args:
        text (str): Sentence to process.
        match_sets (Dict[str, Dict[str, Any]]): Dictionary containing match patterns categorized by labels.
        phrases (bool, optional): Whether to include phrase matching. Defaults to True.
        suppress (Optional[Dict[str, List[str]]], optional): Words to suppress from matching per label. 

            Defaults to None.
    
    Returns:
        Dict[str, Dict[str, Any]]: Dictionary of counts for each match pattern categorized by labels.
    
    Example:
        >>> text = "I love good service and bad support."
        >>> match_sets = {
        ...     'positive': {
        ...         'unigrams': {'good'},
        ...         'bigrams': {'good_service'},
        ...         'phrases': []
        ...     },
        ...     'negative': {
        ...         'unigrams': {'bad'},
        ...         'bigrams': {'bad_support'},
        ...         'phrases': []
        ...     }
        ... }
        >>> suppress = {
        ...     'positive': [],
        ...     'negative': []
        ... }
        >>> counts = count_matches_in_single_sentence(text, match_sets, phrases=True, suppress=suppress)
        >>> print(counts)
        {
            'positive': {
                'total': 2,
                'stats': {'good': 1, 'good_service': 1}
            },
            'negative': {
                'total': 2,
                'stats': {'bad': 1, 'bad_support': 1}
            }
        }
    """
    count_dict = {label : {matchw: 0 for matchw in match_set['unigrams'].union(match_set['bigrams']) } for label, match_set in match_sets.items()}
    total_counts = {label: [] for label in match_sets.keys()}

    for text in texts:
        
        counted = {label: 0 for label in match_sets.keys()}
        unigrams = tokenize_and_lemmatize_text(text, nlp)
        bigrams = ['_'.join(g) for g in generate_ngrams(unigrams, 2)]
        
        text = text.lower()
        for label, match_set in match_sets.items(): 
        
            if any(item in text for item in suppress[label]):
                counted[label] += 0
                continue
                
            for word in unigrams:
                if word in match_set['unigrams']:
                    count_dict[label][word]+=1
                    counted[label] += 1

            for word in bigrams:
                if word in match_set['bigrams']:
                    count_dict[label][word]+=1
                    counted[label] += 1
            
            if phrases:
                if any(phrase in text for phrase in match_set['phrases']):
                    counted[label] += 1
                    continue

        for label in match_sets.keys():
        
            total_counts[label].append(counted[label])

        
    return {label : {'total': total_counts[label], 'stats' : count_dict[label]} for label in match_sets.keys()}


# Create a set of match patterns from match list. This ensures variations such as lemmas & case are handled.
def get_match_set(matches):
  
  bigrams = set([word.lower() for word in matches if len(word.split('_'))==2] + [word.lower().replace(" ", '_') for word in matches if len(word.split(' '))==2] + ['_'.join(matchTokenize(word)) for word in matches if len(word.split(' '))==2])
 
  unigrams = set([matchTokenize(match)[0] for match in matches if ('_' not in match) and (len(match.split(' '))==1) and (len(matchTokenize(match)))] + [match.lower() for match in matches if ('_' not in match) and (len(match.split(' '))==1)])

#  Phrase matching correction
  phrases = [phrase.lower() for phrase in matches if len(phrase.split(" "))>2]
  
  return {'original': matches, 'unigrams': unigrams, 'bigrams': bigrams, 'phrases': phrases}

def match_count_lowStat_negation(texts, match_sets, suppress = None):

  total_counts = {label: [] for label in match_sets.keys()}

  for text in texts:

    text = text.lower()
    unigrams = wordTokenize(text)
    bigrams = ['_'.join(g) for g in find_ngrams(unigrams, 2)]
    
    for label, word_set in match_sets.items():
      if (set(unigrams) & set(word_set['unigrams']) or set(bigrams) & set(word_set['bigrams']) or any(phrase in text for phrase in word_set['phrases'])) and (not any(item in text for item in suppress[label])):
        total_counts[label].append(1)
      else:
        total_counts[label].append(0)
    
  return total_counts