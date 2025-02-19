I have two functions, referred to as Func A and Func B, below. I believe that the functionality of Func B is already included in Func A, meaning that Func A offers more capabilities than Func B. Occasionally, I only need to calculate the statistics provided by Func B, while at other times, I require Func A to obtain additional statistics. Since there is some overlapping logic, I would like to combine both functions into a single function with an optional argument that allows the user to specify whether they want only the statistics from Func B or the full set



Function A:
def count_matches_in_single_sentence(
    texts: str,
    match_sets: Dict[str, Dict[str, Any]],
    nlp: spacy.Language,
    phrases: bool = True,
    suppress: Optional[Dict[str, List[str]]] = None
) -> Dict[str, Dict[str, Any]]:
    
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


Function B:
def calculate_binary_match_flags(texts, match_sets, suppress = None):

  total_counts = {label: [] for label in match_sets.keys()}

  for text in texts:

    text = text.lower()
    unigrams = tokenize_and_lemmatize_text(text)
    bigrams = ['_'.join(g) for g in generate_ngrams(unigrams, 2)]
    
    for label, word_set in match_sets.items():
      if (set(unigrams) & set(word_set['unigrams']) or set(bigrams) & set(word_set['bigrams']) or any(phrase in text for phrase in word_set['phrases'])) and (not any(item in text for item in suppress[label])):
        total_counts[label].append(1)
      else:
        total_counts[label].append(0)
    
  return total_counts