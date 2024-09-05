# processing/matching_and_scoring.py
from typing import List, Dict, Set
from collections import Counter

class MatchingScoring:
    @staticmethod
    def create_match_set(matches: List[str]) -> Dict[str, Set[str]]:
        """Create a match set of unigrams and bigrams from a list of phrases."""
        unigrams = {match.lower() for match in matches if ' ' not in match}
        bigrams = {'_'.join(match.lower().split()) for match in matches if len(match.split()) == 2}
        return {"unigrams": unigrams, "bigrams": bigrams}

    @staticmethod
    def match_count(text: List[str], match_set: Dict[str, Set[str]]) -> Dict[str, int]:
        """Count unigram and bigram matches in the tokenized text."""
        unigram_count = sum(1 for word in text if word in match_set['unigrams'])
        bigram_count = sum(1 for word in text if word in match_set['bigrams'])
        return {"unigrams": unigram_count, "bigrams": bigram_count}

    @staticmethod
    def calculate_relevance(total_sentences: int, matched_sentences: int) -> float:
        """Calculate relevance score based on sentence matching."""
        if total_sentences == 0:
            return 0.0
        return matched_sentences / total_sentences

    @staticmethod
    def calculate_sentiment_score(a: List[int], b: List[int], weighted: bool = True) -> float:
        """Calculate sentiment score using the provided sentiment and relevance values."""
        if len(a) != len(b) or len(a) == 0:
            return None
        positive_sentences = sum(1 for x in a if x > 0)
        if positive_sentences == 0:
            return None
        return sum(a[i] * b[i] for i in range(len(a))) / positive_sentences if weighted else sum(a) / positive_sentences
    

    @staticmethod
    def inference_summary(text1, text2, inference_result, threshold=0.91):
        result_dict = {label: [] for label in text2}
        total_dict = {label: [] for label in text2}
        for i, sentence in enumerate(text1):
            for res in inference_result[i]:
                if res['label'] == 'entailment' and res['score'] > threshold:
                    result_dict[text2[i]].append(sentence)
                    total_dict[text2[i]].append(1)
                else:
                    total_dict[text2[i]].append(0)
        return result_dict, total_dict

    @staticmethod
    def sentscore(a, b, weight=True):
        """Calculates a sentiment score."""
        length = len(a)
        if length == 0 or length != len(b):
            return None
        num = len([x for x in a if x > 0])
        if num == 0:
            return None
        return np.dot(a, b) / num if weight else np.dot([1 if x > 0 else 0 for x in a], b)
        
    @staticmethod
    def match_count_lowStat(self,texts, match_sets, phrases=True, suppress=None):
        """Performs match counting across texts with optional phrase matching."""
        count_dict = {label: {matchw: 0 for matchw in match_set['unigrams'].union(match_set['bigrams'])} for label, match_set in match_sets.items()}
        total_counts = {label: [] for label in match_sets.keys()}

        for text in texts:
            counted = {label: 0 for label in match_sets.keys()}
            unigrams = self.word_tokenize(text)
            bigrams = ['_'.join(g) for g in self.find_ngrams(unigrams, 2)]
            text = text.lower()

            for label, match_set in match_sets.items():
                if any(item in text for item in suppress[label]):
                    continue

                for word in unigrams:
                    if word in match_set['unigrams']:
                        count_dict[label][word] += 1
                        counted[label] += 1

                for word in bigrams:
                    if word in match_set['bigrams']:
                        count_dict[label][word] += 1
                        counted[label] += 1

                if phrases and any(phrase in text for phrase in match_set['phrases']):
                    counted[label] += 1

            for label in match_sets.keys():
                total_counts[label].append(counted[label])

        return {label: {'total': total_counts[label], 'stats': count_dict[label]} for label in match_sets.keys()}