# Databricks notebook source
# MAGIC %run "./../preprocessor/dictionary_model_preprocessor"

# COMMAND ----------

!pip install nutter

# COMMAND ----------

from unittest.mock import MagicMock, patch
import re
from runtime.nutterfixture import NutterFixture
    
class TestDictionaryModelPreprocessor(NutterFixture):
    
    def __init__(self):
        self.preprocessor = DictionaryModelPreprocessor()
        self.preprocessor.stop_words_List = ['a', 'an', 'the']  # Example stop words list
        NutterFixture.__init__(self)

    
    @patch('spacy.load')
    def assertion_word_tokenizer(self, mock_spacy_load):
        mock_spacy = MagicMock()
        mock_spacy_load.return_value = mock_spacy
        mock_doc = MagicMock()
        mock_doc.__iter__.return_value = [
            MagicMock(lemma_='word1'),
            MagicMock(lemma_='word2'),
            MagicMock(lemma_='the')
        ]
        mock_spacy.return_value = mock_doc
        
        text = "Word1 word2 the"
        expected_output = ['word1', 'word2']
        
        result = self.preprocessor.word_tokenizer(text)
        assert result == expected_output, f"Expected {expected_output}, but got {result}"
    
    def assertion_preprocess_text(self):
        text = 'This is a "sample" text with single letters a b c.'
        expected_output = (
            'this is sample text with single letters',
            ['this', 'be', 'sample', 'text', 'with', 'single', 'letter'],
            7
        )
        
        result = self.preprocessor.preprocess_text(text)
        assert result == expected_output, f"Expected {expected_output}, but got {result}"
    
    def assertion_preprocess_text_list(self):
        text_list = [
            'This is a "sample" text.',
            'Another text with single letters a b c.'
        ]
        expected_output = (
            [
                'this is sample text',
                'another text with single letters'
            ],
            [
                ['this', 'be', 'sample', 'text'],
                ['another', 'text', 'with', 'single', 'letter']
            ],
            [4, 5]
        )
        
        result = self.preprocessor.preprocess_text_list(text_list)
        assert result == expected_output, f"Expected {expected_output}, but got {result}"
    
    def assertion_clean_text(self):
        text = 'This is a "sample" text with single letters a b c.'
        expected_output = 'this is sample text with single letters'
        
        result = self.preprocessor._clean_text(text)
        assert result == expected_output, f"Expected {expected_output}, but got {result}"
    
    def assertion_expand_contractions(self):
        text = "ain't going to test this."
        expected_output = "is not going to test this."
        
        result = self.preprocessor.expand_contractions(text)
        assert result == expected_output, f"Expected {expected_output}, but got {result}"

# Run the tests
result = TestDictionaryModelPreprocessor().execute_tests()
print(result.to_string())
