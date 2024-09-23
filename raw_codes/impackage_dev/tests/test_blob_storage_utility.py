# Databricks notebook source
# MAGIC %pip install nutter

# COMMAND ----------

# MAGIC %run "./../filesystem/blob_storage_utility"

# COMMAND ----------

"""
Nutter Fixture for testing the blob storage utility module.
"""

import unittest
from unittest.mock import patch, mock_open
from runtime.nutterfixture import NutterFixture

class BlobStorageUtilityFixture(NutterFixture):
    """
    This BlobStorageUtility fixture is used for unit testing all the methods that are used in the blob_storage_utility.py module 
    """
    def __init__(self):
        """
        Helps in initializing all the instance variables
        """
        self.blob_storage_utility = BlobStorageUtility()
        NutterFixture.__init__(self)
    
    def assertion_load_content_from_txt(self):
        """
        This method is used for unit testing BlobStorageUtility.load_content_from_txt method 
        """
        mock_file_path = 'mock_file.txt'
        mock_content = 'This is a test content.'
        
        with patch('__main__.open', mock_open(read_data=mock_content)):
            content = self.blob_storage_utility.load_content_from_txt(mock_file_path)
            assert content == mock_content
    
    def assertion_load_content_from_txt_file_not_found(self):
        """
        This method is used for unit testing BlobStorageUtility.load_content_from_txt method when file is not found
        """
        mock_file_path = 'non_existent_file.txt'
        
        with patch('__main__.open', side_effect=FileNotFoundError):
            try:
                self.blob_storage_utility.load_content_from_txt(mock_file_path)
            except FilesNotLoadedException:
                assert True
            else:
                assert False
    
    def assertion_load_list_from_txt(self):
        """
        This method is used for unit testing BlobStorageUtility.load_list_from_txt method 
        """
        mock_file_path = 'mock_file.txt'
        mock_content = 'Line1\nLine2\nLine3'
        
        with patch('__main__.open', mock_open(read_data=mock_content)):
            result = self.blob_storage_utility.load_list_from_txt(mock_file_path)
            expected_result = {'line1', 'line2', 'line3'}
            assert result == expected_result
    
    def assertion_load_list_from_txt_no_lower(self):
        """
        This method is used for unit testing BlobStorageUtility.load_list_from_txt method with is_lower=False
        """
        mock_file_path = 'mock_file.txt'
        mock_content = 'Line1\nLine2\nLine3'
        
        with patch('__main__.open', mock_open(read_data=mock_content)):
            result = self.blob_storage_utility.load_list_from_txt(mock_file_path, is_lower=False)
            expected_result = {'Line1', 'Line2', 'Line3'}
            assert result == expected_result
    
    def assertion_read_syllable_count(self):
        """
        This method is used for unit testing BlobStorageUtility.read_syllable_count method 
        """
        mock_file_path = 'mock_file.txt'
        mock_content = 'word1 1\nword2 2\nword3 3'
        
        with patch('__main__.open', mock_open(read_data=mock_content)):
            result = self.blob_storage_utility.read_syllable_count(mock_file_path)
            expected_result = {'word1': 1, 'word2': 2, 'word3': 3}
            assert result == expected_result
    
    def assertion_read_syllable_count_file_not_found(self):
        """
        This method is used for unit testing BlobStorageUtility.read_syllable_count method when file is not found
        """
        mock_file_path = 'non_existent_file.txt'
        
        with patch('__main__.open', side_effect=FileNotFoundError):
            try:
                self.blob_storage_utility.read_syllable_count(mock_file_path)
            except FilesNotLoadedException:
                assert True
            else:
                assert False

result = BlobStorageUtilityFixture().execute_tests()
print(result.to_string())
