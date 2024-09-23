# Databricks notebook source
# MAGIC %run "./../preprocessor/topicx_preprocessor"

# COMMAND ----------

"""
Nutter Fixture for testing the TopicXPreprocessor module.
"""

from runtime.nutterfixture import NutterFixture
class TopicXPreprocessorFixture(NutterFixture):
   """
   This TopicXPreprocessor fixture is used for unit testing all the methods that are used in the topicx_preprocessor.py module 
   """
   def __init__(self):
      """
      helps in intializing all the instance variables
      """
      self.topicx_preprocessor_obj=TopicXPreprocessor()
      NutterFixture.__init__(self)
      
     
   def assertion_find_ngrams(self):
      """
      This method is used for unit testing TopicXPreprocessor.find_ngrams method
      """
      assert (list(self.topicx_preprocessor_obj.find_ngrams(["ajay","babu","data scientist"],2) )== [('ajay','babu'), ('babu', 'data scientist')])

   def assertion_process_sentence(self):
      """
      This method is used for unit testing TopicXPreprocessor.process_sentence method
      """
      assert (self.topicx_preprocessor_obj.process_sentence("thank you, good morning, welcome to the house of voya financial services") == None)
   def assertion_process_sentence_not_None(self):
      """
      This method is used for unit testing TopicXPreprocessor.process_sentence method
      """
      assert (self.topicx_preprocessor_obj.process_sentence("thank you to the house of voya financial services") == " to the house of voya financial services")

   def assertion_word_tokenizer(self):
      """
      This method is used for unit testing TopicXPreprocessor.word_tokenizer method
      """
      assert (self.topicx_preprocessor_obj.word_tokenizer("thank you ii to the house of voya financial services") == ['thank', 'ii', 'house', 'voya', 'financial', 'service'])

   def assertion_tokenizer_for_matched_words(self):
      """
      This method is used for unit testing TopicXPreprocessor.tokenizer_for_matched_words method
      """
      assert (self.topicx_preprocessor_obj.tokenizer_for_matched_words("Thank you, to the House of VOYA 9999 financial services") == ['thank', 'house', 'voya', 'financial', 'service'])
   

result = TopicXPreprocessorFixture().execute_tests()
print(result.to_string())
