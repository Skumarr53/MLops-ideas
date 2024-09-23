# Databricks notebook source
# MAGIC %run "./../dfutils/dataframe_utils"

# COMMAND ----------

import pandas as pd

# COMMAND ----------

"""
Nutter Fixture for testing the dataframe_utils module.
"""

from runtime.nutterfixture import NutterFixture
class DFUtilsFixture(NutterFixture):
   """
   This DFUtils fixture is used for unit testing all the methods that are used in the dataframe_utils.py module 
   """
   def __init__(self):
      """
      helps in intializing all the instance variables

      instance variable: dfutils object is created
      """
      self.df_utils_obj=DFUtils()
      sent_lst=["ajay","ajay","babu","babu"]
      self.currdf = pd.DataFrame({"FILT_MD": [["ajay","babu","ajay","babu"]],
                   "FILT_QA": [["ajay","babu","ajay","babu"]],
                   "FILT_CEO_MD": [["ajay","babu","ajay","babu"]],
                   "FILT_CEO_QA": [["ajay","babu","ajay","babu"]],
                   "FILT_EXEC_QA": [["ajay","babu","ajay","babu"]],
                   "FILT_EXEC_MD": [["ajay","babu","ajay","babu"]],
                   "FILT_ANL_QA": [["ajay","babu","ajay","babu"]]})
      self.empty_df = pd.DataFrame({"FILT_MD": [[]],
                   "FILT_QA": [["ajay","babu"]],
                   "FILT_CEO_MD": [[]],
                   "FILT_CEO_QA": [[]],
                   "FILT_EXEC_QA": [[]],
                   "FILT_EXEC_MD": [[]],
                   "FILT_ANL_QA": [[]]})
      self.list_df = pd.DataFrame({
                   "FILT_QA":  [np.array([0,1,2,3])]
                   })
      NutterFixture.__init__(self)
      
    
   def assertion_df_get_percentage_of_duplicate_text_in_sentence_list(self):
      """
      This method is used for unit testing DFUtils.df_get_percentage_of_duplicate_text_in_sentence_list method.
      """
      currdf=self.df_utils_obj.df_get_percentage_of_duplicate_text_in_sentence_list(self.currdf)
      assert(currdf["SIGNIFICANT_DUPLICATE_FILT_MD"][0]==50.0)

   def assertion_df_get_percentage_of_duplicate_text_in_sentence_list_empty(self):
      """
      This method is used for unit testing DFUtils.df_get_percentage_of_duplicate_text_in_sentence_list on empty list method.
      """
      empty_df=self.df_utils_obj.df_get_percentage_of_duplicate_text_in_sentence_list(self.empty_df)
      assert(empty_df["SIGNIFICANT_DUPLICATE_FILT_MD"][0]==0)

   def assertion_df_remove_duplicate_text_from_sentence_list(self):
      """
      This method is used for unit testing DFUtils.df_remove_duplicate_text_from_sentence_list method.
      """
      currdf=self.df_utils_obj.df_remove_duplicate_text_from_sentence_list(self.currdf)
      assert(len(currdf["FILT_MD"][0])==2)

   def assertion_df_remove_duplicate_text_from_sentence_list_no_duplicate(self):
      """
      This method is used for unit testing DFUtils.df_remove_duplicate_text_from_sentence_list method.
      """
      print("length of list in FILT_QA value before applying deduplication is  {0}".format(len(self.empty_df["FILT_QA"][0])))
      empty_df=self.df_utils_obj.df_remove_duplicate_text_from_sentence_list(self.empty_df)
      print("length of list in FILT_QA value after applying deduplication is  {0}".format(len(self.empty_df["FILT_QA"][0])))
      assert(len(empty_df["FILT_QA"][0])==2)

   def assertion_df_column_convert_to_list(self):
      """
      This method is used for unit testing DFUtils.df_column_convert_to_list method.
      """
      list_df=self.df_utils_obj.df_column_convert_to_list(self.list_df,["FILT_QA"])
      assert(type(list_df["FILT_QA"][0])==list)
   
   

result = DFUtilsFixture().execute_tests()
print(result.to_string())
