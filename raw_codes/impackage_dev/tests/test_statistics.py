# Databricks notebook source
# MAGIC %run "./../algorithms/statistics"

# COMMAND ----------

import numpy as np

# COMMAND ----------

"""
Nutter Fixture for testing the config module.
"""

from runtime.nutterfixture import NutterFixture
class StatisticsFixture(NutterFixture):
   """
   This Statistics fixture is used for unit testing all the methods that are used in the statistics.py module 
   """
   def __init__(self):
      """
      helps in intializing all the instance variables

      """
      self.stat_obj=Statistics()
      NutterFixture.__init__(self)

   def assertion_get_dollar_amounts(self):
      """
      This method is used for unit testing StatisticsFixture.get_dollar_amounts method.
      """
      txt="The new M2 chip makes the 13‑inch MacBook Pro more capable than ever.MacBook Pro cost is $100. The same compact design supports up to 20 hours of battery life1 and an active cooling system to sustain enhanced performance. Featuring a brilliant Retina display, a FaceTime HD camera, and studio‑quality mics, it’s our most portable pro laptop."
      val=self.stat_obj.get_dollar_amounts(txt)
      assert (val[0]==1 and val[1]==36)

   def assertion_get_dollar_amounts_empty_text(self):
      """
      This method is used for unit testing StatisticsFixture.get_dollar_amounts empty text method.
      """
      txt=""
      val=self.stat_obj.get_dollar_amounts(txt)
      assert (np.isnan(val[0]))

   def assertion_get_numbers(self):
      """
      This method is used for unit testing StatisticsFixture.get_dollar_amounts empty text method.
      """
      txt_lst=["You can get a good laptop on a budget $1000","you just have to know which one to pick", "Here are what we recommend for those looking to spend anywhere from $300 to $800 on a new laptop for school, work, or play"]
      val=self.stat_obj.get_numbers(txt_lst)
      assert (val[0]==3 and val[1]==13) 

   def assertion_get_numbers_empty(self):
      """
      This method is used for unit testing StatisticsFixture.get_numbers empty text method.
      """
      txt=[]
      val=self.stat_obj.get_numbers(txt)
      assert (np.isnan(val[0]))

   def assertion_combine_sent_returns_zero(self):
      """
      This method is used for unit testing StatisticsFixture.combine_sent method.
      """
      val=self.stat_obj.combine_sent(1,-1)
      assert (val==0)

   def assertion_combine_sent(self):
      """
      This method is used for unit testing StatisticsFixture.combine_sent method.
      """
      val=self.stat_obj.combine_sent(2,1)
      assert (round(val,3)==0.333)

   def assertion_nan_sum(self):
      """
      This method is used for unit testing StatisticsFixture.combine_sent method.
      """
      val=self.stat_obj.nan_sum(2,1,np.nan,4)
      assert (val==7.0)
   def assertion_section_level_bert_sentiment_score(self):
      """
      This method is used for unit testing StatisticsFixture.combine_sent method.
      """
      val=self.stat_obj.section_level_bert_sentiment_score([1])
      assert(val[0]==1.0 and val[1]==0 and val[2]==1 and val[3]==1)

   def assertion_section_level_bert_sentiment_score(self):
      """
      This method is used for unit testing StatisticsFixture.combine_sent method.
      """
      val=self.stat_obj.section_level_bert_sentiment_score([])
      assert(np.isnan(val[0]))

result = StatisticsFixture().execute_tests()
print(result.to_string())
