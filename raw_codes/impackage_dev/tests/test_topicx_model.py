# Databricks notebook source
# MAGIC %run "./../algorithms/topicx_model"

# COMMAND ----------

# MAGIC %run "./../utilities/config_utility"

# COMMAND ----------

import pandas as pd

# COMMAND ----------

"""
Nutter Fixture for testing the Topicx model module.
"""

from runtime.nutterfixture import NutterFixture
class TopicxFixture(NutterFixture):
   """
   This Topicx fixture is used for unit testing all the methods that are used in the topicx_model.py module 
   """
   def __init__(self):
      """
      helps in intializing all the instance variables
      """
      self.topicx_obj=TopicXModel()
      self.match_df = pd.read_csv(config.CT_fundamentals_path)
      self.word_set_dict = {topic.upper() : self.topicx_obj.get_match_set(self.match_df[self.match_df['label']==topic]['word'].values) for topic in self.match_df['label'].unique()}
      self.currdf_filt=pd.read_parquet("/dbfs/mnt/ajaya/CT_unit_testing/CT_unit_test_filt_df.parquet")
      self.currdf=pd.read_parquet("/dbfs/mnt/ajaya/CT_unit_testing/CT_unit_test_df.parquet")
      self.combined_currdf=pd.read_parquet("/dbfs/mnt/ajaya/CT_unit_testing/CT_unit_test_combined_df.parquet")
      
      NutterFixture.__init__(self)
      
   def assertion_get_match_set(self):
      """
      This method is used for unit testing TopicxModel.get_match_set method
      """
      word_set_dict = {topic.upper() : self.topicx_obj.get_match_set(self.match_df[self.match_df['label']==topic]['word'].values) for topic in self.match_df['label'].unique()}
      assert ('revenue' in word_set_dict['REVENUE']['original'] and 'topline' in word_set_dict['REVENUE']['unigrams'] and 'top_line' in word_set_dict['REVENUE']['bigrams'])

   def assertion_match_count(self):
      """
      This method is used for unit testing TopicxModel.match_count method
      """

      text_lst=[
  "As always, Christoph will be available afterwards for subsequent questions regarding our figures that you may have.",
  "Before we begin, I'd like to share with you some information that relate to the very transmission of this conference.",
  "As already described in our invitation to this call, we recommend that you only use one of the dial-in options given.",
  "Either follow the presentation live via webcast or by telephone.",
  "If you dial in via webcast and via telephone at the same time, there will be a technical delay and please understand that you can only ask questions over the phone.",
  "And now, let's focus on the presentation.",
  "I will first begin with a short overview of the Q1 figures 2023, and afterwards, I'll provide you with more details on our current financial situation.",
  "As always, later on or last time, Sarah will give you some insights into the development of our operational business.",
  "So, let's start with the key figures for the first quarter of this year.",
  "Yeah, as we all know, market environment continues to be largely determined by the war in Ukraine and the associated economic consequences.",
  "Despite those ongoing challenging market conditions, however, we are still continuing to keep on track both strategically and operationally and had a good start, as we think, into 2023.",
  "Compared to the first quarter of 2022, we had an increase in rental income of more than 6%, mainly as a result of rent increases from property additions and indexation effects.",
  "FFO increased by over 23% and amounted to €13.6 million, which corresponds to an FFO per share of €0.17.",
  "NAV per share slightly increased during Q1 by 1.4% to €12.03.",
  "And the operational figures remained very solid with a WALT of 6.5 years and a vacancy rate of 3.3%.",
  "Yeah, and the vacancy rate has been affected by a temporary vacancy in our manage-to-core property in Mainz.",
  "More details on that later on by Sarah.",
  "Our financing structure remained very solid, which is reflected in a REIT equity ratio of 60.4% and an LTV which compared to year-end 2022 has dropped to 38.2%.",
  "Let's now have a closer look at the operating and financial figures for the first quarter of this year.",
  "I continue with a closer look at our FFO development.",
  "As already mentioned, the increase in rental income was mainly caused by property additions and indexations.",
  "Total maintenance expenses decreased to around €1.5 million year-on-year.",
  "Those expenses relate to regular ongoing maintenance and various smaller plant measures in the first quarter of this year.",
  "We expect maintenance activities to rise during the next month and still assume total costs for 2023 at around previous year's level.",
  "Personnel expenses increased mainly due to onboarding of additional employees in the course of 2022.",
  "Other operating income amounts to €1.7 million, mainly due to the compensation payment for the early termination of the contract of a tenant in the property in Mainz.",
  "The increase in interest expenses is mainly the result of interest payments for loans refinanced in 2022, as well as of higher expenses due to increased interest rates for the floating-rate bonded loans.",
  "Rising interest payments were partly offset by further interest income from cash deposits in an amount of €600,000.",
  "On the next slide, yeah, we provide an overview regarding our current financial situation as of end of first quarter of this year.",
  "The REIT equity ratio as well as LTV level both remained very solid.",
  "Further debt indicators such as net debt-to-EBITDA and EBITDA-to-interest coverage still remain at comfortable levels of 9.7 and 4.4, respectively.",
  "Compared to year-end 2022, our financial liabilities have reduced as a result of the scheduled repayment of the bonded loans due in March 2023.",
  "Given the current interest environment as well as our comfortable liquidity situation, we decided to fully repay the loan and to procure the potential raise of additional debt depending on further development of transaction markets in the course of this year.",
  "Our average financing costs are still on a comfortable level at 1.8%, with an average remaining term of loans of 4.8 years.",
  "So, all in all, our financial situation remains very solid.",
  "Yeah, and Sarah, she will now provide you with more details on our operational business development.",
  ", Sarah, and let me conclude the presentation with a short outlook.",
  "Yeah, given the positive business development in Q1 and despite the continuing uncertainty, we are able to confirm our full year guidance for 2023.",
  "Rental income is still expected to be between €88 million and €89.5 million, and the operating result in a range between €50 million and €52 million.",
  "Development of rents and FFO will be mainly affected by our transaction activities in the course of this year.",
  "Corresponding to our original guidance, we still assume a net investment volume of around €50 million.",
  "Besides investment activities, our earnings situation will be mainly influenced by further development of rent markets, interest environment and corresponding cost effects.",
  "And with reference to NAV per share, we aim at a number slightly below previous year's level.",
  "This assumption includes further possible value adjustments in our property portfolio in the remainder of the year.",
  "And with that, ladies and gentlemen,  very much for your attention, and we are now looking forward to your questions.",
  "Let me start with a few comments on the recent letting activities in our manage-to-core property in Mainz.",
  "As you might have seen in our yesterday's press announcement, we were able to sign a new lease for our property in Mainz, which leads to several impact on our portfolio figures shown on the following slides.",
  "We acquired this property as part of our manage-to-core strategy in 2021 and the asset was fully let to a single tenant from the insurance sector.",
  "And we recently agreed with this tenant on an early prolongation of the contract that was originally due in Q1 2024.",
  "We were now able to sign a lease contract with the city of Mainz, a tenant with a best credit rating, and at a rent level that is slightly above market and a lease term initially of at least 4.8 years.",
  "Due to the current general legal restrictions regarding the accommodation of refugees, we agreed a special termination right for the tenant as of end of December 2027.",
  "In case of changing the regulations, we intend to continue our cooperation with the city of Mainz beyond 2027.",
  "To underline the importance of sustainability topics once again, this lease contains so-called green lease classes, that includes, for example, sharing of consumption data as well as the commitment to operate the property with renewable energy in the future.",
  "The signing of the new lease in Q4 2022 led to a significant value enhancement of the property by around 21.5% compared to its purchase price in 2021.",
  "All things considered, this project demonstrates our approach on how to generate value with manage-to-core properties and is a success not just from an economic perspective, but socially as well.",
  "Let me now continue with the view on the portfolio key metrics.",
  "In the first quarter of this year, there were no changes within our property portfolio.",
  "However, portfolio value increased slightly following the signing of the sales contracts for a smaller retail property in Mosbach.",
  "The fully vacant asset was sold to the local authority at a purchase price around €0.5 million above the latest market value.",
  "The transfer of ownership is expected within the second quarter of this year.",
  "The impairment adjustment of the property caused a slight increase of our portfolio value to €1.61 billion by end of March.",
  "The average remaining lease term remains at a consistently high level of 6.5 years, and the EPRA vacancy rate slightly increased to 3.3% year-to-date, which is mainly due to the temporary vacancy, while the construction works in Mainz are ongoing before we hand over this property to the new tenant.",
  "The vacancy is also reflected in the ratios of our manage-to-core portfolio shown on the lower-right hand side of this chart.",
  "And following our letting activities in Q1, the figures within our core portfolio improved with an office WALT of 5.2 years and a low vacancy rate of just 0.9%.",
  "Coming to the rent development, as you can see in the development of our annualized rents, we saw positive like-for-like effects of 3.9% of the total portfolio.",
  "Again, this metric is also influenced by the current vacancy in Mainz.",
  "The like-for-like adjustment within the core portfolio were 6%.",
  "We could benefit from rent increases due to indexation amounting to 4.7% for the overall portfolio.",
  "Depending on further inflation development, we can expect some additional positive effects for the rest of this year.",
  "Let's look on the next slide showing the current leasing situation.",
  "As already mentioned, we secured several expiring and new leases in the first quarter of 2023.",
  "In Q1, contracts were signed for rental space of around 23,200 square meter, an increase by 33% in comparison to the first quarter of the last year.",
  "Office spaces account for 75%.",
  "The leases currently outstanding for renewal in 2023 are at 4.3%, and we stay confident to sign further agreements over this year.",
  "Finally, a few comments on our tenant structure and tenant base.",
  "Compared to year-end 2022, there is a change in our top-10 tenants overview caused by the lease termination with the previous tenant in Mainz.",
  "So, the tenant was replaced by Immobilien Freistaat Bayern, the real estate company of the Federal State of Bavaria with office spaces in our properties in Erlangen, the location at Wetterkreuz and in Ingolstadt.",
  "There are no significant changes in our sector distribution, as food retailers still account for one-third of company's total annual rent, and we still have a diversified tenant base.",
  "Around 12% were generated with tenants from the DIY sector.",
  "In total, office tenants contribute with around 43% to total annual rent.",
  "To summarize, our tenant structure is still very solid and reliable.",
  "And with that, I hand back to Niclas for a short outlook and guidance update for the rest of the year."
]
      count_lst=[]
      for sentence in text_lst:
        count_lst.append(self.topicx_obj.match_count(sentence, self.word_set_dict, phrases = False))
      assert(len(count_lst)==82)
   
   def assertion_match_count_phrases_true(self):
      """
      This method is used for unit testing TopicxModel.match_count method
      """

      text_lst=[
  "As always, Christoph will be available afterwards for subsequent questions regarding our figures that you may have.",
  "Before we begin, I'd like to share with you some information that relate to the very transmission of this conference.",
  "As already described in our invitation to this call, we recommend that you only use one of the dial-in options given.",
  "Either follow the presentation live via webcast or by telephone.",
  "If you dial in via webcast and via telephone at the same time, there will be a technical delay and please understand that you can only ask questions over the phone.",
  "And now, let's focus on the presentation.",
  "I will first begin with a short overview of the Q1 figures 2023, and afterwards, I'll provide you with more details on our current financial situation.",
  "As always, later on or last time, Sarah will give you some insights into the development of our operational business.",
  "So, let's start with the key figures for the first quarter of this year.",
  "Yeah, as we all know, market environment continues to be largely determined by the war in Ukraine and the associated economic consequences.",
  "Despite those ongoing challenging market conditions, however, we are still continuing to keep on track both strategically and operationally and had a good start, as we think, into 2023.",
  "Compared to the first quarter of 2022, we had an increase in rental income of more than 6%, mainly as a result of rent increases from property additions and indexation effects.",
  "FFO increased by over 23% and amounted to €13.6 million, which corresponds to an FFO per share of €0.17.",
  "NAV per share slightly increased during Q1 by 1.4% to €12.03.",
  "And the operational figures remained very solid with a WALT of 6.5 years and a vacancy rate of 3.3%.",
  "Yeah, and the vacancy rate has been affected by a temporary vacancy in our manage-to-core property in Mainz.",
  "More details on that later on by Sarah.",
  "Our financing structure remained very solid, which is reflected in a REIT equity ratio of 60.4% and an LTV which compared to year-end 2022 has dropped to 38.2%.",
  "Let's now have a closer look at the operating and financial figures for the first quarter of this year.",
  "I continue with a closer look at our FFO development.",
  "As already mentioned, the increase in rental income was mainly caused by property additions and indexations.",
  "Total maintenance expenses decreased to around €1.5 million year-on-year.",
  "Those expenses relate to regular ongoing maintenance and various smaller plant measures in the first quarter of this year.",
  "We expect maintenance activities to rise during the next month and still assume total costs for 2023 at around previous year's level.",
  "Personnel expenses increased mainly due to onboarding of additional employees in the course of 2022.",
  "Other operating income amounts to €1.7 million, mainly due to the compensation payment for the early termination of the contract of a tenant in the property in Mainz.",
  "The increase in interest expenses is mainly the result of interest payments for loans refinanced in 2022, as well as of higher expenses due to increased interest rates for the floating-rate bonded loans.",
  "Rising interest payments were partly offset by further interest income from cash deposits in an amount of €600,000.",
  "On the next slide, yeah, we provide an overview regarding our current financial situation as of end of first quarter of this year.",
  "The REIT equity ratio as well as LTV level both remained very solid.",
  "Further debt indicators such as net debt-to-EBITDA and EBITDA-to-interest coverage still remain at comfortable levels of 9.7 and 4.4, respectively.",
  "Compared to year-end 2022, our financial liabilities have reduced as a result of the scheduled repayment of the bonded loans due in March 2023.",
  "Given the current interest environment as well as our comfortable liquidity situation, we decided to fully repay the loan and to procure the potential raise of additional debt depending on further development of transaction markets in the course of this year.",
  "Our average financing costs are still on a comfortable level at 1.8%, with an average remaining term of loans of 4.8 years.",
  "So, all in all, our financial situation remains very solid.",
  "Yeah, and Sarah, she will now provide you with more details on our operational business development.",
  ", Sarah, and let me conclude the presentation with a short outlook.",
  "Yeah, given the positive business development in Q1 and despite the continuing uncertainty, we are able to confirm our full year guidance for 2023.",
  "Rental income is still expected to be between €88 million and €89.5 million, and the operating result in a range between €50 million and €52 million.",
  "Development of rents and FFO will be mainly affected by our transaction activities in the course of this year.",
  "Corresponding to our original guidance, we still assume a net investment volume of around €50 million.",
  "Besides investment activities, our earnings situation will be mainly influenced by further development of rent markets, interest environment and corresponding cost effects.",
  "And with reference to NAV per share, we aim at a number slightly below previous year's level.",
  "This assumption includes further possible value adjustments in our property portfolio in the remainder of the year.",
  "And with that, ladies and gentlemen,  very much for your attention, and we are now looking forward to your questions.",
  "Let me start with a few comments on the recent letting activities in our manage-to-core property in Mainz.",
  "As you might have seen in our yesterday's press announcement, we were able to sign a new lease for our property in Mainz, which leads to several impact on our portfolio figures shown on the following slides.",
  "We acquired this property as part of our manage-to-core strategy in 2021 and the asset was fully let to a single tenant from the insurance sector.",
  "And we recently agreed with this tenant on an early prolongation of the contract that was originally due in Q1 2024.",
  "We were now able to sign a lease contract with the city of Mainz, a tenant with a best credit rating, and at a rent level that is slightly above market and a lease term initially of at least 4.8 years.",
  "Due to the current general legal restrictions regarding the accommodation of refugees, we agreed a special termination right for the tenant as of end of December 2027.",
  "In case of changing the regulations, we intend to continue our cooperation with the city of Mainz beyond 2027.",
  "To underline the importance of sustainability topics once again, this lease contains so-called green lease classes, that includes, for example, sharing of consumption data as well as the commitment to operate the property with renewable energy in the future.",
  "The signing of the new lease in Q4 2022 led to a significant value enhancement of the property by around 21.5% compared to its purchase price in 2021.",
  "All things considered, this project demonstrates our approach on how to generate value with manage-to-core properties and is a success not just from an economic perspective, but socially as well.",
  "Let me now continue with the view on the portfolio key metrics.",
  "In the first quarter of this year, there were no changes within our property portfolio.",
  "However, portfolio value increased slightly following the signing of the sales contracts for a smaller retail property in Mosbach.",
  "The fully vacant asset was sold to the local authority at a purchase price around €0.5 million above the latest market value.",
  "The transfer of ownership is expected within the second quarter of this year.",
  "The impairment adjustment of the property caused a slight increase of our portfolio value to €1.61 billion by end of March.",
  "The average remaining lease term remains at a consistently high level of 6.5 years, and the EPRA vacancy rate slightly increased to 3.3% year-to-date, which is mainly due to the temporary vacancy, while the construction works in Mainz are ongoing before we hand over this property to the new tenant.",
  "The vacancy is also reflected in the ratios of our manage-to-core portfolio shown on the lower-right hand side of this chart.",
  "And following our letting activities in Q1, the figures within our core portfolio improved with an office WALT of 5.2 years and a low vacancy rate of just 0.9%.",
  "Coming to the rent development, as you can see in the development of our annualized rents, we saw positive like-for-like effects of 3.9% of the total portfolio.",
  "Again, this metric is also influenced by the current vacancy in Mainz.",
  "The like-for-like adjustment within the core portfolio were 6%.",
  "We could benefit from rent increases due to indexation amounting to 4.7% for the overall portfolio.",
  "Depending on further inflation development, we can expect some additional positive effects for the rest of this year.",
  "Let's look on the next slide showing the current leasing situation.",
  "As already mentioned, we secured several expiring and new leases in the first quarter of 2023.",
  "In Q1, contracts were signed for rental space of around 23,200 square meter, an increase by 33% in comparison to the first quarter of the last year.",
  "Office spaces account for 75%.",
  "The leases currently outstanding for renewal in 2023 are at 4.3%, and we stay confident to sign further agreements over this year.",
  "Finally, a few comments on our tenant structure and tenant base.",
  "Compared to year-end 2022, there is a change in our top-10 tenants overview caused by the lease termination with the previous tenant in Mainz.",
  "So, the tenant was replaced by Immobilien Freistaat Bayern, the real estate company of the Federal State of Bavaria with office spaces in our properties in Erlangen, the location at Wetterkreuz and in Ingolstadt.",
  "There are no significant changes in our sector distribution, as food retailers still account for one-third of company's total annual rent, and we still have a diversified tenant base.",
  "Around 12% were generated with tenants from the DIY sector.",
  "In total, office tenants contribute with around 43% to total annual rent.",
  "To summarize, our tenant structure is still very solid and reliable.",
  "And with that, I hand back to Niclas for a short outlook and guidance update for the rest of the year."
]
      count_lst=[]
      for sentence in text_lst:
        count_lst.append(self.topicx_obj.match_count(sentence, self.word_set_dict))
      assert(len(count_lst)==82 and "phrase" in count_lst[0]["REVENUE"].keys())
   def assertion_generate_topic_statistics(self):
      """
      This method is used for unit testing TopicxModel.generate_topic_statistics method.
      """
      self.currdf=self.topicx_obj.generate_topic_statistics(self.currdf,self.word_set_dict)

      assert ({'FILT_MD','FILT_QA','FILT_EXEC_QA','FILT_EXEC_MD','LEN_FILT_MD','RAW_LEN_FILT_MD','REVENUE_TOTAL_FILT_MD','REVENUE_STATS_FILT_MD','REVENUE_STATS_LIST_FILT_MD'}.issubset(self.currdf.columns))
   def assertion_generate_sentence_relevance_score(self):
      """
      This method is used for unit testing TopicxModel.generate_sentence_relevance_score method.
      """

      self.combined_currdf=self.topicx_obj.generate_sentence_relevance_score(self.combined_currdf,self.word_set_dict)
      assert ({'FILT_MD','FILT_QA','FILT_EXEC_QA','FILT_EXEC_MD','LEN_FILT_MD','RAW_LEN_FILT_MD','REVENUE_TOTAL_FILT_MD','REVENUE_STATS_FILT_MD','REVENUE_STATS_LIST_FILT_MD','SENT_FILT_MD','NET_SENT_FILT_MD','REVENUE_RELEVANCE_FILT_MD','REVENUE_SENT_FILT_MD','REVENUE_SENT_REL_FILT_MD'}.issubset(self.combined_currdf.columns))

   def assertion_merge_no_match(self):
      """
      This method is used for unit testing DBFSUtility.merge method.
      """

      merge_msg=self.topicx_obj.mergeCount([{'revenue': 0, 'topline': 0, 'operational_sales': 0, 'top_line': 0, 'net_revenue': 0, 'sale': 0, 'operational_sale': 0, 'sales': 0},{'revenue':0, 'topline': 0, 'operational_sales': 0, 'top_line': 0, 'net_revenue': 0, 'sale': 0, 'operational_sale': 0, 'sales': 0}])
      assert(merge_msg["NO_MATCH"]==1)
   def assertion_merge(self):
      """
      This method is used for unit testing DBFSUtility.merge method.
      """

      merge_msg=self.topicx_obj.mergeCount([{'revenue': 0, 'topline': 0, 'operational_sales': 0, 'top_line': 0, 'net_revenue': 0, 'sale': 0, 'operational_sale': 0, 'sales': 0},{'revenue': 1, 'topline': 0, 'operational_sales': 0, 'top_line': 0, 'net_revenue': 0, 'sale': 0, 'operational_sale': 0, 'sales': 0}])
      assert(merge_msg["revenue"]==1)

   def assertion_merge_exception(self):
      """
      This method is used for unit testing DBFSUtility.merge method.
      """

      merge_msg=self.topicx_obj.mergeCount([])
      assert(merge_msg["ERROR"]==1)

   def assertion_sentscore_empty_list(self):
      """
      This method is used for unit testing DBFSUtility.sentscore method.
      """

      sent_score=self.topicx_obj.sentscore([],[])
      assert(sent_score is None)

   def assertion_sentscore_unequal_list(self):
      """
      This method is used for unit testing DBFSUtility.sentscore method.
      """

      sent_score=self.topicx_obj.sentscore([1],[1,1])
      assert(sent_score is None)
   def assertion_sentscore_equal_list(self):
      """
      This method is used for unit testing DBFSUtility.sentscore method.
      """

      sent_score=self.topicx_obj.sentscore([1,2],[1,2])
      assert(sent_score==2.5)
   
   def assertion_sentscore_unequal_list_with_False(self):
      """
      This method is used for unit testing DBFSUtility.sentscore method.
      """

      sent_score=self.topicx_obj.sentscore([1,2],[1,2],False)
      assert(sent_score==1.5)
   
   def assertion_netscore_empty_list(self):
      """
      This method is used for unit testing DBFSUtility.sentscore method.
      """

      net_score=self.topicx_obj.sentscore([],[])
      assert(net_score is None)

   def assertion_netscore_unequal_list(self):
      """
      This method is used for unit testing DBFSUtility.sentscore method.
      """

      net_score=self.topicx_obj.netscore([1],[1,1])
      assert(net_score is None)
   def assertion_netscore_equal_list(self):
      """
      This method is used for unit testing DBFSUtility.sentscore method.
      """

      net_score=self.topicx_obj.netscore([1,2],[1,2])
      assert(net_score==3)
   
result = TopicxFixture().execute_tests()
print(result.to_string())
