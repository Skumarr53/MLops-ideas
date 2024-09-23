# Databricks notebook source
# MAGIC %run "./../preprocessor/text_preprocessor"

# COMMAND ----------

"""
Nutter Fixture for testing the TextPreprocessor module.
"""

from runtime.nutterfixture import NutterFixture
class TextPreprocessorFixture(NutterFixture):
   """
   This TextPreprocessor fixture is used for unit testing all the methods that are used in the text_preprocessor.py module 
   """
   def __init__(self):
      """
      helps in intializing all the instance variables

      instance variable: TextPreprocessor is created
      """
      self.txt_preprocess_obj=TextPreprocessor()
      NutterFixture.__init__(self)
     
   def assertion_expand_contractions(self):
      """
      This method is used for unit testing TextPreprocessor.expand_contractions method 
      """
      txt="I'll"
      assert (self.txt_preprocess_obj.expand_contractions(txt) == "i will")
   def assertion_check_stopword(self):
      """
      This method is used for unit testing TextPreprocessor.check_stopword method.
      """
      txt="colon"
      assert (self.txt_preprocess_obj.check_stopword(txt) == 0)

   def assertion_check_stopword_not(self):
      """
      This method is used for unit testing TextPreprocessor.check_stopword method.
      """
      txt="finance"
      assert (self.txt_preprocess_obj.check_stopword(txt) == 1)
   def assertion_remove_accented_chars(self):
      """
      This method is used for unit testing TextPreprocessor.remove_accented_chars method.
      """
      txt="àjay"
      ret_txt=self.txt_preprocess_obj.remove_accented_chars(txt)
      assert (ret_txt == "ajay")

   def assertion_check_for_acquisition_merger(self):
      """
      This method is used for unit testing TextPreprocessor.check_for_acquisition_merger method.
      """
      txt=["recent","acquisition","voya","with","slk"]
      assert (self.txt_preprocess_obj.check_for_acquisition_merger(txt) == 1)

   def assertion_check_for_acquisition_merger_not(self):
      """
      This method is used for unit testing TextPreprocessor.check_for_acquisition_merger method.
      """
      txt=["recent","voya","acquired","slk"]
      assert (self.txt_preprocess_obj.check_for_acquisition_merger(txt) == 0)

   def assertion_word_negated(self):
      """
      This method is used for unit testing TextPreprocessor.word_negated method.
      """
      txt="cannot"
      assert (self.txt_preprocess_obj.word_negated(txt) == True)

   def assertion_word_negated_not(self):
      """
      This method is used for unit testing TextPreprocessor.word_negated method.
      """
      txt="Software"
      assert (self.txt_preprocess_obj.word_negated(txt) == False)

   def assertion_sentence_tokenizer(self):
      """
      This method is used for unit testing TextPreprocessor.sentence_tokenizer method.
      """
      txt="This is call transcript module. Performing unit testing for this module using Nutter."
      sent_txt=self.txt_preprocess_obj.sentence_tokenizer(txt)
      assert (len(sent_txt) == 2)

   def assertion_correctID(self):
      """
      This method is used for unit testing TextPreprocessor.correctID method.
      """
      lst=["Nutter"]
      assert (self.txt_preprocess_obj.correctID(lst) == ["Nutter"])
   def assertion_correctID_list(self):
      """
      This method is used for unit testing TextPreprocessor.correctID method.
      """
      lst=["Nutter","databricks","unit testing","package"]
      assert (self.txt_preprocess_obj.correctID(lst) == ["Nutter","databricks","unit testing","package"])

   def assertion_check_datatype(self):
      """
      This method is used for unit testing TextPreprocessor.check_datatype method.
      """
      lst=["Nutter","databricks","unit testing","package"]
      assert (self.txt_preprocess_obj.check_datatype(lst) == "Nutter databricks unit testing package")

   def assertion_check_datatype_str(self):
      """
      This method is used for unit testing TextPreprocessor.check_datatype method.
      """
      str="Nutter databricks unit testing package"
      assert (self.txt_preprocess_obj.check_datatype(str) == "Nutter databricks unit testing package")

   def assertion_check_datatype_not_str_or_list(self):
      """
      This method is used for unit testing TextPreprocessor.check_datatype method.
      """
      assert (self.txt_preprocess_obj.check_datatype("") == False)

   def assertion_check_sentence_for_operator_less_than_three(self):
      """
      This method is used for unit testing TextPreprocessor.check_sentence_for_operator method.
      """
      assert (self.txt_preprocess_obj.check_sentence_for_operator("call transcript") == 0)

   def assertion_check_sentence_for_operator(self):
      """
      This method is used for unit testing TextPreprocessor.check_sentence_for_operator method.
      """
      assert (self.txt_preprocess_obj.check_sentence_for_operator("call transcript operator returns") == 0)

   def assertion_check_sentence_for_no_operator(self):
      """
      This method is used for unit testing TextPreprocessor.check_sentence_for_operator method.
      """
      assert (self.txt_preprocess_obj.check_sentence_for_operator("call transcript use case") == 1)

   def assertion_process_sentence(self):
      """
      This method is used for unit testing TextPreprocessor.process_sentence method.
      """
      assert (self.txt_preprocess_obj.process_sentence("Thank you Bea") == " Bea")
   def assertion_checkCEO(self):
      """
      This method is used for unit testing TextPreprocessor.checkCEO method.
      """
      assert (self.txt_preprocess_obj.checkCEO("ceo of the voya is") ==1)
   def assertion_checkCEO_not(self):
      """
      This method is used for unit testing TextPreprocessor.checkCEO method.
      """
      assert (self.txt_preprocess_obj.checkCEO("voya is good company") ==0)
   def assertion_check_earnings(self):
      """
      This method is used for unit testing TextPreprocessor.check_earnings method.
      """
      assert (self.txt_preprocess_obj.check_earnings("voya has great earnings") ==1)
   def assertion_check_earnings_not(self):
      """
      This method is used for unit testing TextPreprocessor.check_earnings method.
      """
      assert (self.txt_preprocess_obj.check_earnings("voya is good company") ==0)
   def assertion_sentence_tokenizer(self):
      """
      This method is used for unit testing TextPreprocessor.sentence_tokenizer method.
      This text is taken from ENTITY_ID='05ZCW4-E' AND VERSION_ID=7081729 
      """
      len(self.txt_preprocess_obj.sentence_tokenizer("First of all, on January 1 of 2024, the Noto Peninsula earthquake occurred, and I would like to express our heartfelt condolences to those who lost their lives in that earthquake. And also, I would like to express our deepest sympathies to all those affected by the disaster. And in order to contribute to the relief efforts and disaster victims for the restructuring of the affected areas, the company has decided to donate a total of ¥10 million to Peace Winds Japan. And so we sincerely pray for the swift recovery of the disaster-affected areas."))==4

result = TextPreprocessorFixture().execute_tests()
print(result.to_string())
