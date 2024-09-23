# Databricks notebook source
# MAGIC %run "./../mlflowutils/mlflow_model"

# COMMAND ----------

"""
Nutter Fixture for testing the mlflow model module.
"""

from runtime.nutterfixture import NutterFixture
class MLFlowModelFixture(NutterFixture):
   """
   This MLFlow Model fixture is used for unit testing all the methods that are used in the mlflow_model module 
   """
   def __init__(self):
      """
      helps in intializing all the instance variables

      instance variable: Config is created
      """
      self.mlflow_model_obj=MLFlowModel("finbert-better-transformer-sentiment-classification_model","None",True,"","","")
      NutterFixture.__init__(self)
 
     
   def assertion_get_latest_model_version_based_on_env(self):
      """
      This method is used for unit testing get_latest_model_version_based_on_env in mlflow_model module
      """
      assert (self.mlflow_model_obj.get_latest_model_version_based_on_env("None") !="")

   def assertion_get_latest_model_version_details_based_on_env(self):
      """
      This method is used for unit testing get_latest_model_version_details_based_on_env in mlflow_model module
      """
      model_version_details=self.mlflow_model_obj.get_latest_model_version_details_based_on_env("None")
      assert (model_version_details.current_stage=="None")

   def assertion_get_latest_model_tags_based_on_env(self):
      """
      This method is used for unit testing get_latest_model_tags_based_on_env in mlflow_model module
      """
      self.mlflow_model_obj.set_model_tag("None")
      model_tag=self.mlflow_model_obj.get_latest_model_tags_based_on_env("None")
      assert('deployment_env' in model_tag.keys())

   def assertion_delete_model_tags_based_on_env(self):
      """
      This method is used for unit testing delete_model_tags_based_on_env in mlflow_model module
      """
      self.mlflow_model_obj.set_model_tag("None")
      model_tag=self.mlflow_model_obj.delete_model_tags_based_on_env("None")
      assert(model_tag)

   def assertion_search_model_based_on_names(self):
      """
      This method is used for unit testing search_model_based_on_names in mlflow_model module
      """
      assert(self.mlflow_model_obj.search_model_based_on_names())

   def assertion_get_models_count(self):
      """
      This method is used for unit testing get_models_count in mlflow_model module
      """
      assert(self.mlflow_model_obj.get_models_count()!=0)

   def assertion_wait_until_model_is_ready(self):
      """
      This method is used for unit testing wait_until_model_is_ready in mlflow_model module
      """
      assert(self.mlflow_model_obj.wait_until_model_is_ready("None"))

   def assertion_load_model(self):
      """
      This method is used for unit testing load_model in mlflow_model module
      """
      assert(self.mlflow_model_obj.load_model() is not None)

   def assertion_perform_model_transition(self):
      """
      This method is used for unit testing perform_model_transition in mlflow_model module
      """
      mlflow_model_obj=MLFlowModel("finbert-better-transformer-sentiment-classification_model","None",True,"","None","Staging")
      assert(mlflow_model_obj.perform_model_transition())

   def assertion_perform_model_transition_from_stg(self):
      """
      This method is used for unit testing perform_model_transition in mlflow_model module
      """
      mlflow_model_obj=MLFlowModel("finbert-better-transformer-sentiment-classification_model","staging",True,"","staging","None")
      assert(mlflow_model_obj.perform_model_transition())

   def assertion_perform_model_transition_from_stg_with_exception(self):
      """
      This method is used for unit testing perform_model_transition in mlflow_model module
      """
      exception_raised=False
      try:
         mlflow_model_obj=MLFlowModel("finbert-better-transformer-sentiment-classification_model","stag",True,"","stag","None")
         mlflow_model_obj.perform_model_transition()
      except:
         exception_raised=True
      assert(exception_raised)
 
   

result = MLFlowModelFixture().execute_tests()
print(result.to_string())
