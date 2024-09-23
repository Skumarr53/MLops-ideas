# Databricks notebook source
# MAGIC %run "./../filesystem/dbfs_utility"

# COMMAND ----------

"""
Nutter Fixture for testing the DBFSUtility module.
"""

from runtime.nutterfixture import NutterFixture
class DBFSUtilityFixture(NutterFixture):
   """
   This DBFSUtility fixture is used for unit testing all the methods that are used in the dbfs_utility.py module 
   """
   def __init__(self):
      """
      helps in intializing all the instance variables

      instance variable: Config is created
      """
      self.dbfs_utility_obj=DBFSUtility()
      NutterFixture.__init__(self)
      
   def assertion_get_directory_content(self):
      """
      This method is used for unit testing DBFSUtility.get_directory_content method.
      """
      assert (len(self.dbfs_utility_obj.get_directory_content("dbfs:/FileStore/AjayaDevalla/resources/"))>0)

   def assertion_add_directory(self):
      """
      This method is used for unit testing DBFSUtility.add_directory method.
      """
      self.dbfs_utility_obj.add_directory("dbfs:/FileStore/AjayaDevalla/resources/","ajay")
      assert (len(dbutils.fs.ls("dbfs:/FileStore/AjayaDevalla/resources/ajay/"))==1)
   def assertion_remove_file(self):
      """
      This method is used for unit testing DBFSUtility.remove_file method.
      """
      self.dbfs_utility_obj.write_file("/FileStore/AjayaDevalla/resources/CT_test/","test.txt","This is unit testing using nutter for call transcript package.")
      self.dbfs_utility_obj.remove_file("dbfs:/FileStore/AjayaDevalla/resources/CT_test/","test.txt")
      file_removed=False
      try:
        dbutils.fs.ls("dbfs:/FileStore/AjayaDevalla/resources/CT_test/test.txt")
      except:
        file_removed=True
      assert(file_removed)
   def assertion_write_file(self):
      """
      This method is used for unit testing DBFSUtility.remove_file method.
      """
      self.dbfs_utility_obj.write_file("/FileStore/AjayaDevalla/resources/CT_test/","ajay.txt","This is unit testing using nutter for call transcript package.")
      assert (self.dbfs_utility_obj.read_file("/FileStore/AjayaDevalla/resources/ajay/","ajay.txt")!="")

   def assertion_read_file_not_present(self):
      """
      This method is used for unit testing DBFSUtility.read_file method.
      """
      exception_raised=False
      try:
         self.dbfs_utility_obj.read_file("dbfs:/FileStore/AjayaDevalla/resources/CT_test1/","quant.txt")
      except:
         exception_raised=True
      assert(exception_raised)
   def assertion_write_file_to_invalid_path(self):
      """
      This method is used for unit testing DBFSUtility.write_file method.
      """
      exception_raised=False
      try:
         self.dbfs_utility_obj.write_file("/FileStore/AjayaDevalla/resources/CT_test1/","ajay.txt","This is unit testing using nutter for call transcript package.")
      except:
         exception_raised=True
      assert(exception_raised)
       

result = DBFSUtilityFixture().execute_tests()
print(result.to_string())
