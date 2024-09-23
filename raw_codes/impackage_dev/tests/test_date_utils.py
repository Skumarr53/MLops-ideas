# Databricks notebook source
# MAGIC %pip install nutter freezegun

# COMMAND ----------

# MAGIC %run ../dateutils/date_utils

# COMMAND ----------

from datetime import datetime
from unittest.mock import patch
from freezegun import freeze_time
from runtime.nutterfixture import NutterFixture, tag

class DateUtilityFixture(NutterFixture):
    """
    This DateUtility fixture is used for unit testing all the methods that are used in the DateUtility module.
    """
    def __init__(self):
        """
        Helps in initializing all the instance variables.
        """
        self.date_utility = DateUtility()
        self.result = None
        NutterFixture.__init__(self)
    
    @tag('test_get_date_range_timespan_1')
    @freeze_time("2023-10-01")
    def run_test_get_date_range_timespan_1(self):
        """
        This method is used for unit testing DateUtility.get_date_range method with timespan = 1 year.
        """
        timespan = 1
        self.result = self.date_utility.get_date_range(timespan)
    
    def assertion_test_get_date_range_timespan_1(self):
        expected_min_date = "'2022-10-01'"
        expected_max_date = "'2023-10-01'"
        assert self.result == (expected_min_date, expected_max_date)
    
    @tag('test_get_date_range_timespan_5')
    @freeze_time("2023-10-01")
    def run_test_get_date_range_timespan_5(self):
        """
        This method is used for unit testing DateUtility.get_date_range method with timespan = 5 years.
        """
        timespan = 5
        self.result = self.date_utility.get_date_range(timespan)
    
    def assertion_test_get_date_range_timespan_5(self):
        expected_min_date = "'2018-10-01'"
        expected_max_date = "'2023-10-01'"
        assert self.result == (expected_min_date, expected_max_date)
    
    @tag('test_get_date_range_timespan_0')
    @freeze_time("2023-10-01")
    def run_test_get_date_range_timespan_0(self):
        """
        This method is used for unit testing DateUtility.get_date_range method with timespan = 0 years.
        """
        timespan = 0
        self.result = self.date_utility.get_date_range(timespan)
    
    def assertion_test_get_date_range_timespan_0(self):
        expected_min_date = "'2023-10-01'"
        expected_max_date = "'2023-10-01'"
        assert self.result == (expected_min_date, expected_max_date)

# Run the tests
result = DateUtilityFixture().execute_tests()
print(result.to_string())
