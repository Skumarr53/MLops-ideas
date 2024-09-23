# Databricks notebook source
from datetime import datetime
from typing import Tuple

class DateUtility:
    """Utility class for handling date-related operations."""

    @staticmethod
    def get_date_range(years_back: int = 0, months_back: int = 0) -> Tuple[str, str]:
        """
        Calculate the date range for the query based on months or years.

        Args:
            months_back (int): The number of months to go back from the current date (default is 0).
            years_back (int): The number of years to go back from the current year (default is 0).

        Returns:
            Tuple[str, str]: A tuple containing the minimum and maximum dates in the format 'YYYY-MM-DD'.
        """
        end_date = datetime.now()
        
        # Calculate start date based on months_back
        if months_back > 0:
            start_date = end_date - relativedelta(months=months_back)
        else:
            start_date = end_date

        # Calculate start date based on years_back
        if years_back > 0:
            start_year = end_date.year - years_back
            min_date = f"{start_year}-{end_date.month:02d}-01"
        else:
            min_date = f"{start_date.year}-{start_date.month:02d}-01"

        # Max date is always the start of the current month
        max_date = f"{end_date.year}-{end_date.month:02d}-01"

        return f"'{min_date}'", f"'{max_date}'"
