# Databricks notebook source
"""Here we define all the custom exception during the Call Transcript pipeline
"""
"""
NAME : CUSTOMEXCEPTION

DESCRIPTION:
This module serves the below functionalities:
               Generates custom function based o the functionality
              
"""
import sys


class BaseCustomException(Exception):
    """Exception base class"""

    def __init__(self, *args, **kwargs):

        #: track information from ``sys.exc_info``
        self.traceback = sys.exc_info()
        if not hasattr(self, "error_code"):
            self.error_code = 0
        if not hasattr(self, "message"):
            self.message = "Base Exception"

        super().__init__(self.message, *args)

    def __str__(self):
        return "{}: {}".format(self.__class__.__name__, self.message)

    def get_error_code(self):
        """
        Get the Exception error code
        Returns:
            int: The error code information

        """
        return self.error_code


class InvalidInputException(BaseCustomException):
    """
    Exception when the input parameters are invalid
    """

    def __init__(self, *args, **kwargs):

        if not hasattr(self, "error_code"):
            self.error_code = 10000
        if not hasattr(self, "message"):
            self.message = "Invalid Input Exception"
        super().__init__(*args, **kwargs)

class AlgoNotSupported(InvalidInputException):
    """Class which raises when specific algorithm is not available for
    inference"""

    def __init__(self, *args, **kwargs):
        self.message = "Algorithm is not available for inference"
        self.error_code = 10002
        super().__init__(*args, **kwargs)


class MandatoryFieldMissingError(InvalidInputException):
    """Class which raises error when mandatory fields are
    missing in the http request"""

    def __init__(self, *args, **kwargs):

        if not hasattr(self, "error_code"):
            self.error_code = 11000
        if not hasattr(self, "message"):
            self.message = "Missing one or more mandatory fields"
        super().__init__(*args, **kwargs)



class SystemConfigException(BaseCustomException):
    """
    Exception Category for the micro-service system is not configured properly
    """

    def __init__(self, *args, **kwargs):
        if not hasattr(self, "error_code"):
            self.error_code = 20000
        if not hasattr(self, "message"):
            self.message = "Some of the configurations are incorrect, please check"

        super().__init__(*args, **kwargs)


class FilesNotLoadedException(SystemConfigException):
    """Class which raises error files are not loaded"""

    def __init__(self, *args, **kwargs):
        self.message = "Error while loading  files(pkl,json, ini etc),  probably " "due to missing files or invalid path"
        self.error_code = 20002
        super().__init__(*args, **kwargs)
        
class KeyNotFoundException(SystemConfigException):
    """Class which raises error key are not present"""

    def __init__(self, *args, **kwargs):
        self.message = "key not present in config file"
        self.error_code = 20003
        super().__init__(*args, **kwargs)



