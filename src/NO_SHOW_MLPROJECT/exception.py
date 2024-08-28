import sys
from src.NO_SHOW_MLPROJECT.logger import logging

def error_message_detail(error, error_details: sys) -> str:
    """
    Generate detailed error message including file name, line number, and error message.
    """
    exc_type, exc_value, exc_tb = error_details.exc_info()
    file_name = 'Unknown'
    line_number = 'Unknown'
    
    if exc_tb is not None:
        file_name = exc_tb.tb_frame.f_code.co_filename
        line_number = exc_tb.tb_lineno

    error_message = f"Error occurred in script: {file_name} at line number: {line_number}. Error message: {error}"
    return error_message

class CustomException(Exception):
    def __init__(self, error_message: str, error_details: sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_details)

    def __str__(self) -> str:
        return self.error_message
