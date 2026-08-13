"""Custom exception utilities for the project.

This module helps raise errors with useful context such as the file name,
line number, and original error message.
"""
from src.logger import logging
import sys


def error_message_detail(error, error_detail: sys):
    """Build a detailed error message using traceback information."""
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    line_number = exc_tb.tb_lineno
    error_message = f"Error occurred in file: {file_name}, line: {line_number}, message: {str(error)}"
    return error_message


class CustomException(Exception):
    """Exception with extra context for debugging."""

    def __init__(self, error_message: str, error_detail: sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail)

    def __str__(self):
        return f"CustomException: {self.error_message}"

