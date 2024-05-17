# utils.py
import logging
import os


def create_directory(directory: str) -> None:
    """
    Create a directory if it doesn't exist.

    Args:
        directory (str): The directory to create.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)


def setup_logging() -> None:
    """
    Set up logging for the application.
    """
    logging.basicConfig(filename="app.log", level=logging.INFO)
