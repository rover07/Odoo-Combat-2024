import sys
import os

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import pandas as pd
from src.logger import logging
from src.exception import CustomException

def load_data(file_path):
    try:
        data = pd.read_csv(file_path)
        logging.info(f"Data loaded successfully from {file_path}")
        return data
    except Exception as e:
        logging.error(f"Error loading data from {file_path}: {str(e)}")
        raise CustomException(e, sys)

if __name__ == '__main__':
    load_data('notebook/data/Cleaned_data.csv')  # Replace with your actual data file path
