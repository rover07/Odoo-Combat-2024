import os
import pickle
import sys
from src.exception import CustomException

def save_object(file_path, obj):
    try:
        if not file_path:
            raise ValueError("The file_path provided is empty.")
        
        dir_path = os.path.dirname(file_path)
        
        if dir_path:  # Ensure dir_path is not empty
            os.makedirs(dir_path, exist_ok=True)
        
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
