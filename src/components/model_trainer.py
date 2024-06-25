import sys
import os

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score
from src.logger import logging
from src.exception import CustomException
from src.utils import save_object
from src.components.data_ingestion import load_data
from src.components.data_transformation import transform_data

def train_model(data_file):
    try:
        data = load_data(data_file)
        X_train, X_test, y_train, y_test = transform_data(data)

        pipe = Pipeline([
            ('ridge', Ridge())
        ])

        pipe.fit(X_train, y_train)
        logging.info("Model training successful")

        y_pred = pipe.predict(X_test)
        score = r2_score(y_test, y_pred)
        logging.info(f'R2 Score: {score}')

        # Specify a directory in the file path
        save_object('models/RidgeModel.pkl', pipe)
        logging.info("Model saved as RidgeModel.pkl")
    except Exception as e:
        logging.error(f"Error in model training: {str(e)}")
        raise CustomException(e, sys)

if __name__ == '__main__':
    train_model('notebook/data/Cleaned_data.csv')  # Replace with your actual data file path
