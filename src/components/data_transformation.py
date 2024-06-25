from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.exception import CustomException
from src.logger import logging
import sys
def transform_data(data):
    try:
        X = data.drop(columns=['price'])  # Assuming 'target' is the target column
        y = data['price']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        column_transformer = ColumnTransformer(
            transformers=[
                ('onehotencoder', OneHotEncoder(sparse_output=False), ['location'])  # Assuming 'location' needs encoding
            ],
            remainder='passthrough'
        )

        X_train_transformed = column_transformer.fit_transform(X_train)
        X_test_transformed = column_transformer.transform(X_test)

        scaler = StandardScaler()
        X_train_transformed = scaler.fit_transform(X_train_transformed)
        X_test_transformed = scaler.transform(X_test_transformed)

        logging.info("Data transformation successful")
        return X_train_transformed, X_test_transformed, y_train, y_test
    except Exception as e:
        logging.error(f"Error during data transformation: {str(e)}")
        raise CustomException(e, sys)
