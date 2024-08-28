import numpy as np
from src.NO_SHOW_MLPROJECT.logger import logging
from src.NO_SHOW_MLPROJECT.exception import CustomException
import sys
from src.NO_SHOW_MLPROJECT.components.data_ingestion import DataIngestion
from src.NO_SHOW_MLPROJECT.components.data_transformation import DataTransformation
from src.NO_SHOW_MLPROJECT.components.model_tranier import ModelTrainer

if __name__ == "__main__":
    logging.info("The execution has started")

    try:
        # Data Ingestion
        data_ingestion = DataIngestion()
        train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()
        
        # Data Transformation
        data_transformation = DataTransformation()
        X_train, X_test, y_train, y_test = data_transformation.initiate_data_transformation(train_data_path, test_data_path)
        
        # Model Training
        model_trainer = ModelTrainer()
        result = model_trainer.initiate_model_trainer(X_train, X_test, y_train, y_test)
        print(result)
    
    except CustomException as ce:
        logging.error(ce)
        raise
    except Exception as e:
        logging.error('An unexpected error occurred')
        raise CustomException(e, sys)
