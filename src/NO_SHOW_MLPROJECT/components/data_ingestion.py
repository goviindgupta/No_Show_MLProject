import os
import sys
from src.NO_SHOW_MLPROJECT.exception import CustomException
from src.NO_SHOW_MLPROJECT.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    reduced_data_path: str = os.path.join('artifacts', 'reduced_data.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        try:
            # Load the raw data
            reduced_data_path = os.path.join('notebook/data', 'reduced_data.csv')
            df = pd.read_csv(reduced_data_path)
            logging.info(f"Data successfully loaded from {reduced_data_path}")

            # Verify the presence of 'target_no_show' column
            if 'target_no_show' not in df.columns:
                raise CustomException("Column 'target_no_show' is missing in the raw data", sys)
            
            logging.info(f"Column 'target_no_show' is present in the raw data")

            # Check for missing values in 'target_no_show'
            if df['target_no_show'].isnull().sum() > 0:
                logging.warning("There are missing values in the 'target_no_show' column")

            # Ensure the artifacts directory exists
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            # Save the raw data to the designated path
            df.to_csv(self.ingestion_config.reduced_data_path, index=False, header=True)
            logging.info(f"Raw data saved to {self.ingestion_config.reduced_data_path}")

            # Split the data into training and testing sets
            train_set, test_set = train_test_split(df, test_size=0.3, random_state=42)

            # Save the training and testing sets
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            logging.info(f"Training data saved to {self.ingestion_config.train_data_path}")
            logging.info(f"Testing data saved to {self.ingestion_config.test_data_path}")

            return self.ingestion_config.train_data_path, self.ingestion_config.test_data_path

        except Exception as e:
            logging.error(f"Error in data ingestion: {e}")
            raise CustomException(e, sys)
