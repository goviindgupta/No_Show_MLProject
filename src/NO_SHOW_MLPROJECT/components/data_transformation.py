import sys
from dataclasses import dataclass
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from src.NO_SHOW_MLPROJECT.utils import save_object
from src.NO_SHOW_MLPROJECT.exception import CustomException
from src.NO_SHOW_MLPROJECT.logger import logging

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self) -> ColumnTransformer:
        try:
            # Define numerical and categorical columns
            numerical_columns = ["Age"]
            categorical_columns = [
                "Gender", "Alcohol_Consumption", "Hypertension", "Diabetes", 
                "Clinic_Location", "Specialty", "Neighborhood"
            ]

            # Create pipelines for numerical and categorical features
            numerical_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),  # Handle missing values
                ('scaler', StandardScaler())  # Standardize numerical features
            ])

            categorical_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),  # Handle missing values
                ('onehot', OneHotEncoder(handle_unknown='ignore'))  # One-hot encode categorical features
            ])

            logging.info(f"Categorical Columns: {categorical_columns}")
            logging.info(f"Numerical Columns: {numerical_columns}")

            # Combine pipelines into a single ColumnTransformer
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numerical_pipeline, numerical_columns),
                    ('cat', categorical_pipeline, categorical_columns)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)  # Handle exceptions

    def initiate_data_transformation(self, train_path: str, test_path: str):
        try:
            # Load datasets
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Reading the train and test files")

            preprocessor_obj = self.get_data_transformer_object()

            target_column_name = "target_no_show"

            # Split datasets into features and target
            input_features_train_df = train_df.drop(columns=[target_column_name])
            target_features_train_df = train_df[target_column_name]

            input_features_test_df = test_df.drop(columns=[target_column_name])
            target_features_test_df = test_df[target_column_name]

            logging.info('Applying preprocessing on training and test dataframes')

            # Apply preprocessing
            input_feature_train_arr = preprocessor_obj.fit_transform(input_features_train_df).toarray()
            input_feature_test_arr = preprocessor_obj.transform(input_features_test_df).toarray()

            # Ensure target arrays are 2D
            target_features_train_arr = target_features_train_df.values.reshape(-1, 1)
            target_features_test_arr = target_features_test_df.values.reshape(-1, 1)

            # Print shapes and types for debugging
            logging.info(f"Shape of input_feature_train_arr: {input_feature_train_arr.shape}")
            logging.info(f"Shape of target_features_train_arr: {target_features_train_arr.shape}")
            logging.info(f"Shape of input_feature_test_arr: {input_feature_test_arr.shape}")
            logging.info(f"Shape of target_features_test_arr: {target_features_test_arr.shape}")

            logging.info(f"Type of input_feature_train_arr: {type(input_feature_train_arr)}")
            logging.info(f"Type of target_features_train_arr: {type(target_features_train_arr)}")
            logging.info(f"Type of input_feature_test_arr: {type(input_feature_test_arr)}")
            logging.info(f"Type of target_features_test_arr: {type(target_features_test_arr)}")

            # Concatenate features and target arrays
            train_arr = np.concatenate([input_feature_train_arr, target_features_train_arr], axis=1)
            test_arr = np.concatenate([input_feature_test_arr, target_features_test_arr], axis=1)

            logging.info("Saving preprocessing object")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e, sys)




if __name__ == "__main__":
    try:
        train_data_path = 'path_to_train.csv'
        test_data_path = 'path_to_test.csv'
        data_transformation = DataTransformation()
        train_arr, test_arr, preprocessor_path = data_transformation.initiate_data_transformation(train_data_path, test_data_path)

        print(f"Training data shape: {train_arr.shape}")
        print(f"Test data shape: {test_arr.shape}")
        print(f"Preprocessor object saved at: {preprocessor_path}")

    except Exception as e:
        logging.exception(f"Error occurred: {e}")
        raise CustomException(e, sys)
