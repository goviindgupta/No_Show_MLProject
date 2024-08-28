import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from src.NO_SHOW_MLPROJECT.exception import CustomException
from src.NO_SHOW_MLPROJECT.logger import logging
import sys
from src.NO_SHOW_MLPROJECT.utils import save_object
import os
from sklearn.preprocessing import LabelEncoder

class DataTransformation:
    def __init__(self):
        self.pipeline = None

    def transform_data(self, data: pd.DataFrame) -> pd.DataFrame:
        try:
            # Check for and handle missing values (if any)
            data = data.dropna()

            # Create age group categories
            data['age_group'] = pd.cut(data['Age'], bins=[0, 30, 40, 50, 60, 100], labels=['<30', '30-40', '40-50', '50-60', '>60'])
            # Convert to string to avoid issues with categorical data
            # data['age_group'] = data['age_group'].astype(str)

            # Drop the original Age column
            data.drop('Age', axis='columns', inplace=True)

            # Apply One-Hot Encoding for nominal variables including 'age_group'
            # data = pd.get_dummies(data, columns=['Clinic_Location', 'Specialty', 'Neighborhood', 'age_group'], drop_first=True)

            # Map 'Alcohol Consumption' to numeric values
            mapping_dict = {'0/week': 0, '1/week': 1, '5/week': 2, '10/week': 3, '> 14/week': 4}
            data['Alcohol_Consumption'] = data['Alcohol_Consumption'].map(mapping_dict)

            # Convert 'Hypertension' and 'Diabetes' to integer
            data['Hypertension'] = data['Hypertension'].astype(int)
            data['Diabetes'] = data['Diabetes'].astype(int)

            # Convert date columns to datetime format and extract features
            data['Appointment_Date'] = pd.to_datetime(data['Appointment_Date'])
            data['Schedule_Date'] = pd.to_datetime(data['Schedule_Date'])
            data['days_until_appointment'] = (data['Appointment_Date'] - data['Schedule_Date']).dt.days

            # Drop the original date columns
            data.drop(columns=['Appointment_Date', 'Schedule_Date'], inplace=True)

            # Convert 'Gender' to label encoding
            le = LabelEncoder()
            data['Gender'] = le.fit_transform(data['Gender'])

            # Convert boolean columns to integers
            bool_columns = data.select_dtypes(include=['bool']).columns
            for column in bool_columns:
                data[column] = data[column].astype(int)

            # # Apply One-Hot Encoding for nominal variables
            # data = pd.get_dummies(data, columns=['Clinic_Location', 'Specialty', 'Neighborhood', 'age_group'], drop_first=True)
                
             # Print the column types for debugging
            print("Column types before checking if all data is numeric:")
            print(data.dtypes)
            print('............................................')

            # Ensure all data is numeric
            # assert data.apply(lambda x: np.issubdtype(x.dtype, np.number)).all(), "Non-numeric data found in dataset"


            return data

        except Exception as e:
            logging.error(f"Error in data transformation: {e}")
            raise CustomException(e, sys)

    def save_preprocessor(self):
        """ Save the preprocessor pipeline to a file """
        if self.pipeline is not None:
            save_object(file_path=os.path.join("artifacts", "preprocessor.pkl"), obj=self.pipeline)
        else:
            raise CustomException("Pipeline is not initialized, cannot save preprocessor.", sys)

    def initiate_data_transformation(self, train_path: str, test_path: str):
        try:
            logging.info("Loading train and test data")
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            # Check if 'target_no_show' column exists
            if 'target_no_show' not in train_df.columns:
                raise CustomException("Column 'target_no_show' is missing in train data", sys)
            if 'target_no_show' not in test_df.columns:
                raise CustomException("Column 'target_no_show' is missing in test data", sys)

            logging.info("Performing data transformations")

            # Separate target variable
            y_train = train_df['target_no_show']
            y_test = test_df['target_no_show']
            
            # Drop target variable from feature data
            train_features = train_df.drop(columns=['target_no_show'])
            test_features = test_df.drop(columns=['target_no_show'])

            # Apply transformations
            train_features_transformed = self.transform_data(train_features)
            test_features_transformed = self.transform_data(test_features)


            print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
            print('datatype of train feature transform',train_features_transformed.dtypes)
            print('datatype of test feature transform',test_features_transformed.dtypes)
            print('777777777&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&')

            # Initialize the pipeline with correct column names
            numeric_features = ['days_until_appointment', 'Alcohol_Consumption','Hypertension','Diabetes','Gender']
            # Since 'Clinic_Location', 'Specialty', 'Neighborhood', 'age_group' were one-hot encoded, they will have multiple columns
            categorical_features = [col for col in train_features_transformed.columns if col not in numeric_features]


             # Print the identified numeric and categorical features for debugging
            print("Numeric features:", numeric_features)
            print("Categorical features:", categorical_features)

            print('&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&')

            # self.pipeline = Pipeline([
            #     ('preprocessor', ColumnTransformer(
            #         transformers=[
            #             ('num', StandardScaler(), numeric_features),
            #             ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
            #         ]
            #     ))
            # ])

            self.pipeline = Pipeline([
    ('preprocessor', ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)  # Set sparse_output=False
        ]
    ))
])
            # Apply pipeline transformations
            X_train = self.pipeline.fit_transform(train_features_transformed)
            X_test = self.pipeline.transform(test_features_transformed)

            # Save the preprocessor pipeline
            self.save_preprocessor()

            return X_train, X_test, y_train.values, y_test.values

        except Exception as e:
            logging.error(f"Error in initiating data transformation: {e}")
            raise CustomException(e, sys)
