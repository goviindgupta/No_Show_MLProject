import sys
import pandas as pd
import tensorflow as tf
# from tensorflow import keras
from keras import layers, models , Model
from sklearn.compose import ColumnTransformer
from src.NO_SHOW_MLPROJECT.exception import CustomException
from src.NO_SHOW_MLPROJECT.logger import logging
from src.NO_SHOW_MLPROJECT.exception import CustomException
from src.NO_SHOW_MLPROJECT.utils import load_object
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
 
import joblib
from sklearn.preprocessing import LabelEncoder
import numpy as np
 
class CustomData:
    def __init__(
        self,
        Gender: str,
        Age: int,
        Alcohol_Consumption: str,
        Hypertension: str,
        Diabetes: str,
        Appointment_Date: str,
        Schedule_Date: str,
        Clinic_Location: str,
        Specialty: str,
        Neighborhood: str,
    ):
        self.Gender = Gender
        self.Age = Age
        self.Alcohol_Consumption = Alcohol_Consumption
        self.Hypertension = Hypertension
        self.Diabetes = Diabetes
        self.Appointment_Date = Appointment_Date
        self.Schedule_Date = Schedule_Date
        self.Clinic_Location = Clinic_Location
        self.Specialty = Specialty
        self.Neighborhood = Neighborhood
 
    def get_data_as_dataframe(self):
        try:
            data = {
                "Gender": [self.Gender],
                "Age": [self.Age],
                "Alcohol_Consumption": [self.Alcohol_Consumption],
                "Hypertension": [self.Hypertension],
                "Diabetes": [self.Diabetes],
                "Appointment_Date": [self.Appointment_Date],
                "Schedule_Date": [self.Schedule_Date],
                "Clinic_Location": [self.Clinic_Location],
                "Specialty": [self.Specialty],
                "Neighborhood": [self.Neighborhood],
            }
            return pd.DataFrame(data)
        except Exception as e:
            raise CustomException(e, sys)
 
class PredictPipeline:
    def __init__(self):
        pass
 
    def predict(self, features):
        try:
            features = pd.DataFrame(features)
            logging.info('first line in predict method')
            # Load the model and preprocessor
            model_path = 'artifacts/model.keras'
            preprocessor_path = 'artifacts/preprocessor.pkl'
            
            # model = models.load_model(model_path)
            # preprocessor = joblib.load(preprocessor_path)
 
            preprocessor: Pipeline = joblib.load(preprocessor_path)
            model: Model = models.load_model(model_path)
 
 
            # Check if preprocessor is a valid Scikit-learn object
            if not hasattr(preprocessor, 'transform'):
                raise CustomException("Preprocessor does not have 'transform' method", sys)
            
            # Apply the necessary transformations to the new data
            logging.info('before calling transform_data of predict pipeline')
            features_transformed = self.transform_data(features)
            logging.info('after calling transform_data of predict pipeline')
 
            # Transform features using the preprocessor
            data_scaled = preprocessor.transform(features_transformed)
            # data_scaled = features_transformed
            # Make predictions
            preds = model.predict(data_scaled)
 
            # Return the predictions in a readable format
            return ["No-Show" if pred > 0.5 else "Show" for pred in preds]
        
        except Exception as e:
            # Use CustomException for error handling
            raise CustomException(e, sys)
 
    def transform_data(self, data:pd.DataFrame) -> pd.DataFrame:
        # Apply the same transformations as you did for the training data
        try:
            # Handle missing values
            data = data.dropna()
 
            # # Create age group categories
            # data['age_group'] = pd.cut(data['Age'], bins=[0, 30, 40, 50, 60, 100], labels=['<30', '30-40', '40-50', '50-60', '>60'])
            # data.drop('Age', axis='columns', inplace=True)
 
            # Create age group categories
            data['age_group'] = pd.cut(data['Age'], bins=[0, 30, 40, 50, 60, 100], labels=['<30', '30-40', '40-50', '50-60', '>60'])
            # Convert to string to avoid issues with categorical data
            # data['age_group'] = data['age_group'].astype(str)
 
            # Drop the original Age column
            data.drop('Age', axis='columns', inplace=True)
 
            # Map 'Alcohol Consumption' to numeric values
            mapping_dict = {'0/week': 0, '1/week': 1, '5/week': 2, '10/week': 3, '> 14/week': 4}
            data['Alcohol_Consumption'] = data['Alcohol_Consumption'].map(mapping_dict)
 
 
            # # Convert necessary columns to integers
            # data['Hypertension'] = data['Hypertension'].astype(int)
            # data['Diabetes'] = data['Diabetes'].astype(int)
 
               # Map 'Hypertension' and 'Diabetes' to numeric values
            data['Hypertension'] = data['Hypertension'].map({'No': 0, 'Yes': 1})
            data['Diabetes'] = data['Diabetes'].map({'No': 0, 'Yes': 1})
 
 
            # # Apply One-Hot Encoding for nominal variables including 'age_group'
            # data = pd.get_dummies(data, columns=['Clinic_Location', 'Specialty', 'Neighborhood', 'age_group'], drop_first=True)
 
 
        
            # # Convert date columns to datetime format and extract features
            # data['Appointment_Date'] = pd.to_datetime(data['Appointment_Date'])
            # data['Schedule_Date'] = pd.to_datetime(data['Schedule_Date'])
            # data['days_until_appointment'] = (data['Appointment_Date'] - data['Schedule_Date']).dt.days
 
            # # Drop the original date columns
            # data.drop(columns=['Appointment_Date', 'Schedule_Date'], inplace=True)
 
             # Convert date columns to datetime format and extract features
            data['Appointment_Date'] = pd.to_datetime(data['Appointment_Date'])
            data['Schedule_Date'] = pd.to_datetime(data['Schedule_Date'])
            data['days_until_appointment'] = (data['Appointment_Date'] - data['Schedule_Date']).dt.days
 
            # Drop the original date columns
            data.drop(columns=['Appointment_Date', 'Schedule_Date'], inplace=True)
 
 
            # Apply Label Encoding for binary variables
            le = LabelEncoder()
            data['Gender'] = le.fit_transform(data['Gender'])
 
            # Convert boolean columns to integers
            bool_columns = data.select_dtypes(include=['bool']).columns
            for column in bool_columns:
                data[column] = data[column].astype(int)
 
            # # Ensure all data is numeric
            # assert data.apply(lambda x: np.issubdtype(x.dtype, np.number)).all(), "Non-numeric data found in dataset"
 
 
             # Initialize the pipeline with correct column names
            numeric_features = ['days_until_appointment', 'Alcohol_Consumption','Hypertension','Diabetes','Gender']
            # Since 'Clinic_Location', 'Specialty', 'Neighborhood', 'age_group' were one-hot encoded, they will have multiple columns
            categorical_features = [col for col in data.columns if col not in numeric_features]
            print('data type of prediction _ pipeline ')
            print(data.dtypes)
 
#             # Create a preprocessing pipeline with StandardScaler and OneHotEncoder
#             preprocess = Pipeline([
#     ('preprocessor', ColumnTransformer(
#         transformers=[
#             ('num', StandardScaler(), numeric_features),
#             ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)  # Set sparse_output=False
#         ]
#     ))
# ])
               # Apply transformations
            # transformed_data = preprocess.transform(data)
 
    
 
            return data
 
        except Exception as e:
            raise CustomException(e, sys)