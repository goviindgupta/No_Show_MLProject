import os
import sys
import pandas as pd
import pyodbc
from dotenv import load_dotenv
from src.NO_SHOW_MLPROJECT.exception import CustomException
from src.NO_SHOW_MLPROJECT.logger import logging
import dill

# Load environment variables from a .env file
load_dotenv()

# Fetch database credentials from environment variables
server = os.getenv("host")
username = os.getenv("user")
password = os.getenv("password")
database = os.getenv("db")

def read_sql_data():
    logging.info("Reading data from SQL Server started.")
    try:
        # Adjusted connection string for SQL Server Authentication
        cnxn = pyodbc.connect(
            f'DRIVER={{ODBC Driver 17 for SQL Server}};'
            f'SERVER={server};'
            f'DATABASE={database};'
            f'UID={username};'
            f'PWD={password};'
        )

        cursor = cnxn.cursor()

        if cursor:
            logging.info('Good connection established.')
        else:
            logging.error('Connection failed.')

        # Example query - Replace with your own SQL query
        query = "SELECT * FROM NoShow_Updated"
        df = pd.read_sql(query, cnxn)
        logging.info(f"Data read successfully. Sample data:\n{df.head()}")

        return df

    except Exception as ex:
        raise CustomException(ex, sys)
    
def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
        logging.info(f"Object saved successfully at {file_path}")

    except Exception as e:
        raise CustomException(e, sys)
    
def load_object(file_path: str):
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)

# Example function call (You can run this to test the connection)
if __name__ == "__main__":
    read_sql_data()
