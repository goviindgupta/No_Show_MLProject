
import os
import sys
import pandas as pd
import pyodbc
from dotenv import load_dotenv
from src.NO_SHOW_MLPROJECT.exception import CustomException
from src.NO_SHOW_MLPROJECT.logger import logging

import pickle
import numpy as np 

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
        # Set up the connection using SQL Server Authentication
        # server = 'localhost,1433'  # Note the correction: comma instead of a space
        # database = 'college'
        # username = 'SA'
        # password = 'govind@123'
        
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
            print('Good connection')
        logging.info('Connection established successfully.')

        # Example query - Replace with your own SQL query
        query = "SELECT * FROM NoShow_Updated"
        df = pd.read_sql(query, cnxn)
        print(df.head())

        return df

    except Exception as ex:
        raise CustomException(ex, sys)
    
def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path,exist_ok=True)
        with open(file_path,"wb") as file_obj:
            pickle.dump(obj,file_obj)

    except Exception as e:
        raise CustomException(e,sys)

# Example function call (You can run this to test the connection)
if __name__ == "__main__":
    read_sql_data()

