import sys
from src.NO_SHOW_MLPROJECT.utils import load_object
from src.NO_SHOW_MLPROJECT.exception import CustomException

try:
    preprocessor = load_object(file_path='artifacts/preprocessor.pkl')
    print("Loaded preprocessor:")
    print(preprocessor)
    print("Preprocessor steps:")
    if hasattr(preprocessor, 'steps'):
        for step in preprocessor.steps:
            print(step)
    else:
        print("The preprocessor does not have steps. It might not be a Pipeline.")

except Exception as e:
    raise CustomException(e, sys)
