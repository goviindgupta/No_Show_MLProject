import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers, models
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, accuracy_score
from src.NO_SHOW_MLPROJECT.exception import CustomException
from src.NO_SHOW_MLPROJECT.logger import logging
from src.NO_SHOW_MLPROJECT.utils import save_object, load_object
from imblearn.over_sampling import SMOTE

class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.keras")
    preprocessor_file_path: str = os.path.join("artifacts", "preprocessor.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        self.preprocessor = load_object(self.model_trainer_config.preprocessor_file_path)

    def build_model(self, input_shape):
        model = models.Sequential([
            layers.Dense(64, activation='relu', input_shape=(input_shape,)),
            layers.Dense(32, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def initiate_model_trainer(self, X_train, X_test, y_train, y_test):
        try:
            logging.info("Starting TensorFlow model training")

            # Initialize SMOTE
            smote = SMOTE(random_state=42)
            X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

            # Convert to TensorFlow-compatible format
            X_train_smote = np.array(X_train_smote, dtype=np.float32)
            y_train_smote = np.array(y_train_smote, dtype=np.float32)
            X_test = np.array(X_test, dtype=np.float32)
            y_test = np.array(y_test, dtype=np.float32)

            # Build and train the model
            model = self.build_model(X_train_smote.shape[1])

            # Use class weights to handle imbalance
            class_weights = {0: 1., 1: 10.}  # Adjust weights based on your needs
            history = model.fit(X_train_smote, y_train_smote, epochs=10, batch_size=32, validation_split=0.2, class_weight=class_weights)

            # Evaluate the model
            loss, accuracy = model.evaluate(X_test, y_test)
            logging.info(f"Test Accuracy: {accuracy:.2f}")

            y_pred = (model.predict(X_test) > 0.5).astype("int32")

            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)

            logging.info(f"Precision: {precision:.2f}")
            logging.info(f"Recall: {recall:.2f}")
            logging.info(f"F1 Score: {f1:.2f}")
            logging.info(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")

            LABELS = ["Show", "No Show"]
            logging.info("Classification Report:\n" + classification_report(y_test, y_pred, target_names=LABELS))

            # Ensure the artifacts directory exists
            if not os.path.exists(os.path.dirname(self.model_trainer_config.trained_model_file_path)):
                os.makedirs(os.path.dirname(self.model_trainer_config.trained_model_file_path))

            # Save the model
            model.save(self.model_trainer_config.trained_model_file_path)
            logging.info(f"TensorFlow model saved to {self.model_trainer_config.trained_model_file_path}")

            return {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1
            }

        except Exception as e:
            logging.error(f"Error in TensorFlow model training: {e}")
            raise CustomException(e, sys)
