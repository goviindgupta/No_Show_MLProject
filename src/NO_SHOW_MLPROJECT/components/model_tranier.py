import os
import sys
from dataclasses import dataclass
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV
from src.NO_SHOW_MLPROJECT.exception import CustomException
from src.NO_SHOW_MLPROJECT.logger import logging
from src.NO_SHOW_MLPROJECT.utils import save_object

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def evaluate_model(self, model, X_train, y_train, X_test, y_test):
        model.fit(X_train, y_train)
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        train_metrics = {
            "accuracy": accuracy_score(y_train, y_train_pred),
            "precision": precision_score(y_train, y_train_pred, average='weighted'),
            "recall": recall_score(y_train, y_train_pred, average='weighted'),
            "f1_score": f1_score(y_train, y_train_pred, average='weighted')
        }

        test_metrics = {
            "accuracy": accuracy_score(y_test, y_test_pred),
            "precision": precision_score(y_test, y_test_pred, average='weighted'),
            "recall": recall_score(y_test, y_test_pred, average='weighted'),
            "f1_score": f1_score(y_test, y_test_pred, average='weighted')
        }

        return train_metrics, test_metrics

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            models = {
                "Random Forest": RandomForestClassifier(),
                "Decision Tree": DecisionTreeClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(),
                "AdaBoost": AdaBoostClassifier(),
            }

            # Hyperparameters for tuning
            params = {
                "Random Forest": {
                    'n_estimators': [100, 200],
                    'max_depth': [None, 10, 20]
                },
                "Decision Tree": {
                    'criterion': ['gini', 'entropy'],
                    'max_depth': [None, 10, 20]
                },
                "Gradient Boosting": {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.1, 0.05],
                    'max_depth': [3, 5]
                },
                "AdaBoost": {
                    'n_estimators': [50, 100],
                    'learning_rate': [0.1, 1]
                }
            }

            best_model = None
            best_score = 0

            for model_name, model in models.items():
                logging.info(f"Training {model_name}")
                
                gs = GridSearchCV(model, params[model_name], cv=3, n_jobs=-1, scoring='f1_weighted')
                gs.fit(X_train, y_train)

                train_metrics, test_metrics = self.evaluate_model(gs.best_estimator_, X_train, y_train, X_test, y_test)

                logging.info(f"{model_name} Train Metrics: {train_metrics}")
                logging.info(f"{model_name} Test Metrics: {test_metrics}")

                if test_metrics['f1_score'] > best_score:
                    best_score = test_metrics['f1_score']
                    best_model = gs.best_estimator_

            logging.info(f"Best model: {best_model}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            return best_model, best_score

        except Exception as e:
            raise CustomException(e, sys)
