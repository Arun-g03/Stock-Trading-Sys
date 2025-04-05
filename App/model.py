# model.py

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
from Logger import System_Log

# Setup the logger
system_logger = System_Log.setup_logger('model')

class Model:
    @staticmethod
    def train_model(data, target_column='Signal'):
        """
        Train a machine learning model to generate trading signals.
        """
        try:
            if target_column not in data.columns:
                raise KeyError(f"'{target_column}' column not found in the dataset.")

            feature_columns = [col for col in data.columns if col not in ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', target_column]]
            X = data[feature_columns]
            y = data[target_column]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            system_logger.info(f"Model trained successfully with accuracy: {accuracy:.2f}")

            return model, accuracy
        except KeyError as e:
            system_logger.error(f"Missing target column: {e}")
            raise
        except Exception as e:
            system_logger.error(f"Error training model: {e}")
            raise


    @staticmethod
    def save_model(model, file_path):
        """
        Save the trained model to a file.
        """
        try:
            joblib.dump(model, file_path)
            system_logger.info(f"Model saved successfully to {file_path}")
        except Exception as e:
            system_logger.error(f"Error saving model: {e}")
            raise

    @staticmethod
    def load_model(file_path):
        """
        Load a trained model from a file.
        """
        try:
            model = joblib.load(file_path)
            system_logger.info(f"Model loaded successfully from {file_path}")
            return model
        except Exception as e:
            system_logger.error(f"Error loading model: {e}")
            raise

    @staticmethod
    def apply_model(data, model):
        """
        Apply a trained model to generate trading signals.
        """
        try:
            feature_columns = [col for col in data.columns if col not in ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Signal']]
            X = data[feature_columns]

            # Apply model prediction
            data['Model_Signal'] = model.predict(X)

            # Ensure `Signal` column is created
            if 'Signal' not in data.columns:
                data['Signal'] = data['Model_Signal']

            system_logger.info("Model signals applied successfully.")
            return data
        except Exception as e:
            system_logger.error(f"Error applying model: {e}")
            raise

