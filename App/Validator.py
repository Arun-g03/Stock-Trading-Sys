# validator.py

import pandas as pd
from Logger import System_Log
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Setup the logger
system_logger = System_Log.setup_logger('validator')

class Validator:
    @staticmethod
    def validate_data_integrity(data):
        """
        Validate the integrity of the data.
        Checks for missing values, duplicates, and correct data types.
        """
        try:
            # Check for missing values
            missing_values = data.isnull().sum()
            if missing_values.any():
                system_logger.warning(f"Missing values found:\n{missing_values}")
            else:
                system_logger.info("No missing values found.")

            # Check for duplicates
            duplicate_rows = data.duplicated().sum()
            if duplicate_rows > 0:
                system_logger.warning(f"Found {duplicate_rows} duplicate rows.")
            else:
                system_logger.info("No duplicate rows found.")

            # Check data types
            system_logger.info(f"Data types:\n{data.dtypes}")

        except Exception as e:
            system_logger.error(f"Error validating data integrity: {e}")
            raise

    @staticmethod
    def validate_data_quality(data):
        """
        Validate the quality of the data.
        Checks for outliers, inconsistencies, and statistical summaries.
        """
        try:
            # Statistical summary
            summary = data.describe()
            system_logger.info(f"Data statistical summary:\n{summary}")

            # Check for outliers using IQR
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))).sum()
            if outliers.any():
                system_logger.warning(f"Outliers found:\n{outliers}")
            else:
                system_logger.info("No outliers found.")

        except Exception as e:
            system_logger.error(f"Error validating data quality: {e}")
            raise

    @staticmethod
    def validate_processed_data(data):
        """
        Validate the processed data.
        Checks if the processed data meets the expected structure and contains no NaN values.
        """
        try:
            if data.isnull().sum().any():
                system_logger.error("Processed data contains NaN values.")
            else:
                system_logger.info("Processed data contains no NaN values.")

            if not all(data.dtypes == 'float64') and not all(data.dtypes == 'int64'):
                system_logger.error("Processed data contains incorrect data types.")
            else:
                system_logger.info("Processed data contains correct data types.")

        except Exception as e:
            system_logger.error(f"Error validating processed data: {e}")
            raise

    @staticmethod
    def validate_model_results(data, target_column='Signal'):
        """
        Validate the model results.
        Checks the accuracy, confusion matrix, and classification report of the model predictions.
        """
        try:
            if 'Model_Signal' not in data.columns:
                raise ValueError("Model_Signal column not found in data.")

            y_true = data[target_column]
            y_pred = data['Model_Signal']

            accuracy = accuracy_score(y_true, y_pred)
            conf_matrix = confusion_matrix(y_true, y_pred)
            class_report = classification_report(y_true, y_pred)

            system_logger.info(f"Model Accuracy: {accuracy:.2f}")
            system_logger.info(f"Confusion Matrix:\n{conf_matrix}")
            system_logger.info(f"Classification Report:\n{class_report}")

            return accuracy, conf_matrix, class_report

        except Exception as e:
            system_logger.error(f"Error validating model results: {e}")
            raise

# Example usage:
# data = pd.read_csv('path_to_your_csv')
# Validator.validate_data_integrity(data)
# Validator.validate_data_quality(data)
# processed_data = FeatureEngineering.engineer_features(data)
# Validator.validate_processed_data(processed_data)
# model, _ = Model.train_model(processed_data)
# processed_data = Model.apply_model(processed_data, model)
# Validator.validate_model_results(processed_data)
