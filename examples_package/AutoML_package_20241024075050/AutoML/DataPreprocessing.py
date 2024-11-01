import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from imblearn.over_sampling import SMOTE

class DataPreprocessing:
    def __init__(self):
        pass

    def handle_missing_values(self, data):
        # Handle missing values using mean imputation
        data.fillna(data.mean(), inplace=True)
        return data

    def encode_categorical_variables(self, data):
        # Encode categorical variables using one-hot encoding
        categorical_cols = data.select_dtypes(include=['object']).columns
        data = pd.get_dummies(data, columns=categorical_cols)
        return data

    def scale_and_normalize(self, data):
        # Scale numerical variables using standard scaling
        numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns
        scaler = StandardScaler()
        data[numerical_cols] = scaler.fit_transform(data[numerical_cols])
        return data

    def detect_outliers(self, data):
        # Detect outliers using isolation forest
        outlier_detector = IsolationForest(contamination=0.1)
        outlier_detector.fit(data)
        outliers = outlier_detector.predict(data)
        outliers = np.where(outliers == -1, True, False)
        return outliers

    def remove_outliers(self, data, outliers):
        # Remove outliers from the data
        data = data[~outliers]
        return data

    def handle_imbalanced_data(self, X, y):
        """Handle imbalanced datasets using SMOTE"""
        smote = SMOTE(random_state=42)
        X_balanced, y_balanced = smote.fit_resample(X, y)
        return X_balanced, y_balanced

    def advanced_imputation(self, data):
        """Perform advanced imputation using iterative imputer"""
        imputer = IterativeImputer(random_state=42)
        data_imputed = pd.DataFrame(
            imputer.fit_transform(data),
            columns=data.columns
        )
        return data_imputed

    def robust_scaling(self, data):
        """Scale data using RobustScaler for handling outliers"""
        numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns
        scaler = RobustScaler()
        data[numerical_cols] = scaler.fit_transform(data[numerical_cols])
        return data
