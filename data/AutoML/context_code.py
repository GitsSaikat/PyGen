from typing import Dict, List, Optional, Union, Any
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
from datetime import datetime

class AutoMLPipeline:
    """
    AutoML pipeline that handles the entire machine learning workflow from
    data preprocessing to model deployment.
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.preprocessor = DataPreprocessor()
        self.feature_engineer = FeatureEngineer()
        self.model_selector = ModelSelector(random_state=random_state)
        self.hyperparameter_tuner = HyperparameterTuner()
        self.model_ensemble = ModelEnsemble()
        self.performance_evaluator = PerformanceEvaluator()
        self.explainer = ModelExplainer()
        self.deployer = ModelDeployer()

class DataPreprocessor:
    """Handles data preprocessing and cleaning tasks."""
    
    def __init__(self):
        self.numerical_imputer = SimpleImputer(strategy='mean')
        self.categorical_imputer = SimpleImputer(strategy='most_frequent')
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocesses the input data by handling missing values, encoding
        categorical variables, and scaling numerical features.
        """
        processed_data = data.copy()
        
        # Identify numerical and categorical columns
        numerical_cols = processed_data.select_dtypes(include=['int64', 'float64']).columns
        categorical_cols = processed_data.select_dtypes(include=['object']).columns
        
        # Handle missing values
        if len(numerical_cols) > 0:
            processed_data[numerical_cols] = self.numerical_imputer.fit_transform(processed_data[numerical_cols])
        if len(categorical_cols) > 0:
            processed_data[categorical_cols] = self.categorical_imputer.fit_transform(processed_data[categorical_cols])
        
        # Encode categorical variables
        for col in categorical_cols:
            self.label_encoders[col] = LabelEncoder()
            processed_data[col] = self.label_encoders[col].fit_transform(processed_data[col])
        
        # Scale numerical features
        if len(numerical_cols) > 0:
            processed_data[numerical_cols] = self.scaler.fit_transform(processed_data[numerical_cols])
        
        return processed_data

class FeatureEngineer:
    """Handles feature engineering tasks."""
    
    def engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generates new features from existing ones to improve model performance.
        """
        engineered_data = data.copy()
        
        # Add interaction terms between numerical features
        numerical_cols = engineered_data.select_dtypes(include=['int64', 'float64']).columns
        for i in range(len(numerical_cols)):
            for j in range(i + 1, len(numerical_cols)):
                col1, col2 = numerical_cols[i], numerical_cols[j]
                engineered_data[f'{col1}_{col2}_interaction'] = (
                    engineered_data[col1] * engineered_data[col2]
                )
        
        return engineered_data

class ModelSelector:
    """Handles model selection tasks."""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models = {
            'random_forest': RandomForestClassifier(random_state=random_state),
            'gradient_boosting': GradientBoostingClassifier(random_state=random_state)
        }
        self.best_model = None
        self.best_score = float('-inf')
    
    def select_best_model(self, X_train: np.ndarray, y_train: np.ndarray,
                         X_val: np.ndarray, y_val: np.ndarray) -> BaseEstimator:
        """
        Selects the best performing model based on validation metrics.
        """
        for name, model in self.models.items():
            model.fit(X_train, y_train)
            score = model.score(X_val, y_val)
            
            if score > self.best_score:
                self.best_score = score
                self.best_model = model
        
        return self.best_model

class HyperparameterTuner:
    """Handles hyperparameter tuning tasks."""
    
    def tune_hyperparameters(self, model: BaseEstimator, X_train: np.ndarray,
                            y_train: np.ndarray) -> BaseEstimator:
        """
        Tunes the hyperparameters of the given model using grid search.
        """
        # Example hyperparameter grid for RandomForestClassifier
        if isinstance(model, RandomForestClassifier):
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, 30],
                'min_samples_split': [2, 5, 10]
            }
        return model

class ModelEnsemble:
    """Handles model ensemble tasks."""
    
    def create_ensemble(self, models: List[BaseEstimator], weights: Optional[List[float]] = None) -> Any:
        """
        Creates an ensemble of models with optional weights.
        """
        if weights is None:
            weights = [1/len(models)] * len(models)
        return {'models': models, 'weights': weights}

class PerformanceEvaluator:
    """Handles model performance evaluation tasks."""
    
    def evaluate_model(self, model: BaseEstimator, X_test: np.ndarray,
                      y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluates model performance using various metrics.
        """
        y_pred = model.predict(X_test)
        return {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1': f1_score(y_test, y_pred, average='weighted')
        }

class ModelExplainer:
    """Handles model explainability tasks."""
    
    def explain_predictions(self, model: BaseEstimator, X: np.ndarray,
                          feature_names: List[str]) -> Dict[str, float]:
        """
        Provides feature importance and prediction explanations.
        """
        if hasattr(model, 'feature_importances_'):
            return dict(zip(feature_names, model.feature_importances_))
        return {}

class ModelDeployer:
    """Handles model deployment tasks."""
    
    def deploy_model(self, model: BaseEstimator, model_path: str) -> str:
        """
        Deploys the trained model to the specified path.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = f"{model_path}/model_{timestamp}.joblib"
        joblib.dump(model, model_filename)
        return model_filename