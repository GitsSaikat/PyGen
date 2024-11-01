import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import pickle

class AutoML:
    def __init__(self):
        self.pipeline = None
        self.best_model = None
        
    def create_pipeline(self, numeric_features, categorical_features):
        """Create a preprocessing and modeling pipeline"""
        numeric_pipeline = Pipeline([
            ('imputer', KNNImputer(n_neighbors=5)),
            ('scaler', StandardScaler()),
            ('feature_selector', SelectKBest(score_func=f_classif, k=10))
        ])
        
        categorical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))
        ])
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_pipeline, numeric_features),
                ('cat', categorical_pipeline, categorical_features)
            ])
            
        self.pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('dim_reduction', PCA(n_components=0.95)),
            ('model', self.model)
        ])
        
    def optimize_pipeline(self, X, y, param_grid, cv=5):
        """Optimize pipeline using grid search"""
        grid_search = GridSearchCV(
            self.pipeline,
            param_grid,
            cv=cv,
            scoring='accuracy',
            n_jobs=-1
        )
        grid_search.fit(X, y)
        self.best_model = grid_search.best_estimator_
        return grid_search.best_params_, grid_search.best_score_

    def get_feature_importance(self, X):
        """Get feature importance from the model"""
        if hasattr(self.best_model, 'feature_importances_'):
            return pd.DataFrame({
                'feature': X.columns,
                'importance': self.best_model.feature_importances_
            }).sort_values('importance', ascending=False)
        return None

    def save_pipeline(self, filepath):
        """Save the pipeline to disk"""
        joblib.dump(self.pipeline, filepath)

    def load_pipeline(self, filepath):
        """Load a saved pipeline"""
        self.pipeline = joblib.load(filepath)

# Initialize the AutoML object
automl = AutoML()

# Load the dataset
data = automl.load_data('path/to/dataset.csv')

# Preprocess the data
data = automl.preprocess_data(data)

# Generate new features
data = automl.generate_features(data)

# Select the best model
model = automl.select_model(data)

# Tune hyperparameters
model = automl.tune_hyperparameters(model)

# Deploy the model
automl.deploy_model(model)

# Evaluate the model
evaluation_results = automl.evaluate_model(model)

print(evaluation_results)
