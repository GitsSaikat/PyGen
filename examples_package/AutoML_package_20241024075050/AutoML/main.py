import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from sklearn.impute import KNNImputer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA

class AutoML:
    def __init__(self):
        self.data = None
        self.model = None

    def load_data(self, file_path):
        self.data = pd.read_csv(file_path)
        return self.data

    def preprocess_data(self, data):
        # Handle missing values
        data.fillna(data.mean(), inplace=True)

        # Encode categorical variables
        categorical_cols = data.select_dtypes(include=['object']).columns
        data[categorical_cols] = data[categorical_cols].apply(lambda x: x.astype('category'))

        # Scale numerical variables
        numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns
        scaler = StandardScaler()
        data[numerical_cols] = scaler.fit_transform(data[numerical_cols])

        return data

    def generate_features(self, data):
        # Generate new features using mathematical operations
        data['new_feature_1'] = data['A'] * data['B']
        data['new_feature_2'] = data['A'] + data['B']

        return data

    def select_model(self, data):
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(data.drop('target', axis=1), data['target'], test_size=0.2, random_state=42)

        # Define models
        models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Gradient Boosting': xgb.XGBClassifier(objective='binary:logistic', n_estimators=100),
            'LightGBM': lgb.LGBMClassifier(objective='binary', n_estimators=100),
            'CatBoost': cb.CatBoostClassifier(iterations=100, random_state=42)
        }

        # Evaluate models
        evaluation_results = {}
        for model_name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            evaluation_results[model_name] = accuracy_score(y_test, y_pred)

        # Select the best model
        best_model_name = max(evaluation_results, key=evaluation_results.get)
        best_model = models[best_model_name]

        return best_model

    def tune_hyperparameters(self, model):
        # Define hyperparameter tuning space
        param_grid = {
            'n_estimators': [10, 100, 1000],
            'max_depth': [3, 5, 10]
        }

        # Perform hyperparameter tuning
        grid_search = GridSearchCV(model, param_grid, cv=3, scoring='accuracy')
        grid_search.fit(X_train, y_train)

        # Get the best model
        best_model = grid_search.best_estimator_

        return best_model

    def deploy_model(self, model):
        # Deploy the model to a production environment
        model_name = 'best_model'
        model_path = f'"{model_name}"'
        model.save(model_path)

    def advanced_preprocessing(self, data):
        """Advanced preprocessing with KNN imputation"""
        # KNN imputation for missing values
        imputer = KNNImputer(n_neighbors=5)
        data_imputed = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)
        return data_imputed

    def feature_selection(self, X, y, k=10):
        """Select top k features using ANOVA F-value"""
        selector = SelectKBest(score_func=f_classif, k=k)
        X_selected = selector.fit_transform(X, y)
        selected_features = X.columns[selector.get_support()].tolist()
        return X_selected, selected_features

    def dimensionality_reduction(self, X, n_components=0.95):
        """Reduce dimensionality using PCA"""
        pca = PCA(n_components=n_components)
        X_reduced = pca.fit_transform(X)
        return X_reduced, pca.explained_variance_ratio_
