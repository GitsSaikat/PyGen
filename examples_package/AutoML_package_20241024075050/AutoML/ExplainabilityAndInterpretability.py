import pandas as pd
from AutoML.main import AutoML
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import eli5
from pdpbox import pdp
import shap

class ExplainabilityAndInterpretability:
    def __init__(self):
        self.automl = AutoML()

    def calculate_feature_importance(self, model, X, y):
        # Initialize an empty dictionary to store feature importances
        feature_importances = {}

        # Loop through each feature
        for feature in X.columns:
            # Create a copy of the input data
            X_copy = X.copy()

            # Permute the feature values randomly
            X_copy[feature] = np.random.permutation(X_copy[feature])

            # Predict outcomes with the modified data
            predicted_outcomes_permuted = model.predict(X_copy)

            # Predict outcomes with the original data
            predicted_outcomes_original = model.predict(X)

            # Calculate the difference between the predicted outcomes
            diff = np.abs(predicted_outcomes_permuted - predicted_outcomes_original)

            # Calculate the feature importance
            feature_importance = np.mean(diff)

            # Store the feature importance in the dictionary
            feature_importances[feature] = feature_importance

        return feature_importances

    def create_partial_dependence_plot(self, model, X, feature):
        # Initialize an empty list to store feature values
        feature_values = []

        # Initialize an empty list to store predicted outcomes
        predicted_outcomes = []

        # Loop through each unique value of the feature
        for value in X[feature].unique():
            # Create a copy of the input data
            X_copy = X.copy()

            # Set the feature value to the current value
            X_copy[feature] = value

            # Predict outcomes with the modified data
            predicted_outcome = model.predict(X_copy)

            # Store the feature value and predicted outcome
            feature_values.append(value)
            predicted_outcomes.append(predicted_outcome)

        # Create a line plot of the feature values and predicted outcomes
        plt.plot(feature_values, predicted_outcomes)
        plt.xlabel(feature)
        plt.ylabel('Predicted Outcome')
        plt.show()

    def calculate_shap_values(self, model, X):
        # Create a SHAP explainer
        explainer = shap.Explainer(model)

        # Calculate SHAP values for the input data
        shap_values = explainer(X)

        return shap_values

    def generate_lime_explanations(self, model, X, y):
        # Create a LIME explainer
        explainer = lime_tabular.LimeTabularExplainer(X, feature_names=X.columns)

        # Generate LIME explanations for the input data
        explanations = []
        for i in range(len(X)):
            exp = explainer.explain_instance(X.iloc[i], model.predict, num_features=10)
            explanations.append(exp)

        return explanations

    def plot_feature_importance_chart(self, feature_importances):
        """Plot feature importance chart"""
        importance_df = pd.DataFrame({
            'feature': feature_importances.keys(),
            'importance': feature_importances.values()
        })
        importance_df = importance_df.sort_values('importance', ascending=False)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(data=importance_df, x='importance', y='feature')
        plt.title('Feature Importance')
        plt.tight_layout()
        return plt

    def generate_eli5_explanation(self, model, X):
        """Generate model explanation using ELI5"""
        explanation = eli5.explain_prediction(model, X)
        return explanation

    def plot_pdp_interaction(self, model, X, feature1, feature2):
        """Plot partial dependence interaction between two features"""
        pdp_interact = pdp.pdp_interact(
            model=model,
            dataset=X,
            model_features=X.columns,
            features=[feature1, feature2]
        )
        pdp.pdp_interact_plot(pdp_interact, feature1, feature2)
        return plt

# Usage example
if __name__ == '__main__':
    explainability_interpretability = ExplainabilityAndInterpretability()

    # Load data
    data = pd.read_csv('path/to/dataset.csv')

    # Calculate feature importances
    model = ExplainabilityAndInterpretability()
    feature_importances = explainability_interpretability.calculate_feature_importance(model, data.drop('target', axis=1), data['target'])
    print(feature_importances)

    # Create partial dependence plots
    feature = 'feature_name'
    explainability_interpretability.create_partial_dependence_plot(model, data.drop('target', axis=1), feature)

    # Calculate SHAP values
    shap_values = explainability_interpretability.calculate_shap_values(model, data.drop('target', axis=1))
    print(shap_values)

    # Generate LIME explanations
    explanations = explainability_interpretability.generate_lime_explanations(model, data.drop('target', axis=1), data['target'])
    print(explanations)
