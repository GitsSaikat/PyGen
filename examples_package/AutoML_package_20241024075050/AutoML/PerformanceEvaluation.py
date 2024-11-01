import pandas as pd
from AutoML.main import AutoML
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, mean_absolute_error, roc_curve, auc, confusion_matrix, roc_auc_score, cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns

class PerformanceEvaluation:
    def __init__(self):
        self.automl = AutoML()

    def evaluate_performance(self, model, X_test, y_test):
        """Enhanced performance evaluation with cross-validation"""
        predictions = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        metrics = {
            'accuracy': accuracy_score(y_test, predictions),
            'precision': precision_score(y_test, predictions, average='weighted'),
            'recall': recall_score(y_test, predictions, average='weighted'),
            'f1_score': f1_score(y_test, predictions, average='weighted'),
            'roc_auc': roc_auc_score(y_test, proba) if proba is not None else None
        }
        
        # Add cross-validation scores
        cv_scores = cross_val_score(model, X_test, y_test, cv=5)
        metrics['cv_mean'] = cv_scores.mean()
        metrics['cv_std'] = cv_scores.std()
        
        return metrics

    def compare_models(self, models):
        evaluation_results = {}
        for model_name, model in models.items():
            evaluation_results[model_name] = self.evaluate_performance(model, X_test, y_test)
        return evaluation_results

    def plot_roc_curve(self, model, X_test, y_test):
        """Plot ROC curve and calculate AUC"""
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        return plt

    def plot_confusion_matrix(self, y_true, y_pred):
        """Plot confusion matrix using seaborn"""
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        return plt

# Compare Multiple Models   
