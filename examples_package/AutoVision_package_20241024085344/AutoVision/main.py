import cv2
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import os
import sys

class AutoVision:
    def __init__(self, model_path=None):
        """Initialize AutoVision with optional pre-trained model"""
        self.model = None
        self.feature_extractors = {}
        self.classes = []
        
        # Get absolute path for model files
        if model_path:
            self.model_path = self._get_resource_path(model_path)
            self.load_model(self.model_path)

    def _get_resource_path(self, relative_path):
        """Get absolute path to resource, works for dev and PyInstaller"""
        base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
        return os.path.join(base_path, relative_path)

    def load_image(self, img_path):
        """Load and validate image"""
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Failed to load image from {img_path}")
        return img

    def preprocess_image(self, img, target_size=(224, 224)):
        """Enhanced preprocessing pipeline"""
        from .image_preprocessing import preprocess_pipeline
        return preprocess_pipeline(img, target_size)

    def detect_objects(self, img):
        """Enhanced object detection with confidence scores"""
        from .object_detection import detect_objects
        return detect_objects(img)

    def extract_features(self, img):
        """Extract comprehensive feature set"""
        from .feature_extraction import extract_features
        return extract_features(img)

    def train(self, images, labels, test_size=0.2):
        """Train the model with extracted features"""
        features = []
        for img in images:
            processed_img = self.preprocess_image(img)
            img_features = self.extract_features(processed_img)
            # Concatenate all feature types
            feature_vector = np.concatenate([
                img_features['hog'].flatten(),
                img_features['color'].flatten(),
                img_features['sift'].flatten() if img_features['sift'] is not None else np.zeros(128)
            ])
            features.append(feature_vector)

        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=test_size, random_state=42
        )

        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)

        # Evaluate model
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        
        return {
            'accuracy': accuracy,
            'report': report,
            'model': self.model
        }

    def predict(self, img):
        """Predict class for new image"""
        if self.model is None:
            raise ValueError("Model not trained. Please train the model first.")
            
        processed_img = self.preprocess_image(img)
        features = self.extract_features(processed_img)
        
        # Concatenate features similar to training
        feature_vector = np.concatenate([
            features['hog'].flatten(),
            features['color'].flatten(),
            features['sift'].flatten() if features['sift'] is not None else np.zeros(128)
        ])
        
        prediction = self.model.predict([feature_vector])[0]
        confidence = np.max(self.model.predict_proba([feature_vector]))
        
        return {
            'prediction': prediction,
            'confidence': confidence
        }

    def save_model(self, path):
        """Save trained model"""
        if self.model is None:
            raise ValueError("No model to save. Please train the model first.")
        import joblib
        joblib.dump(self.model, path)

    def load_model(self, path):
        """Load pre-trained model"""
        import joblib
        self.model = joblib.load(path)
