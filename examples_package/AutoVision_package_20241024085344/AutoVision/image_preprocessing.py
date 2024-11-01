import cv2
import numpy as np

def resize_image(image, target_size):
    return cv2.resize(image, target_size)

def normalize_image(image):
    return image / 255.0

def preprocess_pipeline(image, target_size=(224, 224)):
    """Complete preprocessing pipeline"""
    # Resize
    image = resize_image(image, target_size)
    
    # Denoise
    image = cv2.fastNlMeansDenoisingColored(image)
    
    # Normalize
    image = normalize_image(image)
    
    # Enhance contrast
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    enhanced = cv2.merge((l,a,b))
    image = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    
    return image

def apply_data_augmentation(image):
    """Enhanced data augmentation with more transformations"""
    augmented = image.copy()
    
    # Random rotation
    angle = np.random.uniform(-30, 30)
    height, width = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((width/2, height/2), angle, 1.0)
    augmented = cv2.warpAffine(augmented, matrix, (width, height))
    
    # Random brightness/contrast
    alpha = np.random.uniform(0.8, 1.2)  # Contrast
    beta = np.random.uniform(-30, 30)    # Brightness
    augmented = cv2.convertScaleAbs(augmented, alpha=alpha, beta=beta)
    
    # Random noise
    if np.random.random() > 0.5:
        noise = np.random.normal(0, 0.05, augmented.shape)
        augmented = np.clip(augmented + noise, 0, 1)
        
    return augmented
