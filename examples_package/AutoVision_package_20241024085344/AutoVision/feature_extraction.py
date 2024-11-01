import cv2
import numpy as np
from sklearn.feature_extraction import image as image_utils

def extract_features(image):
    # Extract multiple types of features
    features = {}
    
    # HOG features for shape detection
    hog = cv2.HOGDescriptor()
    features['hog'] = hog.compute(image)
    
    # SIFT features for keypoints
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)
    features['sift'] = descriptors
    
    # Color histogram features
    color_hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    features['color'] = color_hist.flatten()
    
    return features
