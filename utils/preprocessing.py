"""
Data preprocessing utilities for the web application
"""

import numpy as np
import pandas as pd

def preprocess_features(features):
    """
    Preprocess input features for model prediction
    
    Args:
        features: List of feature values [phq9_1..phq9_9, age, gender, institution, year]
    
    Returns:
        Processed feature array
    """
    # Ensure features are numeric
    features = [float(f) for f in features]
    
    # Normalize PHQ-9 scores if needed
    # (No normalization needed as scores are already 0-3)
    
    # Scale age if needed (optional)
    # age = features[9] / 100  # Normalize age
    
    # Convert to numpy array
    return np.array(features)

def calculate_phq9_score(phq9_scores):
    """Calculate total PHQ-9 score"""
    return sum(phq9_scores)

def get_risk_level(phq9_total):
    """Get risk level based on PHQ-9 total score"""
    if phq9_total <= 9:
        return 'Low'
    elif phq9_total <= 14:
        return 'Moderate'
    else:
        return 'High'

def validate_features(features):
    """Validate input features"""
    errors = []
    
    # Check PHQ-9 scores (0-3)
    for i in range(9):
        if not (0 <= features[i] <= 3):
            errors.append(f"PHQ-9 question {i+1} must be between 0 and 3")
    
    # Check age
    if not (18 <= features[9] <= 100):
        errors.append("Age must be between 18 and 100")
    
    # Check gender (0, 1, 2)
    if features[10] not in [0, 1, 2]:
        errors.append("Invalid gender value")
    
    # Check institution
    if not (0 <= features[11] <= 4):
        errors.append("Invalid institution value")
    
    # Check year
    if not (1 <= features[12] <= 4):
        errors.append("Year must be between 1 and 4")
    
    return errors