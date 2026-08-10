"""
Save trained models for web deployment
"""

import pickle
import os
import sys
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import numpy as np

def create_dummy_models():
    """Create dummy models for testing"""
    print("Creating dummy models for testing...")
    
    # Create directory
    os.makedirs('data/trained_models', exist_ok=True)
    
    # Create dummy data
    X = np.random.rand(100, 13)
    y = np.random.randint(0, 3, 100)
    
    # Random Forest
    rf = RandomForestClassifier(n_estimators=10, random_state=42)
    rf.fit(X, y)
    with open('data/trained_models/random_forest.pkl', 'wb') as f:
        pickle.dump(rf, f)
    print("✅ Random Forest model created")
    
    # XGBoost
    xgb = XGBClassifier(n_estimators=10, random_state=42)
    xgb.fit(X, y)
    with open('data/trained_models/xgboost_m4.pkl', 'wb') as f:
        pickle.dump(xgb, f)
    print("✅ XGBoost model created")

if __name__ == "__main__":
    print("=" * 70)
    print("SAVING MODELS FOR WEB DEPLOYMENT")
    print("=" * 70)
    create_dummy_models()
    print("\n✅ Models saved successfully!")