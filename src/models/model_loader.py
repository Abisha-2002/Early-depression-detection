"""
Model loading and management for the web application
"""

import pickle
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

class ModelLoader:
    """Load and manage trained models"""
    
    def __init__(self, model_dir='data/trained_models'):
        self.model_dir = model_dir
        self.models = {}
        self.model_info = {}
        self.risk_levels = ['Low', 'Moderate', 'High']
        
    def load_model(self, model_name):
        """Load a specific model"""
        model_path = os.path.join(self.model_dir, f'{model_name}.pkl')
        
        if not os.path.exists(model_path):
            print(f"⚠️ Model not found: {model_path}")
            return None
        
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            
            self.models[model_name] = model
            self.model_info[model_name] = {
                'loaded': True,
                'path': model_path,
                'type': type(model).__name__
            }
            print(f"✅ Loaded: {model_name} ({type(model).__name__})")
            return model
        except Exception as e:
            print(f"❌ Error loading {model_name}: {e}")
            return None
    
    def load_all_models(self):
        """Load all available models"""
        # Find all .pkl files in the model directory
        if not os.path.exists(self.model_dir):
            print(f"⚠️ Model directory not found: {self.model_dir}")
            return self.models
        
        model_files = [f.replace('.pkl', '') for f in os.listdir(self.model_dir) 
                      if f.endswith('.pkl')]
        
        for model_name in model_files:
            self.load_model(model_name)
        
        return self.models
    
    def get_available_models(self):
        """Get list of available models"""
        available = []
        for name, info in self.model_info.items():
            if info.get('loaded', False):
                available.append({
                    'name': name,
                    'type': info.get('type', 'Unknown'),
                    'label': name.replace('_', ' ').title()
                })
        return available
    
    def get_model_status(self):
        """Get status of all models"""
        return {
            name: info.get('loaded', False) 
            for name, info in self.model_info.items()
        }
    
    def predict(self, model_name, features):
        """Make prediction using loaded model"""
        if model_name not in self.models:
            print(f"❌ Model not loaded: {model_name}")
            return None
        
        model = self.models[model_name]
        features_array = np.array(features).reshape(1, -1)
        
        try:
            # Make prediction
            prediction = model.predict(features_array)
            risk_level = self.risk_levels[prediction[0]]
            
            # Get probabilities if available
            probabilities = None
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(features_array)[0].tolist()
            
            return {
                'risk_level': risk_level,
                'risk_score': int(prediction[0]),
                'probabilities': probabilities
            }
        except Exception as e:
            print(f"❌ Error making prediction: {e}")
            return None