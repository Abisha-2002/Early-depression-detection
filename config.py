"""
Configuration settings for the web application
"""

import os

class Config:
    """Base configuration"""
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-12345'
    DEBUG = True
    HOST = '0.0.0.0'
    PORT = 5000
    
    # Model settings
    MODEL_DIR = 'data/trained_models'
    ALLOWED_MODELS = ['random_forest', 'xgboost']
    
    # PHQ-9 settings
    PHQ9_MAX_SCORE = 27
    RISK_THRESHOLDS = {
        'low': 9,
        'moderate': 14,
        'high': 27
    }
    
    # Data settings
    MAX_AGE = 100
    MIN_AGE = 18

class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True

class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    SECRET_KEY = os.environ.get('SECRET_KEY')
    
    # Use production settings
    HOST = '0.0.0.0'
    PORT = int(os.environ.get('PORT', 5000))

# Select configuration based on environment
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}