"""
Depression Risk Prediction Web Application
SOC-AI-Health-Web-System
Student: Wesly Jeyananthan Abisha (ITBIN-2313-0003)
"""

from flask import Flask, render_template, request, jsonify, session, redirect, url_for
import pandas as pd
import numpy as np
import pickle
import os
import json
from datetime import datetime
from config import Config
from models.model_loader import ModelLoader
from utils.preprocessing import preprocess_features

app = Flask(__name__)
app.config.from_object(Config)

# Initialize model loader
model_loader = ModelLoader()

# PHQ-9 Questions
PHQ9_QUESTIONS = [
    "Little interest or pleasure in doing things?",
    "Feeling down, depressed, or hopeless?",
    "Trouble falling or staying asleep, or sleeping too much?",
    "Feeling tired or having little energy?",
    "Poor appetite or overeating?",
    "Feeling bad about yourself — or that you are a failure or have let yourself or your family down?",
    "Trouble concentrating on things, such as reading the newspaper or watching television?",
    "Moving or speaking so slowly that other people could have noticed? Or the opposite — being so fidgety or restless that you have been moving around a lot more than usual?",
    "Thoughts that you would be better off dead, or of hurting yourself?"
]

@app.route('/')
def index():
    """Home page with PHQ-9 questionnaire"""
    return render_template('index.html', 
                         questions=PHQ9_QUESTIONS,
                         models=model_loader.get_available_models())

@app.route('/about')
def about():
    """About page"""
    return render_template('about.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction request"""
    try:
        # Get form data
        data = request.form if request.form else request.get_json()
        
        # Extract PHQ-9 scores
        phq9_scores = []
        for i in range(1, 10):
            score_key = f'phq9_{i}'
            score = int(data.get(score_key, 0))
            phq9_scores.append(score)
        
        # Extract demographic data
        age = int(data.get('age', 25))
        gender = data.get('gender', '0')
        institution = data.get('institution', '0')
        year = int(data.get('year', 1))
        
        # Create feature vector
        features = phq9_scores + [age, gender, institution, year]
        
        # Calculate PHQ-9 total
        phq9_total = sum(phq9_scores)
        
        # Get model type
        model_type = data.get('model_type', 'xgboost')
        
        # Preprocess features
        features_processed = preprocess_features(features)
        
        # Make prediction
        result = model_loader.predict(model_type, features_processed)
        
        if result is None:
            return render_template('error.html', 
                                 error="Model not available. Please try again later.")
        
        # Prepare response
        response = {
            'phq9_total': phq9_total,
            'risk_level': result['risk_level'],
            'risk_score': result['risk_score'],
            'probabilities': result.get('probabilities'),
            'model_used': model_type,
            'features': features,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Store in session for result page
        session['prediction_result'] = response
        
        # Check if it's an API request
        if request.is_json:
            return jsonify(response)
        else:
            return render_template('result.html', result=response)
            
    except Exception as e:
        print(f"Error in prediction: {e}")
        if request.is_json:
            return jsonify({'error': str(e)}), 400
        else:
            return render_template('error.html', error=str(e))

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """REST API endpoint for predictions"""
    return predict()

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'models_loaded': model_loader.get_model_status(),
        'timestamp': datetime.now().isoformat()
    })

@app.errorhandler(404)
def not_found(error):
    return render_template('error.html', error="Page not found"), 404

@app.errorhandler(500)
def server_error(error):
    return render_template('error.html', error="Server error. Please try again later."), 500

if __name__ == '__main__':
    print("=" * 70)
    print("SOC-AI-HEALTH-WEB-SYSTEM")
    print("Depression Risk Prediction Web Application")
    print("Student: Wesly Jeyananthan Abisha (ITBIN-2313-0003)")
    print("=" * 70)
    
    # Load models
    model_loader.load_all_models()
    
    # Run app
    app.run(debug=app.config['DEBUG'], 
            host=app.config['HOST'], 
            port=app.config['PORT'])