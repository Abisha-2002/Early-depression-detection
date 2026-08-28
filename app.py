"""
Depression Risk Prediction Web Application
MindAssist - Complete System
"""

from flask import Flask, render_template, request, jsonify, session
import numpy as np
import pickle
import os
from datetime import datetime

# Try to import NLP modules, but continue if not available
try:
    from nlp.unicode_normalizer import UnicodeNormalizer
    from nlp.language_identifier import LanguageIdentifier
    from nlp.transliterator import Transliterator
    from nlp.feature_extractor import NLPFeatureExtractor
    NLP_AVAILABLE = True
    print("✅ NLP modules loaded successfully")
except ImportError as e:
    print(f"⚠️ NLP modules not available: {e}")
    NLP_AVAILABLE = False

app = Flask(__name__)
app.secret_key = 'mindassist-secret-key-2024'

# Initialize NLP components if available
if NLP_AVAILABLE:
    normalizer = UnicodeNormalizer()
    language_identifier = LanguageIdentifier()
    transliterator = Transliterator()
    feature_extractor = NLPFeatureExtractor()

# PHQ-9 Questions
PHQ9_DATA = [
    {"id": 1, "question": "Little interest or pleasure in doing things?", "category": "Anhedonia"},
    {"id": 2, "question": "Feeling down, depressed, or hopeless?", "category": "Mood"},
    {"id": 3, "question": "Trouble falling or staying asleep, or sleeping too much?", "category": "Sleep"},
    {"id": 4, "question": "Feeling tired or having little energy?", "category": "Energy"},
    {"id": 5, "question": "Poor appetite or overeating?", "category": "Appetite"},
    {"id": 6, "question": "Feeling bad about yourself — or that you are a failure or have let yourself or your family down?", "category": "Self-worth"},
    {"id": 7, "question": "Trouble concentrating on things, such as reading the newspaper or watching television?", "category": "Concentration"},
    {"id": 8, "question": "Moving or speaking so slowly that other people could have noticed? Or the opposite — being so fidgety or restless that you have been moving around a lot more than usual?", "category": "Psychomotor"},
    {"id": 9, "question": "Thoughts that you would be better off dead, or of hurting yourself?", "category": "Suicidality"}
]

# University Options
UNIVERSITY_OPTIONS = [
    {"value": "horizon", "label": "Horizon Campus"},
    {"value": "jaffna", "label": "University of Jaffna"},
    {"value": "moratuwa", "label": "University of Moratuwa"},
    {"value": "eastern", "label": "University of Eastern"},
    {"value": "other", "label": "Other"}
]

# Year of Study Options
YEAR_OPTIONS = [
    {"value": "1", "label": "First Year"},
    {"value": "2", "label": "Second Year"},
    {"value": "3", "label": "Third Year"},
    {"value": "4", "label": "Fourth Year"},
    {"value": "other", "label": "Other"}
]

# Gender Options
GENDER_OPTIONS = [
    {"value": "0", "label": "Prefer not to say"},
    {"value": "1", "label": "Male"},
    {"value": "2", "label": "Female"}
]

# Load models
models = {}
model_metadata = {}

def load_models():
    """Load trained models with metadata"""
    model_dir = 'data/trained_models'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir, exist_ok=True)
        return
    
    model_files = {
        'xgboost': {'file': 'xgboost_m4.pkl', 'name': 'XGBoost (M4)', 'type': 'Gradient Boosting', 'accuracy': 0.89, 'status': 'Proposed'},
        'random_forest': {'file': 'random_forest.pkl', 'name': 'Random Forest', 'type': 'Ensemble', 'accuracy': 0.85, 'status': 'Baseline'}
    }
    
    for key, info in model_files.items():
        try:
            with open(os.path.join(model_dir, info['file']), 'rb') as f:
                models[key] = pickle.load(f)
                model_metadata[key] = info
                print(f"✅ Loaded: {info['name']}")
        except Exception as e:
            print(f"⚠️ Could not load {info['name']}: {e}")

def process_text(text):
    """Process code-mixed text through NLP pipeline"""
    if not NLP_AVAILABLE:
        # Fallback: Just split by spaces and tag as unknown
        return {
            'original': text,
            'normalized': text,
            'tokens': [(word, 'unknown') for word in text.split()],
            'transliterated': text,
            'features': None
        }
    
    try:
        # Step 1: Normalize Unicode
        normalized = normalizer.normalize(text)
        
        # Step 2: Identify languages
        tokens = language_identifier.identify_tokens(normalized)
        
        # Step 3: Transliterate if needed
        transliterated = []
        for token, lang in tokens:
            if lang == 'sinhala':
                transliterated.append(transliterator.sinhala_to_roman(token))
            elif lang == 'tamil':
                transliterated.append(transliterator.tamil_to_roman(token))
            else:
                transliterated.append(token)
        
        processed_text = ' '.join(transliterated)
        
        # Step 4: Extract features
        features = feature_extractor.extract_features([processed_text]) if processed_text else None
        
        return {
            'original': text,
            'normalized': normalized,
            'tokens': tokens,
            'transliterated': processed_text,
            'features': features
        }
    except Exception as e:
        print(f"⚠️ NLP Processing failed, using fallback: {e}")
        # Ultimate Fallback: Return raw text so the page never crashes
        return {
            'original': text,
            'normalized': text,
            'tokens': [(word, 'unknown') for word in text.split()],
            'transliterated': text,
            'features': None
        }

def calculate_phq9_score(responses):
    """Calculate total PHQ-9 score and severity"""
    total = sum(responses)
    if total <= 4:
        severity = "Minimal"
        description = "You are experiencing minimal depression symptoms."
        color = "#28a745"
    elif total <= 9:
        severity = "Mild"
        description = "You are experiencing mild depression symptoms."
        color = "#ffc107"
    elif total <= 14:
        severity = "Moderate"
        description = "You are experiencing moderate depression symptoms."
        color = "#fd7e14"
    elif total <= 19:
        severity = "Moderately Severe"
        description = "You are experiencing moderately severe depression symptoms."
        color = "#dc3545"
    else:
        severity = "Severe"
        description = "You are experiencing severe depression symptoms."
        color = "#721c24"
    
    return {
        'total': total,
        'severity': severity,
        'description': description,
        'color': color,
        'max_score': 27
    }

@app.route('/')
def index():
    """Home page with PHQ-9 questionnaire"""
    return render_template('index.html', 
                         phq9_data=PHQ9_DATA,
                         models=model_metadata,
                         universities=UNIVERSITY_OPTIONS,
                         years=YEAR_OPTIONS,
                         genders=GENDER_OPTIONS,
                         year=datetime.now().year)

@app.route('/about')
def about():
    """About page"""
    return render_template('about.html', year=datetime.now().year)

@app.route('/contact')
def contact():
    """Contact page"""
    return render_template('contact.html', year=datetime.now().year)

# ROUTE 1: SHOWS THE INPUT FORM (GET Request)
@app.route('/text-analysis')
def text_analysis():
    """Text analysis page"""
    # Pass an empty result so the page loads without crashing
    empty_result = {
        'text_analysis': {
            'original': "No text submitted yet.",
            'normalized': "No text submitted yet.",
            'transliterated': "No text submitted yet.",
            'tokens': []
        },
        'language_breakdown': {'sinhala': 0, 'tamil': 0, 'english': 0},
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    return render_template('text_analysis.html', 
                         result=empty_result,
                         universities=UNIVERSITY_OPTIONS,
                         years=YEAR_OPTIONS,
                         genders=GENDER_OPTIONS,
                         year=datetime.now().year,
                         nlp_available=NLP_AVAILABLE)

# ROUTE 2: PROCESSES THE TEXT (POST Request)
@app.route('/analyze_text', methods=['POST'])
def analyze_text():
    """Analyze code-mixed text for depression risk"""
    try:
        # Get text from the form
        user_text = request.form.get('user_text', '')
        
        if not user_text:
            return render_template('error.html', error="Please enter some text to analyze.")
        
        # Process text (if NLP works, use it; if not, it just returns the text)
        processed = process_text(user_text)
        
        # Count language tokens
        lang_counts = {'sinhala': 0, 'tamil': 0, 'english': 0}
        for _, lang in processed['tokens']:
            if lang in lang_counts:
                lang_counts[lang] += 1
        
        # Create the exact structure HTML expects
        result = {
            'text_analysis': {
                'original': processed.get('original', user_text),
                'normalized': processed.get('normalized', user_text),
                'transliterated': processed.get('transliterated', user_text),
                'tokens': processed.get('tokens', [])
            },
            'language_breakdown': lang_counts,
            'nlp_available': NLP_AVAILABLE,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Render the template with the actual result
        return render_template('text_analysis.html', 
                             result=result,
                             universities=UNIVERSITY_OPTIONS,
                             years=YEAR_OPTIONS,
                             genders=GENDER_OPTIONS,
                             year=datetime.now().year)
        
    except Exception as e:
        print(f"❌ Error in text analysis: {e}")
        import traceback
        traceback.print_exc()
        return render_template('error.html', error=str(e))

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction request"""
    try:
        # Get form data
        data = request.form
        
        # Extract PHQ-9 scores
        responses = []
        for i in range(1, 10):
            score = int(data.get(f'phq9_{i}', 0))
            responses.append(score)
        
        # Calculate PHQ-9 score
        phq9_result = calculate_phq9_score(responses)
        
        # Extract demographic data
        demographics = {
            'age': int(data.get('age', 25)),
            'gender': data.get('gender', '0'),
            'institution': data.get('institution', 'horizon'),
            'year': data.get('year', '1')
        }
        
        # Get model type
        model_type = data.get('model_type', 'xgboost')
        model = models.get(model_type)
        
        # ML Prediction
        ml_prediction = None
        if model:
            try:
                # Convert demographics to numeric for ML
                gender_map = {'0': 0, '1': 1, '2': 2}
                institution_map = {'horizon': 0, 'jaffna': 1, 'moratuwa': 2, 'eastern': 3, 'other': 4}
                year_map = {'1': 1, '2': 2, '3': 3, '4': 4, 'other': 0}
                
                features = responses + [
                    demographics['age'],
                    gender_map.get(demographics['gender'], 0),
                    institution_map.get(demographics['institution'], 0),
                    year_map.get(demographics['year'], 1)
                ]
                features_array = np.array(features).reshape(1, -1)
                
                prediction = model.predict(features_array)
                risk_levels = ['Low', 'Moderate', 'High']
                ml_risk = risk_levels[prediction[0]]
                
                probabilities = None
                if hasattr(model, 'predict_proba'):
                    probabilities = model.predict_proba(features_array)[0].tolist()
                
                ml_prediction = {
                    'risk': ml_risk,
                    'score': int(prediction[0]),
                    'probabilities': probabilities
                }
            except Exception as e:
                print(f"⚠️ ML Prediction error: {e}")
        
        # Get university label
        university_label = "Horizon Campus"
        for uni in UNIVERSITY_OPTIONS:
            if uni['value'] == demographics['institution']:
                university_label = uni['label']
                break
        
        # Get year label
        year_label = "First Year"
        for y in YEAR_OPTIONS:
            if y['value'] == demographics['year']:
                year_label = y['label']
                break
        
        result = {
            'phq9': phq9_result,
            'demographics': {
                'age': demographics['age'],
                'gender': demographics['gender'],
                'institution': demographics['institution'],
                'institution_label': university_label,
                'year': demographics['year'],
                'year_label': year_label
            },
            'ml_prediction': ml_prediction,
            'model_used': model_type if model else 'Rule-based',
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        return render_template('result.html', result=result, year=datetime.now().year)
        
    except Exception as e:
        print(f"❌ Error in prediction: {e}")
        import traceback
        traceback.print_exc()
        return render_template('error.html', error=str(e))

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'models_loaded': list(models.keys()),
        'nlp_available': NLP_AVAILABLE,
        'timestamp': datetime.now().isoformat()
    })

if __name__ == '__main__':
    print("=" * 70)
    print("🧠 MindAssist - Depression Risk Prediction System")
    print("Student: Wesly Jeyananthan Abisha (ITBIN-2313-0003)")
    print("=" * 70)
    
    # Load models
    load_models()
    
    # Run app
    app.run(debug=True, host='0.0.0.0', port=5000)