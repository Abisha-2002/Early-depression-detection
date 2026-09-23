"""
Depression Risk Prediction Web Application
MindAssist - Complete System

Flask web application for:
- PHQ-9 depression symptom assessment
- ML-based depression risk prediction
- Sinhala/Tamil/English code-mixed text analysis
- XGBoost and Random Forest model prediction
"""

from flask import Flask, render_template, request, jsonify
import numpy as np
import pickle
import os
from datetime import datetime


# ============================================================
# APPLICATION CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)

# Use environment variable in production.
# Fallback is only for local development.
app.secret_key = os.environ.get(
    "SECRET_KEY",
    "mindassist-development-secret-key"
)


# ============================================================
# NLP MODULES
# ============================================================

NLP_AVAILABLE = False

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


# ============================================================
# INITIALIZE NLP COMPONENTS
# ============================================================

normalizer = None
language_identifier = None
transliterator = None
feature_extractor = None

if NLP_AVAILABLE:
    try:
        normalizer = UnicodeNormalizer()
        language_identifier = LanguageIdentifier()
        transliterator = Transliterator()
        feature_extractor = NLPFeatureExtractor()

        print("✅ NLP components initialized successfully")

    except Exception as e:
        print(f"⚠️ NLP initialization failed: {e}")

        NLP_AVAILABLE = False
        normalizer = None
        language_identifier = None
        transliterator = None
        feature_extractor = None


# ============================================================
# PHQ-9 QUESTIONS
# ============================================================

PHQ9_DATA = [
    {
        "id": 1,
        "question": "Little interest or pleasure in doing things?",
        "category": "Anhedonia"
    },
    {
        "id": 2,
        "question": "Feeling down, depressed, or hopeless?",
        "category": "Mood"
    },
    {
        "id": 3,
        "question": "Trouble falling or staying asleep, or sleeping too much?",
        "category": "Sleep"
    },
    {
        "id": 4,
        "question": "Feeling tired or having little energy?",
        "category": "Energy"
    },
    {
        "id": 5,
        "question": "Poor appetite or overeating?",
        "category": "Appetite"
    },
    {
        "id": 6,
        "question": (
            "Feeling bad about yourself — or that you are a failure "
            "or have let yourself or your family down?"
        ),
        "category": "Self-worth"
    },
    {
        "id": 7,
        "question": (
            "Trouble concentrating on things, such as reading the newspaper "
            "or watching television?"
        ),
        "category": "Concentration"
    },
    {
        "id": 8,
        "question": (
            "Moving or speaking so slowly that other people could have noticed? "
            "Or the opposite — being so fidgety or restless that you have been "
            "moving around a lot more than usual?"
        ),
        "category": "Psychomotor"
    },
    {
        "id": 9,
        "question": (
            "Thoughts that you would be better off dead, "
            "or of hurting yourself?"
        ),
        "category": "Suicidality"
    }
]


# ============================================================
# UNIVERSITY OPTIONS
# ============================================================

UNIVERSITY_OPTIONS = [
    {
        "value": "horizon",
        "label": "Horizon Campus"
    },
    {
        "value": "jaffna",
        "label": "University of Jaffna"
    },
    {
        "value": "moratuwa",
        "label": "University of Moratuwa"
    },
    {
        "value": "eastern",
        "label": "University of Eastern"
    },
    {
        "value": "other",
        "label": "Other"
    }
]


# ============================================================
# YEAR OPTIONS
# ============================================================

YEAR_OPTIONS = [
    {
        "value": "1",
        "label": "First Year"
    },
    {
        "value": "2",
        "label": "Second Year"
    },
    {
        "value": "3",
        "label": "Third Year"
    },
    {
        "value": "4",
        "label": "Fourth Year"
    },
    {
        "value": "other",
        "label": "Other"
    }
]


# ============================================================
# GENDER OPTIONS
# ============================================================

GENDER_OPTIONS = [
    {
        "value": "0",
        "label": "Prefer not to say"
    },
    {
        "value": "1",
        "label": "Male"
    },
    {
        "value": "2",
        "label": "Female"
    }
]


# ============================================================
# MODEL CONFIGURATION
# ============================================================

models = {}
model_metadata = {}

MODEL_DIR = os.path.join(
    BASE_DIR,
    "data",
    "trained_models"
)

MODEL_FILES = {
    "xgboost": {
        "file": "xgboost_m4.pkl",
        "name": "XGBoost (M4)",
        "type": "Gradient Boosting",
        "accuracy": 0.89,
        "status": "Proposed"
    },

    "random_forest": {
        "file": "random_forest.pkl",
        "name": "Random Forest",
        "type": "Ensemble",
        "accuracy": 0.85,
        "status": "Baseline"
    }
}


# ============================================================
# LOAD TRAINED MODELS
# ============================================================

def load_models():
    """
    Load trained ML models from data/trained_models.
    """

    global models
    global model_metadata

    models.clear()
    model_metadata.clear()

    if not os.path.exists(MODEL_DIR):
        print(f"⚠️ Model directory not found: {MODEL_DIR}")
        return

    for key, info in MODEL_FILES.items():

        model_path = os.path.join(
            MODEL_DIR,
            info["file"]
        )

        if not os.path.exists(model_path):
            print(
                f"⚠️ Model file not found: "
                f"{model_path}"
            )
            continue

        try:

            with open(model_path, "rb") as file:
                loaded_model = pickle.load(file)

            models[key] = loaded_model
            model_metadata[key] = info

            print(
                f"✅ Loaded model: "
                f"{info['name']}"
            )

        except Exception as e:

            print(
                f"❌ Could not load "
                f"{info['name']}: {e}"
            )


# ============================================================
# TEXT PROCESSING
# ============================================================

def process_text(text):
    """
    Process Sinhala/Tamil/English code-mixed text.

    If NLP modules are unavailable, a simple
    fallback processing method is used.
    """

    if text is None:
        text = ""

    text = str(text).strip()

    # --------------------------------------------------------
    # FALLBACK
    # --------------------------------------------------------

    if not NLP_AVAILABLE:

        return {
            "original": text,
            "normalized": text,
            "tokens": [
                (word, "unknown")
                for word in text.split()
            ],
            "transliterated": text,
            "features": None
        }

    try:

        # ----------------------------------------------------
        # STEP 1: Unicode normalization
        # ----------------------------------------------------

        normalized = normalizer.normalize(text)

        # ----------------------------------------------------
        # STEP 2: Language identification
        # ----------------------------------------------------

        tokens = language_identifier.identify_tokens(
            normalized
        )

        # ----------------------------------------------------
        # STEP 3: Transliteration
        # ----------------------------------------------------

        transliterated_tokens = []

        for token, language in tokens:

            if language == "sinhala":

                transliterated_tokens.append(
                    transliterator.sinhala_to_roman(token)
                )

            elif language == "tamil":

                transliterated_tokens.append(
                    transliterator.tamil_to_roman(token)
                )

            else:

                transliterated_tokens.append(token)

        processed_text = " ".join(
            transliterated_tokens
        )

        # ----------------------------------------------------
        # STEP 4: Feature extraction
        # ----------------------------------------------------

        features = None

        if processed_text:

            features = feature_extractor.extract_features(
                [processed_text]
            )

        return {
            "original": text,
            "normalized": normalized,
            "tokens": tokens,
            "transliterated": processed_text,
            "features": features
        }

    except Exception as e:

        print(
            f"⚠️ NLP processing failed: {e}"
        )

        return {
            "original": text,
            "normalized": text,
            "tokens": [
                (word, "unknown")
                for word in text.split()
            ],
            "transliterated": text,
            "features": None
        }


# ============================================================
# PHQ-9 SCORE CALCULATION
# ============================================================

def calculate_phq9_score(responses):
    """
    Calculate PHQ-9 total score and severity.
    """

    # Make sure values are integers.
    clean_responses = []

    for value in responses:

        try:
            value = int(value)

        except (TypeError, ValueError):
            value = 0

        # PHQ-9 item score must be 0-3.
        value = max(0, min(3, value))

        clean_responses.append(value)

    total = sum(clean_responses)

    if total <= 4:

        severity = "Minimal"
        description = (
            "You are experiencing minimal depression symptoms."
        )
        color = "#28a745"

    elif total <= 9:

        severity = "Mild"
        description = (
            "You are experiencing mild depression symptoms."
        )
        color = "#ffc107"

    elif total <= 14:

        severity = "Moderate"
        description = (
            "You are experiencing moderate depression symptoms."
        )
        color = "#fd7e14"

    elif total <= 19:

        severity = "Moderately Severe"
        description = (
            "You are experiencing moderately severe "
            "depression symptoms."
        )
        color = "#dc3545"

    else:

        severity = "Severe"
        description = (
            "You are experiencing severe depression symptoms."
        )
        color = "#721c24"

    return {
        "total": total,
        "severity": severity,
        "description": description,
        "color": color,
        "max_score": 27
    }


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
def index():

    return render_template(
        "index.html",
        phq9_data=PHQ9_DATA,
        models=model_metadata,
        universities=UNIVERSITY_OPTIONS,
        years=YEAR_OPTIONS,
        genders=GENDER_OPTIONS,
        year=datetime.now().year
    )


# ============================================================
# ABOUT
# ============================================================

@app.route("/about")
def about():

    return render_template(
        "about.html",
        year=datetime.now().year
    )


# ============================================================
# CONTACT
# ============================================================

@app.route("/contact")
def contact():

    return render_template(
        "contact.html",
        year=datetime.now().year
    )


# ============================================================
# TEXT ANALYSIS PAGE
# ============================================================

@app.route("/text-analysis")
def text_analysis():

    empty_result = {
        "text_analysis": {
            "original": "No text submitted yet.",
            "normalized": "No text submitted yet.",
            "transliterated": "No text submitted yet.",
            "tokens": []
        },

        "language_breakdown": {
            "sinhala": 0,
            "tamil": 0,
            "english": 0
        },

        "timestamp": datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )
    }

    return render_template(
        "text_analysis.html",
        result=empty_result,
        universities=UNIVERSITY_OPTIONS,
        years=YEAR_OPTIONS,
        genders=GENDER_OPTIONS,
        year=datetime.now().year,
        nlp_available=NLP_AVAILABLE
    )


# ============================================================
# ANALYZE TEXT
# ============================================================

@app.route(
    "/analyze_text",
    methods=["POST"]
)
def analyze_text():

    try:

        user_text = request.form.get(
            "user_text",
            ""
        ).strip()

        if not user_text:

            return render_template(
                "error.html",
                error="Please enter some text to analyze."
            )

        # Process text.
        processed = process_text(
            user_text
        )

        # Count languages.
        language_breakdown = {
            "sinhala": 0,
            "tamil": 0,
            "english": 0
        }

        for token_info in processed.get(
            "tokens",
            []
        ):

            try:

                _, language = token_info

                language = str(
                    language
                ).lower()

                if language in language_breakdown:
                    language_breakdown[language] += 1

            except (ValueError, TypeError):

                continue

        result = {
            "text_analysis": {
                "original": processed.get(
                    "original",
                    user_text
                ),

                "normalized": processed.get(
                    "normalized",
                    user_text
                ),

                "transliterated": processed.get(
                    "transliterated",
                    user_text
                ),

                "tokens": processed.get(
                    "tokens",
                    []
                )
            },

            "language_breakdown": language_breakdown,

            "nlp_available": NLP_AVAILABLE,

            "timestamp": datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            )
        }

        return render_template(
            "text_analysis.html",
            result=result,
            universities=UNIVERSITY_OPTIONS,
            years=YEAR_OPTIONS,
            genders=GENDER_OPTIONS,
            year=datetime.now().year,
            nlp_available=NLP_AVAILABLE
        )

    except Exception as e:

        print(
            f"❌ Error in text analysis: {e}"
        )

        return render_template(
            "error.html",
            error=str(e)
        )


# ============================================================
# PREDICTION
# ============================================================

@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    try:

        # ----------------------------------------------------
        # GET FORM DATA
        # ----------------------------------------------------

        data = request.form

        # ----------------------------------------------------
        # PHQ-9 RESPONSES
        # ----------------------------------------------------

        responses = []

        for i in range(1, 10):

            try:

                score = int(
                    data.get(
                        f"phq9_{i}",
                        0
                    )
                )

            except (TypeError, ValueError):

                score = 0

            # Restrict PHQ-9 response to 0-3.
            score = max(
                0,
                min(3, score)
            )

            responses.append(score)

        # ----------------------------------------------------
        # PHQ-9 RESULT
        # ----------------------------------------------------

        phq9_result = calculate_phq9_score(
            responses
        )

        # ----------------------------------------------------
        # DEMOGRAPHICS
        # ----------------------------------------------------

        try:

            age = int(
                data.get(
                    "age",
                    25
                )
            )

        except (TypeError, ValueError):

            age = 25

        demographics = {
            "age": age,

            "gender": data.get(
                "gender",
                "0"
            ),

            "institution": data.get(
                "institution",
                "horizon"
            ),

            "year": data.get(
                "year",
                "1"
            )
        }

        # ----------------------------------------------------
        # MODEL TYPE
        # ----------------------------------------------------

        model_type = data.get(
            "model_type",
            "xgboost"
        )

        # Prevent invalid model names.
        if model_type not in MODEL_FILES:

            model_type = "xgboost"

        model = models.get(
            model_type
        )

        # ----------------------------------------------------
        # ML PREDICTION
        # ----------------------------------------------------

        ml_prediction = None

        if model is not None:

            try:

                # --------------------------------------------
                # Encode categorical variables
                # --------------------------------------------

                gender_map = {
                    "0": 0,
                    "1": 1,
                    "2": 2
                }

                institution_map = {
                    "horizon": 0,
                    "jaffna": 1,
                    "moratuwa": 2,
                    "eastern": 3,
                    "other": 4
                }

                year_map = {
                    "1": 1,
                    "2": 2,
                    "3": 3,
                    "4": 4,
                    "other": 0
                }

                # --------------------------------------------
                # Create feature vector
                # --------------------------------------------

                features = responses + [
                    demographics["age"],

                    gender_map.get(
                        demographics["gender"],
                        0
                    ),

                    institution_map.get(
                        demographics["institution"],
                        0
                    ),

                    year_map.get(
                        demographics["year"],
                        1
                    )
                ]

                features_array = np.array(
                    features,
                    dtype=float
                ).reshape(
                    1,
                    -1
                )

                # --------------------------------------------
                # Prediction
                # --------------------------------------------

                prediction = model.predict(
                    features_array
                )

                predicted_class = int(
                    prediction[0]
                )

                risk_levels = [
                    "Low",
                    "Moderate",
                    "High"
                ]

                # Protect against invalid model output.
                if (
                    predicted_class < 0
                    or predicted_class >= len(risk_levels)
                ):

                    predicted_class = 0

                ml_risk = risk_levels[
                    predicted_class
                ]

                # --------------------------------------------
                # Probabilities
                # --------------------------------------------

                probabilities = None

                if hasattr(
                    model,
                    "predict_proba"
                ):

                    probabilities = (
                        model.predict_proba(
                            features_array
                        )[0]
                        .tolist()
                    )

                ml_prediction = {
                    "risk": ml_risk,
                    "score": predicted_class,
                    "probabilities": probabilities
                }

            except Exception as e:

                print(
                    f"⚠️ ML prediction error: {e}"
                )

                ml_prediction = None

        # ----------------------------------------------------
        # UNIVERSITY LABEL
        # ----------------------------------------------------

        university_label = "Horizon Campus"

        for university in UNIVERSITY_OPTIONS:

            if (
                university["value"]
                == demographics["institution"]
            ):

                university_label = university["label"]
                break

        # ----------------------------------------------------
        # YEAR LABEL
        # ----------------------------------------------------

        year_label = "First Year"

        for year_option in YEAR_OPTIONS:

            if (
                year_option["value"]
                == demographics["year"]
            ):

                year_label = year_option["label"]
                break

        # ----------------------------------------------------
        # FINAL RESULT
        # ----------------------------------------------------

        result = {

            "phq9": phq9_result,

            "demographics": {

                "age": demographics["age"],

                "gender": demographics["gender"],

                "institution": demographics[
                    "institution"
                ],

                "institution_label": university_label,

                "year": demographics["year"],

                "year_label": year_label
            },

            "ml_prediction": ml_prediction,

            "model_used": (
                model_type
                if model is not None
                else "Rule-based"
            ),

            "timestamp": datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            )
        }

        return render_template(
            "result.html",
            result=result,
            year=datetime.now().year
        )

    except Exception as e:

        print(
            f"❌ Error in prediction: {e}"
        )

        return render_template(
            "error.html",
            error=str(e)
        )


# ============================================================
# HEALTH CHECK
# ============================================================

@app.route("/health")
def health():

    return jsonify({
        "status": "healthy",

        "models_loaded": list(
            models.keys()
        ),

        "nlp_available": NLP_AVAILABLE,

        "timestamp": datetime.now().isoformat()
    })


# ============================================================
# APPLICATION STARTUP
# ============================================================

# Load models when Flask starts.
load_models()


# ============================================================
# LOCAL DEVELOPMENT
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print(
        "🧠 MindAssist - Depression Risk Prediction System"
    )
    print(
        "Student: Wesly Jeyananthan Abisha"
    )
    print(
        "Student ID: ITBIN-2313-0003"
    )
    print("=" * 70)

    print(
        f"📁 Base directory: {BASE_DIR}"
    )

    print(
        f"📁 Model directory: {MODEL_DIR}"
    )

    print(
        f"🤖 Models loaded: {list(models.keys())}"
    )

    print(
        f"🔤 NLP available: {NLP_AVAILABLE}"
    )

    app.run(
        debug=True,
        host="0.0.0.0",
        port=int(
            os.environ.get(
                "PORT",
                5000
            )
        )
    )