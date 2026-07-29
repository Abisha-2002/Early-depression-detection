"""
Data Loading and Preprocessing for Survey Data
"""

import pandas as pd
import numpy as np
from typing import Tuple, List

def load_survey_data(filepath: str = "data/raw/survey_data_2026.csv") -> pd.DataFrame:
    """
    Load survey data from CSV file
    """
    df = pd.read_csv(filepath)
    return df

def clean_survey_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the survey data
    """
    df_clean = df.copy()
    
    # Rename columns for easier access
    # PHQ-9 columns (questions 1-9)
    phq_columns = [
        'phq9_1', 'phq9_2', 'phq9_3', 'phq9_4', 'phq9_5',
        'phq9_6', 'phq9_7', 'phq9_8', 'phq9_9'
    ]
    
    # Calculate PHQ-9 total score
    df_clean['phq9_total'] = df_clean[phq_columns].sum(axis=1)
    
    # Classify risk tiers
    df_clean['risk_tier'] = pd.cut(
        df_clean['phq9_total'],
        bins=[0, 9, 14, 27],
        labels=['Low', 'Moderate', 'High']
    )
    
    return df_clean

def get_features_and_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Extract features and target for modeling
    """
    # Features
    feature_cols = [
        'phq9_1', 'phq9_2', 'phq9_3', 'phq9_4', 'phq9_5',
        'phq9_6', 'phq9_7', 'phq9_8', 'phq9_9',
        'age', 'gender', 'institution', 'year'
    ]
    
    X = df[feature_cols].copy()
    
    # Target
    y = df['phq9_total'].copy()
    
    return X, y

if __name__ == "__main__":
    # Test loading
    df = load_survey_data()
    print(f"Loaded {len(df)} responses")
    print(f"Columns: {df.columns.tolist()}")
    
    df_clean = clean_survey_data(df)
    print(f"\nRisk distribution:")
    print(df_clean['risk_tier'].value_counts())