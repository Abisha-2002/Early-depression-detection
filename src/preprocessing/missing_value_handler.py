import pandas as pd
import numpy as np

class MissingValueHandler:
    """Handle missing values in structured survey data."""
    
    def __init__(self, numerical_strategy='median', categorical_strategy='mode'):
        self.numerical_strategy = numerical_strategy
        self.categorical_strategy = categorical_strategy
        self.numerical_impute_values = {}
        self.categorical_impute_values = {}
    
    def fit(self, df, numerical_cols, categorical_cols):
        """Fit imputation values from training data."""
        for col in numerical_cols:
            self.numerical_impute_values[col] = df[col].median()
        for col in categorical_cols:
            self.categorical_impute_values[col] = df[col].mode()[0]
        return self
    
    def transform(self, df):
        """Apply imputation to data."""
        df_imputed = df.copy()
        for col, value in self.numerical_impute_values.items():
            df_imputed[col] = df_imputed[col].fillna(value)
        for col, value in self.categorical_impute_values.items():
            df_imputed[col] = df_imputed[col].fillna(value)
        return df_imputed