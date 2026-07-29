from sklearn.preprocessing import StandardScaler
import pandas as pd

class Standardiser:
    """Standardise numerical features to zero mean, unit variance."""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_names = None
    
    def fit(self, df, numerical_cols):
        """Fit scaler to training data."""
        self.feature_names = numerical_cols
        self.scaler.fit(df[numerical_cols])
        return self
    
    def transform(self, df):
        """Apply standardisation to data."""
        df_scaled = df.copy()
        df_scaled[self.feature_names] = self.scaler.transform(df[self.feature_names])
        return df_scaled