import pandas as pd
from sklearn.preprocessing import OneHotEncoder

class EncodingUtils:
    """Handle categorical encoding for survey features."""
    
    def __init__(self, encoding_type='one_hot'):
        self.encoding_type = encoding_type
        self.encoder = None
    
    def one_hot_encode(self, df, columns):
        """Apply one-hot encoding to specified columns."""
        df_encoded = df.copy()
        for col in columns:
            if col in df.columns:
                dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                df_encoded = pd.concat([df_encoded, dummies], axis=1)
                df_encoded = df_encoded.drop(columns=[col])
        return df_encoded
    
    def target_encode(self, df, column, target_col):
        """Apply target encoding for specified column."""
        pass