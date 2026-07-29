from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import EditedNearestNeighbours

class SMOTEENNUtil:
    """Apply SMOTE-ENN for class imbalance correction."""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.smote_enn = SMOTEENN(random_state=random_state)
        self.smote = SMOTE(random_state=random_state)
        self.enn = EditedNearestNeighbours()
    
    def apply_smote_enn(self, X, y):
        """Apply SMOTE-ENN to training data."""
        X_resampled, y_resampled = self.smote_enn.fit_resample(X, y)
        return X_resampled, y_resampled
    
    def apply_smote_only(self, X, y):
        """Apply SMOTE only (without ENN)."""
        X_resampled, y_resampled = self.smote.fit_resample(X, y)
        return X_resampled, y_resampled
    
    def apply_enn_only(self, X, y):
        """Apply ENN only (without SMOTE)."""
        X_resampled, y_resampled = self.enn.fit_resample(X, y)
        return X_resampled, y_resampled