import numpy as np
from sklearn.metrics import roc_curve

class ThresholdCalibration:
    """Calibrate thresholds using Youden's J statistic."""
    
    @staticmethod
    def youdens_j(y_true, y_proba):
        """Calculate Youden's J statistic for threshold selection."""
        fpr, tpr, thresholds = roc_curve(y_true, y_proba)
        
        j_scores = tpr - fpr
        optimal_idx = np.argmax(j_scores)
        
        return {
            'threshold': thresholds[optimal_idx],
            'j_score': j_scores[optimal_idx],
            'sensitivity': tpr[optimal_idx],
            'specificity': 1 - fpr[optimal_idx]
        }