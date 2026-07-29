from sklearn.metrics import (
    f1_score, roc_auc_score, accuracy_score,
    confusion_matrix, classification_report
)
import numpy as np

class MetricsCalculator:
    """Calculate evaluation metrics for model performance."""
    
    @staticmethod
    def calculate_all_metrics(y_true, y_pred, y_proba=None):
        """Calculate all evaluation metrics."""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'f1_macro': f1_score(y_true, y_pred, average='macro'),
            'f1_weighted': f1_score(y_true, y_pred, average='weighted'),
            'confusion_matrix': confusion_matrix(y_true, y_pred)
        }
        
        if y_proba is not None:
            n_classes = y_proba.shape[1]
            if n_classes > 2:
                metrics['roc_auc'] = roc_auc_score(y_true, y_proba, multi_class='ovr')
            else:
                metrics['roc_auc'] = roc_auc_score(y_true, y_proba[:, 1])
        
        return metrics
    
    @staticmethod
    def sensitivity_at_threshold(y_true, y_proba, threshold=0.5):
        """Calculate sensitivity at a given threshold."""
        y_pred = (y_proba[:, 1] >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        return tp / (tp + fn) if (tp + fn) > 0 else 0
    
    @staticmethod
    def specificity_at_threshold(y_true, y_proba, threshold=0.5):
        """Calculate specificity at a given threshold."""
        y_pred = (y_proba[:, 1] >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        return tn / (tn + fp) if (tn + fp) > 0 else 0