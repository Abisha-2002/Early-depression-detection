import numpy as np
from scipy import stats

class StatisticalTests:
    """Statistical tests for model comparison."""
    
    @staticmethod
    def mcnemar_test(y_true, y_pred_model1, y_pred_model2):
        """Perform McNemar's test to compare two models."""
        a = np.sum((y_pred_model1 == y_true) & (y_pred_model2 == y_true))
        b = np.sum((y_pred_model1 == y_true) & (y_pred_model2 != y_true))
        c = np.sum((y_pred_model1 != y_true) & (y_pred_model2 == y_true))
        d = np.sum((y_pred_model1 != y_true) & (y_pred_model2 != y_true))
        
        chi2 = ((b - c) ** 2) / (b + c) if (b + c) > 0 else 0
        p_value = 1 - stats.chi2.cdf(chi2, 1)
        
        return {
            'chi2': chi2,
            'p_value': p_value,
            'significant': p_value < 0.05
        }
    
    @staticmethod
    def wilcoxon_test(scores1, scores2):
        """Perform Wilcoxon signed-rank test."""
        statistic, p_value = stats.wilcoxon(scores1, scores2)
        return {
            'statistic': statistic,
            'p_value': p_value,
            'significant': p_value < 0.05
        }