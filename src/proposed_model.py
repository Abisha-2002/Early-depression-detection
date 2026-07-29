"""
Proposed Model: Random Forest Classifier with Stratified 5-Fold Cross-Validation
Student: Wesly Jeyananthan Abisha (ITBIN-2313-0003)
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, classification_report,
    confusion_matrix
)
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

class ProposedRandomForestModel:
    """
    Proposed Model: Random Forest with Stratified 5-Fold Cross-Validation
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        self.cv_results = {}
        self.best_params = None
        
    def train_with_cv(self, X, y, n_splits=5):
        """
        Train Random Forest with Stratified 5-Fold Cross-Validation
        """
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Initialize Random Forest
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=self.random_state,
            n_jobs=-1
        )
        
        # Perform Stratified 5-Fold CV
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
        
        # Store results for each fold
        self.cv_results = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': [],
            'roc_auc': []
        }
        
        fold = 1
        for train_idx, val_idx in skf.split(X, y):
            X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Train on this fold
            fold_model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=self.random_state,
                n_jobs=-1
            )
            fold_model.fit(X_train, y_train)
            
            # Predict
            y_pred = fold_model.predict(X_val)
            y_proba = fold_model.predict_proba(X_val)[:, 1] if len(np.unique(y)) == 2 else None
            
            # Store metrics
            self.cv_results['accuracy'].append(accuracy_score(y_val, y_pred))
            self.cv_results['precision'].append(precision_score(y_val, y_pred, average='weighted'))
            self.cv_results['recall'].append(recall_score(y_val, y_pred, average='weighted'))
            self.cv_results['f1'].append(f1_score(y_val, y_pred, average='weighted'))
            
            if y_proba is not None:
                self.cv_results['roc_auc'].append(roc_auc_score(y_val, y_proba))
            
            print(f"Fold {fold} - Accuracy: {self.cv_results['accuracy'][-1]:.4f}, "
                  f"F1: {self.cv_results['f1'][-1]:.4f}")
            fold += 1
        
        # Train final model on ALL training data
        self.model.fit(X_scaled, y)
        
        return self.cv_results
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate the trained model on test data
        """
        X_test_scaled = self.scaler.transform(X_test)
        y_pred = self.model.predict(X_test_scaled)
        y_proba = self.model.predict_proba(X_test_scaled)[:, 1] if len(np.unique(y_test)) == 2 else None
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1': f1_score(y_test, y_pred, average='weighted'),
            'classification_report': classification_report(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
        
        if y_proba is not None:
            metrics['roc_auc'] = roc_auc_score(y_test, y_proba)
        
        return metrics
    
    def compare_with_baseline(self, baseline_metrics):
        """
        Compare this model's performance with baseline
        """
        comparison = {
            'model': 'Random Forest (Proposed)',
            'cv_mean_accuracy': np.mean(self.cv_results['accuracy']),
            'cv_std_accuracy': np.std(self.cv_results['accuracy']),
            'cv_mean_f1': np.mean(self.cv_results['f1']),
            'cv_std_f1': np.std(self.cv_results['f1'])
        }
        
        if baseline_metrics:
            comparison['baseline_accuracy'] = baseline_metrics.get('accuracy', 'N/A')
            comparison['baseline_f1'] = baseline_metrics.get('f1', 'N/A')
            comparison['improvement'] = (
                comparison['cv_mean_accuracy'] - baseline_metrics.get('accuracy', 0)
            )
        
        return comparison
    
    def plot_confusion_matrix(self, y_true, y_pred, title="Confusion Matrix"):
        """
        Visualize confusion matrix
        """
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(title)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()
    
    def get_feature_importance(self, feature_names):
        """
        Get feature importance from Random Forest
        """
        importance = pd.DataFrame({
            'feature': feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance


# =====================
# Example Usage
# =====================
if __name__ == "__main__":
    print("="*60)
    print("PROPOSED MODEL: Random Forest with Stratified 5-Fold CV")
    print("Student: Wesly Jeyananthan Abisha (ITBIN-2313-0003)")
    print("="*60)
    
    # Example with sample data
    from sklearn.datasets import make_classification
    
    # Generate sample data
    X, y = make_classification(n_samples=200, n_features=10, n_classes=2, random_state=42)
    
    # Create and train model
    model = ProposedRandomForestModel()
    cv_results = model.train_with_cv(X, y, n_splits=5)
    
    print("\n" + "="*60)
    print("CROSS-VALIDATION RESULTS")
    print("="*60)
    print(f"Mean Accuracy: {np.mean(cv_results['accuracy']):.4f} (±{np.std(cv_results['accuracy']):.4f})")
    print(f"Mean F1-Score: {np.mean(cv_results['f1']):.4f} (±{np.std(cv_results['f1']):.4f})")
    print(f"Mean Precision: {np.mean(cv_results['precision']):.4f}")
    print(f"Mean Recall: {np.mean(cv_results['recall']):.4f}")