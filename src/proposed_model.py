"""
Proposed Model: Random Forest Classifier with Stratified 5-Fold Cross-Validation
Student: Wesly Jeyananthan Abisha (ITBIN-2313-0003)
Milestone 2: Random Forest Model for Depression Risk Prediction (Comparison)
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
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
    Used for comparison with XGBoost (M4)
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        self.cv_results = {}
        
    def train_with_cv(self, X, y, n_splits=5, use_smote=True):
        """
        Train Random Forest with Stratified 5-Fold Cross-Validation
        """
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Apply SMOTE if requested
        if use_smote:
            from imblearn.over_sampling import SMOTE
            from imblearn.under_sampling import EditedNearestNeighbours
            from imblearn.combine import SMOTEENN
            
            smote_enn = SMOTEENN(random_state=self.random_state)
            X_resampled, y_resampled = smote_enn.fit_resample(X_scaled, y)
            print(f"✅ SMOTE-ENN Applied: {len(X)} → {len(X_resampled)} samples")
        else:
            X_resampled, y_resampled = X_scaled, y
        
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
        
        self.cv_results = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': [],
            'roc_auc': []
        }
        
        print("="*70)
        print("Random Forest Training with Stratified 5-Fold CV")
        print("="*70)
        
        fold = 1
        for train_idx, val_idx in skf.split(X_resampled, y_resampled):
            X_train, X_val = X_resampled[train_idx], X_resampled[val_idx]
            y_train, y_val = y_resampled[train_idx], y_resampled[val_idx]
            
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
            y_proba = fold_model.predict_proba(X_val)
            
            # Store metrics
            self.cv_results['accuracy'].append(accuracy_score(y_val, y_pred))
            self.cv_results['precision'].append(precision_score(y_val, y_pred, average='weighted'))
            self.cv_results['recall'].append(recall_score(y_val, y_pred, average='weighted'))
            self.cv_results['f1'].append(f1_score(y_val, y_pred, average='weighted'))
            
            if y_proba.shape[1] > 1:
                try:
                    self.cv_results['roc_auc'].append(
                        roc_auc_score(y_val, y_proba, multi_class='ovr')
                    )
                except:
                    self.cv_results['roc_auc'].append(None)
            
            print(f"Fold {fold} - Accuracy: {self.cv_results['accuracy'][-1]:.4f}, "
                  f"F1: {self.cv_results['f1'][-1]:.4f}")
            fold += 1
        
        # Train final model on ALL training data
        self.model.fit(X_resampled, y_resampled)
        
        print("\n" + "="*70)
        print("CROSS-VALIDATION SUMMARY")
        print("="*70)
        print(f"Mean Accuracy:  {np.mean(self.cv_results['accuracy']):.4f} (±{np.std(self.cv_results['accuracy']):.4f})")
        print(f"Mean F1-Score:   {np.mean(self.cv_results['f1']):.4f} (±{np.std(self.cv_results['f1']):.4f})")
        
        return self.cv_results
    
    def evaluate(self, X_test, y_test):
        """Evaluate the trained model on test data"""
        X_test_scaled = self.scaler.transform(X_test)
        y_pred = self.model.predict(X_test_scaled)
        y_proba = self.model.predict_proba(X_test_scaled)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1': f1_score(y_test, y_pred, average='weighted'),
            'classification_report': classification_report(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
        
        if y_proba.shape[1] > 1:
            try:
                metrics['roc_auc'] = roc_auc_score(y_test, y_proba, multi_class='ovr')
            except:
                metrics['roc_auc'] = None
        
        print("\n" + "="*70)
        print("RANDOM FOREST - FINAL EVALUATION")
        print("="*70)
        print(f"Accuracy:  {metrics['accuracy']:.4f}")
        print(f"F1-Score:  {metrics['f1']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        
        return metrics
    
    def plot_feature_importance(self, feature_names=None, top_n=10):
        """Plot feature importance from Random Forest"""
        if feature_names is None:
            feature_names = [f'Feature_{i}' for i in range(len(self.model.feature_importances_))]
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        plt.figure(figsize=(10, 6))
        top_features = importance_df.head(top_n)
        plt.barh(top_features['feature'], top_features['importance'])
        plt.xlabel('Importance')
        plt.title(f'Top {top_n} Feature Importances (Random Forest)')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()
        
        return importance_df


# =====================
# Example Usage
# =====================
if __name__ == "__main__":
    from sklearn.datasets import make_classification
    
    print("="*70)
    print("Random Forest: Stratified 5-Fold CV for Depression Risk Prediction")
    print("Student: Wesly Jeyananthan Abisha (ITBIN-2313-0003)")
    print("="*70)
    
    # Generate sample data (simulating PHQ-9 features)
    X, y = make_classification(
        n_samples=200, 
        n_features=15, 
        n_informative=10,
        n_redundant=2,
        n_classes=3,
        random_state=42
    )
    
    print(f"\n📊 Dataset Shape: {X.shape}")
    print(f"📊 Class Distribution: {np.bincount(y)}")
    
    model = ProposedRandomForestModel()
    cv_results = model.train_with_cv(X, y, n_splits=5)
    
    print("\n" + "="*70)
    print("✅ RANDOM FOREST TRAINING COMPLETE")
    print("="*70)
    print(f"📈 Mean Accuracy:  {np.mean(cv_results['accuracy']):.4f} (±{np.std(cv_results['accuracy']):.4f})")
    print(f"📈 Mean F1-Score:   {np.mean(cv_results['f1']):.4f} (±{np.std(cv_results['f1']):.4f})")