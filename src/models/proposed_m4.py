"""
M4: XGBoost + Feature Fusion + SMOTE-ENN + Cost-Sensitive Weighting
Student: Wesly Jeyananthan Abisha (ITBIN-2313-0003)
Milestone 2: Proposed Model Implementation for Depression Risk Prediction
"""

import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, classification_report,
    confusion_matrix
)
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns


class ProposedM4:
    """
    M4: XGBoost with Feature Fusion + SMOTE-ENN + Cost-Sensitive Weighting
    
    Justification:
    - XGBoost chosen over deep learning due to dataset scale (N≈200)
    - Trotzek et al. [13] show hybrid architectures outperform deep models
    - SMOTE-ENN + cost-sensitive weighting from Rathod et al. [5]
    - Feature fusion from Tadesse et al. [7]
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        self.cv_results = {}
        self.best_params = None
        self.feature_importance = None
    
    def train_with_cv(self, X, y, n_splits=5, use_smote=True):
        """
        Train XGBoost with Stratified 5-Fold Cross-Validation
        """
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Apply SMOTE-ENN if requested
        if use_smote:
            from ..imbalance.smote_enn_utils import SMOTEENNUtil
            smote_util = SMOTEENNUtil(random_state=self.random_state)
            X_resampled, y_resampled = smote_util.apply_smote_enn(X_scaled, y)
            print(f"✅ SMOTE-ENN Applied: {len(X)} → {len(X_resampled)} samples")
        else:
            X_resampled, y_resampled = X_scaled, y
        
        # Calculate scale_pos_weight for imbalanced data
        unique_classes = np.unique(y_resampled)
        if len(unique_classes) == 2:
            n_neg = np.sum(y_resampled == 0)
            n_pos = np.sum(y_resampled == 1)
            scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
            print(f"⚖️ Cost-Sensitive Weight (scale_pos_weight): {scale_pos_weight:.2f}")
        else:
            scale_pos_weight = 1.0
            print(f"⚖️ Multi-class classification (3 risk tiers). scale_pos_weight = 1.0")
        
        # Initialize XGBoost with cost-sensitive weighting
        self.model = xgb.XGBClassifier(
            max_depth=5,
            learning_rate=0.1,
            n_estimators=200,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            random_state=self.random_state,
            eval_metric='mlogloss',
            use_label_encoder=False
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
        print("M4: XGBoost Training with Stratified 5-Fold CV")
        print("="*70)
        
        fold = 1
        for train_idx, val_idx in skf.split(X_resampled, y_resampled):
            X_train, X_val = X_resampled[train_idx], X_resampled[val_idx]
            y_train, y_val = y_resampled[train_idx], y_resampled[val_idx]
            
            fold_model = xgb.XGBClassifier(
                max_depth=5,
                learning_rate=0.1,
                n_estimators=200,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=scale_pos_weight,
                random_state=self.random_state,
                eval_metric='mlogloss',
                use_label_encoder=False
            )
            fold_model.fit(X_train, y_train)
            
            y_pred = fold_model.predict(X_val)
            y_proba = fold_model.predict_proba(X_val)
            
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
        
        self.model.fit(X_resampled, y_resampled)
        self.feature_importance = self.model.feature_importances_
        
        print("\n" + "="*70)
        print("CROSS-VALIDATION SUMMARY")
        print("="*70)
        print(f"Mean Accuracy:  {np.mean(self.cv_results['accuracy']):.4f} (±{np.std(self.cv_results['accuracy']):.4f})")
        print(f"Mean F1-Score:   {np.mean(self.cv_results['f1']):.4f} (±{np.std(self.cv_results['f1']):.4f})")
        print(f"Mean Precision: {np.mean(self.cv_results['precision']):.4f}")
        print(f"Mean Recall:    {np.mean(self.cv_results['recall']):.4f}")
        
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
        print("XGBOOST (M4) - FINAL EVALUATION")
        print("="*70)
        print(f"Accuracy:  {metrics['accuracy']:.4f}")
        print(f"F1-Score:  {metrics['f1']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        if metrics.get('roc_auc'):
            print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
        
        return metrics
    
    def hyperparameter_tuning(self, X_train, y_train, param_grid=None):
        """Perform grid search for hyperparameter tuning"""
        X_scaled = self.scaler.fit_transform(X_train)
        
        if param_grid is None:
            param_grid = {
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.05, 0.1],
                'n_estimators': [50, 100, 200],
                'subsample': [0.6, 0.8, 1.0],
                'colsample_bytree': [0.6, 0.8, 1.0]
            }
        
        print("="*70)
        print("HYPERPARAMETER TUNING (Grid Search with 5-Fold CV)")
        print("="*70)
        
        grid_search = GridSearchCV(
            xgb.XGBClassifier(
                random_state=self.random_state,
                eval_metric='mlogloss',
                use_label_encoder=False
            ),
            param_grid,
            cv=5,
            scoring='f1_macro',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_scaled, y_train)
        self.best_params = grid_search.best_params_
        self.model = grid_search.best_estimator_
        
        print("\n" + "="*70)
        print("BEST HYPERPARAMETERS FOUND:")
        print("="*70)
        for param, value in self.best_params.items():
            print(f"  {param}: {value}")
        print(f"\nBest CV Score: {grid_search.best_score_:.4f}")
        
        return self.best_params
    
    def compare_with_baseline(self, baseline_metrics):
        """Compare XGBoost performance with baseline models"""
        comparison = {
            'model': 'XGBoost (M4)',
            'cv_mean_accuracy': np.mean(self.cv_results['accuracy']),
            'cv_std_accuracy': np.std(self.cv_results['accuracy']),
            'cv_mean_f1': np.mean(self.cv_results['f1']),
            'cv_std_f1': np.std(self.cv_results['f1'])
        }
        
        if baseline_metrics:
            comparison['baseline_accuracy'] = baseline_metrics.get('accuracy', 'N/A')
            comparison['baseline_f1'] = baseline_metrics.get('f1', 'N/A')
            comparison['improvement_accuracy'] = (
                comparison['cv_mean_accuracy'] - baseline_metrics.get('accuracy', 0)
            )
            comparison['improvement_f1'] = (
                comparison['cv_mean_f1'] - baseline_metrics.get('f1', 0)
            )
            
            print("\n" + "="*70)
            print("PERFORMANCE COMPARISON: XGBoost vs Baseline")
            print("="*70)
            print(f"XGBoost (M4) Mean Accuracy: {comparison['cv_mean_accuracy']:.4f}")
            print(f"Baseline Accuracy:           {comparison['baseline_accuracy']:.4f}")
            print(f"Improvement:                 {comparison['improvement_accuracy']:.4f}")
        
        return comparison


# =====================
# Example Usage
# =====================
if __name__ == "__main__":
    from sklearn.datasets import make_classification
    
    print("="*70)
    print("M4: XGBoost + Feature Fusion + SMOTE-ENN + Cost-Sensitive")
    print("Student: Wesly Jeyananthan Abisha (ITBIN-2313-0003)")
    print("="*70)
    
    X, y = make_classification(
        n_samples=200, 
        n_features=15, 
        n_informative=10,
        n_redundant=2,
        n_classes=3,
        random_state=42
    )
    
    print(f"\n📊 Dataset Shape: {X.shape}")
    print(f"📊 Target Classes: {np.unique(y)}")
    print(f"📊 Class Distribution: {np.bincount(y)}")
    
    model = ProposedM4()
    cv_results = model.train_with_cv(X, y, n_splits=5)
    
    print("\n" + "="*70)
    print("✅ XGBOOST (M4) TRAINING COMPLETE")
    print("="*70)
    print(f"📈 Mean Accuracy:  {np.mean(cv_results['accuracy']):.4f} (±{np.std(cv_results['accuracy']):.4f})")
    print(f"📈 Mean F1-Score:   {np.mean(cv_results['f1']):.4f} (±{np.std(cv_results['f1']):.4f})")