"""
Model training.

Author: Danial Malik
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

class ChurnPredictor:
    """
    Trains churn prediction models.
    """
    
    def __init__(self):
        """Initialize the ChurnPredictor."""
        self.outputs_dir = Path("data/outputs")
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        
        # Model configurations
        self.models = {
            'logistic_regression': LogisticRegression(random_state=42, max_iter=1000),
            'random_forest': RandomForestClassifier(random_state=42)
        }
        
        self.trained_models = {}
        
    def train_logistic_regression(self, X_train: pd.DataFrame, y_train: pd.Series, 
                                 dataset_name: str):
        """
        Train a Logistic Regression model for churn prediction.
        """
        print(f"Training Logistic Regression model for {dataset_name}...")
        
        model = self.models['logistic_regression']
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Cross-validation score
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1')
        
        results = {
            'model_name': 'logistic_regression',
            'dataset_name': dataset_name,
            'model': model,
            'cv_scores': cv_scores,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std()
        }
        
        print(f"Logistic Regression trained. CV F1: {cv_scores.mean():.4f}")
        
        return results
    
    def train_random_forest(self, X_train: pd.DataFrame, y_train: pd.Series, 
                           dataset_name: str):
        """
        Train a Random Forest model for churn prediction.
        """
        print(f"Training Random Forest model for {dataset_name}...")
        
        model = self.models['random_forest']
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Cross-validation score
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1')
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        results = {
            'model_name': 'random_forest',
            'dataset_name': dataset_name,
            'model': model,
            'cv_scores': cv_scores,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'feature_importance': feature_importance
        }
        
        print(f"Random Forest trained. CV F1: {cv_scores.mean():.4f}")
        
        return results
    
    def train_models_for_dataset(self, data_dict: Dict, dataset_name: str):
        """
        Train all models for a specific dataset.
        """
        print(f"Training models for {dataset_name}...")
        
        X_train = data_dict['X_train']
        y_train = data_dict['y_train']
        
        models = {}
        
        # Train Logistic Regression
        lr_results = self.train_logistic_regression(X_train, y_train, dataset_name)
        models['logistic_regression'] = lr_results
        
        # Train Random Forest
        rf_results = self.train_random_forest(X_train, y_train, dataset_name)
        models['random_forest'] = rf_results
        
        return models
    
    def train_models(self, analysis_ready_data: Dict[str, Dict]):
        """
        Train models for all datasets.
        """
        print("Training models for all datasets...")
        
        all_models = {}
        
        for dataset_name, data_dict in analysis_ready_data.items():
            dataset_models = self.train_models_for_dataset(data_dict, dataset_name)
            all_models[dataset_name] = dataset_models
        
        print(f"Model training complete. Trained models for {len(all_models)} datasets.")
        
        return all_models

def main():
    """Main function to run model training independently."""
    predictor = ChurnPredictor()
    
    # Example usage
    print("Model training module")

if __name__ == "__main__":
    main() 