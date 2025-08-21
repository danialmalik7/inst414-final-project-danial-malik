"""
Model training.

Author: Danial Malik
"""

import pandas as pd
import numpy as np
import logging
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
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Model configurations - we'll create new instances for each dataset
        self.model_configs = {
            'logistic_regression': lambda: LogisticRegression(random_state=42, max_iter=1000),
            'random_forest': lambda: RandomForestClassifier(random_state=42)
        }
        
        self.trained_models = {}
        
    def train_logistic_regression(self, X_train: pd.DataFrame, y_train: pd.Series, 
                                 dataset_name: str):
        """
        Train a Logistic Regression model for churn prediction.
        """
        self.logger.info(f"Training Logistic Regression model for {dataset_name}...")
        
        try:
            model = self.model_configs['logistic_regression']()
            
            # Train the model
            model.fit(X_train, y_train)
            self.logger.info(f"Logistic Regression model fitted successfully for {dataset_name}")
            
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
            
            self.logger.info(f"Logistic Regression trained successfully. CV F1: {cv_scores.mean():.4f}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error training Logistic Regression for {dataset_name}: {str(e)}")
            raise
    
    def train_random_forest(self, X_train: pd.DataFrame, y_train: pd.Series, 
                           dataset_name: str):
        """
        Train a Random Forest model for churn prediction.
        """
        self.logger.info(f"Training Random Forest model for {dataset_name}...")
        
        try:
            model = self.model_configs['random_forest']()
            
            # Train the model
            model.fit(X_train, y_train)
            self.logger.info(f"Random Forest model fitted successfully for {dataset_name}")
            
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
            
            self.logger.info(f"Random Forest trained successfully. CV F1: {cv_scores.mean():.4f}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error training Random Forest for {dataset_name}: {str(e)}")
            raise
    
    def train_models_for_dataset(self, data_dict: Dict, dataset_name: str):
        """
        Train all models for a specific dataset.
        """
        self.logger.info(f"Training models for {dataset_name}...")
        
        try:
            X_train = data_dict['X_train']
            y_train = data_dict['y_train']
            
            models = {}
            
            # Train Logistic Regression
            lr_results = self.train_logistic_regression(X_train, y_train, dataset_name)
            models['logistic_regression'] = lr_results
            
            # Train Random Forest
            rf_results = self.train_random_forest(X_train, y_train, dataset_name)
            models['random_forest'] = rf_results
            
            self.logger.info(f"Successfully trained all models for {dataset_name}")
            return models
            
        except Exception as e:
            self.logger.error(f"Error training models for {dataset_name}: {str(e)}")
            raise
    
    def train_models(self, analysis_ready_data: Dict[str, Dict]):
        """
        Train models for all datasets.
        """
        self.logger.info("Starting model training for all datasets...")
        
        try:
            all_models = {}
            
            for dataset_name, data_dict in analysis_ready_data.items():
                self.logger.info(f"Processing dataset: {dataset_name}")
                dataset_models = self.train_models_for_dataset(data_dict, dataset_name)
                all_models[dataset_name] = dataset_models
            
            self.logger.info(f"Model training completed successfully. Trained models for {len(all_models)} datasets.")
            
            return all_models
            
        except Exception as e:
            self.logger.error(f"Error in model training pipeline: {str(e)}")
            raise

def main():
    """Main function to run model training independently."""
    # Setup basic logging for standalone execution
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    predictor = ChurnPredictor()
    
    # Example usage
    print("Model training module")

if __name__ == "__main__":
    main() 