"""
Model evaluation.

Author: Danial Malik
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report

class ModelEvaluator:
    """
    Evaluates model performance.
    """
    
    def __init__(self):
        """Initialize the ModelEvaluator."""
        self.outputs_dir = Path("data/outputs")
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        
    def evaluate_model(self, model, X_test: pd.DataFrame, y_test: pd.Series, 
                      model_name: str, dataset_name: str):
        """
        Evaluate a single model.
        """
        print(f"Evaluating {model_name} on {dataset_name}...")
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        # ROC AUC (if probabilities available)
        roc_auc = None
        if y_pred_proba is not None:
            roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        
        results = {
            'model_name': model_name,
            'dataset_name': dataset_name,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'confusion_matrix': cm,
            'classification_report': report
        }
        
        print(f"{model_name} results - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
        
        return results
    
    def evaluate_dataset_models(self, dataset_models: Dict, data_dict: Dict, dataset_name: str):
        """
        Evaluate all models for a specific dataset.
        """
        print(f"Evaluating models for {dataset_name}...")
        
        X_test = data_dict['X_test']
        y_test = data_dict['y_test']
        
        evaluation_results = {}
        
        for model_name, model_results in dataset_models.items():
            model = model_results['model']
            eval_results = self.evaluate_model(model, X_test, y_test, model_name, dataset_name)
            evaluation_results[model_name] = eval_results
        
        return evaluation_results
    
    def evaluate_all_models(self, models: Dict[str, Dict], analysis_ready_data: Dict[str, Dict]):
        """
        Evaluate all models for all datasets.
        """
        print("Evaluating all models...")
        
        all_evaluation_results = {}
        
        for dataset_name, dataset_models in models.items():
            data_dict = analysis_ready_data[dataset_name]
            dataset_eval_results = self.evaluate_dataset_models(dataset_models, data_dict, dataset_name)
            all_evaluation_results[dataset_name] = dataset_eval_results
        
        print(f"Model evaluation complete. Evaluated models for {len(all_evaluation_results)} datasets.")
        
        return all_evaluation_results

def main():
    """Main function to run model evaluation independently."""
    evaluator = ModelEvaluator()
    
    # Example usage
    print("Model evaluation module")

if __name__ == "__main__":
    main() 