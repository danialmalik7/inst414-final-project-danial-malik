"""
Model evaluation.

Author: Danial Malik
"""

import pandas as pd
import numpy as np
import logging
import json
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
        self.evaluation_dir = self.outputs_dir / "evaluation_results"
        self.evaluation_dir.mkdir(exist_ok=True)
        self.logger = logging.getLogger(__name__)
        
    def save_evaluation_results(self, results: Dict, model_name: str, dataset_name: str):
        """
        Save evaluation results to files.
        """
        try:
            # Create dataset-specific directory
            dataset_dir = self.evaluation_dir / dataset_name
            dataset_dir.mkdir(exist_ok=True)
            
            # Save metrics to CSV
            metrics_df = pd.DataFrame({
                'Metric': ['Accuracy', 'Precision', 'Recall', 'F1_Score', 'ROC_AUC'],
                'Value': [
                    results['accuracy'],
                    results['precision'],
                    results['recall'],
                    results['f1_score'],
                    results['roc_auc'] if results['roc_auc'] is not None else 'N/A'
                ]
            })
            
            metrics_path = dataset_dir / f"{model_name}_metrics.csv"
            metrics_df.to_csv(metrics_path, index=False)
            self.logger.info(f"Saved metrics to {metrics_path}")
            
            # Save confusion matrix
            cm_df = pd.DataFrame(
                results['confusion_matrix'],
                columns=['Predicted_0', 'Predicted_1'],
                index=['Actual_0', 'Actual_1']
            )
            cm_path = dataset_dir / f"{model_name}_confusion_matrix.csv"
            cm_df.to_csv(cm_path)
            self.logger.info(f"Saved confusion matrix to {cm_path}")
            
            # Save classification report
            report_df = pd.DataFrame(results['classification_report']).transpose()
            report_path = dataset_dir / f"{model_name}_classification_report.csv"
            report_df.to_csv(report_path)
            self.logger.info(f"Saved classification report to {report_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving evaluation results for {model_name}: {str(e)}")
            print(f"Error saving evaluation results: {str(e)}")
    
    def evaluate_model(self, model, X_test: pd.DataFrame, y_test: pd.Series, 
                      model_name: str, dataset_name: str):
        """
        Evaluate a single model.
        """
        self.logger.info(f"Evaluating {model_name} on {dataset_name}...")
        print(f"Evaluating {model_name} on {dataset_name}...")
        
        try:
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
            
            # Save results to files
            self.save_evaluation_results(results, model_name, dataset_name)
            
            self.logger.info(f"{model_name} evaluation completed successfully. Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
            print(f"{model_name} results - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error evaluating {model_name} on {dataset_name}: {str(e)}")
            print(f"Error evaluating {model_name}: {str(e)}")
            raise
    
    def evaluate_dataset_models(self, dataset_models: Dict, data_dict: Dict, dataset_name: str):
        """
        Evaluate all models for a specific dataset.
        """
        self.logger.info(f"Evaluating models for {dataset_name}...")
        print(f"Evaluating models for {dataset_name}...")
        
        try:
            X_test = data_dict['X_test']
            y_test = data_dict['y_test']
            
            evaluation_results = {}
            
            for model_name, model_results in dataset_models.items():
                model = model_results['model']
                eval_results = self.evaluate_model(model, X_test, y_test, model_name, dataset_name)
                evaluation_results[model_name] = eval_results
            
            self.logger.info(f"Successfully evaluated all models for {dataset_name}")
            return evaluation_results
            
        except Exception as e:
            self.logger.error(f"Error evaluating models for {dataset_name}: {str(e)}")
            print(f"Error evaluating models for {dataset_name}: {str(e)}")
            raise
    
    def evaluate_all_models(self, models: Dict[str, Dict], analysis_ready_data: Dict[str, Dict]):
        """
        Evaluate all models for all datasets.
        """
        self.logger.info("Starting evaluation of all models...")
        print("Evaluating all models...")
        
        try:
            all_evaluation_results = {}
            
            for dataset_name, dataset_models in models.items():
                self.logger.info(f"Processing dataset: {dataset_name}")
                data_dict = analysis_ready_data[dataset_name]
                dataset_eval_results = self.evaluate_dataset_models(dataset_models, data_dict, dataset_name)
                all_evaluation_results[dataset_name] = dataset_eval_results
            
            # Create summary report
            self.create_evaluation_summary(all_evaluation_results)
            
            self.logger.info(f"Model evaluation completed successfully. Evaluated models for {len(all_evaluation_results)} datasets.")
            print(f"Model evaluation complete. Evaluated models for {len(all_evaluation_results)} datasets.")
            
            return all_evaluation_results
            
        except Exception as e:
            self.logger.error(f"Error in model evaluation pipeline: {str(e)}")
            print(f"Error in model evaluation pipeline: {str(e)}")
            raise
    
    def create_evaluation_summary(self, all_evaluation_results: Dict):
        """
        Create a summary of all evaluation results.
        """
        try:
            summary_data = []
            
            for dataset_name, dataset_results in all_evaluation_results.items():
                for model_name, model_results in dataset_results.items():
                    summary_data.append({
                        'Dataset': dataset_name,
                        'Model': model_name,
                        'Accuracy': model_results['accuracy'],
                        'Precision': model_results['precision'],
                        'Recall': model_results['recall'],
                        'F1_Score': model_results['f1_score'],
                        'ROC_AUC': model_results['roc_auc'] if model_results['roc_auc'] is not None else 'N/A'
                    })
            
            summary_df = pd.DataFrame(summary_data)
            summary_path = self.evaluation_dir / "evaluation_summary.csv"
            summary_df.to_csv(summary_path, index=False)
            
            self.logger.info(f"Created evaluation summary at {summary_path}")
            
        except Exception as e:
            self.logger.error(f"Error creating evaluation summary: {str(e)}")

def main():
    """Main function to run model evaluation independently."""
    # Setup basic logging for standalone execution
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    evaluator = ModelEvaluator()
    
    # Example usage
    print("Model evaluation module")

if __name__ == "__main__":
    main() 