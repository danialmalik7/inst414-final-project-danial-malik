"""
Data and model visualizations.

Author: Danial Malik
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from pathlib import Path
from typing import Dict

class VisualizationGenerator:
    """
    Creates visualizations.
    """
    
    def __init__(self):
        """Initialize the VisualizationGenerator."""
        self.outputs_dir = Path("data/outputs")
        self.vis_dir = Path("data/outputs/visualizations")
        self.vis_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
        
        # Set up plotting style
        plt.style.use('default')
        
        # Color scheme for churn analysis
        self.colors = {
            'churned': '#ff6b6b',
            'retained': '#4ecdc4',
            'neutral': '#95a5a6'
        }
    
    def create_data_exploration_plots(self, analysis_ready_data: Dict[str, Dict]):
        """
        Create data exploration visualizations.
        """
        self.logger.info("Starting data exploration visualization creation...")
        print("Creating data exploration visualizations...")
        
        try:
            for dataset_name, data_dict in analysis_ready_data.items():
                self.logger.info(f"Processing dataset: {dataset_name}")
                data = data_dict['data']
                
                # Create dataset-specific visualizations
                if dataset_name == "uci_online_retail":
                    self.create_uci_exploration_plots(data, dataset_name)
                elif dataset_name == "ecommerce_churn":
                    self.create_ecommerce_exploration_plots(data, dataset_name)
            
            self.logger.info("Data exploration visualizations created successfully")
            
        except Exception as e:
            self.logger.error(f"Error creating data exploration visualizations: {str(e)}")
            print(f"Error creating data exploration visualizations: {str(e)}")
            raise
    
    def create_uci_exploration_plots(self, data: pd.DataFrame, dataset_name: str):
        """
        Create exploration plots for UCI Online Retail data.
        """
        self.logger.info(f"Creating UCI exploration plots for {dataset_name}...")
        
        try:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle(f'UCI Online Retail Data Exploration', fontsize=16)
            
            # Plot 1: RFM Distribution
            rfm_features = ['Recency', 'Frequency', 'Monetary']
            for i, feature in enumerate(rfm_features):
                if feature in data.columns:
                    axes[0, i].hist(data[feature], bins=30, alpha=0.7, color=self.colors['neutral'])
                    axes[0, i].set_title(f'{feature} Distribution')
                    axes[0, i].set_xlabel(feature)
                    axes[0, i].set_ylabel('Count')
            
            # Plot 2: Churn Rate
            if 'Churned' in data.columns:
                churn_counts = data['Churned'].value_counts()
                axes[1, 0].pie(churn_counts.values, labels=['Retained', 'Churned'], 
                              autopct='%1.1f%%', colors=[self.colors['retained'], self.colors['churned']])
                axes[1, 0].set_title('Churn Distribution')
            
            # Plot 3: RFM vs Churn
            if all(feature in data.columns for feature in rfm_features + ['Churned']):
                for i, feature in enumerate(rfm_features):
                    axes[1, i+1].boxplot([data[data['Churned']==0][feature], 
                                         data[data['Churned']==1][feature]], 
                                        labels=['Retained', 'Churned'])
                    axes[1, i+1].set_title(f'{feature} by Churn Status')
                    axes[1, i+1].set_ylabel(feature)
            
            plt.tight_layout()
            
            # Save plot
            plot_path = self.vis_dir / f"{dataset_name}_exploration.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"UCI exploration plots created successfully for {dataset_name}")
            print(f"Created UCI exploration plots for {dataset_name}")
            
        except Exception as e:
            self.logger.error(f"Error creating UCI exploration plots for {dataset_name}: {str(e)}")
            print(f"Error creating UCI exploration plots: {str(e)}")
            raise
    
    def create_ecommerce_exploration_plots(self, data: pd.DataFrame, dataset_name: str):
        """
        Create exploration plots for E-commerce Customer Churn data.
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'E-commerce Customer Churn Data Exploration', fontsize=16)
        
        # Plot 1: Churn Rate
        target_col = 'Churn' if 'Churn' in data.columns else 'churn'
        if target_col in data.columns:
            churn_counts = data[target_col].value_counts()
            axes[0, 0].pie(churn_counts.values, labels=['Retained', 'Churned'], 
                          autopct='%1.1f%%', colors=[self.colors['retained'], self.colors['churned']])
            axes[0, 0].set_title('Churn Distribution')
        
        # Plot 2: Tenure Distribution
        if 'Tenure' in data.columns:
            axes[0, 1].hist(data['Tenure'], bins=30, alpha=0.7, color=self.colors['neutral'])
            axes[0, 1].set_title('Tenure Distribution')
            axes[0, 1].set_xlabel('Tenure (months)')
            axes[0, 1].set_ylabel('Count')
        
        # Plot 3: Satisfaction Score
        if 'SatisfactionScore' in data.columns:
            satisfaction_counts = data['SatisfactionScore'].value_counts().sort_index()
            axes[1, 0].bar(satisfaction_counts.index, satisfaction_counts.values, color=self.colors['neutral'])
            axes[1, 0].set_title('Satisfaction Score Distribution')
            axes[1, 0].set_xlabel('Satisfaction Score')
            axes[1, 0].set_ylabel('Count')
        
        # Plot 4: Tenure vs Churn
        if all(col in data.columns for col in ['Tenure', target_col]):
            axes[1, 1].boxplot([data[data[target_col]==0]['Tenure'], 
                               data[data[target_col]==1]['Tenure']], 
                              labels=['Retained', 'Churned'])
            axes[1, 1].set_title('Tenure by Churn Status')
            axes[1, 1].set_ylabel('Tenure (months)')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.vis_dir / f"{dataset_name}_exploration.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Created E-commerce exploration plots for {dataset_name}")
    
    def create_model_performance_plots(self, evaluation_results: Dict[str, Dict]):
        """
        Create model performance visualizations.
        """
        print("Creating model performance visualizations...")
        
        for dataset_name, dataset_results in evaluation_results.items():
            self.create_dataset_performance_plots(dataset_results, dataset_name)
    
    def create_dataset_performance_plots(self, dataset_results: Dict, dataset_name: str):
        """
        Create performance plots for a specific dataset.
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'Model Performance - {dataset_name}', fontsize=16)
        
        # Extract metrics for each model
        model_names = []
        accuracies = []
        f1_scores = []
        precisions = []
        recalls = []
        
        for model_name, results in dataset_results.items():
            model_names.append(model_name.replace('_', ' ').title())
            accuracies.append(results['accuracy'])
            f1_scores.append(results['f1_score'])
            precisions.append(results['precision'])
            recalls.append(results['recall'])
        
        # Plot 1: Accuracy comparison
        axes[0, 0].bar(model_names, accuracies, color=[self.colors['retained'], self.colors['churned']])
        axes[0, 0].set_title('Model Accuracy')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Plot 2: F1 Score comparison
        axes[0, 1].bar(model_names, f1_scores, color=[self.colors['retained'], self.colors['churned']])
        axes[0, 1].set_title('Model F1 Score')
        axes[0, 1].set_ylabel('F1 Score')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Plot 3: Precision vs Recall
        axes[1, 0].scatter(precisions, recalls, s=100, c=[self.colors['retained'], self.colors['churned']])
        for i, model in enumerate(model_names):
            axes[1, 0].annotate(model, (precisions[i], recalls[i]), xytext=(5, 5), textcoords='offset points')
        axes[1, 0].set_xlabel('Precision')
        axes[1, 0].set_ylabel('Recall')
        axes[1, 0].set_title('Precision vs Recall')
        
        # Plot 4: Confusion Matrix (for best model)
        best_model = max(dataset_results.keys(), key=lambda x: dataset_results[x]['f1_score'])
        cm = dataset_results[best_model]['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 1])
        axes[1, 1].set_title(f'Confusion Matrix - {best_model.replace("_", " ").title()}')
        axes[1, 1].set_xlabel('Predicted')
        axes[1, 1].set_ylabel('Actual')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.vis_dir / f"{dataset_name}_performance.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Created performance plots for {dataset_name}")
    
    def create_feature_importance_plots(self, analysis_ready_data: Dict[str, Dict]):
        """
        Create feature importance visualizations.
        """
        print("Creating feature importance visualizations...")
        
        for dataset_name, data_dict in analysis_ready_data.items():
            if dataset_name == "uci_online_retail":
                self.create_uci_feature_importance(data_dict['data'], dataset_name)
            elif dataset_name == "ecommerce_churn":
                self.create_ecommerce_feature_importance(data_dict['data'], dataset_name)
    
    def create_uci_feature_importance(self, data: pd.DataFrame, dataset_name: str):
        """
        Create feature importance plot for UCI data.
        """
        # Simple feature importance based on correlation with target
        if 'Churned' in data.columns:
            numeric_features = data.select_dtypes(include=[np.number]).columns
            numeric_features = [col for col in numeric_features if col != 'Churned']
            
            correlations = []
            for feature in numeric_features:
                corr = abs(data[feature].corr(data['Churned']))
                correlations.append(corr)
            
            # Create plot
            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.bar(numeric_features, correlations, color=self.colors['neutral'])
            ax.set_title('Feature Importance (Correlation with Churn)')
            ax.set_xlabel('Features')
            ax.set_ylabel('Absolute Correlation')
            ax.tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            
            # Save plot
            plot_path = self.vis_dir / f"{dataset_name}_feature_importance.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Created feature importance plot for {dataset_name}")
    
    def create_ecommerce_feature_importance(self, data: pd.DataFrame, dataset_name: str):
        """
        Create feature importance plot for E-commerce data.
        """
        # Simple feature importance based on correlation with target
        target_col = 'Churn' if 'Churn' in data.columns else 'churn'
        if target_col in data.columns:
            numeric_features = data.select_dtypes(include=[np.number]).columns
            numeric_features = [col for col in numeric_features if col != target_col]
            
            correlations = []
            for feature in numeric_features:
                corr = abs(data[feature].corr(data[target_col]))
                correlations.append(corr)
            
            # Create plot
            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.bar(numeric_features, correlations, color=self.colors['neutral'])
            ax.set_title('Feature Importance (Correlation with Churn)')
            ax.set_xlabel('Features')
            ax.set_ylabel('Absolute Correlation')
            ax.tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            
            # Save plot
            plot_path = self.vis_dir / f"{dataset_name}_feature_importance.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Created feature importance plot for {dataset_name}")
    
    def create_all_visualizations(self, evaluation_results: Dict[str, Dict], 
                                 analysis_ready_data: Dict[str, Dict]):
        """
        Create all visualizations.
        """
        print("Creating all visualizations...")
        
        # Create data exploration plots
        self.create_data_exploration_plots(analysis_ready_data)
        
        # Create model performance plots
        self.create_model_performance_plots(evaluation_results)
        
        # Create feature importance plots
        self.create_feature_importance_plots(analysis_ready_data)
        
        print("All visualizations created successfully!")

def main():
    """Main function to run visualizations independently."""
    viz_generator = VisualizationGenerator()
    
    # Example usage
    print("Visualization module")

if __name__ == "__main__":
    main() 