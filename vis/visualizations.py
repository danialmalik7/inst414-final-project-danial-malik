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
        
        # Professional color scheme for business analysis
        self.colors = {
            'churned': '#d32f2f',      # Dark red - negative outcome
            'retained': '#2e7d32',      # Dark green - positive outcome
            'neutral': '#757575',       # Dark gray - neutral
            'warning': '#f57c00',       # Orange - warning/attention needed
            'success': '#388e3c',       # Green - success/good performance
            'info': '#1976d2'           # Blue - informational
        }
    
    def create_data_exploration_plots(self, analysis_ready_data: Dict[str, Dict]):
        """
        Create data exploration visualizations.
        """
        self.logger.info("Starting data exploration visualization creation...")
        
        for dataset_name, data_dict in analysis_ready_data.items():
            data = data_dict['data']
            
            # Create dataset-specific visualizations
            if dataset_name == "uci_online_retail":
                self.create_uci_exploration_plots(data, dataset_name)
            elif dataset_name == "ecommerce_churn":
                self.create_ecommerce_exploration_plots(data, dataset_name)
        
        self.logger.info("Data exploration visualizations created successfully")
    
    def create_uci_exploration_plots(self, data: pd.DataFrame, dataset_name: str):
        """
        Create exploration plots for UCI Online Retail data.
        """
        self.logger.info(f"Creating UCI exploration plots for {dataset_name}...")
        
        try:
            # Create a 2x3 grid to accommodate 3 RFM features + 3 other plots
            fig, axes = plt.subplots(2, 3, figsize=(18, 10))
            fig.suptitle(f'UCI Online Retail Data Exploration', fontsize=16)
            
            # Plot 1-3: RFM Distribution (top row)
            rfm_features = ['Recency', 'Frequency', 'Monetary']
            rfm_colors = [self.colors['warning'], self.colors['success'], self.colors['info']]  # Orange, Green, Blue
            for i, feature in enumerate(rfm_features):
                if feature in data.columns:
                    axes[0, i].hist(data[feature], bins=30, alpha=0.7, color=rfm_colors[i])
                    axes[0, i].set_title(f'{feature} Distribution')
                    axes[0, i].set_xlabel(feature)
                    axes[0, i].set_ylabel('Count')
            
            # Plot 4: Churn Rate (bottom left)
            if 'Churned' in data.columns:
                churn_counts = data['Churned'].value_counts()
                axes[1, 0].pie(churn_counts.values, labels=['Retained', 'Churned'], 
                              autopct='%1.1f%%', colors=[self.colors['retained'], self.colors['churned']])
                axes[1, 0].set_title('Churn Distribution')
            
            # Plot 5-6: RFM vs Churn (bottom middle and right)
            if all(feature in data.columns for feature in rfm_features + ['Churned']):
                for i, feature in enumerate(rfm_features[:2]):  # Only plot first 2 RFM features
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
            
        except Exception as e:
            self.logger.error(f"Error creating UCI exploration plots for {dataset_name}: {str(e)}")
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
            axes[0, 1].hist(data['Tenure'], bins=30, alpha=0.7, color=self.colors['info'])
            axes[0, 1].set_title('Tenure Distribution')
            axes[0, 1].set_xlabel('Tenure (months)')
            axes[0, 1].set_ylabel('Count')
        
        # Plot 3: Satisfaction Score
        if 'SatisfactionScore' in data.columns:
            satisfaction_counts = data['SatisfactionScore'].value_counts().sort_index()
            
            # Create color gradient: red (1) -> yellow (2-4) -> green (5)
            colors = []
            for score in satisfaction_counts.index:
                if score == 1:
                    colors.append('#d32f2f')  # Darker red for low satisfaction
                elif score == 2:
                    colors.append('#ff9800')  # Orange-yellow for below average
                elif score == 3:
                    colors.append('#ffc107')  # Amber for neutral
                elif score == 4:
                    colors.append('#ffeb3b')  # Bright yellow for above average
                elif score == 5:
                    colors.append('#4caf50')  # Proper green for high satisfaction
                else:
                    colors.append(self.colors['neutral'])
            
            axes[1, 0].bar(satisfaction_counts.index, satisfaction_counts.values, color=colors)
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
    
    def create_model_performance_plots(self, evaluation_results: Dict[str, Dict]):
        """
        Create model performance visualizations.
        """
        self.logger.info("Starting model performance visualization creation...")
        
        for dataset_name, dataset_results in evaluation_results.items():
            self.create_dataset_performance_plots(dataset_results, dataset_name)
        
        self.logger.info("Model performance visualizations created successfully")
    
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
        axes[0, 0].bar(model_names, accuracies, color=[self.colors['success'], self.colors['info']])
        axes[0, 0].set_title('Model Accuracy')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Plot 2: F1 Score comparison
        axes[0, 1].bar(model_names, f1_scores, color=[self.colors['success'], self.colors['info']])
        axes[0, 1].set_title('Model F1 Score')
        axes[0, 1].set_ylabel('F1 Score')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Plot 3: Precision vs Recall
        axes[1, 0].scatter(precisions, recalls, s=100, c=[self.colors['success'], self.colors['info']])
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
        
        self.logger.info(f"Performance plots created successfully for {dataset_name}")
    
    def create_feature_importance_plots(self, analysis_ready_data: Dict[str, Dict]):
        """
        Create feature importance visualizations.
        """
        self.logger.info("Starting feature importance visualization creation...")
        
        for dataset_name, data_dict in analysis_ready_data.items():
            if dataset_name == "uci_online_retail":
                self.create_uci_feature_importance(data_dict['data'], dataset_name)
            elif dataset_name == "ecommerce_churn":
                self.create_ecommerce_feature_importance(data_dict['data'], dataset_name)
        
        self.logger.info("Feature importance visualizations created successfully")
    
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
            
            # Create plot with better spacing
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Color bars based on correlation strength: low (gray) -> medium (blue) -> high (green)
            colors = []
            for corr in correlations:
                if corr < 0.1:
                    colors.append(self.colors['neutral'])      # Gray for low correlation
                elif corr < 0.3:
                    colors.append(self.colors['info'])         # Blue for medium correlation
                else:
                    colors.append(self.colors['success'])      # Green for high correlation
            
            bars = ax.bar(numeric_features, correlations, color=colors)
            ax.set_title('Feature Importance (Correlation with Churn)', fontsize=14, pad=20)
            ax.set_xlabel('Features', fontsize=12)
            ax.set_ylabel('Absolute Correlation', fontsize=12)
            
            # Rotate x-axis labels for better readability
            plt.xticks(rotation=45, ha='right', fontsize=10)
            plt.yticks(fontsize=10)
            
            plt.tight_layout()
            
            # Save plot
            plot_path = self.vis_dir / f"{dataset_name}_feature_importance.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Feature importance plot created for {dataset_name}")
    
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
            
            # Create plot with larger figure size to accommodate rotated labels
            fig, ax = plt.subplots(figsize=(14, 8))
            
            # Color bars based on correlation strength: low (gray) -> medium (blue) -> high (green)
            colors = []
            for corr in correlations:
                if corr < 0.1:
                    colors.append(self.colors['neutral'])      # Gray for low correlation
                elif corr < 0.3:
                    colors.append(self.colors['info'])         # Blue for medium correlation
                else:
                    colors.append(self.colors['success'])      # Green for high correlation
            
            bars = ax.bar(numeric_features, correlations, color=colors)
            ax.set_title('Feature Importance (Correlation with Churn)', fontsize=14, pad=20)
            ax.set_xlabel('Features', fontsize=12)
            ax.set_ylabel('Absolute Correlation', fontsize=12)
            
            # Rotate x-axis labels for better readability
            plt.xticks(rotation=45, ha='right', fontsize=10)
            plt.yticks(fontsize=10)
            
            # Add some padding and adjust layout
            plt.tight_layout()
            plt.subplots_adjust(bottom=0.2)
            
            # Save plot
            plot_path = self.vis_dir / f"{dataset_name}_feature_importance.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Feature importance plot created for {dataset_name}")
    
    def create_all_visualizations(self, evaluation_results: Dict[str, Dict], 
                                 analysis_ready_data: Dict[str, Dict]):
        """
        Create all visualizations.
        """
        self.logger.info("Starting all visualization creation...")
        
        # Create data exploration plots
        self.create_data_exploration_plots(analysis_ready_data)
        
        # Create model performance plots
        self.create_model_performance_plots(evaluation_results)
        
        # Create feature importance plots
        self.create_feature_importance_plots(analysis_ready_data)
        
        self.logger.info("All visualizations created successfully!")

 