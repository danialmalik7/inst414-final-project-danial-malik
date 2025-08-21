"""
Data loading and preparation.

Author: Danial Malik
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

class DataLoader:
    """
    Loads and prepares data for analysis.
    """
    
    def __init__(self):
        """Initialize the DataLoader."""
        self.processed_dir = Path("data/processed")
        self.outputs_dir = Path("data/outputs")
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
        
        # Model preparation settings
        self.test_size = 0.2
        self.random_state = 42
        
    def load_processed_data(self) -> Dict[str, pd.DataFrame]:
        """
        Load all processed datasets.
        """
        self.logger.info("Starting to load processed datasets...")
        
        try:
            processed_data = {}
            
            # Load UCI Online Retail processed data
            uci_path = self.processed_dir / "uci_online_retail_processed.csv"
            if uci_path.exists():
                uci_data = pd.read_csv(uci_path)
                processed_data["uci_online_retail"] = uci_data
                self.logger.info(f"Successfully loaded UCI data: {len(uci_data)} records")
            else:
                self.logger.warning("UCI processed data file not found")
            
            # Load E-commerce Churn processed data
            churn_path = self.processed_dir / "ecommerce_churn_processed.csv"
            if churn_path.exists():
                churn_data = pd.read_csv(churn_path)
                processed_data["ecommerce_churn"] = churn_data
                self.logger.info(f"Successfully loaded E-commerce Churn data: {len(churn_data)} records")
            else:
                self.logger.warning("E-commerce Churn processed data file not found")
            
            self.logger.info(f"Data loading completed. Loaded {len(processed_data)} datasets")
            return processed_data
            
        except Exception as e:
            self.logger.error(f"Error loading processed data: {str(e)}")
            raise
    
    def prepare_uci_data_for_modeling(self, uci_data: pd.DataFrame):
        """
        Prepare UCI Online Retail data for modeling.
        """
        self.logger.info("Starting UCI data preparation for modeling...")
        
        try:
            # Select features for modeling (exclude DaysSinceFirstPurchase as it's all NaN)
            feature_columns = ['Recency', 'Frequency', 'Monetary', 'AvgOrderValue']
            
            # Ensure all features exist and have valid data
            available_features = []
            for col in feature_columns:
                if col in uci_data.columns and uci_data[col].notna().any():
                    available_features.append(col)
            
            self.logger.info(f"Using features: {available_features}")
            
            X = uci_data[available_features].copy()
            y = uci_data['Churned']
            
            # Handle missing values and infinite values
            X = X.replace([np.inf, -np.inf], np.nan)
            X = X.fillna(X.median())
            
            # Ensure no remaining NaN values
            if X.isnull().any().any():
                self.logger.warning("Still have NaN values after imputation, dropping rows")
                X = X.dropna()
                y = y[X.index]
            
            # Scale features
            scaler = StandardScaler()
            X_scaled = pd.DataFrame(
                scaler.fit_transform(X),
                columns=X.columns,
                index=X.index
            )
            
            self.logger.info(f"UCI data preparation completed successfully. Shape: {X_scaled.shape}")
            
            return X_scaled, y
            
        except Exception as e:
            self.logger.error(f"Error preparing UCI data for modeling: {str(e)}")
            raise
    
    def prepare_ecommerce_churn_data_for_modeling(self, churn_data: pd.DataFrame):
        """
        Prepare E-commerce Customer Churn data for modeling.
        """
        try:
            # Select features for modeling
            numeric_features = ['Tenure', 'HourSpendOnApp', 'NumberOfDeviceRegistered', 
                              'SatisfactionScore', 'NumberOfAddress', 'Complain']
            categorical_features = ['PreferredLoginDevice', 'PreferredPaymentMode', 'Gender', 
                                  'PreferedOrderCat', 'MaritalStatus']
            
            # Ensure features exist and have valid data
            available_numeric = []
            for col in numeric_features:
                if col in churn_data.columns and churn_data[col].notna().any():
                    available_numeric.append(col)
            
            available_categorical = []
            for col in categorical_features:
                if col in churn_data.columns and churn_data[col].notna().any():
                    available_categorical.append(col)
            
            # Prepare numeric features
            X_numeric = churn_data[available_numeric].copy()
            
            # Prepare categorical features
            X_categorical = churn_data[available_categorical].copy()
            
            # Encode categorical variables
            encoders = {}
            for col in X_categorical.columns:
                encoder = LabelEncoder()
                X_categorical[col] = encoder.fit_transform(X_categorical[col].astype(str))
                encoders[col] = encoder
            
            # Combine numeric and categorical features
            X = pd.concat([X_numeric, X_categorical], axis=1)
            y = churn_data['Churn'] if 'Churn' in churn_data.columns else churn_data['churn']
            
            # Handle missing values and infinite values
            X = X.replace([np.inf, -np.inf], np.nan)
            X = X.fillna(X.median())
            
            # Ensure no remaining NaN values
            if X.isnull().any().any():
                self.logger.warning("Still have NaN values after imputation, dropping rows")
                X = X.dropna()
                y = y[X.index]
            
            # Scale features
            scaler = StandardScaler()
            X_scaled = pd.DataFrame(
                scaler.fit_transform(X),
                columns=X.columns,
                index=X.index
            )
            
            return X_scaled, y
            
        except Exception as e:
            self.logger.error(f"Error preparing E-commerce Churn data for modeling: {str(e)}")
            raise
    
    def split_data(self, X: pd.DataFrame, y: pd.Series, dataset_name: str):
        """
        Split data into training and testing sets.
        """
        self.logger.info(f"Splitting {dataset_name} data...")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )
        
        self.logger.info(f"Split complete: {len(X_train)} train, {len(X_test)} test")
        
        return X_train, X_test, y_train, y_test
    
    def load_all_data(self, processed_data_paths: Dict[str, pd.DataFrame]):
        """
        Load and prepare all data for analysis.
        """
        self.logger.info("Loading and preparing all data...")
        
        if not processed_data_paths:
            self.logger.error("No processed data provided for loading")
            raise ValueError("No processed data available - pipeline cannot continue")
        
        analysis_ready_data = {}
        
        # Process UCI Online Retail data
        if "uci_online_retail" in processed_data_paths:
            uci_data = processed_data_paths["uci_online_retail"]['data']
            X_uci, y_uci = self.prepare_uci_data_for_modeling(uci_data)
            
            # Ensure X and y have the same index after cleaning
            common_index = X_uci.index.intersection(y_uci.index)
            X_uci = X_uci.loc[common_index]
            y_uci = y_uci.loc[common_index]
            
            X_train_uci, X_test_uci, y_train_uci, y_test_uci = self.split_data(X_uci, y_uci, "UCI")
            
            analysis_ready_data["uci_online_retail"] = {
                'X_train': X_train_uci,
                'X_test': X_test_uci,
                'y_train': y_train_uci,
                'y_test': y_test_uci,
                'data': uci_data
            }
        else:
            self.logger.error("UCI Online Retail data not found in processed data")
            raise ValueError("UCI Online Retail data not available for loading")
        
        # Process E-commerce Customer Churn data
        if "ecommerce_churn" in processed_data_paths:
            churn_data = processed_data_paths["ecommerce_churn"]['data']
            X_churn, y_churn = self.prepare_ecommerce_churn_data_for_modeling(churn_data)
            
            # Ensure X and y have the same index after cleaning
            common_index = X_churn.index.intersection(y_churn.index)
            X_churn = X_churn.loc[common_index]
            y_churn = y_churn.loc[common_index]
            
            X_train_churn, X_test_churn, y_train_churn, y_test_churn = self.split_data(X_churn, y_churn, "E-commerce Churn")
            
            analysis_ready_data["ecommerce_churn"] = {
                'X_train': X_train_churn,
                'X_test': X_test_churn,
                'y_train': y_train_churn,
                'y_test': y_test_churn,
                'data': churn_data
            }
        else:
            self.logger.error("E-commerce Customer Churn data not found in processed data")
            raise ValueError("E-commerce Customer Churn data not available for loading")
        
        # Validate loading results
        if len(analysis_ready_data) == 0:
            self.logger.error("No datasets were successfully loaded")
            raise ValueError("No analysis-ready data available - pipeline cannot continue")
        
        self.logger.info(f"Data loading complete. Prepared {len(analysis_ready_data)} datasets.")
        
        return analysis_ready_data

def main():
    """Main function to run data loading independently."""
    loader = DataLoader()
    
    # Example usage
    print("Data loading module")

if __name__ == "__main__":
    main() 