"""
Data cleaning and transformation.

Author: Danial Malik
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class DataTransformer:
    """
    Cleans and transforms data.
    """
    
    def __init__(self):
        """Initialize the DataTransformer."""
        self.extracted_dir = Path("data/extracted")
        self.processed_dir = Path("data/processed")
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
        
        # Churn definition parameters
        self.churn_threshold_days = 90  # Customers inactive for 90+ days are considered churned
        
    def clean_uci_online_retail(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean UCI Online Retail data.
        """
        self.logger.info("Starting UCI Online Retail dataset cleaning...")
        print("Cleaning UCI Online Retail dataset...")
        
        try:
            # Make a copy to avoid modifying original
            df_clean = df.copy()
            
            # Remove cancelled invoices (InvoiceNo starting with 'C')
            initial_count = len(df_clean)
            df_clean = df_clean[~df_clean['InvoiceNo'].astype(str).str.startswith('C')]
            self.logger.info(f"Removed {initial_count - len(df_clean)} cancelled invoices")
            print(f"Removed {initial_count - len(df_clean)} cancelled invoices")
            
            # Remove rows with missing CustomerID
            initial_count = len(df_clean)
            df_clean = df_clean.dropna(subset=['CustomerID'])
            self.logger.info(f"Removed {initial_count - len(df_clean)} rows with missing CustomerID")
            print(f"Removed {initial_count - len(df_clean)} rows with missing CustomerID")
            
            # Convert CustomerID to integer
            df_clean['CustomerID'] = df_clean['CustomerID'].astype(int)
            
            # Convert InvoiceDate to datetime
            df_clean['InvoiceDate'] = pd.to_datetime(df_clean['InvoiceDate'])
            
            # Remove rows with negative or zero quantities/prices
            initial_count = len(df_clean)
            df_clean = df_clean[
                (df_clean['Quantity'] > 0) & 
                (df_clean['UnitPrice'] > 0)
            ]
            self.logger.info(f"Removed {initial_count - len(df_clean)} rows with invalid quantities/prices")
            print(f"Removed {initial_count - len(df_clean)} rows with invalid quantities/prices")
            
            # Calculate total amount for each transaction
            df_clean['TotalAmount'] = df_clean['Quantity'] * df_clean['UnitPrice']
            
            self.logger.info(f"UCI Online Retail dataset cleaned successfully. Final records: {len(df_clean)}")
            print(f"Cleaned dataset: {len(df_clean)} records")
            return df_clean
            
        except Exception as e:
            self.logger.error(f"Error cleaning UCI Online Retail dataset: {str(e)}")
            print(f"Error cleaning UCI Online Retail dataset: {str(e)}")
            raise
    
    def calculate_rfm_features(self, df: pd.DataFrame, reference_date: Optional[datetime] = None) -> pd.DataFrame:
        """
        Calculate RFM features.
        """
        self.logger.info("Starting RFM features calculation...")
        print("Calculating RFM features...")
        
        try:
            if reference_date is None:
                reference_date = df['InvoiceDate'].max()
            
            # Group by customer and calculate RFM metrics
            rfm = df.groupby('CustomerID').agg({
                'InvoiceDate': lambda x: (reference_date - x.max()).days,  # Recency
                'InvoiceNo': 'nunique',  # Frequency (unique invoices)
                'TotalAmount': 'sum'  # Monetary
            }).reset_index()
            
            # Rename columns
            rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']
            
            # Calculate additional features
            rfm['AvgOrderValue'] = rfm['Monetary'] / rfm['Frequency']
            rfm['DaysSinceFirstPurchase'] = (reference_date - df.groupby('CustomerID')['InvoiceDate'].min()).dt.days
            
            self.logger.info(f"RFM features calculated successfully for {len(rfm)} customers")
            print(f"RFM features calculated for {len(rfm)} customers")
            return rfm
            
        except Exception as e:
            self.logger.error(f"Error calculating RFM features: {str(e)}")
            print(f"Error calculating RFM features: {str(e)}")
            raise
    
    def define_churn_labels(self, rfm_df: pd.DataFrame) -> pd.DataFrame:
        """
        Define churn labels.
        """
        print("Defining churn labels...")
        
        # Define churn based on recency
        rfm_df['Churned'] = (rfm_df['Recency'] > self.churn_threshold_days).astype(int)
        
        print(f"Churn rate: {rfm_df['Churned'].mean():.2%}")
        return rfm_df
    
    def clean_ecommerce_churn(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean E-commerce Customer Churn data.
        """
        print("Cleaning E-commerce Customer Churn dataset...")
        
        # Make a copy to avoid modifying original
        df_clean = df.copy()
        
        # Handle missing values
        print(f"Missing values: {df_clean.isnull().sum().sum()}")
        
        # Fill missing values with appropriate defaults
        numeric_columns = df_clean.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if df_clean[col].isnull().sum() > 0:
                df_clean[col].fillna(df_clean[col].median(), inplace=True)
        
        # Encode categorical variables
        categorical_columns = df_clean.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            if col != 'CustomerID':  # Skip CustomerID if it's categorical
                df_clean[col] = df_clean[col].astype('category').cat.codes
        
        print(f"Cleaned dataset: {len(df_clean)} records")
        return df_clean
    
    def perform_eda(self, df: pd.DataFrame, dataset_name: str) -> Dict:
        """
        Perform basic EDA.
        """
        print(f"Performing EDA for {dataset_name}...")
        
        eda_results = {
            'dataset_name': dataset_name,
            'shape': df.shape,
            'columns': list(df.columns),
            'missing_values': df.isnull().sum().to_dict(),
            'data_types': df.dtypes.to_dict()
        }
        
        # Basic statistics for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            eda_results['numeric_stats'] = df[numeric_cols].describe().to_dict()
        
        print(f"EDA completed for {dataset_name}")
        return eda_results
    
    def transform_uci_data(self, raw_data_path: str) -> Tuple[pd.DataFrame, Dict]:
        """
        Transform UCI Online Retail data.
        """
        print("Transforming UCI Online Retail data...")
        
        # Load raw data
        df = pd.read_csv(raw_data_path)
        
        # Clean data
        df_clean = self.clean_uci_online_retail(df)
        
        # Calculate RFM features
        rfm_df = self.calculate_rfm_features(df_clean)
        
        # Define churn labels
        final_df = self.define_churn_labels(rfm_df)
        
        # Perform EDA
        eda_results = self.perform_eda(final_df, "uci_online_retail")
        
        # Save processed data
        output_path = self.processed_dir / "uci_online_retail_processed.csv"
        final_df.to_csv(output_path, index=False)
        
        print(f"UCI data transformation complete. Saved to {output_path}")
        
        return final_df, eda_results
    
    def transform_ecommerce_churn_data(self, raw_data_path: str) -> Tuple[pd.DataFrame, Dict]:
        """
        Transform E-commerce Customer Churn data.
        """
        print("Transforming E-commerce Customer Churn data...")
        
        # Load raw data
        df = pd.read_csv(raw_data_path)
        
        # Clean data
        df_clean = self.clean_ecommerce_churn(df)
        
        # Perform EDA
        eda_results = self.perform_eda(df_clean, "ecommerce_churn")
        
        # Save processed data
        output_path = self.processed_dir / "ecommerce_churn_processed.csv"
        df_clean.to_csv(output_path, index=False)
        
        print(f"E-commerce Churn data transformation complete. Saved to {output_path}")
        
        return df_clean, eda_results
    
    def transform_all_data(self, raw_data_paths: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Transform all datasets.
        """
        print("Starting data transformation...")
        
        transformed_data = {}
        
        # Transform UCI Online Retail data
        if "uci_online_retail" in raw_data_paths:
            uci_df, uci_eda = self.transform_uci_data("data/extracted/uci_online_retail_raw.csv")
            transformed_data["uci_online_retail"] = {
                'data': uci_df,
                'eda': uci_eda
            }
        
        # Transform E-commerce Customer Churn data
        if "ecommerce_churn" in raw_data_paths:
            churn_df, churn_eda = self.transform_ecommerce_churn_data("data/extracted/ecommerce_churn_raw.csv")
            transformed_data["ecommerce_churn"] = {
                'data': churn_df,
                'eda': churn_eda
            }
        
        print(f"Transformation complete. Transformed {len(transformed_data)} datasets.")
        
        return transformed_data

def main():
    """Main function to run data transformation independently."""
    transformer = DataTransformer()
    
    # Example usage
    print("Data transformation module")

if __name__ == "__main__":
    main() 