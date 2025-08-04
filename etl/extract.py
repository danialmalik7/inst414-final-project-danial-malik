"""
Data extraction from Excel files.

Author: Danial Malik
"""

import pandas as pd
from pathlib import Path
from typing import Dict, Optional

class DataExtractor:
    """
    Extracts data from Excel files.
    """
    
    def __init__(self):
        """Initialize the DataExtractor."""
        self.data_dir = Path("data/extracted")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
    def extract_uci_online_retail(self) -> Optional[pd.DataFrame]:
        """
        Extract UCI Online Retail data.
        """
        try:
            filepath = self.data_dir / "uci_online_retail.xlsx"
            
            if not filepath.exists():
                print("UCI Online Retail Excel file not found")
                return None
            
            # Load the Excel file
            df = pd.read_excel(filepath)
            print(f"Loaded UCI data: {len(df)} records")
            
            return df
            
        except Exception as e:
            print(f"Error extracting UCI data: {str(e)}")
            return None
    
    def extract_ecommerce_churn(self) -> Optional[pd.DataFrame]:
        """
        Extract E-commerce Churn data.
        """
        try:
            filepath = self.data_dir / "ecommerce_churn.xlsx"
            
            if not filepath.exists():
                print("E-commerce Churn Excel file not found")
                return None
            
            # Load the Excel file
            df = pd.read_excel(filepath)
            print(f"Loaded E-commerce Churn data: {len(df)} records")
            
            return df
            
        except Exception as e:
            print(f"Error extracting E-commerce Churn data: {str(e)}")
            return None
    
    def extract_all_data(self) -> Dict[str, pd.DataFrame]:
        """
        Extract all datasets.
        """
        print("Starting data extraction...")
        
        extracted_data = {}
        
        # Extract UCI Online Retail dataset
        uci_data = self.extract_uci_online_retail()
        if uci_data is not None:
            extracted_data["uci_online_retail"] = uci_data
            # Save raw data
            uci_data.to_csv(self.data_dir / "uci_online_retail_raw.csv", index=False)
        
        # Extract E-commerce Churn dataset
        churn_data = self.extract_ecommerce_churn()
        if churn_data is not None:
            extracted_data["ecommerce_churn"] = churn_data
            # Save raw data
            churn_data.to_csv(self.data_dir / "ecommerce_churn_raw.csv", index=False)
        
        print(f"Extraction complete. Extracted {len(extracted_data)} datasets.")
        
        return extracted_data

def main():
    """Main function to run data extraction independently."""
    extractor = DataExtractor()
    extracted_data = extractor.extract_all_data()
    
    print(f"Extracted {len(extracted_data)} datasets:")
    for name, data in extracted_data.items():
        print(f"  - {name}: {len(data)} records")

if __name__ == "__main__":
    main()
