#!/usr/bin/env python3
"""
Test script for the enhanced churn prediction pipeline.

Author: Danial Malik
"""

import logging
import sys
from pathlib import Path
from datetime import datetime

def setup_test_logging():
    """
    Set up logging for testing.
    """
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"test_pipeline_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Test logging initialized. Log file: {log_file}")
    
    return logger

def test_pipeline_components():
    """
    Test individual pipeline components.
    """
    logger = setup_test_logging()
    
    try:
        logger.info("=" * 50)
        logger.info("Testing Pipeline Components")
        logger.info("=" * 50)
        
        # Test imports
        logger.info("Testing imports...")
        try:
            from etl.extract import DataExtractor
            from etl.transform import DataTransformer
            from etl.load import DataLoader
            from analysis.model import ChurnPredictor
            from analysis.evaluate import ModelEvaluator
            from vis.visualizations import VisualizationGenerator
            logger.info("All imports successful")
        except ImportError as e:
            logger.error(f"Import error: {str(e)}")
            return False
        
        # Test component initialization
        logger.info("Testing component initialization...")
        try:
            extractor = DataExtractor()
            transformer = DataTransformer()
            loader = DataLoader()
            predictor = ChurnPredictor()
            evaluator = ModelEvaluator()
            viz_generator = VisualizationGenerator()
            logger.info("All components initialized successfully")
        except Exception as e:
            logger.error(f"Component initialization error: {str(e)}")
            return False
        
        # Test directory creation
        logger.info("Testing directory creation...")
        test_dirs = [
            "data/extracted",
            "data/processed", 
            "data/outputs",
            "data/outputs/evaluation_results",
            "data/outputs/visualizations",
            "logs"
        ]
        
        for dir_path in test_dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
            logger.info(f"Directory {dir_path} ready")
        
        logger.info("=" * 50)
        logger.info("All pipeline components tested successfully!")
        logger.info("=" * 50)
        
        return True
        
    except Exception as e:
        logger.error(f"Test failed with error: {str(e)}")
        return False

def main():
    """
    Main test function.
    """
    print("Testing Enhanced Churn Prediction Pipeline")
    print("=" * 50)
    
    success = test_pipeline_components()
    
    if success:
        print("\n✅ All tests passed! Pipeline is ready to run.")
        print("\nNext steps:")
        print("1. Add your Excel data files to data/extracted/")
        print("2. Run: python main.py")
        print("3. Check logs/ directory for execution logs")
        print("4. Check data/outputs/ for results")
    else:
        print("\n❌ Some tests failed. Check the logs for details.")
        sys.exit(1)

if __name__ == "__main__":
    main()
