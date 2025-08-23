#!/usr/bin/env python3
"""
Main entry point for the churn prediction pipeline.

Author: Danial Malik
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from etl.extract import DataExtractor
from etl.transform import DataTransformer
from etl.load import DataLoader
from analysis.model import ChurnPredictor
from analysis.evaluate import ModelEvaluator
from vis.visualizations import VisualizationGenerator

def setup_logging():
    """
    Set up logging configuration with file and console handlers.
    """
    # Create logs directory
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Create log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"pipeline_{timestamp}.log"
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized. Log file: {log_file}")
    
    return logger

def main():
    """
    Main function to run the pipeline.
    """
    # Setup logging
    logger = setup_logging()
    
    try:
        logger.info("Starting E-Commerce Churn Prediction Pipeline")

        # Initialize pipeline components
        logger.info("Initializing pipeline components...")
        extractor = DataExtractor()
        transformer = DataTransformer()
        loader = DataLoader()
        predictor = ChurnPredictor()
        evaluator = ModelEvaluator()
        viz_generator = VisualizationGenerator()
        logger.info("Pipeline components initialized successfully")

        # ETL Pipeline
        logger.info("Step 1: Extracting data from sources")
        try:
            raw_data_paths = extractor.extract_all_data()
            logger.info(f"Data extraction completed. Found {len(raw_data_paths)} datasets")
        except Exception as e:
            logger.error(f"Data extraction failed: {str(e)}")
            raise

        logger.info("Step 2: Transforming and cleaning data")
        try:
            processed_data_paths = transformer.transform_all_data(raw_data_paths)
            logger.info(f"Data transformation completed. Processed {len(processed_data_paths)} datasets")
        except Exception as e:
            logger.error(f"Data transformation failed: {str(e)}")
            raise

        logger.info("Step 3: Loading processed data")
        try:
            analysis_ready_data = loader.load_all_data(processed_data_paths)
            logger.info(f"Data loading completed. Loaded {len(analysis_ready_data)} datasets")
        except Exception as e:
            logger.error(f"Data loading failed: {str(e)}")
            raise

        # Analysis Pipeline
        logger.info("Step 4: Building churn prediction models")
        try:
            models = predictor.train_models(analysis_ready_data)
            logger.info(f"Model training completed. Trained models for {len(models)} datasets")
        except Exception as e:
            logger.error(f"Model training failed: {str(e)}")
            raise

        logger.info("Step 5: Evaluating model performance")
        try:
            evaluation_results = evaluator.evaluate_all_models(models, analysis_ready_data)
            logger.info(f"Model evaluation completed. Evaluated models for {len(evaluation_results)} datasets")
        except Exception as e:
            logger.error(f"Model evaluation failed: {str(e)}")
            raise

        # Visualization Pipeline
        logger.info("Step 6: Generating visualizations")
        try:
            viz_generator.create_all_visualizations(evaluation_results, analysis_ready_data)
            logger.info("Visualization generation completed successfully")
        except Exception as e:
            logger.error(f"Visualization generation failed: {str(e)}")
            raise

        logger.info("Pipeline completed successfully!")

    except Exception as e:
        logger.error(f"Pipeline failed with error: {str(e)}")
        raise

if __name__ == "__main__":
    main() 