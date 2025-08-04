#!/usr/bin/env python3
"""
Main entry point for the churn prediction pipeline.

Author: Danial Malik
"""

from etl.extract import DataExtractor
from etl.transform import DataTransformer
from etl.load import DataLoader
from analysis.model import ChurnPredictor
from analysis.evaluate import ModelEvaluator
from vis.visualizations import VisualizationGenerator

def main():
    """
    Main function to run the pipeline.
    """
    print("Starting E-Commerce Churn Prediction Pipeline")

    # Initialize pipeline components
    extractor = DataExtractor()
    transformer = DataTransformer()
    loader = DataLoader()
    predictor = ChurnPredictor()
    evaluator = ModelEvaluator()
    viz_generator = VisualizationGenerator()

    # ETL Pipeline
    print("Step 1: Extracting data from sources")
    raw_data_paths = extractor.extract_all_data()

    print("Step 2: Transforming and cleaning data")
    processed_data_paths = transformer.transform_all_data(raw_data_paths)

    print("Step 3: Loading processed data")
    analysis_ready_data = loader.load_all_data(processed_data_paths)

    # Analysis Pipeline
    print("Step 4: Building churn prediction models")
    models = predictor.train_models(analysis_ready_data)

    print("Step 5: Evaluating model performance")
    evaluation_results = evaluator.evaluate_all_models(models, analysis_ready_data)

    # Visualization Pipeline
    print("Step 6: Generating visualizations")
    viz_generator.create_all_visualizations(evaluation_results, analysis_ready_data)

    print("Pipeline completed!")

if __name__ == "__main__":
    main() 