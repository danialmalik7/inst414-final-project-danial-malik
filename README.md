# E-Commerce Customer Churn Prediction

**Author: Danial Malik**

This project predicts customer churn for e-commerce companies using machine learning. The goal is to identify customers at risk of leaving so the business can take action to retain them.

## Datasets
- UCI Online Retail Dataset: Transaction data from a UK online retailer
- E-commerce Customer Churn Dataset: Customer data with churn labels

## Setup
1. Clone the repository
2. Create virtual environment: `python -m venv venv`
3. Activate: `source venv/bin/activate` (or `venv\Scripts\activate` on Windows)
4. Install dependencies: `pip install -r requirements.txt`
5. Add Excel files to `data/extracted/` directory

## Running
```bash
python main.py
```

## Features

### Enhanced Pipeline
- **Comprehensive Logging**: Full logging system with timestamped log files
- **Error Handling**: Robust error handling throughout the ETL pipeline
- **Model Evaluation**: Detailed model performance metrics and evaluation outputs
- **Data Validation**: Enhanced data quality checks and validation

### Model Evaluation
- Accuracy, Precision, Recall, F1-Score, and ROC-AUC metrics
- Confusion matrix analysis
- Detailed classification reports
- Results stored in organized CSV files

### Logging and Monitoring
- Timestamped log files in `logs/` directory
- Console and file logging
- Detailed progress tracking for each pipeline stage

## Project Structure
```
├── data/
│   ├── extracted/          # Raw data files
│   ├── processed/          # Cleaned data
│   ├── outputs/            # Model outputs and evaluation results
│   │   ├── evaluation_results/  # Detailed model evaluation outputs
│   │   └── visualizations/      # Generated charts and plots
│   └── reference-tables/   # Data dictionaries
├── etl/
│   ├── extract.py          # Data extraction with logging
│   ├── transform.py        # Data cleaning with error handling
│   └── load.py             # Data loading with validation
├── analysis/
│   ├── model.py            # Model training with logging
│   └── evaluate.py         # Model evaluation with detailed outputs
├── vis/
│   └── visualizations.py   # Visualizations with error handling
├── logs/                   # Pipeline execution logs
├── main.py                 # Main pipeline with comprehensive logging
└── requirements.txt        # Dependencies
```

## Output Files

### Model Evaluation Results
- `data/outputs/evaluation_results/`: Contains detailed evaluation metrics
- `data/outputs/evaluation_results/evaluation_summary.csv`: Summary of all model performances
- Individual model results stored in dataset-specific subdirectories

### Logs
- `logs/pipeline_YYYYMMDD_HHMMSS.log`: Detailed execution logs for each pipeline run

## Error Handling
The pipeline includes comprehensive error handling for:
- Data loading and validation
- Model training and evaluation
- Visualization generation
- File I/O operations

All errors are logged with detailed information for debugging and monitoring.
