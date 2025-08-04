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

## Project Structure
```
├── data/
│   ├── extracted/          # Raw data files
│   ├── processed/          # Cleaned data
│   ├── outputs/            # Model outputs
│   └── reference-tables/   # Data dictionaries
├── etl/
│   ├── extract.py          # Data extraction
│   ├── transform.py        # Data cleaning
│   └── load.py             # Data loading
├── analysis/
│   ├── model.py            # Model training
│   └── evaluate.py         # Model evaluation
├── vis/
│   └── visualizations.py   # Visualizations
├── main.py                 # Main pipeline
└── requirements.txt        # Dependencies
```
