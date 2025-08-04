"""
This package contains modules for model training and evaluation.
"""

from .model import ChurnPredictor
from .evaluate import ModelEvaluator

__all__ = ['ChurnPredictor', 'ModelEvaluator'] 