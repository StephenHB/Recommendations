"""
BLP (Berry, Levinsohn, and Pakes) Model Implementation

A TensorFlow-based implementation of the BLP model for demand estimation
using instrumental variables and two-stage least squares.
"""

from .config.base import Base
from .model.estimation import TFEstimator, Validator

__version__ = "1.0.0"
__author__ = "BLP Implementation Team"

__all__ = [
    "Base",
    "TFEstimator", 
    "Validator"
]
