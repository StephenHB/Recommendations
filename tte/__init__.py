"""
Two Tower Embedding (TTE) for Assortment Recommendations

A TensorFlow Recommenders-based implementation of two-tower embedding models
for assortment recommendations with logQ correlation correction.
"""

from .config.base import BaseConfig
from .model.two_tower_model import TwoTowerModel
from .model.single_tower_model import SingleTowerModel
from .model.embedding_model import EmbeddingModel
from .model.log_q_correlation import LogQCorrelation

__version__ = "1.0.0"
__author__ = "TTE Implementation Team"

__all__ = [
    "BaseConfig",
    "TwoTowerModel",
    "SingleTowerModel", 
    "EmbeddingModel",
    "LogQCorrelation"
]
