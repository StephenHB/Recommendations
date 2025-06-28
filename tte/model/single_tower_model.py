from tte.config.base import BaseConfig
from typing import Dict, Text
import numpy as np
import tensorflow as tf
import tensorflow_recommenders as tfrs
from tte.model.embedding_model import EmbeddingModel
from tte.model.l2_norm_layer import L2NormLayer


class SingleTowerModel(tf.keras.Model):
    """Single tower model for user or item embeddings"""
    
    def __init__(self, config: BaseConfig, tower_type: str, vocab_dict: dict, adapted_layers: tf.keras.layers.Layer):
        """
        Model for encoding embedding layers
        Args:
            config: config object
            tower_type: str of the type, can be 'product' or 'account'.
            vocab_dict: a dict of vocabs for each layer.
            adapted_layers: tf.keras.layers.Layer objects for each adapted layer
        """
        super().__init__()

        self.embedding_model = EmbeddingModel(
            config=config,
            tower_type=tower_type,
            vocab_dict=vocab_dict,
            adapted_layers=adapted_layers
        )

        # Construct the layers
        self.dense_layers = tf.keras.Sequential()

        # Add cross layer if specified (assuming these are config attributes)
        if hasattr(config, 'cross_layer') and config.cross_layer:
            self.dense_layers.add(tfrs.layers.dcn.Cross(
                projection_dim=getattr(config, 'projection_dim', 64),
                kernel_initializer="glorot_uniform"
            ))

        # Add dense layers if specified
        if hasattr(config, 'layers') and config.layers:
            # Use ReLU activation for all but the last layer
            for layer_size in config.layers[:-1]:
                self.dense_layers.add(tf.keras.layers.Dense(layer_size, activation='relu'))
            
            # Add final layer without activation
            if len(config.layers) > 0:
                self.dense_layers.add(tf.keras.layers.Dense(config.layers[-1]))

        # Add normalization layer if specified
        if hasattr(config, 'norm_layer') and config.norm_layer:
            self.dense_layers.add(L2NormLayer())
        
        # Add dropout layer if specified
        if hasattr(config, 'dropout_layer') and config.dropout_layer:
            dropout_rate = getattr(config, 'dropout_rate', 0.1)
            self.dense_layers.add(tf.keras.layers.Dropout(dropout_rate))

        # Added product model
        self.product_model = tf.keras.Sequential([...])

    def call(self, features: Dict[str, tf.Tensor]) -> tf.Tensor:
        """
        Forward pass through the single tower model
        
        Args:
            features: Dictionary of feature tensors
            
        Returns:
            Processed embeddings tensor
        """
        feature_embedding = self.embedding_model(features=features)
        return self.dense_layers(feature_embedding)