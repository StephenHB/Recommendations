from tte.config.base import BaseConfig
from typing import Dict, Text
import numpy as np
import tensorflow as tf
import tensorflow_recommenders as tfrs
from embedding_model import EmbeddingModel
from l2_norm_layer import L2NormLayer

class SingleTowerModel(tf.keras.Model):
    def __init__(self, config:dict, tower_type:str, vocab_dict:dict, adapted_layers:tf.keras.layers.Layer):
        """
        Model for encoding embedding layers
        Args:
            config: dict of config vars
            twoer_type: str of the type, can be 'product' or 'account'.
            vocab_dict: a dict of vocabs for each layer.
            adapted_layers: tf.keras.layers.Layer objects for each adapted layer
        """
        super().__init__()

        self.embedding_model = EmbeddingModel(
            config=config,
            tower_type = tower_type,
            vocab_dict = vocab_dict,
            adapted_layers = adapted_layers
        )

        # Construct the layers
        self.dense_layers = tf.keras.Sequential()

        if config['cross_layer']:
            self.dense_layers.add(tfrs.layers.dcn.Cross(
                projection_dim = config['projection_dim'],
                kernel_initializer="glorot_uniform"
            ))

        #Use ReLU activation for all but the last layer
        for layer_size in config['layers'][:-1]:
            self.dense_layers.add(tf.keras.layers.Dense(layer_size))

        if config['norm_layer']:
            self.dense_layers.add(L2NormLayer())
        
        if config['dropout_layer']:
            self.dense_layers.add(tf.keras.layers.Dropout(config['dropout_rate']))

    def call(self, features):
        feature_embedding = self.embedding_model(features=features)
        return self.dense_layers(feature_embedding)