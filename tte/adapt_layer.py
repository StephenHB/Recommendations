from config import BaseConfig
from typing import Dict, Text
import numpy as np
import tensorflow as tf


class AdaptLayers():
    """
    Class to adapt the layers for the embedding model.
    """
    def __init__(self,config, training_features):
        """
        Model for adapting normalizer and string vectorizer layers.
        Args:
            config: config object
            training_features: tf._MapDataset of training features.
        """
        self.config = config
        self.training_features = training_features
        self.normalizer_features = []
        self.str_features = []
        for tower_type in  ["account", "product"]:
            self.normalizer_features += self.config.normalizer_config[tower_type]
            self.str_features += self.config.str_vectorizer_config[tower_type]
        self._all_features = self.normalizer_features + self.str_features
        self._adapted_layers = {}

    def adapt(self) -> Dict[str, tf.keras.layers.Layer]:
        """
        Adapts the layers for the embedding model.
        """
        if len(self.normalizer_features) >0:
            for feature in self.normalizer_features:
                normalizer = tf.keras.layers.Normalization(
                    axis = None,
                    name = feature + "_normalizer"
                )
                # Learn the stats of the data
                normalizer.adapt(np.concatenate(
                    list(self.training_features.map(
                        lambda x: x[feature]
                    ).batch(100))))
        if len(self.str_features)>0:
            for feature in self.str_features:
                vectorizer = tf.keras.layers.TextVectorization(
                    max_tokens = 10000,
                    standardize = 'lower_and_strip_punctuation',
                    split = 'whitespace',
                    ngrams =  None,
                    name = feature + "_textvectorizer"
                )
                vectorizer.adapt(np.concatenate(list(self.training_features.map(lambda x: x[feature]).batch(100))))
        
        return self._adapted_layers