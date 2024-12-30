from config import BaseConfig
from typing import Dict, Text
import numpy as np
import tensorflow as tf

class EmbeddingModel(tf.keras.Model):
    """
    Creates the embedding layers for the tower
    """
    def __init__(self, config:BaseConfig, tower_type:str, vocab_dict:dict, adapted_layers:Dict[str, tf.keras.layers.Layer]):
        """
        Model for encoding embedding layers
        Args:
            config: dict of config vars
            twoer_type: str of the type, can be 'product' or 'account'.
            vocab_dict: a dict of vocabs for each layer.
            adapted_layers: Dict objects for each adapted layer.
        """
        super().__init()
        self.config = config
        self.vocab_dict =  vocab_dict
        self.adapted_layers = adapted_layers
        self.str_lookup_features = self.config.str_lookup_config[tower_type]
        self.int_lookup_features = self.config.int_lookup_config[tower_type]
        self.normalizer_features = self.config.normalizer_config[tower_type]
        self.str_features = self.config.str_vectorizer_config[tower_type]
        self.list_features = self.config.list_vectorizer_config[tower_type]
        self._all_features = self.str_lookup_features + self.int_lookup_features + self.normalizer_features + self.str_features + self.list_features
        self._embeddings = {}

        # Create normalization layers
        if len(self.normalizer_features) >0:
            for feature in self.normalizer_features:
                self._embeddings[feature] = self.adapted_layers[feature]

        # Create String lookup layers
        if len(self.str_lookup_features) >0:
            for feature in self.str_lookup_features:
                vocabulary = self.vocab_dict[feature]
                self._embeddings[feature] = tf.keras.Sequential(
                    [
                        tf.keras.layers.StringLookup(
                            vocabulary = vocabulary,
                            name = feature+"_lookup"
                        ),
                        tf.keras.layers.Embedding(
                            input_dim = len(vocabulary) +1,
                            output_dim = config['layer_output'],
                            mask_zero = True,
                            name=feature+"_emb"
                        )
                    ],
                name = feature+'_layer'
                )

        # Create Int lookup layers
        if len(self.int_lookup_features)>0:
            for feature in self.int_lookup_features:
                vocabulary = self.vocab_dict[feature]
                self._embeddings[feature] = tf.keras.Squential(
                    [
                        tf.keras.layers.IntegerLookup(
                            vocabulary = vocabulary,
                            name = feature+"_lookup"
                        ),
                        tf.keras.layers.Embedding(
                            input_dim = len(vocabulary) +1,
                            output_dim = config['layer_output'],
                            mask_zero = True,
                            name=feature+"_emb"
                        )
                    ],
                name = feature+'_layer'
                )

        # Create Str vectorizer layers
        if len(self.str_features)>0:
            for feature in self.str_features:
                vocabulary = self.vocab_dict[feature]
                self._embeddings[feature] = tf.keras.Squential(
                    [
                        self.adapted_layers[feature],
                        tf.keras.layers.Embedding(
                            input_dim = 10000 +1,
                            output_dim = config['layer_output'],
                            mask_zero = False,
                            name=feature+"_emb"
                        ),
                        tf.keras.layers.GlobalAveragePooling1D(name=feature+"_1d")
                    ],
                name = feature+'_layer'
                )

         # Create list vectorizer layers
        if len(self.list_features)>0:
            for feature in self.list_features:
                vocabulary = self.vocab_dict[feature]
                self._embeddings[feature] = tf.keras.Squential(
                    [
                        tf.keras.layers.IntegerLookup(
                            vocabulary = vocabulary,
                            name = feature+"_lookup"
                        ),
                        tf.keras.layers.Embedding(
                            input_dim = len(vocabulary) +1,
                            output_dim = config['layer_output'],
                            mask_zero = True,
                            name=feature+"_emb"
                        ),
                        tf.keras.layers.GlobalAveragePooling1D(name=feature+"_1d")
                    ],
                name = feature+'_layer'
                )

    def call(self,features):
        self.features = features
        embeddings = []
        for feature in self._all_features:
            embedding_fn = self._embeddings[feature]
            if (feature in self.normalizer_features):
                embeddings.append(tf.reshape(embedding_fn(self.features[feature]),(-1,1)))
            elif (feature in self.str_lookup_features) | (feature in self.int_lookup_features):
                embeddings.append(embedding_fn(self.features[feature]))
            elif (feature in self.str_features):
                self.features[feature] = tf.expand_dims(self.features[feature],-1)
                embeddings.append(embedding_fn(self.features[feature]))
            else:
                embeddings.append(embedding_fn(self.features[feature]))
