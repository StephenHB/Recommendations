import os
import pprint
import tempfile
from typing import Dict, Text
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import keras
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_recommenders as tfrs
from tensorflow.keras.utils import plot_model

ratings = tfds.load ("movielens/100k-ratings",split = "train")
movies = tfds.load ("movielens/100k-movies",split = "train")

config ={
    'str_vectorizer':{
        'account':[],
        'product':['movie_title']
    },
    'list_vectorizer':{
        'account':[],
        'product':[]
    },
    'str_lookup':{
        'account':[],
        'product':[]
    },
    'normalizer':{
        'account':['user_rating'],
        'product':[]
    },
    'embedding_dimension':16,
    'max_history_length':100,
    'product_id_col': 'movie_id',
    'batch_size': 512,
    'validation_freq': 5,
    'num_epochs':300,
}

ratings =ratings.map(lambda x:{
    "movie_title": x["movie_title"],
    "movie_genres": x["movie_genres"],
    "user_id": x["user_id"],
    "movie_id": x["movie_id"],
    "user_rating": x["user_rating"],
    "user_gender": x["user_gender"],
    "user_zip_code": x["user_zip_code"],
    "user_occupation_text": x["user_occupation_text"],
    "bucketized_user_age": x["bucketized_user_age"],
    "raw_user_age": x["raw_user_age"],
    "user_occupation_label": x["user_occupation_label"],
})

movies = movies.mapmap(lambda x:{
    "movie_genres": x["movie_genres"],
    "movie_id": x["movie_id"],
    "movie_title": x["movie_title"],
})

feature_names = [
    "user_occupation_text",
    "movie_id",
    "movie_genres",
    "movie_title",
    "user_id",
    "user_occupation_label",
    "raw_user_age",
    "user_zip_code",
    "user_gender",
]

account_vocab_dict = {}

for feature_name in feature_names:
    if feature_name in ratings.element_spec:
        vocab = ratings.batch(1).map(lambda x: tf.reshape(x[feature_name],[-1]))
        account_vocab_dict[feature_name] = np.unqiue(np.concatenate(list(vocab),axis=0))

def df_to_dataset (dataframe, shuffle=True, batch_size=32):
    """
    Convert pandas df to tf dataset
    """
    df = dataframe.copy()
    labels = df.pop('target')
    # Converts all other columns in df to numpy, and stores in a dict, where each key
    # is the column name and the value is the corresponding np array. The [:,tf.newaxis]
    # adds an extra dimension to the np array, which is required from tf.
    df = {key: value.to_numpy()[:,tf.newaxis] for ket, value in dataframe.items()}
    ds = tf.data.Dataset.from_tensor_slices((dict(df), labels))
    if shuffle:
        ds = ds.shuffle( buffer_size=len(dataframe))
    # Model training batch size
    ds = ds.batch(batch_size)
    # Preparing the next batch of data
    ds = ds.prefetch(batch_size)
    return ds

class AdaptLayers():
    """
    Class to adapt the layers for the embedding model.
    """
    def __init__(self,config, training_features):
        """
        Model for adapting normalizer and string vectorizer layers.
        Args:
            config: config dictionary
            training_features: tf._MapDataset of training features.
        """
        self.config = config
        self.training_features = training_features
        self.normalizer_features = []
        self.str_features = []
        for tower_type in  ["account", "product"]:
            self.normalizer_features += self.config['normalizer'][tower_type]
            self.str_features += self.config['str_vectorizer'][tower_type]
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
    
adapted_layers =  AdaptLayers(config, ratings).adapt()


class EmbeddingModel(tf.keras.Model):
    """
    Creates the embedding layers for the tower
    """
    def __init__(self, config:dict, twoer_type:str, vocab_dict;dict, adapted_layers:Dict[str, tf.keras.layers.Layer]):
