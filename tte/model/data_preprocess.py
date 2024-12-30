import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np
from tte.config.base import BaseConfig


class Datapreprocess:
    """
    Prepare the data for the model, currently loading data from tfds.
    """
    def __init__(self, config):
        self.config = config
        self.account_vocab_dict = {}


    def get_data_and_vocab_dict(self):
        """
        Returns the account_vocab_dict and raw data.
        """
        ratings = tfds.load("movielens/100k-ratings", split="train")
        movies = tfds.load("movielens/100k-movies", split="train")

        self.data_ratings =ratings.map(lambda x:{
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

        self.data_movies = movies.mapmap(lambda x:{
            "movie_genres": x["movie_genres"],
            "movie_id": x["movie_id"],
            "movie_title": x["movie_title"],
        })


        for feature_name in self.config.feature_names:
            if feature_name in self.data_ratings.element_spec:
                vocab = self.data_ratings.batch(1).map(lambda x: tf.reshape(x[feature_name],[-1]))
                self.account_vocab_dict[feature_name] = np.unqiue(np.concatenate(list(vocab),axis=0))

        return self