import tensorflow_datasets as tfds
from config import BaseConfig

ratings = tfds.load ("movielens/100k-ratings",split = "train")
movies = tfds.load ("movielens/100k-movies",split = "train")