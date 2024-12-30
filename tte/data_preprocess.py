import tensorflow_datasets as tfds
from config import BaseConfig

ratings = tfds.load ("movielens/100k-ratings",split = "train")
movies = tfds.load ("movielens/100k-movies",split = "train")

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
