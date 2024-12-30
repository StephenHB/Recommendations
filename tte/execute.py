from tte.config.base import BaseConfig
from typing import Dict, Text
import numpy as np
import tensorflow as tf
import tensorflow_recommenders as tfrs
import tensorflow_datasets as tfds
from tte.model.data_preprocess import Datapreprocess
from tte.model.adapt_layer import AdaptLayers
from tte.model.embedding_model import EmbeddingModel
from tte.model.log_q_correlation import LogQCorrelation
from tte.model.l2_norm_layer import L2NormLayer
from tte.model.single_tower_model import SingleTowerModel
from tte.model.two_tower_model import TwoTowerModel

# Prepare the inlputs
config = BaseConfig()
data_preprocess = Datapreprocess(config).get_data_and_vocab_dict()
account_vocab_dict = data_preprocess.account_vocab_dict
ratings = data_preprocess.data_ratings
movies = data_preprocess.data_movies

# Execute AdaptLayers
adapted_layers =  AdaptLayers(config, ratings).adapt()
# Prepare training data
train_df = tfds.as_dataframe(ratings)
logq = LogQCorrelation(train_df, config)
product_lookup = logq.build_lookups()
product_prob_lookup = logq.get_label_probs_has_table()


# Execute the TTE
task = tfrs.tasks.Retrieval(remove_accidental_hits=True)
model = TwoTowerModel(config=config,
                      vocab_dict=account_vocab_dict,
                      adapted_layers=adapted_layers,
                      label_probs=product_prob_lookup,
                      lookup=product_lookup,
                      task=task
                    )

model.compile(optimizer=tf.keras.optimizers.adagrad(learning_rate=0.1))
cached_train = train_df.batch(1).cache()

# Initialize model inputs
for x in cached_train:
    model(x)
    break

# padded batch for list features
cached_train = ratings.shuffle(buffer_size=2500, reshuffle_each_iteration=True).padded_batch(100)
model.fit(cached_train, epochs=10)

model.summary(expand_nested=True)