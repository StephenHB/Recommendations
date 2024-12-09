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

# Execute AdaptLayers
adapted_layers =  AdaptLayers(config, ratings).adapt()


class EmbeddingModel(tf.keras.Model):
    """
    Creates the embedding layers for the tower
    """
    def __init__(self, config:dict, tower_type:str, vocab_dict:dict, adapted_layers:Dict[str, tf.keras.layers.Layer]):
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
        self.str_lookup_features = self.config['str_lookup'][tower_type]
        self.int_lookup_features = self.config['int_lookup'][tower_type]
        self.normalizer_features = self.config['normalizer'][tower_type]
        self.str_features = self.config['str_vectorizer'][tower_type]
        self.list_features = self.config['list_vectorizer'][tower_type]
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

class L2NormLayer(tf.keras.layers.Layer):
    """
    Simple function that wraps l2 norm transformation in keras layer.
    """
    def __init__(self,**kwargs):
        super(L2NormLayer,self).__init__(**kwargs)

    @tf.function
    def call(self, inputs, mask=None):
        if mask is not None:
            inputs = tf.ragged.boolean_mask(inputs, mask).to_tensor()
            
        return tf.math.l2_normalize(inputs, axis=-1) + tf.keras.backend.epsilon()
    
    def compute_mask(self,inputs,mask=None):
        return mask
    
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

class LogQCorrelation():
    """
    Class to compute the objects needed for log Q corr.
    """
    def __init__(self, train_df:pd.DataFrame, config:dict):
        """
        Args:
            train_df: training data that includes the frequency of product id orders
            config: config dict
        """
        self.train_df = train_df
        self.config = config

    def build_lookups(self)->Dict[str,tf.keras.layers.StringLookup]:
        """
        Function to build the lookup objects for the product ids.
        """
        self.product_lookup = {}
        for cat_variable in [self.config['product_id_col']]:
            unique_values = self.train_df[cat_variable].unique()
            self.product_lookup[cat_variable] = tf.keras.layers.StringLookup(vocabulary=unique_values)
        return self.product_lookup
    
    def get_label_probs_has_table(self) -> tf.lookup.StaticHashTable:
        """
        Function to get the probability has table for log Q corr
        """
        product_counts_dict = self.train_df.groupby(self.config['product_id_col'])[self.config['product_id_col']].count().to_dict()
        nb_transactions = self.train_df.shape[0]
        keys = list(product_counts_dict.keys())
        values = [count / nb_transactions for count in product_counts_dict.values()]

        keys = tf.constant(keys, dtype=tf.string)
        keys = self.product_lookup[self.config['product_id_col']](keys)
        values = tf.constant(values, dtype=tf.float32)

        return tf.lookup.StaticHashTable(tf.lookup.KeyValueTensorInitializer(keys,values),default_value=0.0)
    
# Execute
train_df = tfds.as_dataframe(ratings)
logq = LogQCorrelation(train_df, config)
product_lookup = logq.build_lookups()
product_prob_lookup = logq.get_label_probs_has_table()

class TwoTowerModel(tfrs.Model):
    """
    Inherits from tensorflow-recommenders and contains boilerplate for training the model
    with sampled in-batch negatives and added logQ correlation
    """

    def __init__(self, 
                 config:dict, 
                 vocab_dict:dict, 
                 adapted_layers:tf.keras.layers.Layer, 
                 label_probs:tf.lookup.StaticHashTable, 
                 lookup: tf.keras.layers.StringLookup,
                 task: tfrs.tasks.Retrieval
                 ):
        """
        Model for encoding embedding layers
        Args:
            config: dict of config vars
            vocab_dict: a dict of vocabs for each layer.
            adapted_layers: tf.keras.layers.Layer objects for each adapted layer.
            label_probs: tf.lookup.StaticHashTable object for logQ corr.
            lookup: tf.keras.layers.StringLookup dict for product ids.
            task: tfrs.tasks.Retrieval object for computing loss
        """
        super().__init__()
        self.config = config
        self.label_probs = label_probs
        self.lookup = lookup
        self.task = task
        self.account_model = tf.keras.Sequential([
            SingleTowerModel(config = config,
                             tower_type = 'account',
                             vocab_dict=vocab_dict,
                             adapted_layers=adapted_layers
                             ),
            tf.keras.layers.Dense(config['embedding_dimension'])
        ])

    def compute_loss(self, features:Dict[str,tf.Tensor],training:bool = False)->tf.Tensor:
        user_embeddings,item_embeddings = self(features)
        return self.task(user_embeddings,item_embeddings,compute_metrics=not training)
    
    def call(self, features:Dict[str,tf.Tensor],training:bool = False)->tf.Tensor:
        """
        Overrides the call method of the model to return the embeddings of the accounts and products.
        """
        self.account_embeddings = self.account_model(features)
        self.product_embeddings = self.product_model(features)

        return self.account_embeddings, self.product_embeddings
    
    def train_step(self, features:Dict[str,tf.Tensor],training:bool = False)->tf.Tensor:
        """
        Overrides the train-step method of the model to return the losses and metrics.
        """
        candidate_ids, candidate_sampling_probability = None, None
        self.true_label = self.lookupconfig[['product_id_col']](features[config['product_id_col']])
        self.candidate_id = config['product_id_col']
        if self.candidate_id is not None and self.task._remove_accidental_hits:
            candidate_ids = features[self.candidate_id]

        with tf.GradientTape() as tape:
            self.account_embeddings, self.product_embeddings=self(features)

            loss = self.task(
                self.account_embeddings,
                self.product_embeddings,
                compute_metrics = False,
                candidate_ids = candidate_ids,
                candidate_sampling_probability=candidate_sampling_probability
            )

            regularization_loss = sum(self.losses)
            total_loss = loss + regularization_loss

        gradients = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        metrics = {metric.name: metric.result() for metric in self.metrics}
        metrics["loss"] = loss
        metrics["regularization_loss"] = regularization_loss
        metrics["total_loss"] = total_loss

        return metrics
    
    def test_step(self, features:Dict[str,tf.Tensor],training:bool = False)->tf.Tensor:
        """
        Overrides the test_step method.
        """
        candidate_ids = None

        if self.candidate_id is not None and self.task._remove_accidental_hits:
            candidate_ids = features[config['product_id_col']]

        self.account_embeddings,self.product_embeddings=self(features)
        loss = self.task(self.account_embedding, self.product_embeddings, candidate_ids=candidate_ids)

        regularization_loss = sum(self.losses)
        total_loss = loss + regularization_loss

        metrics = {metric.name: metric.result() for metric in self.metrics}
        metrics["loss"] = loss
        metrics["regularization_loss"] = regularization_loss
        metrics["total_loss"] = total_loss

        return metrics
    
# Execute the tte
task = tfrs.tasks.Retrieval(remove_accidental_hits=True)
model = TwoTowerModel(config=config,
                      vocab_dict=account_vocab_dict,
                      adapted_layers=adapted_layers,
                      label_probs=product_prob_lookup,
                      lookup=product_lookup,
                      task=task
                    )

model.compile(optimizer=tf.keras.optimizers.adagrad(learning_rate=0.1))
cached_train = train.batch(1).cache()

# Initialize model inputs
for x in cached_train:
    model(x)
    break

# padded batch for list features
cached_train = ratings.shuffle(buffer_size=2500, reshuffle_each_iteration=True).padded_batch(100)
model.fit(cached_train, epochs=10)

model.summary(expand_nested=True)