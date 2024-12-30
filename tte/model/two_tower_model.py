from tte.config.base import BaseConfig
from typing import Dict, Text
import numpy as np
import tensorflow as tf
import tensorflow_recommenders as tfrs
from tte.model.embedding_model import EmbeddingModel
from tte.model.l2_norm_layer import L2NormLayer
from tte.model.single_tower_model import SingleTowerModel

class TwoTowerModel(tfrs.Model):
    """
    Inherits from tensorflow-recommenders and contains boilerplate for training the model
    with sampled in-batch negatives and added logQ correlation
    """

    def __init__(self, 
                 config:BaseConfig, 
                 vocab_dict:dict, 
                 adapted_layers:tf.keras.layers.Layer, 
                 label_probs:tf.lookup.StaticHashTable, 
                 lookup: tf.keras.layers.StringLookup,
                 task: tfrs.tasks.Retrieval
                 ):
        """
        Model for encoding embedding layers
        Args:
            config: config object
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
            tf.keras.layers.Dense(config.embedding_dimension)
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
        self.true_label = self.lookupconfig[['product_id_col']](features[self.config.product_id_col])
        self.candidate_id = self.config.product_id_col
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
            candidate_ids = features[self.config.product_id_col]

        self.account_embeddings,self.product_embeddings=self(features)
        loss = self.task(self.account_embedding, self.product_embeddings, candidate_ids=candidate_ids)

        regularization_loss = sum(self.losses)
        total_loss = loss + regularization_loss

        metrics = {metric.name: metric.result() for metric in self.metrics}
        metrics["loss"] = loss
        metrics["regularization_loss"] = regularization_loss
        metrics["total_loss"] = total_loss

        return metrics