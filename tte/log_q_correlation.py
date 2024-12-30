from config import BaseConfig
from typing import Dict, Text
import numpy as np
import tensorflow as tf

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
        for cat_variable in [self.config.product_id_col]:
            unique_values = self.train_df[cat_variable].unique()
            self.product_lookup[cat_variable] = tf.keras.layers.StringLookup(vocabulary=unique_values)
        return self.product_lookup
    
    def get_label_probs_has_table(self) -> tf.lookup.StaticHashTable:
        """
        Function to get the probability has table for log Q corr
        """
        product_counts_dict = self.train_df.groupby(self.config.product_id_col)[self.config.product_id_col].count().to_dict()
        nb_transactions = self.train_df.shape[0]
        keys = list(product_counts_dict.keys())
        values = [count / nb_transactions for count in product_counts_dict.values()]

        keys = tf.constant(keys, dtype=tf.string)
        keys = self.product_lookup[self.config.product_id_col](keys)
        values = tf.constant(values, dtype=tf.float32)

        return tf.lookup.StaticHashTable(tf.lookup.KeyValueTensorInitializer(keys,values),default_value=0.0)