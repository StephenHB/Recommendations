"""
Two Tower Embedding (TTE) Model Execution Script

This script demonstrates how to train a two-tower embedding model for
assortment recommendations using TensorFlow Recommenders.
"""

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


def main():
    """Main execution function for TTE model training"""
    
    print("=== Two Tower Embedding (TTE) Model Training ===")
    
    try:
        # Initialize configuration
        print("1. Initializing configuration...")
        config = BaseConfig()
        
        # Prepare data preprocessing
        print("2. Preparing data preprocessing...")
        data_preprocess = Datapreprocess(config).get_data_and_vocab_dict()
        account_vocab_dict = data_preprocess.account_vocab_dict
        ratings = data_preprocess.data_ratings
        movies = data_preprocess.data_movies
        
        print(f"   - Ratings dataset size: {len(ratings)}")
        print(f"   - Movies dataset size: {len(movies)}")
        
        # Execute AdaptLayers
        print("3. Creating adapted layers...")
        adapted_layers = AdaptLayers(config, ratings).adapt()
        
        # Prepare training data
        print("4. Preparing training data...")
        train_df = tfds.as_dataframe(ratings)
        
        # Setup logQ correlation
        print("5. Setting up logQ correlation...")
        logq = LogQCorrelation(train_df, config)
        product_lookup = logq.build_lookups()
        product_prob_lookup = logq.get_label_probs_has_table()
        
        # Create the TTE model
        print("6. Creating Two Tower Model...")
        task = tfrs.tasks.Retrieval(remove_accidental_hits=True)
        model = TwoTowerModel(
            config=config,
            vocab_dict=account_vocab_dict,
            adapted_layers=adapted_layers,
            label_probs=product_prob_lookup,
            lookup=product_lookup,
            task=task
        )
        
        # Compile the model
        print("7. Compiling model...")
        model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.1))
        
        # Initialize model inputs
        print("8. Initializing model inputs...")
        cached_train = train_df.batch(1).cache()
        
        # Warm up the model
        for x in cached_train:
            model(x)
            break
        
        # Prepare training data with proper batching
        print("9. Preparing training batches...")
        cached_train = ratings.shuffle(
            buffer_size=2500, 
            reshuffle_each_iteration=True
        ).padded_batch(config.batch_size)
        
        # Train the model
        print("10. Starting model training...")
        print(f"    - Epochs: {config.num_epochs}")
        print(f"    - Batch size: {config.batch_size}")
        
        history = model.fit(
            cached_train, 
            epochs=config.num_epochs,
            validation_freq=config.validation_freq
        )
        
        # Model summary
        print("\n11. Model Summary:")
        model.summary(expand_nested=True)
        
        print("\n=== Training Completed Successfully! ===")
        
        return model, history
        
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        print("Please check your configuration and data.")
        raise


if __name__ == "__main__":
    main()