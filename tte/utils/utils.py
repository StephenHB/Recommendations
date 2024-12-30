import tensorflow as tf

class Utils:
    @staticmethod
    def df_to_dataset(dataframe, shuffle=True, batch_size=32):
        """
        Convert a pandas DataFrame to a TensorFlow dataset.
        
        Args:
        - dataframe: A pandas DataFrame containing the data.
        - shuffle: Whether to shuffle the dataset (default is True).
        - batch_size: Batch size for training (default is 32).
        
        Returns:
        - A TensorFlow Dataset ready for model training.
        """
        df = dataframe.copy()
        labels = df.pop('target')
        
        # Convert all columns in the dataframe to numpy arrays and add an extra dimension.
        df = {key: value.to_numpy()[:, tf.newaxis] for key, value in df.items()}
        
        # Create TensorFlow dataset from the features (df) and labels.
        ds = tf.data.Dataset.from_tensor_slices((dict(df), labels))
        
        # Shuffle the dataset if specified.
        if shuffle:
            ds = ds.shuffle(buffer_size=len(dataframe))
        
        # Batch the dataset.
        ds = ds.batch(batch_size)
        
        # Prepare the next batch of data.
        ds = ds.prefetch(tf.data.AUTOTUNE)
        
        return ds
