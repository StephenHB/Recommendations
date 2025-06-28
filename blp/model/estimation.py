import numpy as np
import pandas as pd
from typing import final, Optional
import site
import tensorflow as tf
from blp.config.base import Base as ConfigBase


class Validator:
    """Simple validator for instrument variables"""
    
    @staticmethod
    def validate_instrument_variables(x: tf.Tensor, z: tf.Tensor) -> None:
        """Validate that instrument variables have correct dimensions"""
        if x.shape[0] != z.shape[0]:
            raise ValueError(f"Number of observations must match: x has {x.shape[0]}, z has {z.shape[0]}")
        if z.shape[1] < x.shape[1]:
            raise ValueError(f"Number of instruments ({z.shape[1]}) must be >= number of endogenous variables ({x.shape[1]})")


class TFEstimator:
    """
        BLP 2SLS estimation
    """

    def __init__(self):
        self.config: Optional[ConfigBase] = None
        self.data: Optional[pd.DataFrame] = None
        self.reg_param: float = 0.0
        self.features: Optional[tf.Tensor] = None
        self.target: Optional[tf.Tensor] = None
        self.instruments: Optional[tf.Tensor] = None
        self.coefficients: Optional[pd.Series] = None
        self.xhat_params: Optional[tf.Tensor] = None
        self.params: Optional[tf.Tensor] = None
        self.normalized_cov_params: Optional[tf.Tensor] = None
        self.validator = Validator()

    def set_config(self, config: ConfigBase) -> 'TFEstimator':
        """Set the configuration for the estimator"""
        self.config = config
        self.reg_param = config.reg_param
        return self

    def set_data(self, data: pd.DataFrame) -> 'TFEstimator':
        """Set the data for the estimator"""
        self.data = data
        return self

    def fit(self) -> pd.Series:
        """
        Fit the model with 2SLS
        """
        if self.config is None or self.data is None:
            raise ValueError("Config and data must be set before fitting")
            
        self._process_data()
        y, x, z = self.target, self.features, self.instruments
        
        if self.validator:
            self.validator.validate_instrument_variables(x, z)
            
        # 1st Stage
        xhat_params = self.first_stage_ols(x, z)
        xhat = self.compute_xhat(xhat_params, z)
        # Get the feature names
        feature_names = self.config.exog_ind_names + self.config.exog_dep_names

        return self.second_stage_ols(xhat, y, tuple(feature_names))
    
    def set_target(self, data: pd.DataFrame, target_name: str) -> tf.Tensor:
        """
            Get the target variable from pd df and transform to tf tensor
        Args:
            data: data containing all the features and IV
            target_name: name of the label variable
        Returns:
            A tensor of the target variable
        """
        return tf.reshape(tf.convert_to_tensor(data[target_name], dtype=tf.float32), [-1, 1])
    
    def set_features(self, data: pd.DataFrame, feature_names: tuple) -> tf.Tensor:
        """
            Get the exogenous variables from pd df and transform to tf tensor
        Args:
            data: data containing all the features and IV
            feature_names: Exogenous feature names
        Returns:
            A tensor of the feature variables
        """
        return tf.convert_to_tensor(data[list(feature_names)], dtype=tf.float32)
    
    def set_instruments(self, data: pd.DataFrame, instrument_names: tuple) -> tf.Tensor:
        """
            Get the IVs from pd df and transform to tf tensor
        Args:
            data: data containing all the features and IV
            instrument_names: IV names
        Returns:
            A tensor of the IV
        """
        return tf.convert_to_tensor(data[list(instrument_names)], dtype=tf.float32)
    
    def _process_data(self) -> None:
        """
            Call the functions that set target, exogenous and IVs to tensor
        """
        if self.config is None:
            raise ValueError("Config must be set before processing data")
        
        if self.data is None:
            raise ValueError("Data must be set before processing")
            
        # Check for missing values
        if self.data.isnull().any().any():
            raise ValueError("Data contains missing values. Please handle missing values before estimation.")
            
        # Check for infinite values
        if np.isinf(self.data.select_dtypes(include=[np.number])).any().any():
            raise ValueError("Data contains infinite values. Please handle infinite values before estimation.")
            
        feature_names = self.config.exog_ind_names + self.config.exog_dep_names
        target_name = self.config.target_name
        instrument_names = self.config.exog_ind_names + self.config.instrument_variable_names
        
        # Check if all required columns exist
        missing_cols = []
        for col in feature_names + [target_name] + instrument_names:
            if col not in self.data.columns:
                missing_cols.append(col)
        
        if missing_cols:
            raise ValueError(f"Missing columns in data: {missing_cols}")
        
        self.features = self.set_features(self.data, tuple(feature_names))
        self.target = self.set_target(self.data, target_name)
        self.instruments = self.set_instruments(self.data, tuple(instrument_names))

    @tf.function
    def first_stage_ols(self, x: tf.Tensor, z: tf.Tensor) -> tf.Tensor:
        # The projection of z on (z,x)
        z_t_z = tf.tensordot(tf.transpose(z), z, axes=[1, 0], name="ztz_1st_stage")
        
        # add tikhonov regularization using pure TensorFlow operations
        z_dim = tf.shape(z_t_z)[1]
        regularizer = tf.eye(z_dim, dtype=tf.float32) * self.reg_param
        mask = tf.eye(z_dim, dtype=tf.float32)
        mask = tf.tensor_scatter_nd_update(mask, [[0, 0]], [0.0])
        regularizer = regularizer * mask
        z_t_z = z_t_z + regularizer
        
        z_t_x = tf.tensordot(tf.transpose(z), x, axes=[1, 0], name="ztx_1st_stage")
        
        # Solve coefficient with error handling
        try:
            self.xhat_params = xhat_params = tf.linalg.solve(z_t_z, z_t_x, adjoint=False, name="x_hat_params")
            return xhat_params
        except tf.errors.InvalidArgumentError:
            # If matrix is singular, try with additional regularization
            z_t_z_reg = z_t_z + tf.eye(z_dim, dtype=tf.float32) * 1e-6
            self.xhat_params = xhat_params = tf.linalg.solve(z_t_z_reg, z_t_x, adjoint=False, name="x_hat_params")
            return xhat_params
    
    @tf.function
    def compute_xhat(self, xhat_params: tf.Tensor, z: tf.Tensor) -> tf.Tensor:
        """
            Project IV on the computed xhat_params
        """
        return tf.tensordot(z, xhat_params, axes=1, name="instrumented_x")
    
    def second_stage_ols(self, xhat: tf.Tensor, y: tf.Tensor, feature_names: tuple) -> pd.Series:
        """
            Second Stage OLS
        """
        xhat_x = tf.tensordot(tf.transpose(xhat), xhat, axes=[1, 0])
        xhat_y = tf.tensordot(tf.transpose(xhat), y, axes=[1, 0])
        
        # Add error handling for singular matrix
        try:
            self.params = params = tf.reshape(tf.linalg.solve(xhat_x, xhat_y, adjoint=False, name="params"), [-1])
        except tf.errors.InvalidArgumentError:
            # If matrix is singular, add small regularization
            x_dim = tf.shape(xhat_x)[0]
            xhat_x_reg = xhat_x + tf.eye(x_dim, dtype=tf.float32) * 1e-6
            self.params = params = tf.reshape(tf.linalg.solve(xhat_x_reg, xhat_y, adjoint=False, name="params"), [-1])

        return pd.Series(data=params.numpy(), index=list(feature_names), name="estimated params")
    
    @tf.function
    def predict(self) -> tf.Tensor:
        if self.features is None or self.params is None:
            raise ValueError("Model must be fitted before prediction")
        return tf.tensordot(self.features, self.params, axes=1)
    
    @staticmethod
    def compute_r_squared(actuals: tf.Tensor, preds: tf.Tensor) -> np.float32:
        """
            Compute R2 and the estimated params
        """
        metric = tf.keras.metrics.R2Score()
        # Flatten the data
        actuals = tf.reshape(actuals, [-1]).numpy()
        preds = tf.reshape(preds, [-1]).numpy()
        metric.update_state(actuals, preds)

        return metric.result().numpy()
    
    @final
    def execute(self) -> 'TFEstimator':
        if self.config is None:
            raise ValueError("Config must be set before execution")
            
        pd_df = pd.read_table(self.config.input_table_path)
        pd_df["intercept"] = 1
        self.data = pd_df
        self.coefficients = self.fit()

        return self