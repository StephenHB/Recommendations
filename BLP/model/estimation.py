import numpy as np
import pandas as pd
from typing import final
import site
import tensorflow as tf
from blp.config.base import Base as ConfigBase


class TFEstimator():
    """
        BLP 2SLS estimation
    """

    config: ConfigBase
    data: pd.DataFrame
    reg_param: float
    features: tf.Tensor
    target: tf.Tensor
    instruments: tf.Tensor
    coefficients: tf.Tensor
    xhat_params: tf.Tensor
    params: tf.Tensor
    normalized_cov_params: tf.Tensor

    def __init__(self):
        self.data = None
        self.reg_param = None
        self.features = None
        self.target = None
        self.instruments = None
        self.coefficients = None
        self.xhat_params = None
        self.params = None
        self.normalized_cov_params = None

    def fit(self) -> pd.Series:
        """
        Fit the model with 2SLS
        """
        self._process_data()
        y,x,z = self.target, self.features, self.instruments
        self.validator.validate_instrument_varables(x,z)
        # 1st Stage
        xhat_params = self.first_stage_ols(x,z)
        xhat = self.compute_xhat(xhat_params,z)
        # Get the feature names
        feature_names = self.config.exog_ind_names + self.config.exog_dep_names

        return self.second_stage_ols(xhat, y, tuple(feature_names))
    
    def set_target(self, data:pd.DataFrame, target_name:str) ->tf.Tensor:
        """
            Get the target variable from pd df and transform to tf tensor
        Args:
            data: data containing all the features and IV
            target_name: name of the label variable
        Returns:
            A tensor of the target variable
        """
        return tf.reshape(tf.convert_to_tensor(data[target_name],dtype=tf.float32),[-1,1])
    
    def set_features(self, data:pd.DataFrame, feature_names:tuple) ->tf.Tensor:
        """
            Get the exogenous variables from pd df and transform to tf tensor
        Args:
            data: data containing all the features and IV
            feature_names: Exogenous feature names
        Returns:
            A tensor of the feature variables
        """
        return tf.reshape(tf.convert_to_tensor(data[list(feature_names)],dtype=tf.float32))
    
    def set_instruments(self, data:pd.DataFrame, instrument_names:tuple) ->tf.Tensor:
        """
            Get the IVs from pd df and transform to tf tensor
        Args:
            data: data containing all the features and IV
            feature_names: IV names
        Returns:
            A tensor of the IV
        """
        return tf.reshape(tf.convert_to_tensor(data[list(instrument_names)],dtype=tf.float32))
    
    def _process_data(self) -> None:
        """
            Call the functions that set target, exogenous and IVs to tensor
        """
        feature_names = self.config.exog_ind_names + self.confg.exog_dep_names
        target_name = self.config.target_name
        instrument_names = self.config.exog_ind_names + self.config.instrument_variable_names
        self.features = self.set_features(self.data,tuple(feature_names))
        self.target = self.set_target(self.data, target_name)
        self.instruments = self.set_instruments(self.data, tuple(instrument_names))

        return self
    
    @tf.function
    def first_stage_ols(self, x:tf.Tensor, z:tf.Tensor) -> tf.Tensor:
        # The projection of z on (z,x)
        z_t_z = tf.tensordot(tf.transpose(z),z,axes=[1,0], name="ztz_1st_stage")
        # add tikhonov regularization
        np_regularizer = np.eye(z_t_z.shape[1],dtype=np.float32)*self.reg_param
        # Make the intercept column regularizer zero
        np_regularizer[0,0] = 0
        # add regularizer to the tensor to remove collinearity
        z_t_z = z_t_z + np_regularizer
        z_t_x = tf.tensordot(tf.transpose(z),x,axes=[1,0], name="ztx_1st_stage")
        # Solve coefficient
        self.xhat_params = xhat_params = tf.linalg.solve(z_t_z,z_t_x,adjoint=False, name = "x_hat_params")
        return xhat_params
    
    @tf.function
    def compute_xhat(self, xhat_params:tf.Tensor, z:tf.Tensor ) -> tf.Tensor:
        """
            Project IV on the computed xhat_params
        """
        return tf.tensordot(z,xhat_params,axes=1, name="instrumented_x")
    
    def second_stage_ols(self, xhat:tf.Tensor, y:tf.Tensor, feature_names: tuple) -> pd.Series:
        """
            Second Stage OLS
        """
        xhat_x = tf.tensordot(tf.transpose(xhat),xhat,axes=[1,0])
        xhat_y = tf.tensordot(tf.transpose(xhat),y,axes=[1,0])
        self.params = params = tf.reshape(tf.linalg.solve(xhat_x,xhat_y,adjoint=False, name = "params"),[-1])

        return pd.Series(data=params.numpy(), index=list(feature_names), name = "estimated params")
    
    @tf.function
    def predict(self) -> tf.Tensor:
        return tf.tensordot(self.features, self.params, axes=1)
    
    @staticmethod
    def compute_r_squared(actuals:tf.Tensor, preds:tf.Tensor) -> np.float32:
        """
            Compute R2 and the estimated params
        """
        metric = tf.keras.metrics.R2Score()
        # Flatten the data
        actuals = tf.reshape(actuals,[-1]).numpy()
        preds = tf.reshape(preds,[-1]).numpy()
        metric.update_state(actuals,preds)

        return metric.result().numpy()
    
    @final
    def execute(self) -> "TFEstimator":
        pd_df = pd.read_table(self.config.input_table_path)
        pd_df["intercept"] = 1
        self.coefficients = self.fit()

        return self