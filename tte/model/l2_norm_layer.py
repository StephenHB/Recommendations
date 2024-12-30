import tensorflow as tf
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