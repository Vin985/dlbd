from numpy.lib.arraysetops import isin
import tensorflow as tf
from tensorflow.keras import layers

from .CityNetTF2 import CityNetTF2


class DLBD(CityNetTF2):
    """DLBD Network with one less Dense layer to reduce the number of parameters and overfitting

    Args:
        CityNetTF2 ([type]): [description]

    Returns:
        [type]: [description]
    """

    NAME = "DLBD"

    def get_dilation_rate(self, idx):
        dr = self.opts.get("dilation_rate", 0)
        if dr:
            if isinstance(dr, list):
                if idx <= len(dr):
                    return dr[idx - 1]
            elif isinstance(dr, int):
                return dr
        return 1

    def get_base_layers(self, x=None):
        if x is None:
            x = self.get_preprocessing_layers()

        regularizer = self.get_regularizer()
        # * First block
        conv_filter_height = self.opts.get(
            "conv_filter_height", self.opts["input_height"] - self.opts["wiggle_room"]
        )
        x = layers.Conv2D(
            self.opts.get("num_filters", 128),
            (
                conv_filter_height,
                self.opts["conv_filter_width"],
            ),
            dilation_rate=self.get_dilation_rate(1),
            bias_initializer=None,
            padding="valid",
            activation="relu",
            kernel_regularizer=regularizer,
        )(x)
        x = layers.BatchNormalization()(x)
        # * Second block
        x = layers.Conv2D(
            self.opts.get("num_filters", 128),
            (1, 3),
            bias_initializer=None,
            dilation_rate=self.get_dilation_rate(2),
            padding="valid",
            activation="relu",
            kernel_regularizer=regularizer,
        )(x)
        x = layers.BatchNormalization()(x)
        W = x.shape[2]
        x = layers.MaxPool2D(pool_size=(1, W), strides=(1, 1), name="pool2")(x)
        x = layers.BatchNormalization()(x)
        x = tf.transpose(x, (0, 3, 2, 1))
        x = layers.Flatten(name="pool2_flat")(x)
        x = layers.Dense(
            self.opts["num_dense_units"],
            activation="relu",
            bias_initializer=None,
            kernel_regularizer=regularizer,
        )(x)
        x = layers.BatchNormalization()(x)
        if not self.opts.get("is_lite", True):
            x = layers.Dense(
                self.opts["num_dense_units2"],
                activation="relu",
                bias_initializer=None,
                kernel_regularizer=regularizer,
            )(x)
            x = layers.BatchNormalization()(x)

        return x

    def get_top_layers(self, x=None):
        if x is None:
            x = self.get_base_layers()
        x = layers.Dense(2, activation=None, name="fc8")(x)
        return x
