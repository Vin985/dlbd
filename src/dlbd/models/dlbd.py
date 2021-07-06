import tensorflow as tf
from tensorflow.keras import Input, Model, layers, regularizers

from .CityNetTF2 import CityNetTF2, NormalizeSpectrograms


class DLBDDense(CityNetTF2):
    NAME = "DLBD_dense"

    def add_layers(self, inputs, opts):
        x = layers.Conv2D(
            opts.get("num_filters", 128),
            (opts["spec_height"] - opts["wiggle_room"], opts["conv_filter_width"],),
            bias_initializer=None,
            padding="valid",
            activation=None,
            kernel_regularizer=regularizers.l2(0.001),
        )(inputs)
        # * Second block
        x = layers.Conv2D(
            opts.get("num_filters", 128),
            (1, 3),
            bias_initializer=None,
            padding="valid",
            activation=None,
            kernel_regularizer=regularizers.l2(0.001),
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(alpha=1 / 3, name="conv1_2",)(x)
        W = x.shape[2]
        x = layers.MaxPool2D(pool_size=(1, W), strides=(1, 1), name="pool2")(x)
        x = layers.Dropout(0.5)(x)
        x = tf.transpose(x, (0, 3, 2, 1))
        x = layers.Flatten(name="pool2_flat")(x)
        x = layers.Dense(
            opts["num_dense_units"],
            activation=None,
            bias_initializer=None,
            kernel_regularizer=regularizers.l2(0.001),
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(alpha=1 / 3, name="fc6")(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(
            opts["num_dense_units"],
            activation=None,
            bias_initializer=None,
            # kernel_regularizer=regularizers.l2(0.001),
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(alpha=1 / 3, name="fc7")(x)
        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(
            2, activation=None, name="fc8"  # kernel_regularizer=regularizers.l2(0.001),
        )(x)
        return outputs


class DLBDLite(CityNetTF2):
    """DLBD Network with one less Dense layer to reduce the number of parameters and overfitting

    Args:
        CityNetTF2 ([type]): [description]

    Returns:
        [type]: [description]
    """

    NAME = "DLBD_lite"

    def add_layers(self, inputs, opts):
        x = layers.Conv2D(
            opts.get("num_filters", 128),
            (opts["spec_height"] - opts["wiggle_room"], opts["conv_filter_width"],),
            bias_initializer=None,
            padding="valid",
            activation=None,
            kernel_regularizer=regularizers.l2(0.001),
        )(inputs)
        # * Second block
        x = layers.Conv2D(
            opts.get("num_filters", 128),
            (1, 3),
            bias_initializer=None,
            padding="valid",
            activation=None,
            kernel_regularizer=regularizers.l2(0.001),
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(alpha=1 / 3, name="conv1_2",)(x)
        W = x.shape[2]
        x = layers.MaxPool2D(pool_size=(1, W), strides=(1, 1), name="pool2")(x)
        x = layers.Dropout(0.5)(x)
        x = tf.transpose(x, (0, 3, 2, 1))
        x = layers.Flatten(name="pool2_flat")(x)
        x = layers.Dense(
            opts["num_dense_units"],
            activation=None,
            bias_initializer=None,
            kernel_regularizer=regularizers.l2(0.001),
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(alpha=1 / 3, name="fc6")(x)
        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(
            2, activation=None, name="fc8"  # kernel_regularizer=regularizers.l2(0.001),
        )(x)
        return outputs

