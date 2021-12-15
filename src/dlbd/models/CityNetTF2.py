import tensorflow as tf
import tensorflow.keras as keras

from . import AudioDetector


class CityNetTF2(AudioDetector):
    NAME = "CityNetTF2"

    def get_base_layers(self, x=None):
        if not x:
            x = self.get_preprocessing_layers()
        # * First block
        x = keras.layers.Conv2D(
            self.opts.get("num_filters", 128),
            (
                self.opts["input_height"] - self.opts["wiggle_room"],
                self.opts["conv_filter_width"],
            ),
            bias_initializer=None,
            padding="valid",
            activation=None,
            kernel_regularizer=keras.regularizers.l2(0.001),
        )(x)
        x = keras.layers.LeakyReLU(
            alpha=1 / 3,
            name="conv1_1",
        )(x)
        # * Second block
        x = keras.layers.Conv2D(
            self.opts.get("num_filters", 128),
            (1, 3),
            bias_initializer=None,
            padding="valid",
            activation=None,
            kernel_regularizer=keras.regularizers.l2(0.001),
        )(x)
        x = keras.layers.LeakyReLU(
            alpha=1 / 3,
            name="conv1_2",
        )(x)
        W = x.shape[2]
        x = keras.layers.MaxPool2D(pool_size=(1, W), strides=(1, 1), name="pool2")(x)
        x = tf.transpose(x, (0, 3, 2, 1))
        x = keras.layers.Flatten(name="pool2_flat")(x)
        x = keras.layers.Dense(
            self.opts["num_dense_units"],
            activation=None,
            bias_initializer=None,
            kernel_regularizer=keras.regularizers.l2(0.001),
        )(x)
        x = keras.layers.LeakyReLU(alpha=1 / 3, name="fc6")(x)
        x = keras.layers.Dense(
            self.opts["num_dense_units"],
            activation=None,
            bias_initializer=None,
        )(x)
        x = keras.layers.LeakyReLU(alpha=1 / 3, name="fc7")(x)
        return x

    def get_top_layers(self, x=None):
        if not x:
            x = self.get_base_layers()
        x = keras.layers.Dense(
            2, activation=None, name="fc8"  # kernel_regularizer=regularizers.l2(0.001),
        )(x)
        return x
