import tensorflow as tf
from mouffet import common_utils
from tensorflow.keras import layers

from . import AudioDetector


class DLBD(AudioDetector):
    """DLBD Network with one less Dense layer to reduce the number of parameters and overfitting

    Args:
        CityNetTF2 ([type]): [description]

    Returns:
        [type]: [description]
    """

    NAME = "DLBD"

    def check_options(self):
        num_dense_units2 = self.opts.get("num_dense_units2", 0)
        if num_dense_units2 and (
            num_dense_units2 > self.opts.get("num_dense_units", 128)
        ):
            common_utils.print_error(
                "Error: 'num_dense_units2' is greater than 'num_dense_units'."
            )
            return False

        return True

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
                self.opts.get("conv_filter_width", 4),
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
            (
                self.opts.get("conv2_filter_height", 1),
                self.opts.get("conv2_filter_width", 3),
            ),
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
            self.opts.get("num_dense_units", 128),
            activation="relu",
            bias_initializer=None,
            kernel_regularizer=regularizer,
        )(x)
        x = layers.BatchNormalization()(x)
        dense2 = self.opts.get("num_dense_units2", 0)
        if dense2 > 0:
            x = layers.Dense(
                dense2,
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


class DLBDBinary(DLBD):
    @staticmethod
    def tf_loss(y_true, y_pred):
        bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        return tf.reduce_mean(bce(y_true, y_pred))

    def predict(self, x):
        return tf.keras.activations.sigmoid(self.model(x, training=False)).numpy()
