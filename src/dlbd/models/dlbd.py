import tensorflow as tf
from tensorflow.keras import layers, regularizers

from .CityNetTF2 import CityNetTF2


class DLBDDense(CityNetTF2):
    NAME = "DLBD_dense"

    def get_base_layers(self, x=None):
        if x is None:
            x = self.get_preprocessing_layers()
        # * First block
        x = layers.Conv2D(
            self.opts.get("num_filters", 128),
            (
                self.opts["input_height"] - self.opts["wiggle_room"],
                self.opts["conv_filter_width"],
            ),
            bias_initializer=None,
            padding="valid",
            activation=None,
            kernel_regularizer=regularizers.l2(0.001),
        )(x)
        # * Second block
        x = layers.Conv2D(
            self.opts.get("num_filters", 128),
            (1, 3),
            bias_initializer=None,
            padding="valid",
            activation=None,
            kernel_regularizer=regularizers.l2(0.001),
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(
            alpha=1 / 3,
            name="conv1_2",
        )(x)
        W = x.shape[2]
        x = layers.MaxPool2D(pool_size=(1, W), strides=(1, 1), name="pool2")(x)
        x = layers.Dropout(self.opts.get("dropout", 0.5))(x)
        x = tf.transpose(x, (0, 3, 2, 1))
        x = layers.Flatten(name="pool2_flat")(x)
        x = layers.Dense(
            self.opts["num_dense_units"],
            activation=None,
            bias_initializer=None,
            kernel_regularizer=regularizers.l2(0.001),
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(alpha=1 / 3, name="fc6")(x)
        x = layers.Dropout(self.opts.get("dropout", 0.5))(x)
        x = layers.Dense(
            self.opts["num_dense_units"],
            activation=None,
            bias_initializer=None,
            # kernel_regularizer=regularizers.l2(0.001),
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(alpha=1 / 3, name="fc7")(x)
        x = layers.Dropout(self.opts.get("dropout", 0.5))(x)
        return x

    def get_top_layers(self, x=None):
        if x is None:
            x = self.get_base_layers()
        x = layers.Dense(2, activation=None, name="fc8")(x)
        return x


class DLBDDenseDrop(CityNetTF2):
    NAME = "DLBD_dense"

    def get_base_layers(self, x=None):
        if x is None:
            x = self.get_preprocessing_layers()
        # * First block
        x = layers.Conv2D(
            self.opts.get("num_filters", 128),
            (
                self.opts["input_height"] - self.opts["wiggle_room"],
                self.opts["conv_filter_width"],
            ),
            bias_initializer=None,
            padding="valid",
            activation=None,
            kernel_regularizer=regularizers.l2(0.001),
        )(x)
        # * Second block
        x = layers.Conv2D(
            self.opts.get("num_filters", 128),
            (1, 3),
            bias_initializer=None,
            padding="valid",
            activation=None,
            kernel_regularizer=regularizers.l2(0.001),
        )(x)
        x = layers.LeakyReLU(
            alpha=1 / 3,
            name="conv1_2",
        )(x)
        W = x.shape[2]
        x = layers.MaxPool2D(pool_size=(1, W), strides=(1, 1), name="pool2")(x)
        x = layers.Dropout(self.opts.get("dropout", 0.5))(x)
        x = tf.transpose(x, (0, 3, 2, 1))
        x = layers.Flatten(name="pool2_flat")(x)
        x = layers.Dense(
            self.opts["num_dense_units"],
            activation=None,
            bias_initializer=None,
            kernel_regularizer=regularizers.l2(0.001),
        )(x)
        x = layers.LeakyReLU(alpha=1 / 3, name="fc6")(x)
        x = layers.Dropout(self.opts.get("dropout", 0.5))(x)
        x = layers.Dense(
            self.opts["num_dense_units"],
            activation=None,
            bias_initializer=None,
            # kernel_regularizer=regularizers.l2(0.001),
        )(x)
        x = layers.LeakyReLU(alpha=1 / 3, name="fc7")(x)
        x = layers.Dropout(self.opts.get("dropout", 0.5))(x)
        return x

    def get_top_layers(self, x=None):
        if x is None:
            x = self.get_base_layers()
        x = layers.Dense(2, activation=None, name="fc8")(x)
        return x


class DLBDLite(CityNetTF2):
    """DLBD Network with one less Dense layer to reduce the number of parameters and overfitting

    Args:
        CityNetTF2 ([type]): [description]

    Returns:
        [type]: [description]
    """

    NAME = "DLBD_lite"

    def get_base_layers(self, x=None):
        if x is None:
            x = self.get_preprocessing_layers()
        # * First block
        x = layers.Conv2D(
            self.opts.get("num_filters", 128),
            (
                self.opts["input_height"] - self.opts["wiggle_room"],
                self.opts["conv_filter_width"],
            ),
            bias_initializer=None,
            padding="valid",
            activation=None,
            kernel_regularizer=regularizers.l2(0.001),
        )(x)
        # * Second block
        x = layers.Conv2D(
            self.opts.get("num_filters", 128),
            (1, 3),
            bias_initializer=None,
            padding="valid",
            activation=None,
            kernel_regularizer=regularizers.l2(0.001),
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(
            alpha=1 / 3,
            name="conv1_2",
        )(x)
        W = x.shape[2]
        x = layers.MaxPool2D(pool_size=(1, W), strides=(1, 1), name="pool2")(x)
        x = layers.Dropout(self.opts.get("dropout", 0.5))(x)
        x = tf.transpose(x, (0, 3, 2, 1))
        x = layers.Flatten(name="pool2_flat")(x)
        x = layers.Dense(
            self.opts["num_dense_units"],
            activation=None,
            bias_initializer=None,
            kernel_regularizer=regularizers.l2(0.001),
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(alpha=1 / 3, name="fc6")(x)
        return x

    def get_top_layers(self, x=None):
        if x is None:
            x = self.get_base_layers()
        x = layers.Dense(2, activation=None, name="fc8")(x)
        return x


class DLBDLiteNoReg(CityNetTF2):
    """DLBD Network with one less Dense layer to reduce the number of parameters and overfitting

    Args:
        CityNetTF2 ([type]): [description]

    Returns:
        [type]: [description]
    """

    NAME = "DLBD_lite"

    def get_base_layers(self, x=None):
        if x is None:
            x = self.get_preprocessing_layers()
        # * First block
        x = layers.Conv2D(
            self.opts.get("num_filters", 128),
            (
                self.opts["input_height"] - self.opts["wiggle_room"],
                self.opts["conv_filter_width"],
            ),
            bias_initializer=None,
            padding="valid",
            activation=None,
            kernel_regularizer=regularizers.l2(0.001),
        )(x)
        # * Second block
        x = layers.Conv2D(
            self.opts.get("num_filters", 128),
            (1, 3),
            bias_initializer=None,
            padding="valid",
            activation=None,
            kernel_regularizer=regularizers.l2(0.001),
        )(x)
        x = layers.LeakyReLU(
            alpha=1 / 3,
            name="conv1_2",
        )(x)
        W = x.shape[2]
        x = layers.MaxPool2D(pool_size=(1, W), strides=(1, 1), name="pool2")(x)
        x = tf.transpose(x, (0, 3, 2, 1))
        x = layers.Flatten(name="pool2_flat")(x)
        x = layers.Dense(
            self.opts["num_dense_units"],
            activation=None,
            bias_initializer=None,
            kernel_regularizer=regularizers.l2(0.001),
        )(x)
        x = layers.LeakyReLU(alpha=1 / 3, name="fc6")(x)
        return x

    def get_top_layers(self, x=None):
        if x is None:
            x = self.get_base_layers()
        x = layers.Dense(2, activation=None, name="fc8")(x)
        return x


class DLBDLiteBN(CityNetTF2):
    """DLBD Network with one less Dense layer to reduce the number of parameters and overfitting

    Args:
        CityNetTF2 ([type]): [description]

    Returns:
        [type]: [description]
    """

    NAME = "DLBD_lite"

    def get_base_layers(self, x=None):
        if x is None:
            x = self.get_preprocessing_layers()
        # * First block
        x = layers.Conv2D(
            self.opts.get("num_filters", 128),
            (
                self.opts["input_height"] - self.opts["wiggle_room"],
                self.opts["conv_filter_width"],
            ),
            bias_initializer=None,
            padding="valid",
            activation=None,
            kernel_regularizer=regularizers.l2(0.001),
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(
            alpha=1 / 3,
            name="conv1_1",
        )(x)
        # * Second block
        x = layers.Conv2D(
            self.opts.get("num_filters", 128),
            (1, 3),
            bias_initializer=None,
            padding="valid",
            activation=None,
            kernel_regularizer=regularizers.l2(0.001),
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(
            alpha=1 / 3,
            name="conv1_2",
        )(x)
        W = x.shape[2]
        x = layers.MaxPool2D(pool_size=(1, W), strides=(1, 1), name="pool2")(x)
        x = layers.BatchNormalization()(x)
        x = tf.transpose(x, (0, 3, 2, 1))
        x = layers.Flatten(name="pool2_flat")(x)
        x = layers.Dense(
            self.opts["num_dense_units"],
            activation=None,
            bias_initializer=None,
            kernel_regularizer=regularizers.l2(0.001),
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(alpha=1 / 3, name="fc6")(x)

        return x

    def get_top_layers(self, x=None):
        if x is None:
            x = self.get_base_layers()
        x = layers.Dense(2, activation=None, name="fc8")(x)
        return x


class DLBDLiteBNrelu(CityNetTF2):
    """DLBD Network with one less Dense layer to reduce the number of parameters and overfitting

    Args:
        CityNetTF2 ([type]): [description]

    Returns:
        [type]: [description]
    """

    NAME = "DLBD_lite"

    def get_base_layers(self, x=None):
        if x is None:
            x = self.get_preprocessing_layers()
        # * First block
        x = layers.Conv2D(
            self.opts.get("num_filters", 128),
            (
                self.opts["input_height"] - self.opts["wiggle_room"],
                self.opts["conv_filter_width"],
            ),
            bias_initializer=None,
            padding="valid",
            activation="relu",
            kernel_regularizer=regularizers.l2(0.001),
        )(x)
        x = layers.BatchNormalization()(x)
        # * Second block
        x = layers.Conv2D(
            self.opts.get("num_filters", 128),
            (1, 3),
            bias_initializer=None,
            padding="valid",
            activation="relu",
            kernel_regularizer=regularizers.l2(0.001),
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
            kernel_regularizer=regularizers.l2(0.001),
        )(x)
        x = layers.BatchNormalization()(x)

        return x

    def get_top_layers(self, x=None):
        if x is None:
            x = self.get_base_layers()
        x = layers.Dense(2, activation=None, name="fc8")(x)
        return x


class DLBDDenseBNRelu(CityNetTF2):
    """DLBD Network with one less Dense layer to reduce the number of parameters and overfitting

    Args:
        CityNetTF2 ([type]): [description]

    Returns:
        [type]: [description]
    """

    NAME = "DLBD_lite"

    def get_base_layers(self, x=None):
        if x is None:
            x = self.get_preprocessing_layers()
        # * First block
        x = layers.Conv2D(
            self.opts.get("num_filters", 128),
            (
                self.opts["input_height"] - self.opts["wiggle_room"],
                self.opts["conv_filter_width"],
            ),
            bias_initializer=None,
            padding="valid",
            activation="relu",
            kernel_regularizer=regularizers.l2(0.001),
        )(x)
        x = layers.BatchNormalization()(x)
        # * Second block
        x = layers.Conv2D(
            self.opts.get("num_filters", 128),
            (1, 3),
            bias_initializer=None,
            padding="valid",
            activation="relu",
            kernel_regularizer=regularizers.l2(0.001),
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
            kernel_regularizer=regularizers.l2(0.001),
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dense(
            self.opts["num_dense_units2"],
            activation="relu",
            bias_initializer=None,
            # kernel_regularizer=regularizers.l2(0.001),
        )(x)
        x = layers.BatchNormalization()(x)

        return x

    def get_top_layers(self, x=None):
        if x is None:
            x = self.get_base_layers()
        x = layers.Dense(2, activation=None, name="fc8")(x)
        return x


class DLBDLiteDrop(CityNetTF2):
    """DLBD Network with one less Dense layer to reduce the number of parameters and overfitting

    Args:
        CityNetTF2 ([type]): [description]

    Returns:
        [type]: [description]
    """

    NAME = "DLBD_lite"

    def get_base_layers(self, x=None):
        if x is None:
            x = self.get_preprocessing_layers()
        # * First block
        x = layers.Conv2D(
            self.opts.get("num_filters", 128),
            (
                self.opts["input_height"] - self.opts["wiggle_room"],
                self.opts["conv_filter_width"],
            ),
            bias_initializer=None,
            padding="valid",
            activation=None,
            kernel_regularizer=regularizers.l2(0.001),
        )(x)
        x = layers.LeakyReLU(
            alpha=1 / 3,
            name="conv1_1",
        )(x)
        # * Second block
        x = layers.Conv2D(
            self.opts.get("num_filters", 128),
            (1, 3),
            bias_initializer=None,
            padding="valid",
            activation=None,
            kernel_regularizer=regularizers.l2(0.001),
        )(x)
        x = layers.LeakyReLU(
            alpha=1 / 3,
            name="conv1_2",
        )(x)
        W = x.shape[2]
        x = layers.MaxPool2D(pool_size=(1, W), strides=(1, 1), name="pool2")(x)
        x = layers.Dropout(self.opts.get("dropout", 0.5))(x)
        x = tf.transpose(x, (0, 3, 2, 1))
        x = layers.Flatten(name="pool2_flat")(x)
        x = layers.Dense(
            self.opts["num_dense_units"],
            activation=None,
            bias_initializer=None,
            kernel_regularizer=regularizers.l2(0.001),
        )(x)
        x = layers.LeakyReLU(alpha=1 / 3, name="fc6")(x)
        x = layers.Dropout(self.opts.get("dropout", 0.5))(x)
        return x

    def get_top_layers(self, x=None):
        if x is None:
            x = self.get_base_layers()
        x = layers.Dense(2, activation=None, name="fc8")(x)
        return x


class DLBDDenseDropRelu(CityNetTF2):
    """DLBD Network with one less Dense layer to reduce the number of parameters and overfitting

    Args:
        CityNetTF2 ([type]): [description]

    Returns:
        [type]: [description]
    """

    NAME = "DLBD_lite"

    def get_base_layers(self, x=None):
        if x is None:
            x = self.get_preprocessing_layers()
        # * First block
        x = layers.Conv2D(
            self.opts.get("num_filters", 128),
            (
                self.opts["input_height"] - self.opts["wiggle_room"],
                self.opts["conv_filter_width"],
            ),
            bias_initializer=None,
            padding="valid",
            activation="relu",
            kernel_regularizer=regularizers.l2(0.001),
        )(x)
        # * Second block
        x = layers.Conv2D(
            self.opts.get("num_filters", 128),
            (1, 3),
            bias_initializer=None,
            padding="valid",
            activation="relu",
            kernel_regularizer=regularizers.l2(0.001),
        )(x)
        W = x.shape[2]
        x = layers.MaxPool2D(pool_size=(1, W), strides=(1, 1), name="pool2")(x)
        x = layers.Dropout(self.opts.get("dropout", 0.5))(x)
        x = tf.transpose(x, (0, 3, 2, 1))
        x = layers.Flatten(name="pool2_flat")(x)
        x = layers.Dense(
            self.opts["num_dense_units"],
            activation="relu",
            bias_initializer=None,
            kernel_regularizer=regularizers.l2(0.001),
        )(x)
        x = layers.Dropout(self.opts.get("dropout", 0.5))(x)
        x = layers.Dense(
            self.opts["num_dense_units2"],
            activation="relu",
            bias_initializer=None,
            kernel_regularizer=regularizers.l2(0.001),
        )(x)
        x = layers.Dropout(self.opts.get("dropout", 0.5))(x)

        return x

    def get_top_layers(self, x=None):
        if x is None:
            x = self.get_base_layers()
        x = layers.Dense(2, activation=None, name="fc8")(x)
        return x
