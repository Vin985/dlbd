import numpy as np
import tensorflow as tf
from tensorflow.keras import Input, Model, layers, regularizers

from dlbd.models.TF2Model import TF2Model

from ..training.spectrogram_sampler import SpectrogramSampler


class CityNetTF2(TF2Model):
    NAME = "CityNetTF2"

    def create_net(self):
        print("init_create_net")
        opts = self.opts["net"]
        inputs = Input(
            shape=(opts["spec_height"], opts["hww_x"] * 2, opts["channels"],),
            # batch_size=128,
            dtype=tf.float32,
        )
        # * First block
        x = layers.Conv2D(
            opts.get("num_filters", 128),
            (opts["spec_height"] - opts["wiggle_room"], opts["conv_filter_width"],),
            bias_initializer=None,
            padding="valid",
            activation=None,
            # name="conv1_1",
            kernel_regularizer=regularizers.l2(0.001),
        )(inputs)
        x = layers.LeakyReLU(alpha=1 / 3, name="conv1_1",)(x)
        # * Second block
        x = layers.Conv2D(
            opts.get("num_filters", 128),
            (1, 3),
            bias_initializer=None,
            padding="valid",
            activation=None,
            # name="conv1_2",
            kernel_regularizer=regularizers.l2(0.001),
        )(x)
        # x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(alpha=1 / 3, name="conv1_2",)(x)
        W = x.shape[2]
        x = layers.MaxPool2D(pool_size=(1, W), strides=(1, 1), name="pool2")(x)
        # x = layers.Dropout(0.5)(x)
        x = tf.transpose(x, (0, 3, 2, 1))
        x = layers.Flatten(name="pool2_flat")(x)
        x = layers.Dense(
            opts["num_dense_units"],
            activation=None,
            bias_initializer=None,
            kernel_regularizer=regularizers.l2(0.001),
        )(x)
        # x = layers.Dropout(self.opts["dropout"])(x)
        # x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(alpha=1 / 3, name="fc6")(x)
        # x = layers.Dropout(0.5)(x)
        x = layers.Dense(
            opts["num_dense_units"],
            activation=None,
            bias_initializer=None,
            # kernel_regularizer=regularizers.l2(0.001),
        )(x)
        # x = layers.Dropout(self.opts["dropout"])(x)
        # x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(alpha=1 / 3, name="fc7")(x)
        # x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(
            2, activation=None, name="fc8"  # kernel_regularizer=regularizers.l2(0.001),
        )(x)
        print("end_layers")
        model = Model(inputs, outputs, name=self.NAME)
        print("after model")
        model.summary()
        return model

    def init_metrics(self):
        """Inits the metrics used during model evaluation. Fills the metrics
        attribute which is a dict that should contain the following keys:

        - train_loss
        - train_accuracy
        - validation_loss
        - validation accuracy
        """
        # * Train functions
        self.metrics["train_loss"] = tf.keras.metrics.Mean(name="train_loss")
        self.metrics["train_accuracy"] = tf.keras.metrics.SparseCategoricalAccuracy(
            name="train_accuracy"
        )
        # * Validation functions
        self.metrics["validation_loss"] = tf.keras.metrics.Mean(name="validation_loss")
        self.metrics[
            "validation_accuracy"
        ] = tf.keras.metrics.SparseCategoricalAccuracy(name="validation_accuracy")

    @staticmethod
    def tf_loss(y_true, y_pred):
        return tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)
        )

    def init_samplers(self):
        train_sampler = SpectrogramSampler(self.opts, randomise=True, balanced=True)
        validation_sampler = SpectrogramSampler(
            self.opts, randomise=False, balanced=True
        )
        return train_sampler, validation_sampler

    def init_optimizer(self):
        self.optimizer = tf.keras.optimizers.Adam()

    def modify_spectrogram(self, spec):
        spec = np.log(self.opts["model"]["A"] + self.opts["model"]["B"] * spec)
        spec = spec - np.median(spec, axis=1, keepdims=True)
        return spec

    def prepare_data(self, data):
        specs, tags, infos = data
        if not self.opts["model"]["learn_log"]:
            specs = [self.modify_spectrogram(spec) for spec in specs]
        return specs, tags, infos

    def classify_spectrogram(self, spectrogram, spec_sampler):
        spectrogram = self.modify_spectrogram(spectrogram)
        return super().classify_spectrogram(spectrogram, spec_sampler)

    def predict(self, x):
        return tf.nn.softmax(self.model(x, training=False)).numpy()
