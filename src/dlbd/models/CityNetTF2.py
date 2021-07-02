import numpy as np
import tensorflow as tf
from mouffet.models.TF2Model import TF2Model
from tensorflow.keras import Input, Model, layers, regularizers

from ..training.spectrogram_sampler import SpectrogramSampler
from .audio_dlmodel import AudioDLModel


class NormalizeSpectrograms(tf.keras.layers.Layer):
    """Compute log-magnitude mel-scaled spectrograms."""

    def __init__(self, learn_log, do_augmentation, **kwargs):
        super().__init__(**kwargs)
        self.learn_log = learn_log
        self.do_augmentation = do_augmentation

    def build(self, input_shape):
        # self.non_trainable_weights.append(self.mel_filterbank)
        super().build(input_shape)

    @tf.function  # (input_signature=(tf.TensorSpec(shape=[32, 20], dtype=tf.float32)))
    def normalize(self, x):
        print("Tracing with:", x.shape)
        one = x
        if self.learn_log:
            spec = tf.stack([one, one, one, one])
        else:
            row_mean = tf.expand_dims(tf.math.reduce_mean(x, axis=1), 1)
            row_std = tf.expand_dims(tf.add(tf.math.reduce_std(x, axis=1), 0.001), 1)
            two = (one - row_mean) / row_std

            three = tf.math.divide(
                tf.math.subtract(x, tf.math.reduce_mean(x)), tf.math.reduce_std(x),
            )
            four = tf.math.divide_no_nan(x, tf.math.reduce_max(x))
            spec = tf.stack([one, two, three, four])
        if self.do_augmentation:
            if self.learn_log:
                mult = 1.0 + np.random.randn(1, 1, 1) * 0.1
                mult = np.clip(mult, 0.1, 200)
                spec *= mult
            else:
                spec = tf.math.multiply(spec, 1.0 + np.random.randn(1, 1, 1) * 0.1)
                spec = tf.add(spec, np.random.randn(1, 1, 1) * 0.1)
                # if np.random.rand() > 0.9:
                #     print("in random")
                #     spec = tf.add(
                #         spec, tf.multiply(tf.roll(spec, 1, axis=0), np.random.randn())
                #     )
        spec = tf.transpose(spec, perm=[1, 2, 0])
        return spec

    @tf.function
    def vectorize(self, spec):
        return tf.vectorized_map(self.normalize, spec)

    def call(self, spec):
        res = self.vectorize(spec)
        return res

    def get_config(self):
        config = {
            "do_augmentation": self.do_augmentation,
            "learn_log": self.learn_log,
        }
        config.update(super().get_config())

        return config


class CityNetTF2(TF2Model, AudioDLModel):
    NAME = "CityNetTF2"

    def create_net(self):
        print("init_create_net")
        opts = self.opts["net"]
        inputs = Input(
            shape=(opts["spec_height"], opts["hww_x"] * 2,), dtype=tf.float32,
        )
        x = NormalizeSpectrograms(
            learn_log=self.opts["model"].get("learn_log", False),
            do_augmentation=self.opts["model"].get("do_augmentation", False),
        )(inputs)
        # * First block
        x = layers.Conv2D(
            opts.get("num_filters", 128),
            (opts["spec_height"] - opts["wiggle_room"], opts["conv_filter_width"],),
            bias_initializer=None,
            padding="valid",
            activation=None,
            kernel_regularizer=regularizers.l2(0.001),
        )(x)
        x = layers.LeakyReLU(alpha=1 / 3, name="conv1_1",)(x)
        # * Second block
        x = layers.Conv2D(
            opts.get("num_filters", 128),
            (1, 3),
            bias_initializer=None,
            padding="valid",
            activation=None,
            kernel_regularizer=regularizers.l2(0.001),
        )(x)
        x = layers.LeakyReLU(alpha=1 / 3, name="conv1_2",)(x)
        W = x.shape[2]
        x = layers.MaxPool2D(pool_size=(1, W), strides=(1, 1), name="pool2")(x)
        x = tf.transpose(x, (0, 3, 2, 1))
        x = layers.Flatten(name="pool2_flat")(x)
        x = layers.Dense(
            opts["num_dense_units"],
            activation=None,
            bias_initializer=None,
            kernel_regularizer=regularizers.l2(0.001),
        )(x)
        x = layers.LeakyReLU(alpha=1 / 3, name="fc6")(x)
        x = layers.Dense(
            opts["num_dense_units"], activation=None, bias_initializer=None,
        )(x)
        x = layers.LeakyReLU(alpha=1 / 3, name="fc7")(x)
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
        train_sampler = SpectrogramSampler(
            self.opts,
            randomise=True,
            balanced=self.opts["model"].get("training_balanced", True),
        )
        validation_sampler = SpectrogramSampler(
            self.opts, randomise=False, balanced=True
        )
        return train_sampler, validation_sampler

    def init_optimizer(self):
        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.opts["model"]["learning_rate"]
        )

    def predict(self, x):
        return tf.nn.softmax(self.model(x, training=False)).numpy()
