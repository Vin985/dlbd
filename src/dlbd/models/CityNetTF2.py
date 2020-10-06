import datetime
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras import Input, Model, layers, regularizers
from tqdm import tqdm

from ..training.spectrogram_sampler import SpectrogramSampler
from .dl_model import DLModel


class CityNetTF2(DLModel):
    NAME = "CityNetTF2"

    def __init__(self, opts=None):
        super().__init__(opts)
        self.optimizer = None
        self.loss = {}
        self.accuracy = {}
        self.summary_writer = {}

    def create_net(self):
        print("init_create_net")
        opts = self.opts["net"]
        inputs = Input(
            shape=(opts["spec_height"], opts["hww_x"] * 2, opts["channels"],),
            # batch_size=128,
            dtype=tf.float32,
        )
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
        x = layers.Conv2D(
            opts.get("num_filters", 128),
            (1, 3),
            bias_initializer=None,
            padding="valid",
            activation=None,
            # name="conv1_2",
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
        # x = layers.Dropout(self.opts["dropout"])(x)
        x = layers.LeakyReLU(alpha=1 / 3, name="fc6")(x)
        x = layers.Dense(
            opts["num_dense_units"],
            activation=None,
            bias_initializer=None,
            # kernel_regularizer=regularizers.l2(0.001),
        )(x)
        # x = layers.Dropout(self.opts["dropout"])(x)
        x = layers.LeakyReLU(alpha=1 / 3, name="fc7")(x)
        outputs = layers.Dense(
            2, activation=None, name="fc8"  # kernel_regularizer=regularizers.l2(0.001),
        )(x)
        print("end_layers")
        model = Model(inputs, outputs, name=self.NAME)
        print("after model")
        model.summary()
        return model

    @tf.function
    def train_step(self, data, labels):
        step = "train"
        with tf.GradientTape() as tape:
            # training=True is only needed if there are layers with different
            # behavior during training versus inference (e.g. Dropout).
            predictions = self.model(data, training=True)
            loss = self.tf_loss(labels, predictions)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        self.loss[step](loss)
        self.accuracy[step](labels, predictions)

    @tf.function
    def validation_step(self, data, labels):
        step = "validation"
        # training=False is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        predictions = self.model(data, training=False)
        t_loss = self.tf_loss(labels, predictions)

        self.loss[step](t_loss)
        self.accuracy[step](labels, predictions)

    @staticmethod
    def tf_loss(y_true, y_pred):
        return tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)
        )

    def modify_spectrogram(self, spec):
        spec = np.log(self.opts["model"]["A"] + self.opts["model"]["B"] * spec)
        spec = spec - np.median(spec, axis=1, keepdims=True)
        return spec

    def prepare_data(self, data):
        specs, tags = data
        if not self.opts["model"]["learn_log"]:
            specs = [self.modify_spectrogram(spec) for spec in specs]
        return specs, tags

    def train(self, training_data, validation_data):
        if not self.model:
            self.model = self.create_net()

        train_sampler = SpectrogramSampler(self.opts, randomise=True, balanced=True)
        validation_sampler = SpectrogramSampler(
            self.opts, randomise=False, balanced=True
        )

        # * Create logging writers
        self.create_writers()

        self.optimizer = tf.keras.optimizers.Adam()
        # * Train functions
        self.loss["train"] = tf.keras.metrics.Mean(name="train_loss")
        self.accuracy["train"] = tf.keras.metrics.SparseCategoricalAccuracy(
            name="train_accuracy"
        )
        # * Validation functions
        self.loss["validation"] = tf.keras.metrics.Mean(name="test_loss")
        self.accuracy["validation"] = tf.keras.metrics.SparseCategoricalAccuracy(
            name="validation_accuracy"
        )
        for epoch in range(self.opts["model"]["max_epochs"]):
            # Reset the metrics at the start of the next epoch
            # TODO: save intermediate models?
            self.reset_states()

            self.run_step("train", training_data, epoch, train_sampler)
            self.run_step("validation", validation_data, epoch, validation_sampler)

            template = "Epoch {}, Loss: {}, Accuracy: {}, Validation Loss: {}, Validation Accuracy: {}"
            print(
                template.format(
                    epoch + 1,
                    self.loss["train"].result(),
                    self.accuracy["train"].result() * 100,
                    self.loss["validation"].result(),
                    self.accuracy["validation"].result() * 100,
                )
            )
        self.save_model()

    def create_writers(self):
        # current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir = Path(self.opts["logs"]["log_dir"]) / self.model_name

        # TODO: externalize logging directory
        train_log_dir = log_dir / "train"
        validation_log_dir = log_dir / "validation"
        self.summary_writer["train"] = tf.summary.create_file_writer(str(train_log_dir))
        self.summary_writer["validation"] = tf.summary.create_file_writer(
            str(validation_log_dir)
        )

    def reset_states(self):
        for x in ["train", "validation"]:
            self.loss[x].reset_states()
            self.accuracy[x].reset_states()

    def run_step(self, step_type, data, step, sampler):
        for data, labels in tqdm(sampler(*data)):
            getattr(self, step_type + "_step")(data, labels)
        with self.summary_writer[step_type].as_default():
            tf.summary.scalar("loss", self.loss[step_type].result(), step=step)
            tf.summary.scalar("accuracy", self.accuracy[step_type].result(), step=step)

    def save_weights(self):
        self.model.save_weights(str(self.results_dir / self.model_name))

    def predict(self, x):
        return self.model.predict(x)
