import datetime
import random

import tensorflow as tf
from tensorflow.keras import Input, Model, layers, regularizers
from tqdm import tqdm

from analysis.detection.models.dl_model import DLModel
from analysis.detection.lib.train_helpers import SpecSampler


import sys
from collections import namedtuple
from time import time

import librosa
import numpy as np
import tensorflow as tf
from librosa.feature import melspectrogram
from scipy.io import wavfile

from analysis.detection.lib import train_helpers
from analysis.spectrogram import Spectrogram


class CityNetTF2(DLModel):
    NAME = "CityNetTF2"

    def create_net(self):
        print("init_create_net")
        inputs = Input(
            shape=(
                self.opts["spec_height"],
                self.opts["hww_x"] * 2,
                self.opts["channels"],
            ),
            # batch_size=128,
            dtype=tf.float32,
        )
        x = layers.Conv2D(
            self.opts.get("num_filters", 128),
            (
                self.opts["spec_height"] - self.opts["wiggle_room"],
                self.opts["conv_filter_width"],
            ),
            bias_initializer=None,
            padding="valid",
            activation=None,
            # name="conv1_1",
            kernel_regularizer=regularizers.l2(0.001),
        )(inputs)
        x = layers.LeakyReLU(alpha=1 / 3, name="conv1_1",)(x)
        x = layers.Conv2D(
            self.opts.get("num_filters", 128),
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
            self.opts["num_dense_units"],
            activation=None,
            bias_initializer=None,
            kernel_regularizer=regularizers.l2(0.001),
        )(x)
        # x = layers.Dropout(self.opts["dropout"])(x)
        x = layers.LeakyReLU(alpha=1 / 3, name="fc6")(x)
        x = layers.Dense(
            self.opts["num_dense_units"],
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
        self.model = model
        return model

    @tf.function
    def train_step(self, data, labels):
        with tf.GradientTape() as tape:
            # training=True is only needed if there are layers with different
            # behavior during training versus inference (e.g. Dropout).
            predictions = self.model(data, training=True)
            loss = self.tf_loss(labels, predictions)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        self.train_loss(loss)
        self.train_accuracy(labels, predictions)

    @tf.function
    def test_step(self, data, labels):
        # training=False is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        predictions = self.model(data, training=False)
        t_loss = self.tf_loss(labels, predictions)

        self.test_loss(t_loss)
        self.test_accuracy(labels, predictions)

    @staticmethod
    def tf_loss(y_true, y_pred):
        return tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)
        )

    def train(self, train_data, validation_data=None):
        if not self.model:
            self.model = self.create_net()

        # X: spectrograms, y: labels
        train_x, train_y = train_data
        val_x, val_y = validation_data

        train_sampler = SpecSampler(
            128,
            self.opts["HWW_X"],
            self.opts["HWW_Y"],
            self.opts["do_augmentation"],
            self.opts["learn_log"],
            randomise=True,
            balanced=True,
        )

        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = "logs/gradient_tape/" + current_time + "/train"
        test_log_dir = "logs/gradient_tape/" + current_time + "/test"
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        test_summary_writer = tf.summary.create_file_writer(test_log_dir)

        self.optimizer = tf.keras.optimizers.Adam()
        self.train_loss = tf.keras.metrics.Mean(name="train_loss")
        self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
            name="train_accuracy"
        )
        # tf.keras.metrics.BinaryAccuracy(
        #     name="train_accuracy", threshold=0.8
        # )
        self.test_loss = tf.keras.metrics.Mean(name="test_loss")
        self.test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
            name="test_accuracy"
        )
        # tf.keras.metrics.BinaryAccuracy(
        #     name="test_accuracy", threshold=0.8
        # )
        self.save_params()
        for epoch in range(self.opts["max_epochs"]):
            # Reset the metrics at the start of the next epoch
            self.train_loss.reset_states()
            self.train_accuracy.reset_states()
            self.test_loss.reset_states()
            self.test_accuracy.reset_states()

            for data, labels in tqdm(train_sampler(train_x, train_y)):
                self.train_step(data, labels)
            with train_summary_writer.as_default():
                tf.summary.scalar("loss", self.train_loss.result(), step=epoch)
                tf.summary.scalar("accuracy", self.train_accuracy.result(), step=epoch)

            for test_data, test_labels in tqdm(train_sampler(val_x, val_y)):
                self.test_step(test_data, test_labels)
            with test_summary_writer.as_default():
                tf.summary.scalar("loss", self.test_loss.result(), step=epoch)
                tf.summary.scalar("accuracy", self.test_accuracy.result(), step=epoch)

            template = (
                "Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}"
            )
            print(
                template.format(
                    epoch + 1,
                    self.train_loss.result(),
                    self.train_accuracy.result() * 100,
                    self.test_loss.result(),
                    self.test_accuracy.result() * 100,
                )
            )
        self.model.save_weights(self.results_dir)

    def classify(self, wavpath=None):
        """Apply the classifier"""
        tic = time()

        if wavpath is not None:
            wav, sr = self.load_wav(wavpath, loadmethod="librosa")
            spec = self.compute_spec(wav, sr)

        labels = np.zeros(spec.shape[1])
        # print("Took %0.3fs to load" % (time() - tic))
        tic = time()
        probas = []
        for Xb, _ in self.test_sampler([spec], [labels]):
            pred = self.model.predict(Xb)
            probas.append(pred)
        # print("Took %0.3fs to classify" % (time() - tic))
        print("Classified {0} in {1}".format(wavpath, time() - tic))

        return (np.vstack(probas)[:, 1], sr)

