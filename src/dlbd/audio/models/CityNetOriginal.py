import collections

import numpy as np
import tensorflow as tf
import tf_slim as slim
from dlbd.audio.models.audio_dlmodel import AudioDLModel
from tqdm import tqdm

from ..training.spectrogram_sampler import SpectrogramSampler


class CityNetOriginal(AudioDLModel):
    NAME = "CityNetOriginal"

    def __init__(self, *args, **kwargs):
        tf.compat.v1.disable_eager_execution()
        self._session = None
        super().__init__(*args, **kwargs)

    @property
    def session(self):
        if not self._session:
            self._session = tf.compat.v1.Session()
        return self._session

    def create_net(self):
        opts = self.opts["net"]
        channels = opts["channels"]
        net = collections.OrderedDict()

        net["input"] = tf.compat.v1.placeholder(
            tf.float32,
            (None, opts["spec_height"], opts["hww_x"] * 2, channels),
            name="input",
        )
        net["conv1_1"] = slim.conv2d(
            net["input"],
            opts["num_filters"],
            (opts["spec_height"] - opts["wiggle_room"], opts["conv_filter_width"]),
            padding="valid",
            activation_fn=None,
            biases_initializer=None,
        )
        net["conv1_1"] = tf.nn.leaky_relu(net["conv1_1"], alpha=1 / 3)

        net["conv1_2"] = slim.conv2d(
            net["conv1_1"],
            opts["num_filters"],
            (1, 3),
            padding="valid",
            activation_fn=None,
            biases_initializer=None,
        )
        net["conv1_2"] = tf.nn.leaky_relu(net["conv1_2"], alpha=1 / 3)

        W = net["conv1_2"].shape[2]
        net["pool2"] = slim.max_pool2d(
            net["conv1_2"], kernel_size=(1, W), stride=(1, 1),
        )

        net["pool2"] = tf.transpose(net["pool2"], (0, 3, 2, 1))
        net["pool2_flat"] = slim.flatten(net["pool2"])

        net["fc6"] = slim.fully_connected(
            net["pool2_flat"],
            opts["num_dense_units"],
            activation_fn=None,
            biases_initializer=None,
        )
        net["fc6"] = tf.nn.dropout(net["fc6"], 0.5)
        net["fc6"] = tf.nn.leaky_relu(net["fc6"], alpha=1 / 3)

        net["fc7"] = slim.fully_connected(
            net["fc6"],
            opts["num_dense_units"],
            activation_fn=None,
            biases_initializer=None,
        )
        net["fc7"] = tf.nn.dropout(net["fc7"], 0.5)
        net["fc7"] = tf.nn.leaky_relu(net["fc7"], alpha=1 / 3)

        net["fc8"] = slim.fully_connected(net["fc7"], 2, activation_fn=None)
        # net['fc8'] = tf.nn.leaky_relu(net['fc8'], alpha=1/3)
        net["output"] = tf.nn.softmax(net["fc8"])
        return net

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

        train_sampler = SpectrogramSampler(self.opts, randomise=True, balanced=True)
        validation_sampler = SpectrogramSampler(
            self.opts, randomise=False, balanced=True
        )

        y_in = tf.compat.v1.placeholder(tf.int32, (None))
        x_in = self.model["input"]

        print("todo - fix this up...")
        trn_output = self.model["fc8"]
        test_output = self.model["fc8"]

        _trn_loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=trn_output, labels=y_in
            )
        )
        _test_loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=test_output, labels=y_in
            )
        )
        print(y_in, trn_output, tf.argmax(trn_output, axis=1))

        pred = tf.cast(tf.argmax(trn_output, axis=1), tf.int32)
        _trn_acc = tf.reduce_mean(tf.cast(tf.equal(y_in, pred), tf.float32))

        pred = tf.cast(tf.argmax(test_output, axis=1), tf.int32)
        _test_acc = tf.reduce_mean(tf.cast(tf.equal(y_in, pred), tf.float32))

        optimizer = tf.compat.v1.train.AdamOptimizer(
            learning_rate=self.opts["learning_rate"], beta1=0.5, beta2=0.9
        )

        train_op = slim.learning.create_train_op(_trn_loss, optimizer)

        self.session.run(tf.compat.v1.global_variables_initializer())

        for epoch in range(self.opts["max_epochs"]):

            print("training epoch #", epoch)

            ######################
            # TRAINING
            trn_losses = []
            trn_accs = []

            for xx, yy in tqdm(
                train_sampler(
                    self.get_raw_data(training_data),
                    self.get_ground_truth(training_data),
                )
            ):
                trn_ls, trn_acc, _ = self.session.run(
                    [_trn_loss, _trn_acc, train_op], feed_dict={x_in: xx, y_in: yy}
                )
                trn_losses.append(trn_ls)
                trn_accs.append(trn_acc)

            ######################
            # VALIDATION
            val_losses = []
            val_accs = []

            for xx, yy in tqdm(
                validation_sampler(
                    self.get_raw_data(validation_data),
                    self.get_ground_truth(validation_data),
                )
            ):
                val_ls, val_acc = self.session.run(
                    [_test_loss, _test_acc], feed_dict={x_in: xx, y_in: yy}
                )
                val_losses.append(val_ls)
                val_accs.append(val_acc)

            print(
                " %03d :: %02f  -  %02f  -  %02f  -  %02f"
                % (
                    epoch,
                    np.mean(trn_losses),
                    np.mean(trn_accs),
                    np.mean(val_losses),
                    np.mean(val_accs),
                )
            )
        self.save_model()

    def save_weights(self, path=None):
        saver = tf.compat.v1.train.Saver(max_to_keep=5)
        saver.save(
            self.session, str(self.opts.results_dir / self.opts.name), global_step=1
        )

    def load_weights(self, path=None):
        if not path:
            path = str(self.opts.results_dir / self.opts.name)
        print(path)
        saver = tf.compat.v1.train.Saver()
        saver.restore(self.session, path)

    def classify(self, data, sampler):
        spectrogram, _ = data
        spectrogram = self.modify_spectrogram(spectrogram)
        return super().classify_spectrogram(spectrogram, sampler)

    def predict(self, x):
        return self.session.run(
            self.model["output"], feed_dict={self.model["input"]: x}
        )

