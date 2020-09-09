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


N_FFT = 2048
HOP_LENGTH = 1024  # 512
N_MELS = 32  # 128


class CityNetClassifier1(object):
    def count_number_trainable_params(self):
        """
        Counts the number of trainable variables.
        """
        tot_nb_params = 0
        for trainable_variable in tf.compat.v1.trainable_variables():
            shape = trainable_variable.get_shape()  # e.g [D,F] or [W,H,C]
            current_nb_params = self.get_nb_params_shape(shape)
            tot_nb_params = tot_nb_params + current_nb_params
        return tot_nb_params

    def get_nb_params_shape(self, shape):
        """
        Computes the total number of params for a given shap.
        Works for any number of shapes etc [D,F] or [W,H,C] computes D*F and W*H*C.
        """
        nb_params = 1
        for dim in shape:
            nb_params = nb_params * int(dim)
        return nb_params

    def __init__(self, opts, weights_path):
        """Create the layers of the neural network, with the same options we used in training"""
        self.opts = namedtuple("opts", opts.keys())(*opts.values())
        tf.compat.v1.reset_default_graph()
        tf.compat.v1.disable_eager_execution()

        # config = tf.ConfigProto(device_count={"GPU": 0}, log_device_placement=True)
        # self.sess = tf.Session(config=config)

        self.sess = tf.compat.v1.Session()
        self.sess.run(tf.compat.v1.global_variables_initializer())

        net_options = {xx: opts[xx] for xx in train_helpers.net_params}
        self.net = train_helpers.create_net(SPEC_HEIGHT=N_MELS, **net_options)
        # self.net = train_helpers.create_net(SPEC_HEIGHT=N_MELS, **net_options)
        # self.net.load_weights(weights_path)

        train_saver = tf.compat.v1.train.Saver()
        print("Loading from {}".format(weights_path))
        train_saver.restore(self.sess, weights_path)

        print("loaded")

        print(
            "Number of trainable parameters: %d" % self.count_number_trainable_params()
        )

        # Create an object which will iterate over the test spectrograms
        # appropriately
        self.test_sampler = train_helpers.SpecSampler(
            batch_size=256,
            hww_x=self.opts.HWW_X,
            hww_y=self.opts.HWW_Y,
            do_aug=False,
            learn_log=self.opts.LEARN_LOG,
            randomise=False,
            seed=10,
            balanced=False,
        )

    def __init__new(self, opts, weights_path):
        """Create the layers of the neural network, with the same options we used in training"""
        self.opts = namedtuple("opts", opts.keys())(*opts.values())
        # tf.compat.v1.reset_default_graph()

        # config = tf.ConfigProto(device_count={"GPU": 0}, log_device_placement=True)
        # self.sess = tf.Session(config=config)

        # self.sess = tf.compat.v1.Session()
        # self.sess.run(tf.compat.v1.global_variables_initializer())

        net_options = {xx: opts[xx] for xx in train_helpers.net_params}
        # self.net = train_helpers.create_net_old(SPEC_HEIGHT=N_MELS, **net_options)
        self.net = train_helpers.create_net(SPEC_HEIGHT=N_MELS, **net_options)
        # self.net.load_weights(weights_path)

        train_saver = tf.compat.v1.train.Saver()
        print("Loading from {}".format(weights_path))
        # train_saver.restore(self.sess, weights_path)

        # Create an object which will iterate over the test spectrograms
        # appropriately
        self.test_sampler = train_helpers.SpecSampler(
            batch_size=256,
            hww_x=self.opts.HWW_X,
            hww_y=self.opts.HWW_Y,
            do_aug=False,
            learn_log=self.opts.LEARN_LOG,
            randomise=False,
            seed=10,
            balanced=False,
        )

    def __exit__(self):
        # tear down the tensorflow session
        self.sess.close()

    def load_wav(self, wavpath, loadmethod="librosa"):
        # tic = time()

        if loadmethod == "librosa":
            # a more correct and robust way -
            # this resamples any audio file to 22050Hz
            # TODO: downsample if higher than 22050
            sample_rate = 22050 if self.opts.resample else None
            return librosa.load(wavpath, sr=sample_rate)
        elif loadmethod == "wavfile":
            # a hack for speed - resampling is done assuming raw audio is
            # sampled at 44100Hz. Not recommended for general use.
            sample_rate, wav = wavfile.read(open(wavpath))
            wav = wav[::2] / 32768.0
            sample_rate /= 2
            return (wav, sample_rate)
        else:
            raise Exception("Unknown load method")

    def compute_spec(self, wav, sample_rate):
        # tic = time()
        spec = melspectrogram(
            wav, sr=sample_rate, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS,
        )

        if self.opts.remove_noise:
            spec = Spectrogram.remove_noise(spec)

        spec = np.log(self.opts.A + self.opts.B * spec)
        spec = spec - np.median(spec, axis=1, keepdims=True)
        return spec.astype(np.float32)

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
            pred = self.sess.run(self.net["output"], feed_dict={self.net["input"]: Xb})
            probas.append(pred)
        # print("Took %0.3fs to classify" % (time() - tic))
        print("Classified {0} in {1}".format(wavpath, time() - tic))

        return (np.vstack(probas)[:, 1], sr)
