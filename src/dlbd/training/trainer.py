from pathlib import Path

import librosa
import numpy as np
import tensorflow as tf
import tf_slim as slim
import yaml
from scipy.ndimage.interpolation import zoom
from tqdm import tqdm
import random
import datetime

from ..data.data_handler import DataHandler
from lib.train_helpers import SpecSampler, create_net


class Trainer:
    def __init__(self, opts, model=None):
        self.opts = opts
        self.model = model

        self.data_handler = DataHandler(opts)

    def train_and_test(
        self, train_X, test_X, train_y, test_y, val_X=None, val_y=None,
    ):
        """
        Doesn't do any data loading - assumes the train and test data are passed
        in as parameters!
        """
        if val_X is None:
            n_val = int(self.opts.get("val_prop", 0.2) * len(train_X))
            print(n_val)
            print(len(train_X))
            val_idx = random.sample(range(0, len(train_X)), n_val)
            tmp_train_x, val_X, tmp_train_y, val_y = [], [], [], []
            for i in range(0, len(train_X)):
                if i in val_idx:
                    val_X.append(train_X[i])
                    val_y.append(train_y[i])
                else:
                    tmp_train_x.append(train_X[i])
                    tmp_train_y.append(train_y[i])
            train_X = tmp_train_x
            train_y = tmp_train_y
            print(len(train_X))
            print(len(val_X))

        print("in train and test")
        tf.compat.v1.disable_eager_execution()

        # # creaging samplers and batch iterators
        train_sampler = SpecSampler(
            128,
            self.opts["HWW_X"],
            self.opts["HWW_Y"],
            self.opts["do_augmentation"],
            self.opts["learn_log"],
            randomise=True,
            balanced=True,
        )
        test_sampler = SpecSampler(
            128,
            self.opts["HWW_X"],
            self.opts["HWW_Y"],
            False,
            self.opts["learn_log"],
            randomise=False,
            seed=10,
            balanced=True,
        )

        height = train_X[0].shape[0]
        net = create_net(
            height,
            self.opts["HWW_X"],
            self.opts["HWW_Y"],
            self.opts["num_filters"],
            self.opts["wiggle_room"],
            self.opts["conv_filter_width"],
            self.opts["num_dense_units"],
            self.opts["do_batch_norm"],
        )

        y_in = tf.compat.v1.placeholder(tf.int32, (None))
        x_in = net["input"]

        print("todo - fix this up...")
        trn_output = net["fc8"]
        test_output = net["fc8"]

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

        saver = tf.compat.v1.train.Saver(max_to_keep=5)

        with tf.compat.v1.Session() as sess:

            sess.run(tf.compat.v1.global_variables_initializer())

            for epoch in range(self.opts["max_epochs"]):

                print("training epoch #", epoch)

                ######################
                # TRAINING
                trn_losses = []
                trn_accs = []

                for xx, yy in tqdm(train_sampler(train_X, train_y)):
                    trn_ls, trn_acc, _ = sess.run(
                        [_trn_loss, _trn_acc, train_op], feed_dict={x_in: xx, y_in: yy}
                    )
                    trn_losses.append(trn_ls)
                    trn_accs.append(trn_acc)

                ######################
                # VALIDATION
                val_losses = []
                val_accs = []

                for xx, yy in test_sampler(test_X, test_y):
                    val_ls, val_acc = sess.run(
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

            #######################
            # TESTING
            # results_savedir = self.force_make_dir(
            #     self.paths["logging_dir"] + "results/"
            # )
            # predictions_savedir = self.force_make_dir(
            #     self.paths["logging_dir"] + "per_file_predictions/"
            # )

            # test_sampler = SpecSampler(
            #     128,
            #     self.opts["HWW_X"],
            #     self.opts["HWW_Y"],
            #     False,
            #     self.opts["learn_log"],
            #     randomise=False,
            #     seed=10,
            #     balanced=False,
            # )
            # for fname, spec, y in zip(test_files, test_X, test_y):
            #     probas = []
            #     y_true = []
            #     for Xb, yb in test_sampler([spec], [y]):
            #         preds = sess.run(test_output, feed_dict={x_in: Xb})
            #         probas.append(preds)
            #         y_true.append(yb)

            #     y_pred_prob = np.vstack(probas)
            #     y_true = np.hstack(y_true)
            #     y_pred = np.argmax(y_pred_prob, axis=1)

            #     print("Saving to {}".format(predictions_savedir))
            #     with open(predictions_savedir + fname, "wb") as f:
            #         pickle.dump([y_true, y_pred_prob], f, -1)

            # save weights from network
            saver.save(
                sess, self.paths["results_dir"] + self.opts["model_name"], global_step=1
            )

    def train(self):

        # X: spectrograms, y: labels
        train_X, train_y = self.load_data("train")
        test_X, test_y = self.load_data("test")

        print("data_loaded")

        for idx in range(self.opts["ensemble_members"]):
            print("train ensemble: ", idx)
            self.paths["results_dir"] = (
                self.opts["model_dir"] + self.opts["model_name"] + "/"
            )
            self.force_make_dir(self.paths["results_dir"])
            # sys.stdout = ui.Logger(logging_dir + "log.txt")

            with open(self.paths["results_dir"] + "network_opts.yaml", "w") as f:
                yaml.dump(self.opts, f, default_flow_style=False)

            self.train_and_test(
                train_X, test_X, train_y, test_y,
            )

    # def train_model(self):
    #     if not self.model:
    #         raise AttributeError("No model found")
    #     # X: spectrograms, y: labels
    #     train_X, train_y = self.load_data("train")
    #     # test_X, test_y = self.load_data("test")

    #     print("data_loaded")

    #     # test_files = os.listdir(self.paths["test_spec_dir"])

    #     self.paths["results_dir"] = (
    #         self.opts["model_dir"] + self.opts["model_name"] + "/"
    #     )
    #     self.force_make_dir(self.paths["results_dir"])
    #     # sys.stdout = ui.Logger(logging_dir + "log.txt")

    #     with open(self.paths["results_dir"] + "network_opts.yaml", "w") as f:
    #         yaml.dump(self.opts, f, default_flow_style=False)
    #     model = self.model.net

    #     def tf_loss(y_true, y_pred):
    #         return tf.reduce_mean(
    #             tf.nn.sparse_softmax_cross_entropy_with_logits(
    #                 labels=y_true, logits=y_pred
    #             )
    #         )

    #     log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    #     tensorboard_callback = tf.keras.callbacks.TensorBoard(
    #         log_dir=log_dir, histogram_freq=1
    #     )

    #     train_sampler = SpecSampler(
    #         128,
    #         self.opts["HWW_X"],
    #         self.opts["HWW_Y"],
    #         self.opts["do_augmentation"],
    #         self.opts["learn_log"],
    #         randomise=True,
    #         balanced=True,
    #     )

    #     model.compile(
    #         optimizer="adam", loss=tf_loss, metrics=["accuracy"],
    #     )

    #     model.fit(
    #         train_sampler(train_X, train_y),
    #         epochs=10,
    #         callbacks=[tensorboard_callback],
    #     )

    def train_model2(self):
        if not self.model:
            raise AttributeError("No model found")
        self.model.train(self.load_data("train"), self.load_data("test"))
