import math
from random import randint

import numpy as np
from . import minibatch_generators as mbg


class SpectrogramSampler:
    def __init__(
        self,
        opts,
        randomise=False,
        seed=None,
        balanced=True,
    ):
        self.load_options(opts)
        # self.opts["seed"] = seed
        self.opts["randomise"] = randomise
        self.opts["balanced"] = balanced
        self.specs = None
        self.labels = None
        self.which_spec = None
        self.medians = None
        self.idxs = []
        self.tmp_idxs = []
        self.dims = (0, 0)
        if seed is not None:
            np.random.seed(seed)

    def load_options(self, opts):
        self.opts = {}
        self.opts["learn_log"] = opts.get("learn_log", False)
        # Half-width window for spectrograms
        self.opts["hww_spec"] = opts.get("hww_spec", math.ceil(opts["input_width"] / 2))
        # Half-width window for ground truth. Should ideally be the same as spectrograms
        self.opts["hww_gt"] = opts.get("hww_gt", self.opts["hww_spec"])
        self.opts["batch_size"] = opts["batch_size"]
        self.opts["overlap"] = opts.get("spectrogram_overlap", 0.75)
        self.opts["random_start"] = opts.get("random_start", False)
        self.opts["gt_prop"] = opts.get("gt_prop", 0)

    def __call__(self, X, y=None):
        """Call the spectrogram sampler

        Args:
            X (list of np.array): List of spectrograms
            y ([type], optional): [description]. Defaults to None.

        Returns:
            [type]: [description]
        """
        # must pad X and Y the same amount
        pad_hww = max(self.opts["hww_spec"], self.opts["hww_gt"])

        # * Create a blank spectrogram with same height as spectrograms and length 2 * padding
        blank_spec = np.zeros((X[0].shape[0], 2 * pad_hww))
        # * Stack all spectrograms together and pad with two blank spectrograms at the beginning and
        # * the end
        self.specs = np.hstack([blank_spec] + X + [blank_spec])

        blank_label = np.zeros(2 * pad_hww) - 1
        if y is not None:
            labels = [yy > 0 for yy in y]
        else:
            labels = [np.zeros(self.specs.shape[2] - 4 * pad_hww)]

        self.labels = np.hstack([blank_label] + labels + [blank_label])

        which_spec = [ii * np.ones(xx.shape[1]) for ii, xx in enumerate(X)]
        self.which_spec = np.hstack([blank_label] + which_spec + [blank_label]).astype(
            np.int32
        )

        self.medians = np.zeros((len(X), X[0].shape[0]))
        for idx, spec in enumerate(X):
            self.medians[idx] = np.median(spec, axis=1)

        self.dims = (self.specs.shape[0], self.opts["hww_spec"] * 2)
        # step = max(round((1 - self.opts["overlap"]) * self.dims[1]), 1)
        self.idxs = np.where(self.labels >= 0)[0]

        self.reset_idxs()

        assert self.labels.shape[0] == self.specs.shape[1]
        return self

    def reset_idxs(self):
        step = max(round((1 - self.opts["overlap"]) * self.dims[1]), 1)
        start = (
            randint(0, step)
            if self.opts["randomise"] and self.opts["random_start"]
            else 0
        )
        self.tmp_idxs = self.idxs[start::step]

    def __iter__(self):  # , num_per_class, seed=None
        # num_samples = num_per_class * 2

        for sampled_locs, y in mbg.minibatch_iterator(
            self.tmp_idxs,
            self.labels[self.tmp_idxs],
            self.opts["batch_size"],
            randomise=self.opts["randomise"],
            balanced=self.opts["balanced"],
            class_size="smallest",
        ):

            # extract the specs
            # avoid using self.batch_size as last batch may be smaller
            bs = y.shape[0]
            X = np.zeros((bs, self.dims[0], self.dims[1]), np.float32)
            y = np.zeros(bs) * np.nan
            if self.opts["learn_log"]:
                X_medians = np.zeros((bs, self.dims[0]), np.float32)

            for count, loc in enumerate(sampled_locs):
                which = self.which_spec[loc]

                X[count] = self.specs[
                    :, (loc - self.opts["hww_spec"]) : (loc + self.opts["hww_spec"])
                ]

                if not self.opts["learn_log"]:
                    X[count] = X[count] - self.medians[which][:, None]

                # TODO: Change the way labels are decided?
                y[count] = int(
                    self.labels[
                        (loc - self.opts["hww_gt"]) : (loc + self.opts["hww_gt"])
                    ].mean()
                    > self.opts["gt_prop"]
                )

                if self.opts["learn_log"]:
                    which = self.which_spec[loc]
                    X_medians[count] = self.medians[which]

            if self.opts["learn_log"]:
                xb = {
                    "input": X.astype(np.float32),
                    "input_med": X_medians.astype(np.float32),
                }
                yield xb, y.astype(np.int32)

            else:
                yield X.astype(np.float32), y.astype(np.int32)
        self.reset_idxs()

    def __len__(self):
        if len(self.tmp_idxs):
            class_size = "smallest" if self.opts["balanced"] else "largest"
            return int(
                np.ceil(
                    float(
                        mbg.get_class_size(self.labels[self.tmp_idxs], class_size)
                        * np.unique(self.labels[self.tmp_idxs]).shape[0]
                    )
                    / float(self.opts["batch_size"])
                )
            )
        return 0

    # def reset_sampler(self):
    #     start = randint(0, step) if self.opts["random_start"] else 0
    #     self.tmp_idxs = self.idxs[start::step]

    #     self.iterator = mbg.minibatch_iterator(
    #         self.tmp_idxs,
    #         self.labels[self.tmp_idxs],
    #         self.opts["batch_size"],
    #         randomise=self.opts["randomise"],
    #         balanced=self.opts["balanced"],
    #         class_size="smallest",
    #     )

    # def test(self):  # , num_per_class, seed=None
    #     # num_samples = num_per_class * 2
    #     self.reset_sampler()
    #     while True:
    #         try:
    #             sampled_locs, y = next(self.iterator)
    #             # extract the specs
    #             # avoid using self.batch_size as last batch may be smaller
    #             bs = y.shape[0]
    #             X = np.zeros((bs, self.dims[0], self.dims[1]), np.float32)
    #             y = np.zeros(bs) * np.nan
    #             if self.opts["learn_log"]:
    #                 X_medians = np.zeros((bs, self.dims[0]), np.float32)

    #             for count, loc in enumerate(sampled_locs):
    #                 which = self.which_spec[loc]

    #                 X[count] = self.specs[
    #                     :, (loc - self.opts["hww_spec"]) : (loc + self.opts["hww_spec"])
    #                 ]

    #                 if not self.opts["learn_log"]:
    #                     X[count] = X[count] - self.medians[which][:, None]

    #                 # TODO: Change the way labels are decided?
    #                 y[count] = self.labels[
    #                     (loc - self.opts["hww_gt"]) : (loc + self.opts["hww_gt"])
    #                 ].max()

    #                 if self.opts["learn_log"]:
    #                     which = self.which_spec[loc]
    #                     X_medians[count] = self.medians[which]

    #             if self.opts["learn_log"]:
    #                 xb = {
    #                     "input": X.astype(np.float32),
    #                     "input_med": X_medians.astype(np.float32),
    #                 }
    #                 yield xb, y.astype(np.int32)

    #             else:
    #                 yield X.astype(np.float32), y.astype(np.int32)
    #         except StopIteration:
    #             self.reset_sampler()

    # self.reset_sampler()
