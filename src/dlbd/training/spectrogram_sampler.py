import numpy as np

# from tensorflow.contrib import slim
from . import minibatch_generators as mbg


# Which parameters are used in the network generation?


class SpectrogramSampler:
    def __init__(
        self, opts, randomise=False, seed=None, balanced=True,
    ):
        self.opts = opts
        self.seed = seed
        self.randomise = randomise
        self.balanced = balanced
        self.specs = None
        self.labels = None
        self.which_spec = None
        self.medians = None

    def __call__(self, X, y=None):

        # must pad X and Y the same amount
        pad_hww = max(self.opts["net"]["hww_x"], self.opts["net"]["hww_y"])

        blank_spec = np.zeros((X[0].shape[0], 2 * pad_hww))
        self.specs = np.hstack([blank_spec] + X + [blank_spec])[None, ...]
        # print(X[0].shape)
        # print(blank_spec.shape)
        # print(self.specs.shape)

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

        assert self.labels.shape[0] == self.specs.shape[2]
        return self

    def __iter__(self):  # , num_per_class, seed=None
        # num_samples = num_per_class * 2

        do_augmentation = self.opts["model"].get("do_augmentation", False)
        learn_log = self.opts["model"].get("learn_log", False)
        hww_x = self.opts["net"]["hww_x"]
        hww_y = self.opts["net"]["hww_y"]
        batch_size = self.opts["net"]["batch_size"]

        channels = self.specs.shape[0]
        if not learn_log:
            channels += 3
        height = self.specs.shape[1]

        if self.seed is not None:
            np.random.seed(self.seed)

        idxs = np.where(self.labels >= 0)[0]
        for sampled_locs, y in mbg.minibatch_iterator(
            idxs,
            self.labels[idxs],
            batch_size,
            randomise=self.randomise,
            balanced=self.balanced,
            class_size="smallest",
        ):

            # extract the specs
            # avoid using self.batch_size as last batch may be smaller
            bs = y.shape[0]
            X = np.zeros((bs, channels, height, hww_x * 2), np.float32)
            y = np.zeros(bs) * np.nan
            if learn_log:
                X_medians = np.zeros((bs, channels, height), np.float32)
            count = 0

            for loc in sampled_locs:
                which = self.which_spec[loc]

                X[count] = self.specs[:, :, (loc - hww_x) : (loc + hww_x)]

                if not learn_log:
                    X[count, 1] = X[count, 0] - self.medians[which][:, None]
                    # X[count, 0] = (X[count, 0] - X[count, 0].mean()) / X[count, 0].std()
                    X[count, 0] = (X[count, 1] - X[count, 1].mean(1, keepdims=True)) / (
                        X[count, 1].std(1, keepdims=True) + 0.001
                    )

                    X[count, 2] = (X[count, 1] - X[count, 1].mean()) / X[count, 1].std()
                    x1max = X[count, 1].max()
                    if x1max:
                        X[count, 3] = X[count, 1] / x1max
                    else:
                        X[count, 3] = X[count, 1]

                y[count] = self.labels[(loc - hww_y) : (loc + hww_y)].max()
                if learn_log:
                    which = self.which_spec[loc]
                    X_medians[count] = self.medians[which]

                count += 1

            # doing augmentation
            if do_augmentation:
                if learn_log:
                    mult = 1.0 + np.random.randn(bs, 1, 1, 1) * 0.1
                    mult = np.clip(mult, 0.1, 200)
                    X *= mult
                else:
                    X *= 1.0 + np.random.randn(bs, 1, 1, 1) * 0.1
                    X += np.random.randn(bs, 1, 1, 1) * 0.1
                    if np.random.rand() > 0.9:
                        X += np.roll(X, 1, axis=0) * np.random.randn()

            if learn_log:
                xb = {
                    "input": X.astype(np.float32),
                    "input_med": X_medians.astype(np.float32),
                }
                yield xb, y.astype(np.int32)

            else:
                yield X.astype(np.float32).transpose(0, 2, 3, 1), y.astype(np.int32)
