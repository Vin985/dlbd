import numpy as np
import tensorflow as tf
import tensorflow_io as tfio
from tensorflow_io.python.ops.audio_ops import freq_mask, time_mask


class NormalizeSpectrograms(tf.keras.layers.Layer):
    def __init__(self, learn_log, do_augmentation, **kwargs):
        super().__init__(**kwargs)
        self.learn_log = learn_log
        self.do_augmentation = do_augmentation

    @tf.function(
        experimental_relax_shapes=True
    )  # (input_signature=(tf.TensorSpec(shape=[32, 20], dtype=tf.float32)))
    def normalize(self, x, training=None):
        print("Tracing with:", x.shape)
        one = x
        if self.learn_log:
            spec = tf.stack([one, one, one, one])
        else:
            row_mean = tf.expand_dims(tf.math.reduce_mean(x, axis=1), 1)
            row_std = tf.expand_dims(tf.add(tf.math.reduce_std(x, axis=1), 0.001), 1)
            two = (one - row_mean) / row_std

            three = tf.math.divide(
                tf.math.subtract(x, tf.math.reduce_mean(x)),
                tf.math.reduce_std(x),
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
                if np.random.rand() > 0.9:
                    print("in random")
                    spec = tf.add(
                        spec, tf.multiply(tf.roll(spec, 1, axis=0), np.random.randn())
                    )
        spec = tf.transpose(spec, perm=[1, 2, 0])
        return spec

    @tf.function
    def vectorize(self, spec, training=None):
        return tf.vectorized_map(self.normalize, spec, training)

    def call(self, spec, training=None):
        res = self.vectorize(spec, training)
        return res

    def get_config(self):
        config = {
            "do_augmentation": self.do_augmentation,
            "learn_log": self.learn_log,
        }
        config.update(super().get_config())

        return config


class MaskSpectrograms(tf.keras.layers.Layer):
    """
    .. csv-table::
        :header: "Option name", "Description", "Default", "Type"


        "time_mask", "Options for the time masking augmentation", , "dict"
        "prop", "proportion of the time the masking will occur", 70, "int"
        "value", "",10, "int"
        "freq_mask", "Options for the frequency asking augmentation", , "dict"
        "prop", "proportion of the time the masking will occur", 70, "int"
        "value", "", 30 , "int"

    """

    def __init__(self, time_mask, freq_mask, **kwargs):
        super().__init__(**kwargs)
        self.time_mask = time_mask
        self.freq_mask = freq_mask

    @tf.function(experimental_relax_shapes=True)
    def mask(self, spec):
        if self.time_mask:
            if tf.random.uniform(  # pylint: disable=unexpected-keyword-arg
                shape=(), minval=1, maxval=100, dtype=tf.int32
            ) > self.time_mask.get("prop", 70):
                spec = tfio.audio.time_mask(spec, param=self.time_mask.get("value", 10))
        if self.freq_mask:
            if tf.random.uniform(  # pylint: disable=unexpected-keyword-arg
                shape=(), minval=1, maxval=100, dtype=tf.int32
            ) > self.freq_mask.get("prop", 70):
                spec = tfio.audio.freq_mask(spec, param=self.freq_mask.get("prop", 30))
        return spec

    @tf.function
    def vectorize(self, spec):
        return tf.map_fn(self.mask, spec)

    def call(self, spec, training=None):
        if training:
            spec = self.vectorize(spec)
        return spec

    def get_config(self):
        config = {
            "time_mask": self.time_mask,
            "freq_mask": self.freq_mask,
        }
        config.update(super().get_config())

        return config
