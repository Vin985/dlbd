import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_addons as tfa
from mouffet.models import TF2Model
from tensorflow.keras import regularizers

from ..training import SpectrogramSampler
from .audio_dlmodel import AudioDLModel
from .layers import MaskSpectrograms, NormalizeSpectrograms


class AudioDetector(TF2Model, AudioDLModel):
    NAME = "AudioDetector"

    CALLBACKS_DEFAULTS = {
        "early_stopping": {
            "patience": 3,
            "monitor": "validation_loss",
            "restore_best_weights": True,
        }
    }

    def get_regularizer(self):
        regularizer = None
        if self.opts.get("regularizer", {}):
            reg_type = self.opts["regularizer"].get("type", "l2")
            reg_val = self.opts["regularizer"].get("value", 0.001)
            if reg_type == "l2":
                regularizer = regularizers.L2(reg_val)
            elif reg_type == "l1":
                regularizer = regularizers.L1(reg_val)
            elif reg_type == "l1_l2":
                regularizer = regularizers.L1L2(
                    l1=self.opts["regularizer"].get("l1", reg_val),
                    l2=self.opts["regularizer"].get("l2", reg_val),
                )
        return regularizer

    def get_dilation_rate(self, idx):
        dr = self.opts.get("dilation_rate", 0)
        if dr:
            if isinstance(dr, list):
                if idx <= len(dr):
                    val = dr[idx - 1]
                    if isinstance(val, list):
                        val = tuple(val)
                    return val
            elif isinstance(dr, int):
                return dr
        return 1

    def create_model(self):
        self.inputs = keras.Input(
            shape=(
                self.opts["input_height"],
                self.opts["input_width"],
            ),
            dtype=tf.float32,
        )
        if not self.opts.get("inference", False) and self.opts.get(
            "transfer_learning", False
        ):
            # * Load weights in training mode: do transfer learning and freeze base layers
            base_model = keras.Model(
                self.inputs, self.get_base_layers(), name=self.NAME + "_base"
            )
            base_model.trainable = False
            x = base_model(self.inputs, training=False)

        else:
            x = self.get_top_layers()

        model = keras.Model(self.inputs, x, name=self.NAME)
        model.summary()

        return model

    def get_base_layers(self, x=None):
        return x

    def get_top_layers(self, x=None):
        return x

    def load_weights(self):
        super().load_weights()

        if self.opts.get("transfer_learning", False):
            print("adding layers for transfer learning")
            model = self.get_top_layers(self.model(self.inputs))
            model = keras.Model(self.inputs, model, name=self.NAME)
            model.summary()
            self.model = model

    def get_preprocessing_layers(self, x=None):
        if not x:
            x = self.inputs
        time_mask = self.opts.get("time_mask", True)
        freq_mask = self.opts.get("freq_mask", True)
        x = MaskSpectrograms(time_mask=time_mask, freq_mask=freq_mask)(x)
        x = NormalizeSpectrograms(
            name="Normalize_spectrograms",
            learn_log=self.opts.get("learn_log", False),
            do_augmentation=self.opts.get("do_augmentation", False),
        )(x)
        return x

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

    def init_samplers(self, training_data, validation_data):
        train_sampler = SpectrogramSampler(
            self.opts,
            randomise=True,
            balanced=self.opts.get("training_balanced", True),
        )(self.get_raw_data(training_data), self.get_ground_truth(training_data))
        validation_sampler = SpectrogramSampler(
            self.opts, randomise=False, balanced=True
        )(self.get_raw_data(validation_data), self.get_ground_truth(validation_data))
        return train_sampler, validation_sampler

    def init_optimizer(self, learning_rate=0.01):
        if not self.optimizer:
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
            if self.opts.get("use_ma", False):
                print("using moving average")
                self.optimizer = tfa.optimizers.MovingAverage(self.optimizer)
            elif self.opts.get("use_swa", False):
                print("using stochastic average")
                self.optimizer = tfa.optimizers.SWA(self.optimizer)
        else:
            self.optimizer.lr.assign(learning_rate)

    def predict(self, x):
        return tf.nn.softmax(self.model(x, training=False)).numpy()

    def set_fine_tuning(self, layers=None, count=1):
        start_at = self.opts.get("fine_tuning", {}).get("start_at", 0)
        self.model.trainable = True
        layers = layers if layers else self.model.layers

        for layer in layers:
            if isinstance(layer, keras.Model):
                count = self.set_fine_tuning(layer.layers, count)
            elif isinstance(layer, keras.layers.BatchNormalization) or count < start_at:
                layer.trainable = False
            count += 1

        return count
