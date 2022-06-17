import copy
from abc import abstractmethod
from pathlib import Path

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from mouffet.utils import common_utils
from mouffet.models import DLModel


class TF2Model(DLModel):

    CALLBACKS_DEFAULTS = {"early_stopping": {}}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.optimizer = None
        self.summary_writer = {}
        self.metrics = {}
        self.callbacks = []
        self.logs = {}
        self.optimizer = None

    @property
    def n_parameters(self):
        if self.model:
            res = {}
            res["trainableParams"] = np.sum(
                [
                    np.prod(v.get_shape())
                    for v in self.model.trainable_weights  # pylint: disable=no-member
                ]
            )
            res["nonTrainableParams"] = np.sum(
                [
                    np.prod(v.get_shape())
                    for v in self.model.non_trainable_weights  # pylint: disable=no-member
                ]
            )
            res["totalParams"] = res["trainableParams"] + res["nonTrainableParams"]
            return str(res)
        else:
            return super().n_parameters

    def train_step(self, data, labels):
        self.callbacks.on_train_batch_begin(data, logs=self.logs)
        self.basic_step(data, labels, self.STEP_TRAINING)
        self.logs = {
            self.STEP_TRAINING
            + "_loss": self.metrics[self.STEP_TRAINING + "_loss"].result(),
            self.STEP_TRAINING
            + "accuracy": self.metrics[self.STEP_TRAINING + "_accuracy"].result(),
        }
        self.callbacks.on_train_batch_end(data, logs=self.logs)

    def validation_step(self, data, labels):
        self.callbacks.on_test_batch_begin(data, logs=self.logs)
        self.basic_step(data, labels, self.STEP_VALIDATION)
        self.logs = {
            self.STEP_VALIDATION
            + "_loss": self.metrics[self.STEP_VALIDATION + "_loss"].result(),
            self.STEP_VALIDATION
            + "accuracy": self.metrics[self.STEP_VALIDATION + "_accuracy"].result(),
        }
        self.callbacks.on_test_batch_end(data, logs=self.logs)

    @tf.function
    def basic_step(self, data, labels, step_type):
        training = step_type == self.STEP_TRAINING
        if training:
            with tf.GradientTape() as tape:
                predictions = self.model(data, training=True)
                loss = self.tf_loss(labels, predictions)
            gradients = tape.gradient(
                loss, self.model.trainable_variables  # pylint: disable=no-member
            )  # pylint: disable=no-member
            self.optimizer.apply_gradients(
                zip(
                    gradients, self.model.trainable_variables
                )  # pylint: disable=no-member
            )
        else:
            predictions = self.model(data, training=False)
            loss = self.tf_loss(labels, predictions)

        self.metrics[step_type + "_loss"].update_state(loss)
        self.metrics[step_type + "_accuracy"].update_state(labels, predictions)

    @staticmethod
    @abstractmethod
    def tf_loss(y_true, y_pred):
        """This is the loss function.

        Args:
            y_true ([type]): [description]
            y_pred ([type]): [description]
        """
        raise NotImplementedError()

    @abstractmethod
    def init_metrics(self):
        """Inits the metrics used during model evaluation. Fills the metrics
        attribute which is a dict that should contain the following keys:

        - train_loss
        - train_accuracy
        - validation_loss
        - validation accuracy
        """
        raise NotImplementedError()

    @abstractmethod
    def init_samplers(self, training_data, validation_data):
        raise NotImplementedError()

    @abstractmethod
    def init_optimizer(self, learning_rate=None):
        raise NotImplementedError()

    def add_callbacks(self):
        early_stopping = self.opts.get("early_stopping", {})
        if early_stopping:
            if isinstance(early_stopping, bool):
                early_stopping = {}
            opts = copy.deepcopy(self.CALLBACKS_DEFAULTS["early_stopping"])
            opts.update(early_stopping)
            self.callbacks.append(tf.keras.callbacks.EarlyStopping(**opts))
        # if self.opts.get("use_ma", False):
        #     self.callbacks.append(
        #         tfa.callbacks.AverageModelCheckpoint(
        #             filepath=str(self.opts.results_save_dir / self.opts.model_id),
        #             update_weights=False,
        #         )
        #     )

    def init_callbacks(self):
        self.add_callbacks()
        if isinstance(self.callbacks, list):
            self.callbacks = tf.keras.callbacks.CallbackList(
                self.callbacks, add_history=True, model=self.model
            )

    def init_training(self):
        """This is a function called at the beginning of the training step. In
        this function you should initialize your train and validation samplers,
        as well as the optimizer, loss and accuracy functions for training and
        validation.

        """
        self.opts.add_option("training", True)

        self.logs = {}

        self.init_model()

        self.init_optimizer()

        self.init_metrics()

        self.init_callbacks()

    def run_epoch(
        self,
        epoch,
        training_sampler,
        validation_sampler,
        epoch_save_step=None,
    ):
        # * Reset the metrics at the start of the next epoch
        self.callbacks.on_epoch_begin(epoch, logs=self.logs)
        self.reset_states()

        self.run_step("train", epoch, training_sampler)
        self.run_step("validation", epoch, validation_sampler)

        template = (
            "Epoch {}, Loss: {}, Accuracy: {},"
            " Validation Loss: {}, Validation Accuracy: {}"
        )
        print(
            template.format(
                epoch,
                self.metrics["train_loss"].result(),
                self.metrics["train_accuracy"].result() * 100,
                self.metrics["validation_loss"].result(),
                self.metrics["validation_accuracy"].result() * 100,
            )
        )

        if epoch_save_step is not None and epoch % epoch_save_step == 0:
            self.save_model(self.opts.get_intermediate_path(epoch))
        self.callbacks.on_epoch_end(epoch, logs=self.logs)

    def train(self, training_data, validation_data):

        self.init_training()

        self.callbacks.on_train_begin(logs=self.logs)

        print("Training model", self.opts.model_id)

        # early_stopping = self.opts.get("early_stopping", {})
        stop = False
        # if early_stopping:
        #     patience = early_stopping.get("patience", 3)
        #     count = 1

        training_stats = {}

        training_sampler, validation_sampler = self.init_samplers(
            training_data, validation_data
        )

        epoch_save_step = self.opts.get("epoch_save_step", None)

        epoch_batches = []

        # * Create logging writers
        self.create_writers()

        epoch_batches = self.get_epoch_batches()

        for epoch_batch in epoch_batches:
            lr = epoch_batch["learning_rate"]

            common_utils.print_info(
                (
                    "Starting new batch of epochs from epoch number {}, with learning rate {} for {} iterations"
                ).format(epoch_batch["start"], lr, epoch_batch["length"])
            )

            if epoch_batch.get("fine_tuning", False):
                print("Doing fine_tuning")
                self.set_fine_tuning()
                self.model.summary()  # pylint: disable=no-member

            self.init_optimizer(learning_rate=lr)
            for epoch in range(epoch_batch["start"], epoch_batch["end"] + 1):
                print("Running epoch ", epoch)
                self.run_epoch(
                    epoch,
                    training_sampler,
                    validation_sampler,
                    epoch_save_step,
                )
                # train_loss = self.metrics["train_loss"].result()
                # val_loss = self.metrics["validation_loss"].result()

                if self.model.stop_training:  # pylint: disable=no-member
                    stop = True
                    training_stats["stopped"] = epoch
                    common_utils.print_info(
                        "Early stopping: stopping at epoch {}".format(epoch)
                    )
                    break

                # diff = train_loss - val_loss

                # if diff <= 0:
                #     if not training_stats["crossed"]:
                #         training_stats["crossed"] = True
                #         training_stats["crossed_at"] = epoch
                #         self.save_model(self.opts.get_intermediate_path(epoch))

                #     if early_stopping:
                #         if count < patience:
                #             count += 1
                #         else:
                #             stop = True
                #             break
                # else:
                #     count = 0

            if stop:
                break
                # training_stats["train_loss"] = train_loss
                # training_stats["val_loss"] = val_loss

        # self.save_model()

        self.callbacks.on_train_end(logs=self.logs)

        return training_stats

    def create_writers(self):
        log_dir = Path(self.opts.logs["log_dir"]) / (
            self.opts.model_id + "_v" + str(self.opts.save_version)
        )

        # TODO: externalize logging directory
        train_log_dir = log_dir / "train"
        validation_log_dir = log_dir / "validation"
        self.summary_writer["train"] = tf.summary.create_file_writer(str(train_log_dir))
        self.summary_writer["validation"] = tf.summary.create_file_writer(
            str(validation_log_dir)
        )

    def reset_states(self):
        for x in ["train", "validation"]:
            self.metrics[x + "_loss"].reset_states()
            self.metrics[x + "_accuracy"].reset_states()

    def run_step(self, step_type, step, sampler):
        for data, labels in tqdm(sampler, ncols=50):
            self.callbacks.on_batch_begin(data, logs=self.logs)
            getattr(self, step_type + "_step")(data, labels)
            self.callbacks.on_batch_end(data, logs=self.logs)

        with self.summary_writer[step_type].as_default():
            tf.summary.scalar(
                "loss", self.metrics[step_type + "_loss"].result(), step=step
            )
            tf.summary.scalar(
                "accuracy", self.metrics[step_type + "_accuracy"].result(), step=step
            )

    def save_weights(self, path=None):
        if not path:
            path = str(self.opts.results_save_dir / self.opts.model_id)
        self.model.save_weights(path)  # pylint: disable=no-member

    def load_weights(self):
        print("Loading pre-trained weights")
        self.model.load_weights(  # pylint: disable=no-member
            self.opts.get_weights_path()
        )

    @abstractmethod
    def predict(self, x):
        """This function calls the model to have a predictions

        Args:
            x (data): The input data to be classified

            NotImplementedError: No basic implementation is provided and it should therefore be
            provided in child classes
        """
        raise NotImplementedError()
