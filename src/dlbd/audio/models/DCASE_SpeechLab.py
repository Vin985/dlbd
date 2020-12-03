import tensorflow as tf
from tensorflow.keras import Model, layers, regularizers, Input

from analysis.detection.models.dl_model import DLModel


class DCASESpeechLab(DLModel):
    NAME = "DCASESpeechLab"

    def create_net(self):
        print("init_create_net")
        inputs = Input(
            shape=(
                # self.opts["spec_height"],
                # self.opts["hww_x"] * 2,
                # self.opts["channels"],
                700,
                80,
                1,
            ),
            dtype=tf.float32,
        )

        x = layers.Conv2D(16, (3, 3), padding="valid", name="conv1_1",)(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(alpha=0.001)(x)
        x = layers.MaxPooling2D(pool_size=(3, 3))(x)
        x = layers.Conv2D(16, (3, 3), padding="valid")(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(alpha=0.001)(x)
        x = layers.MaxPooling2D(pool_size=(3, 3))(x)
        x = layers.Conv2D(16, (3, 3), padding="valid")(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(alpha=0.001)(x)
        x = layers.MaxPooling2D(pool_size=(3, 1))(x)
        x = layers.Conv2D(
            16, (3, 3), padding="valid", kernel_regularizer=regularizers.l2(0.01)
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(alpha=0.001)(x)
        x = layers.MaxPooling2D(pool_size=(3, 1))(x)

        x = layers.Flatten()(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(256)(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(alpha=0.001)(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(32)(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(alpha=0.001)(x)
        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(1, activation="sigmoid")(x)

        print("end_layers")
        model = Model(inputs, outputs, name=self.NAME)
        print("after model")
        model.summary()
        return model

    def run_net(self, inputs):
        pass
