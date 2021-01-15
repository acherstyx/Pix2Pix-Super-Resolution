import tensorflow as tf

from tensorflow.keras import layers, Model, optimizers, losses, metrics
from templates import ModelTemplate


class Pix2PixLight(ModelTemplate):
    def __init__(self, input_shape=(100, 100, 3)):
        self.__INPUT_SHAPE = input_shape

        super(Pix2PixLight, self).__init__(None)

    def build(self, *args):
        input_layer = layers.Input(shape=self.__INPUT_SHAPE, dtype=tf.float32)

        def multiple_kernel(pre_layer, filters):
            hidden_layer_1 = layers.Conv2DTranspose(filters=filters,
                                                    kernel_size=(3, 3),
                                                    strides=2,
                                                    padding="SAME",
                                                    activation="relu")(pre_layer)

            hidden_layer_2 = layers.Conv2DTranspose(filters=filters,
                                                    kernel_size=(8, 8),
                                                    strides=2,
                                                    padding="SAME",
                                                    activation="relu")(pre_layer)

            hidden_layer_3 = layers.Conv2DTranspose(filters=filters,
                                                    kernel_size=(16, 16),
                                                    strides=2,
                                                    padding="SAME",
                                                    activation="relu")(pre_layer)
            return layers.Concatenate()([hidden_layer_1, hidden_layer_2, hidden_layer_3])

        hidden_layer = multiple_kernel(input_layer, 3)
        hidden_layer = layers.Conv2DTranspose(filters=16,
                                              kernel_size=(1, 1),
                                              padding="SAME",
                                              activation="relu")(hidden_layer)
        hidden_layer = multiple_kernel(hidden_layer, 8)
        output_layer = layers.Conv2DTranspose(filters=3,
                                              kernel_size=(1, 1),
                                              padding="SAME",
                                              activation="sigmoid")(hidden_layer)

        self.model = Model(inputs=input_layer,
                           outputs=output_layer)

        self.model.compile(optimizer=optimizers.Adam(learning_rate=0.1),
                           loss=losses.MeanSquaredError(),
                           metrics=[metrics.MeanSquaredError()])


if __name__ == '__main__':
    my_instance = Pix2PixLight()
    my_instance.show_summary(with_plot=True)
