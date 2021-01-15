import tensorflow as tf

from tensorflow.keras import layers, Model, optimizers, losses, metrics, Sequential
from templates import ModelTemplate


class Pix2PixLight(ModelTemplate):
    def __init__(self, input_shape=(100, 100, 3)):
        self.__INPUT_SHAPE = input_shape

        super(Pix2PixLight, self).__init__(None)

    @staticmethod
    def __down_sample(filters, size, batch_norm=True, dropout=False, dropout_rate=0.5):
        down_sample_layer = Sequential()
        initializer = tf.random_normal_initializer(0., 0.02)

        down_sample_layer.add(
            layers.Conv2D(filters=filters,
                          kernel_size=size,
                          strides=2,
                          padding="SAME",
                          kernel_initializer=initializer,
                          use_bias=False)
        )

        if batch_norm:
            down_sample_layer.add(layers.BatchNormalization())
        if dropout:
            down_sample_layer.add(layers.Dropout(dropout_rate))

        down_sample_layer.add(layers.LeakyReLU())

        return down_sample_layer

    @staticmethod
    def __up_sample(filters, size, batch_norm=False, dropout=False, dropout_rate=0.5):
        up_sample_layer = Sequential()
        initializer = tf.random_normal_initializer(.0, 0.02)

        up_sample_layer.add(
            layers.Conv2DTranspose(filters=filters,
                                   kernel_size=size,
                                   strides=2,
                                   padding="SAME",
                                   kernel_initializer=initializer,
                                   use_bias=False)
        )

        if batch_norm:
            up_sample_layer.add(layers.BatchNormalization())
        if dropout:
            up_sample_layer.add(layers.Dropout(dropout_rate))

        up_sample_layer.add(layers.ReLU())

        return up_sample_layer

    def build(self):
        inputs = layers.Input(shape=[512, 512, 3])

        down_stack = [
            self.__down_sample(64, 4, batch_norm=False),  # 128, 128, 64
            self.__down_sample(128, 4),  # 64, 64, 128
            self.__down_sample(256, 4),  # 32, 32, 256
            self.__down_sample(512, 4),  # 16, 16, 512
            self.__down_sample(512, 4),  # 8, 8, 512
            self.__down_sample(512, 4),  # 4, 4, 512
            self.__down_sample(512, 4),  # 2, 2, 512
            self.__down_sample(512, 4),  # 1, 1, 512
        ]

        up_stack = [
            self.__up_sample(512, 4, dropout=True),  # 2, 2, 512
            self.__up_sample(512, 4, dropout=True),  # 4, 4, 512
            self.__up_sample(512, 4, dropout=True),  # 8, 8, 512
            self.__up_sample(512, 4),  # 16, 16, 512
            self.__up_sample(256, 4),  # 32, 32, 256
            self.__up_sample(128, 4),  # 64, 64, 128
            self.__up_sample(64, 4),  # 128, 128, 64
        ]

        initializer = tf.random_normal_initializer(0., 0.02)

        # jump out of `for`
        last = tf.keras.layers.Conv2DTranspose(filters=3,
                                               kernel_size=4,
                                               strides=2,
                                               padding="SAME",
                                               kernel_initializer=initializer,
                                               activation='tanh')  # 256, 256, OUTPUT_CHANNEL

        # down sample
        hidden_layer = inputs
        down_outputs = []
        for down in down_stack:
            hidden_layer = down(hidden_layer)
            down_outputs.append(hidden_layer)

        down_outputs = reversed(down_outputs[:-1])  # skip the last layer

        # up sample
        for output, up in zip(down_outputs, up_stack):
            hidden_layer = up(hidden_layer)
            hidden_layer = layers.Concatenate()([hidden_layer, output])

        outputs = last(hidden_layer)

        self.model = Model(inputs=inputs,
                           outputs=outputs)

        self.model.compile(optimizer=optimizers.Adam(learning_rate=0.0001),
                           loss=losses.MeanAbsoluteError(),
                           metrics=[metrics.MeanAbsoluteError()])

        return self


if __name__ == '__main__':
    my_instance = Pix2PixLight()
    my_instance.show_summary(with_plot=True)
