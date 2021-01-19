import tensorflow as tf

from tensorflow.keras import layers, Model, optimizers, losses, metrics, Sequential
from templates import ModelTemplate


class Pix2Pix(ModelTemplate):
    def __init__(self, learning_rate=0.001, epoch=1):
        self.__LR = learning_rate
        self.__EPOCH = epoch

        super(Pix2Pix, self).__init__(None)

    def build(self):
        def create_generator():
            initializer = tf.random_normal_initializer(0., 0.02)

            def down_sample(filters, size, batch_norm=True, dropout=False, dropout_rate=0.5):
                down_sample_layer = Sequential()

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

            def up_sample(filters, size, batch_norm=False, dropout=False, dropout_rate=0.5):
                up_sample_layer = Sequential()

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

            inputs = layers.Input(shape=[512, 512, 3])

            down_stack = [
                down_sample(64, 4, batch_norm=False),  # 128, 128, 64
                down_sample(128, 4),  # 64, 64, 128
                down_sample(256, 4),  # 32, 32, 256
                down_sample(512, 4),  # 16, 16, 512
                down_sample(512, 4),  # 8, 8, 512
                down_sample(512, 4),  # 4, 4, 512
                down_sample(512, 4),  # 2, 2, 512
                down_sample(512, 4),  # 1, 1, 512
            ]

            up_stack = [
                up_sample(512, 4, dropout=True),  # 2, 2, 512
                up_sample(512, 4, dropout=True),  # 4, 4, 512
                up_sample(512, 4, dropout=True),  # 8, 8, 512
                up_sample(512, 4),  # 16, 16, 512
                up_sample(256, 4),  # 32, 32, 256
                up_sample(128, 4),  # 64, 64, 128
                up_sample(64, 4),  # 128, 128, 64
            ]

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

            # create model
            generator = Model(inputs=inputs,
                              outputs=outputs)
            # compile
            generator.compile(optimizer=optimizers.Adam(learning_rate=self.__LR),
                              loss=losses.MeanAbsoluteError(),
                              metrics=[metrics.MeanAbsoluteError()])

            return generator

        def create_discriminator():
            initializer = tf.random_normal_initializer(0., 0.02)

            init_inputs = layers.Input(shape=(512, 512, 3), dtype=tf.float32)

            target_inputs = layers.Input(shape=(512, 512, 3), dtype=tf.float32)

            hidden_layer = layers.Concatenate()([init_inputs, target_inputs])

            hidden_layer = layers.Conv2D(filters=64,
                                         kernel_size=4,
                                         strides=2,
                                         padding="SAME",
                                         kernel_initializer=initializer,
                                         activation=None)(hidden_layer)
            hidden_layer = layers.LeakyReLU()(hidden_layer)

            hidden_layer = layers.Conv2D(filters=128,
                                         kernel_size=4,
                                         strides=2,
                                         padding="SAME",
                                         kernel_initializer=initializer,
                                         activation=None)(hidden_layer)
            hidden_layer = layers.BatchNormalization()(hidden_layer)
            hidden_layer = layers.LeakyReLU()(hidden_layer)

            hidden_layer = layers.Conv2D(filters=256,
                                         kernel_size=4,
                                         strides=2,
                                         padding="SAME",
                                         kernel_initializer=initializer,
                                         activation=None)(hidden_layer)
            hidden_layer = layers.BatchNormalization()(hidden_layer)
            hidden_layer = layers.LeakyReLU()(hidden_layer)

            hidden_layer = layers.Conv2D(filters=512,
                                         kernel_size=4,
                                         strides=2,
                                         padding="SAME",
                                         kernel_initializer=initializer,
                                         activation=None)(hidden_layer)
            hidden_layer = layers.BatchNormalization()(hidden_layer)
            hidden_layer = layers.LeakyReLU()(hidden_layer)
            assert tuple(hidden_layer.shape) == (None, 32, 32, 512)

            hidden_layer = layers.ZeroPadding2D(padding=(1, 1))(hidden_layer)
            assert tuple(hidden_layer.shape) == (None, 34, 34, 512)

            hidden_layer = layers.Conv2D(filters=512,
                                         kernel_size=4,
                                         kernel_initializer=initializer,
                                         activation=None)(hidden_layer)
            hidden_layer = layers.BatchNormalization()(hidden_layer)
            hidden_layer = layers.LeakyReLU()(hidden_layer)
            assert tuple(hidden_layer.shape) == (None, 31, 31, 512)

            hidden_layer = layers.ZeroPadding2D(padding=(1, 1))(hidden_layer)
            assert tuple(hidden_layer.shape) == (None, 33, 33, 512)

            hidden_layer = layers.Conv2D(filters=1,
                                         kernel_size=4,
                                         kernel_initializer=initializer,
                                         activation="tanh")(hidden_layer)
            assert tuple(hidden_layer.shape) == (None, 30, 30, 1)

            output_layer = hidden_layer

            return Model(inputs=[init_inputs, target_inputs],
                         outputs=output_layer)

        self.model = create_discriminator()

        self.show_summary(with_plot=True)

        self.model = (create_generator(), create_discriminator())

        return self

    def train(self, dataset):
        for epoch in range(self.__EPOCH):
            pass


if __name__ == '__main__':
    my_instance = Pix2Pix()
    my_instance.show_summary(with_plot=True)
