import logging
import cv2
import os
import numpy as np
import tensorflow as tf

from tensorflow.keras import layers, Model, optimizers, losses, metrics, Sequential
from tqdm import tqdm

logger = logging.getLogger(__name__)


class Pix2Pix256:
    def __init__(self, learning_rate=0.001, epoch=1):
        self.__LR = learning_rate
        self.__EPOCH = epoch

        self._generator = None
        self._discriminator = None

        self.__build()

        self.__generator_optimizer = optimizers.Adam(learning_rate, beta_1=0.5)
        self.__discriminator_optimizer = optimizers.Adam(learning_rate, beta_1=0.5)

    def __build(self):
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

            inputs = layers.Input(shape=[256, 256, 3])

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

            return generator

        def create_discriminator():
            initializer = tf.random_normal_initializer(0., 0.02)

            init_inputs = layers.Input(shape=(256, 256, 3), dtype=tf.float32)

            target_inputs = layers.Input(shape=(256, 256, 3), dtype=tf.float32)

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
            assert tuple(hidden_layer.shape) == (None, 32, 32, 256)

            hidden_layer = layers.ZeroPadding2D(padding=(1, 1))(hidden_layer)
            assert tuple(hidden_layer.shape) == (None, 34, 34, 256)

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

        self._generator = create_generator()
        self._discriminator = create_discriminator()

        tf.keras.utils.plot_model(self._generator, to_file="generator.png", expand_nested=True,
                                  show_shapes=True,
                                  dpi=50)
        tf.keras.utils.plot_model(self._discriminator, to_file="discriminator.png", expand_nested=True,
                                  show_shapes=True,
                                  dpi=50)

    @staticmethod
    def __generator_loss(real_image, generate_image, discriminator_output):
        discriminate_loss = losses.BinaryCrossentropy(from_logits=True)(tf.ones_like(discriminator_output),
                                                                        discriminator_output)
        generate_loss = losses.MeanAbsoluteError()(real_image, generate_image)
        return discriminate_loss * 4 + generate_loss * 100

    @staticmethod
    def __discriminator_loss(discriminator_real_output, discriminator_fake_output):
        real_loss = losses.BinaryCrossentropy(from_logits=True)(tf.ones_like(discriminator_real_output),
                                                                discriminator_real_output)
        fake_loss = losses.BinaryCrossentropy(from_logits=True)(tf.zeros_like(discriminator_fake_output),
                                                                discriminator_fake_output)
        return real_loss + fake_loss

    def __train_step(self, input_image, target_image, with_preview):
        self._generator: Model
        self._discriminator: Model
        with tf.GradientTape() as generator_tape, tf.GradientTape() as discriminator_tape:
            generate_image = self._generator(input_image, training=True)

            if with_preview:
                combine = np.concatenate([(input_image.numpy()[0] + 1) / 2,
                                          (generate_image.numpy()[0] + 1) / 2,
                                          (target_image.numpy()[0] + 1) / 2], axis=1)
                cv2.imshow("Training", combine)
                cv2.waitKey(1)

            discriminator_fake_output = self._discriminator([input_image, generate_image], training=True)
            discriminator_real_output = self._discriminator([input_image, target_image], training=True)

            generator_loss = self.__generator_loss(target_image, generate_image, discriminator_fake_output)
            discriminator_loss = self.__discriminator_loss(discriminator_real_output, discriminator_fake_output)

        generator_gradient = generator_tape.gradient(generator_loss,
                                                     self._generator.trainable_variables)
        discriminator_gradient = discriminator_tape.gradient(discriminator_loss,
                                                             self._discriminator.trainable_variables)

        self.__generator_optimizer.apply_gradients(
            zip(generator_gradient, self._generator.trainable_variables)
        )
        self.__discriminator_optimizer.apply_gradients(
            zip(discriminator_gradient, self._discriminator.trainable_variables)
        )

        return generator_loss, discriminator_loss

    def train(self, dataset, epoch=None, with_preview=False):
        log = []

        if epoch is None:
            epoch = self.__EPOCH
        for i in range(epoch):
            bar = tqdm(dataset)
            for input_image, target_image in bar:
                generator_loss, discriminator_loss = self.__train_step(input_image, target_image, with_preview)
                bar.set_description("Epoch {}/{}".format(i + 1, epoch))
                bar.set_postfix(gen_loss=generator_loss.numpy(), disc_loss=discriminator_loss.numpy())
                log.append([generator_loss, discriminator_loss])
            bar.close()
        if with_preview:
            cv2.destroyWindow("Training")
        return log

    def predict(self, sample_image):
        sample_image = (sample_image.copy() * 2.0) / 255 - 1.0
        sample_image_target = cv2.resize(sample_image.copy(), (256, 256))
        sample_image_resized = cv2.resize(sample_image.copy(), (128, 128))
        sample_image_resized = cv2.resize(sample_image_resized.copy(), (256, 256))
        sample_image_reshaped = np.reshape(sample_image_resized.copy(), (1, 256, 256, 3))

        gen_output = self._generator(sample_image_reshaped, training=True).numpy()[0]

        gen_output = (gen_output + 1.0) / 2.0
        sample_image_target = (sample_image_target + 1.0) / 2.0
        sample_image_resized = (sample_image_resized + 1.0) / 2.0

        return (sample_image_target,
                sample_image_resized,
                gen_output)

    def save(self, gen_save_path, disc_save_path):
        for a_dir in [gen_save_path, disc_save_path]:
            directory = os.path.split(a_dir)[0]

            if not os.path.exists(directory):
                os.makedirs(directory)

        self._generator: tf.keras.Model
        self._discriminator: tf.keras.Model

        self._generator.save(gen_save_path)
        self._discriminator.save(disc_save_path)

    def load(self, gen_save_path, disc_save_path):
        self._generator: tf.keras.Model
        self._discriminator: tf.keras.Model

        self._generator.load_weights(gen_save_path)
        self._discriminator.load_weights(disc_save_path)
