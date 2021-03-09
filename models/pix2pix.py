import logging
import cv2
import os
import datetime
import numpy as np
import tensorflow as tf

from abc import ABC, abstractmethod
from tensorflow.keras import layers, Model, optimizers, losses, Sequential
from tqdm import tqdm

logger = logging.getLogger(__name__)


class Pix2Pix(ABC):
    def __init__(self,
                 epoch=1,
                 learning_rate=0.001,
                 learning_rate_scheduler=None,
                 tf_board_path=None):
        self._LR_SCHEDULER = learning_rate_scheduler
        self._EPOCH = epoch

        self._generator = None
        self._discriminator = None

        self._build()

        self._generator_optimizer = optimizers.Adam(learning_rate, beta_1=0.5)
        self._discriminator_optimizer = optimizers.Adam(learning_rate, beta_1=0.5)

        self._TF_BOARD_PATH = tf_board_path
        if self._TF_BOARD_PATH is not None:
            self._TF_BOARD = tf.summary.create_file_writer(
                self._TF_BOARD_PATH + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        else:
            self._TF_BOARD = None

        self._train_step_count = 0

    def _update_learning_rate(self, step):
        if self._LR_SCHEDULER is not None:
            logger.info("Set learning rate to %s for epoch %s", self._LR_SCHEDULER(step), step)
            self._generator.lr = self._LR_SCHEDULER(step)
            self._discriminator.lr = self._LR_SCHEDULER(step)

    @staticmethod
    def _generator_loss(real_image, generate_image, discriminator_output):
        discriminate_loss = losses.BinaryCrossentropy(from_logits=True)(tf.ones_like(discriminator_output),
                                                                        discriminator_output)
        generate_loss = losses.MeanAbsoluteError()(real_image, generate_image)
        return discriminate_loss * 3 + generate_loss * 100, discriminate_loss * 3, generate_loss * 100

    @staticmethod
    def _discriminator_loss(discriminator_real_output, discriminator_fake_output):
        real_loss = losses.BinaryCrossentropy(from_logits=True)(tf.ones_like(discriminator_real_output),
                                                                discriminator_real_output)
        fake_loss = losses.BinaryCrossentropy(from_logits=True)(tf.zeros_like(discriminator_fake_output),
                                                                discriminator_fake_output)
        return real_loss + fake_loss

    def _train_step(self, input_image, target_image, with_preview):
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

            generator_loss, generator_gen_loss, generator_disc_loss = self._generator_loss(target_image, generate_image,
                                                                                           discriminator_fake_output)
            discriminator_loss = self._discriminator_loss(discriminator_real_output, discriminator_fake_output)

        generator_gradient = generator_tape.gradient(generator_loss,
                                                     self._generator.trainable_variables)
        discriminator_gradient = discriminator_tape.gradient(discriminator_loss,
                                                             self._discriminator.trainable_variables)

        self._generator_optimizer.apply_gradients(
            zip(generator_gradient, self._generator.trainable_variables)
        )
        self._discriminator_optimizer.apply_gradients(
            zip(discriminator_gradient, self._discriminator.trainable_variables)
        )

        return generator_loss, discriminator_loss, generator_gen_loss, generator_disc_loss

    def train(self, dataset, epoch=None, with_preview=False):
        if epoch is None:
            epoch = self._EPOCH

        if self._TF_BOARD is not None:
            metric_total_loss = tf.metrics.Mean("total_loss", dtype=tf.float32)
            metric_gen_loss = tf.metrics.Mean("gen_loss", dtype=tf.float32)
            metric_disc_loss = tf.metrics.Mean("disc_loss", dtype=tf.float32)
            with self._TF_BOARD.as_default():
                for i in range(epoch):
                    # apply learning rate scheduler if exist
                    self._update_learning_rate(step=self._train_step_count + 1)
                    # task bar
                    bar = tqdm(dataset)
                    bar.set_description("Epoch {}/{}".format(i + 1, epoch))
                    for input_image, target_image in bar:
                        # train
                        generator_loss, discriminator_loss, gen_loss, disc_loss = self._train_step(input_image,
                                                                                                   target_image,
                                                                                                   with_preview)
                        # show taskbar
                        bar.set_postfix(gen_loss=generator_loss.numpy(), disc_loss=discriminator_loss.numpy())
                        # tensorboard
                        self._train_step_count += 1
                        metric_gen_loss(generator_loss)
                        metric_disc_loss(discriminator_loss)
                        metric_total_loss(generator_loss + discriminator_loss)
                        tf.summary.scalar("Step/generator loss", generator_loss, step=self._train_step_count)
                        tf.summary.scalar("Step/discriminator loss", discriminator_loss, step=self._train_step_count)
                        tf.summary.scalar("Step/total loss", generator_loss + discriminator_loss,
                                          step=self._train_step_count)
                        tf.summary.scalar("Step/generator loss/gen", gen_loss, step=self._train_step_count)
                        tf.summary.scalar("Step/generator loss/disc", disc_loss, step=self._train_step_count)

                    bar.close()
                    tf.summary.scalar("Epoch/total loss", metric_total_loss.result(), step=i)
                    tf.summary.scalar("Epoch/generator loss", metric_gen_loss.result(), step=i)
                    tf.summary.scalar("Epoch/discriminator loss", metric_disc_loss.result(), step=i)

        else:
            for i in range(epoch):
                # apply learning rate scheduler if exist
                self._update_learning_rate(step=self._train_step_count + 1)
                # task bar
                bar = tqdm(dataset)
                bar.set_description("Epoch {}/{}".format(i + 1, epoch))
                for input_image, target_image in bar:
                    generator_loss, discriminator_loss = self._train_step(input_image, target_image, with_preview)
                    bar.set_postfix(gen_loss=generator_loss.numpy(), disc_loss=discriminator_loss.numpy())
                bar.close()
        if with_preview:
            cv2.destroyWindow("Training")

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

    def predict(self, sample_image):
        init_shape = np.shape(sample_image)
        shape = self._predict_size(sample_image)
        resized = cv2.resize(sample_image, (shape[1], shape[0]))
        reshaped = self.image_uint2norm(np.reshape(resized.copy(), (1, shape[0], shape[1], -1)))
        logger.info("Generator input shape: %s", reshaped.shape)
        generate = self._generator(reshaped, training=True).numpy()[0]

        generate = cv2.resize(generate, (init_shape[1], init_shape[0]))

        return self.image_norm2uint(generate)

    def predict_sample(self, sample_image):
        init_shape = np.shape(sample_image)
        shape = self._predict_size(sample_image)

        target = cv2.resize(sample_image, (shape[1], shape[0]))
        resized = cv2.blur(target, (6, 6))
        reshaped = self.image_uint2norm(np.reshape(resized.copy(), (1, shape[0], shape[1], -1)))
        logger.info("Generator input shape: %s", reshaped.shape)
        generate = self._generator(reshaped, training=True).numpy()[0]

        generate = (generate + 1.0) / 2.0
        resized = cv2.resize(resized, (init_shape[1], init_shape[0]))
        generate = cv2.resize(generate, (init_shape[1], init_shape[0]))

        return sample_image.astype(np.uint8), resized.astype(np.uint8), (generate * 255).astype(np.uint8)

    def show_summary(self):
        tf.keras.utils.plot_model(self._generator, to_file="generator.png", expand_nested=True,
                                  show_shapes=True,
                                  dpi=100)
        tf.keras.utils.plot_model(self._discriminator, to_file="discriminator.png", expand_nested=True,
                                  show_shapes=True,
                                  dpi=100)

    @staticmethod
    def image_norm2uint(image):
        return ((image + 1.0) * (255 / 2.0)).astype(np.uint8)

    @staticmethod
    def image_uint2norm(image):
        return (image * 2.0) / 255 - 1.0

    @staticmethod
    def _predict_size(image, factor=256):
        if image is None:
            raise FileNotFoundError
        init_shape = np.shape(image)
        shape = [factor, factor]
        while True:
            if shape[0] < init_shape[0]:
                shape[0] += factor
            else:
                break
        while True:
            if shape[1] < init_shape[1]:
                shape[1] += factor
            else:
                break
        return shape

    @abstractmethod
    def _build(self):
        pass


class Pix2Pix256(Pix2Pix):
    @staticmethod
    def _down_sample(filters, size, batch_norm=True, dropout=False, dropout_rate=0.5):
        initializer = tf.random_normal_initializer(0., 0.02)
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

    @staticmethod
    def _up_sample(filters, size, batch_norm=False, dropout=False, dropout_rate=0.5):
        initializer = tf.random_normal_initializer(0., 0.02)
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

    @staticmethod
    def _create_generator(down_stack, up_stack):
        initializer = tf.random_normal_initializer(0., 0.02)

        inputs = layers.Input(shape=[256, 256, 3])

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

    @staticmethod
    def _create_discriminator(input_shape=(None, None, 3)):
        initializer = tf.random_normal_initializer(0., 0.02)

        # default: 256, 256, 3
        init_inputs = layers.Input(shape=input_shape, dtype=tf.float32)
        # default: 256, 256, 3
        target_inputs = layers.Input(shape=input_shape, dtype=tf.float32)

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

        hidden_layer = layers.ZeroPadding2D(padding=(1, 1))(hidden_layer)

        hidden_layer = layers.Conv2D(filters=512,
                                     kernel_size=4,
                                     kernel_initializer=initializer,
                                     activation=None)(hidden_layer)
        hidden_layer = layers.BatchNormalization()(hidden_layer)
        hidden_layer = layers.LeakyReLU()(hidden_layer)

        hidden_layer = layers.ZeroPadding2D(padding=(1, 1))(hidden_layer)

        hidden_layer = layers.Conv2D(filters=1,
                                     kernel_size=4,
                                     kernel_initializer=initializer,
                                     activation="tanh")(hidden_layer)

        output_layer = hidden_layer

        return Model(inputs=[init_inputs, target_inputs],
                     outputs=output_layer)

    def _build(self):

        self._generator = self._create_generator(
            down_stack=[
                self._down_sample(64, 4, batch_norm=False),  # 128, 128, 64
                self._down_sample(128, 4),  # 64, 64, 128
                self._down_sample(256, 4),  # 32, 32, 256
                self._down_sample(512, 4),  # 16, 16, 512
                self._down_sample(512, 4),  # 8, 8, 512
                self._down_sample(512, 4),  # 4, 4, 512
                self._down_sample(512, 4),  # 2, 2, 512
                self._down_sample(512, 4),  # 1, 1, 512
            ],
            up_stack=[
                self._up_sample(512, 4, dropout=True),  # 2, 2, 512
                self._up_sample(512, 4, dropout=True),  # 4, 4, 512
                self._up_sample(512, 4, dropout=True),  # 8, 8, 512
                self._up_sample(512, 4),  # 16, 16, 512
                self._up_sample(256, 4),  # 32, 32, 256
                self._up_sample(128, 4),  # 64, 64, 128
                self._up_sample(64, 4),  # 128, 128, 64
            ]
        )
        self._discriminator = self._create_discriminator(input_shape=(256, 256, 3))


class Pix2Pix64(Pix2Pix256):
    def _build(self):
        self._generator = self._create_generator(
            down_stack=[
                self._down_sample(64, 4, batch_norm=False),  # 128, 128, 64
                self._down_sample(128, 4),  # 64, 64, 128
                self._down_sample(256, 4),  # 32, 32, 256
                self._down_sample(512, 4),  # 16, 16, 512
                self._down_sample(512, 4),  # 8, 8, 512
                self._down_sample(512, 4),  # 4, 4, 512
            ],
            up_stack=[
                self._up_sample(512, 4, dropout=True),  # 8, 8, 512
                self._up_sample(512, 4),  # 16, 16, 512
                self._up_sample(256, 4),  # 32, 32, 256
                self._up_sample(128, 4),  # 64, 64, 128
                self._up_sample(64, 4),  # 128, 128, 64
            ]
        )
        self._discriminator = self._create_discriminator(input_shape=(64, 64, 3))

        self.show_summary()
