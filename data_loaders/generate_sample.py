import os
import logging
import random
import numpy as np
import cv2 as cv
import tensorflow as tf

logger = logging.getLogger(__name__)


class ImageSampler:
    def __init__(self,
                 origin_image_dir,
                 batch_size,
                 image_size,
                 blur_kernel_size,
                 blur_kernel_size_delta=0,
                 shuffle=10,
                 prefetch=1,
                 low_memory=True,
                 skip_small_image=True):
        super(ImageSampler, self).__init__()

        self.__ORIGIN_IMAGE_DIR = origin_image_dir
        self.__UP_SIZE = image_size
        self.__SKIP_SMALL_IMAGE = skip_small_image
        # td.dataset
        self.__BATCH_SIZE = batch_size
        self.__PREFETCH = prefetch
        self.__SHUFFLE = shuffle
        # set blur image
        self.__BLUR_KERNEL_SIZE = blur_kernel_size
        self.__BLUR_KERNEL_SIZE_DELTA = blur_kernel_size_delta
        # save result of first epoch in mem
        self.__LOW_MEM = low_memory
        self.__CACHE = []
        self._dataset = None
        self.__load()

    @staticmethod
    def __generate_sample_and_save(in_path, out_path, new_size, write_image=True):
        image = cv.imread(in_path)
        try:
            shape = image.shape
        except:
            raise cv.error
        if shape[0] < new_size[0] or shape[1] < new_size[1]:
            image = cv.resize(image, new_size)
        else:
            offset_1 = random.randint(0, shape[0] - new_size[0] - 1)
            offset_2 = random.randint(0, shape[1] - new_size[1] - 1)

            image = image[offset_1:offset_1 + new_size[0], offset_2:offset_2 + new_size[1], :]

        if write_image:
            cv.imwrite(out_path, image)
        return image

    def __down_sample(self):
        org_sample_image = os.listdir(self.__ORIGIN_IMAGE_DIR)
        if not self.__CACHE:
            for index, image_file_name in enumerate(org_sample_image):
                image_file_path = os.path.join(self.__ORIGIN_IMAGE_DIR, image_file_name)
                if not os.path.isfile(image_file_path):
                    continue
                # read image and check
                image_origin = cv.imread(image_file_path)
                if image_origin is None:
                    logger.debug("Cannot open image: %s, skip.", image_file_path)
                    continue
                shape = image_origin.shape
                # get up image
                if shape[0] < self.__UP_SIZE[0] or shape[1] < self.__UP_SIZE[1]:
                    if self.__SKIP_SMALL_IMAGE:
                        continue
                    image_up = cv.resize(image_origin, self.__UP_SIZE)
                else:
                    offset_1 = random.randint(0, shape[0] - self.__UP_SIZE[0] - 1)
                    offset_2 = random.randint(0, shape[1] - self.__UP_SIZE[1] - 1)

                    image_up = image_origin[
                               offset_1:offset_1 + self.__UP_SIZE[0],
                               offset_2:offset_2 + self.__UP_SIZE[1], :]
                # get down image
                kernel = [max(1, x + random.randint(-self.__BLUR_KERNEL_SIZE_DELTA,
                                                    self.__BLUR_KERNEL_SIZE_DELTA)) for x in self.__BLUR_KERNEL_SIZE]
                image_down = cv.blur(image_up, tuple(kernel))
                image_up = (image_up * 2.0) / 255 - 1.0
                image_down = (image_down * 2.0) / 255 - 1.0
                if not self.__LOW_MEM:
                    self.__CACHE.append([image_down, image_up])
                yield image_down, image_up
        else:
            logger.debug("Data loader is using cache.")
            for image_down, image_up in self.__CACHE:
                yield image_down, image_up

    def __load(self):
        self._dataset = tf.data.Dataset.from_generator(
            generator=lambda: self.__down_sample(),
            output_types=(tf.float32, tf.float32),
            output_shapes=(self.__UP_SIZE + (3,),
                           self.__UP_SIZE + (3,))
        ).shuffle(self.__SHUFFLE).batch(batch_size=self.__BATCH_SIZE, drop_remainder=True)
        self._dataset.prefetch(self.__PREFETCH)

    def get_dataset(self):
        return self._dataset


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    loader = ImageSampler(origin_image_dir="../data/origin",
                          batch_size=1,
                          image_size=(256, 256),
                          blur_kernel_size=(5, 5),
                          blur_kernel_size_delta=3).get_dataset()

    for batch_data in loader:
        a = batch_data[0][0]
        b = batch_data[1][0]
        cv.imshow("a", (a.numpy() + 1) / 2)
        cv.imshow("b", (b.numpy() + 1) / 2)
        cv.waitKey(0)
