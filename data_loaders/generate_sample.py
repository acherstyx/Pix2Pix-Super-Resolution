import os
import logging
import cv2 as cv
import tensorflow as tf

logger = logging.getLogger(__name__)


class ImageSampler:
    def __init__(self,
                 origin_image_dir,
                 batch_size,
                 up_size,
                 down_size,
                 shuffle=10,
                 prefetch=1,
                 low_memory=True):
        super(ImageSampler, self).__init__()

        self.__BATCH_SIZE = batch_size
        self.__ORIGIN_IMAGE_DIR = origin_image_dir
        self.__UP_SIZE = up_size
        self.__DOWN_SIZE = down_size
        self.__PREFETCH = prefetch
        self.__SHUFFLE = shuffle
        # save result of first epoch in mem
        self.__LOW_MEM = low_memory
        self.__CACHE = []
        self._dataset = None
        self.__load()

    @staticmethod
    def __generate_sample_and_save(in_path, out_path, new_size, write_image=True):
        image = cv.imread(in_path)
        image = cv.resize(image, new_size)
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
                # size 1
                image_up = self.__generate_sample_and_save(image_file_path,
                                                           None,
                                                           self.__UP_SIZE,
                                                           False)
                image_up = (image_up * 2.0) / 255 - 1.0
                # size 2
                image_down = cv.resize(self.__generate_sample_and_save(image_file_path,
                                                                       None,
                                                                       self.__DOWN_SIZE,
                                                                       False), self.__UP_SIZE)
                image_down = (image_down * 2.0) / 255 - 1.0
                if not self.__LOW_MEM:
                    self.__CACHE.append([image_down, image_up])
                yield image_down, image_up
        else:
            logger.debug("Data loader is using cache.")
            for image_down, image_up in self.__CACHE:
                yield image_down, image_up

    def __load(self, *args):
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
                          up_size=(256, 256),
                          down_size=(128, 128)).get_dataset()

    for batch_data in loader:
        a = batch_data[0][0]
        b = batch_data[1][0]
        # print(a.numpy().shape, b.numpy().shape)
        cv.imshow("a", (a.numpy() + 1) / 2)
        cv.imshow("b", (b.numpy() + 1) / 2)
        cv.waitKey(0)
