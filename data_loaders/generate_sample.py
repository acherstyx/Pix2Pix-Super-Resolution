import os
import logging
import cv2 as cv
import tensorflow as tf
from templates import DataLoaderTemplate, ConfigTemplate
from templates.utils import mkdir

logger = logging.getLogger(__name__)


class ImageSampler(DataLoaderTemplate):
    def __init__(self, config):
        self.__BATCH_SIZE = 1

        super(ImageSampler, self).__init__(config)
        self.config: ImageSamplerConfig

    @staticmethod
    def __generate_sample_and_save(in_path, out_path, new_size, write_image=True):
        image = cv.imread(in_path)
        image = cv.resize(image, new_size)
        if write_image:
            mkdir(out_path)
            cv.imwrite(out_path, image)
        return image

    def down_sample(self):
        org_sample_image = os.listdir(self.config.ORIGIN_IMAGE_DIR)
        for index, image_file_name in enumerate(org_sample_image):
            image_file_path = os.path.join(self.config.ORIGIN_IMAGE_DIR, image_file_name)
            if not os.path.isfile(image_file_path):
                continue
            # size 1
            image_up = self.__generate_sample_and_save(image_file_path,
                                                       None,
                                                       self.config.UP_SIZE,
                                                       False)
            # size 2
            image_down = cv.resize(self.__generate_sample_and_save(image_file_path,
                                                                   None,
                                                                   self.config.DOWN_SIZE,
                                                                   False), self.config.UP_SIZE)
            yield image_down / 255, image_up / 255

    def load(self, *args):
        self.config: ImageSamplerConfig
        self.dataset = tf.data.Dataset.from_generator(
            generator=lambda: self.down_sample(),
            output_types=(tf.float32, tf.float32),
            output_shapes=(self.config.UP_SIZE + (3,),
                           self.config.UP_SIZE + (3,))
        ).batch(batch_size=self.__BATCH_SIZE, drop_remainder=True)
        self.dataset.prefetch(1)


class ImageSamplerConfig(ConfigTemplate):
    def __init__(self,
                 origin_image_dir,
                 up_size,
                 down_size):
        self.ORIGIN_IMAGE_DIR = origin_image_dir
        self.UP_SIZE = up_size
        self.DOWN_SIZE = down_size


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    my_instance = ImageSampler(ImageSamplerConfig(origin_image_dir="./data/origin",
                                                  up_size=(512, 512),
                                                  down_size=(200, 200)))

    loader = my_instance.get_dataset()

    for batch_data in loader:
        a = batch_data[0][0]
        b = batch_data[1][0]
        # print(a.numpy().shape, b.numpy().shape)
        cv.imshow("a", a.numpy())
        cv.waitKey(10)

        cv.imshow("b", b.numpy())
        cv.waitKey(0)
