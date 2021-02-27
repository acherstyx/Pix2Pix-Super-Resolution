import logging
import cv2
import numpy as np
from data_loaders.generate_sample import ImageSampler
from models.pix2pix import Pix2Pix256

logger = logging.getLogger(__name__)

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    dataset = ImageSampler(origin_image_dir="../data/origin",
                           batch_size=3,
                           prefetch=10,
                           shuffle=50,
                           up_size=(256, 256),
                           down_size=(128, 128)).get_dataset()
    model = Pix2Pix256(epoch=50, learning_rate=0.001)

    try:
        model.load("../logs/super_resolution/save_weight/generator.h5",
                   "../logs/super_resolution/save_weight/discriminator.h5")
    except OSError:
        pass
    model.train(dataset, epoch=5, with_preview=True)
    model.save("../logs/super_resolution/save_weight/generator.h5",
               "../logs/super_resolution/save_weight/discriminator.h5")

    sample_image = cv2.imread("../data/origin/1.jpg")
    target_output, gen_input, gen_output = model.predict(sample_image)

    # logger.debug("gen_output: %s", gen_output)
    combine_output = np.concatenate([gen_input, gen_output, target_output], axis=1)
    cv2.imshow("Result", combine_output)
    cv2.waitKey(0)
