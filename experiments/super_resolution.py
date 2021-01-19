from data_loaders.generate_sample import ImageSampler, ImageSamplerConfig
from models.pix2pix import Pix2Pix

if __name__ == '__main__':
    data_loader_config = ImageSamplerConfig("./data/origin", (512, 512), (100, 100))

    data_loader = ImageSampler(data_loader_config).get_dataset()
    model = Pix2Pix().get_model()

    model.fit(data_loader, epochs=50)

    import cv2 as cv
    import numpy as np

    sample = cv.resize(
        cv.imread("data/origin/111.jpg"),
        (100, 100)) / 255
    sample = cv.resize(sample, (512, 512))
    cv.imshow("sample", sample)
    sample = np.reshape(sample, (1, 512, 512, 3))
    sample_4x = model.predict(sample)[0]
    cv.imshow("sample 4x", sample_4x)
    sample_target = cv.resize(
        cv.imread("data/origin/111.jpg"),
        (512, 512)) / 255
    cv.imshow("sample 4x target", sample_target)
    cv.waitKey()
