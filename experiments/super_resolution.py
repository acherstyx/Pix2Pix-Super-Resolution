from data_loaders.generate_sample import ImageSampler, ImageSamplerConfig
from models.pix2pix_light import Pix2PixLight

if __name__ == '__main__':
    data_loader_config = ImageSamplerConfig("./data/origin", (400, 400), (100, 100))

    data_loader = ImageSampler(data_loader_config).get_dataset()
    model = Pix2PixLight().get_model()

    model.fit(data_loader, epochs=2)

    import cv2 as cv
    import numpy as np

    sample = cv.resize(cv.imread("data/origin/wallhaven-lqmrwl.png"), (100, 100)) / 255
    print(sample)
    cv.imshow("sample", sample)
    sample = np.reshape(sample, (1, 100, 100, 3))
    print(sample.shape)
    sample_4x = model.predict_step(sample)[0].numpy()
    cv.imshow("sample 4x", sample_4x)
    cv.waitKey()
