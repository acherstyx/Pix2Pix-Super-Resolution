from data_loaders.generate_sample import ImageSampler, ImageSamplerConfig
from models.pix2pix_light import Pix2PixLight

if __name__ == '__main__':
    data_loader_config = ImageSamplerConfig("./data/origin", (512, 512), (100, 100))

    data_loader = ImageSampler(data_loader_config).get_dataset()
    model = Pix2PixLight().get_model()

    model.fit(data_loader, epochs=100)

    import cv2 as cv
    import numpy as np

    sample = cv.resize(
        cv.imread("data/origin/anime-fantasy-world-anime-girls-witchs-broom-sky-birds-reverse-city-anime-15393.jpg"),
        (100, 100)) / 255
    sample = cv.resize(sample, (512, 512))
    cv.imshow("sample", sample)
    sample = np.reshape(sample, (1, 512, 512, 3))
    sample_4x = model.predict_step(sample)[0].numpy()
    cv.imshow("sample 4x", sample_4x)
    sample_target = cv.resize(
        cv.imread("data/origin/anime-fantasy-world-anime-girls-witchs-broom-sky-birds-reverse-city-anime-15393.jpg"),
        (512, 512)) / 255
    cv.imshow("sample 4x target", sample_target)
    cv.waitKey()
