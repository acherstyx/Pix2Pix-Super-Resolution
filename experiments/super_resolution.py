from data_loaders.generate_sample import ImageSampler
from models.pix2pix import Pix2Pix256

if __name__ == '__main__':
    dataset = ImageSampler(origin_image_dir="../data/origin",
                           batch_size=1,
                           up_size=(256, 256),
                           down_size=(128, 128)).get_dataset()
    model = Pix2Pix256(epoch=50, learning_rate=0.001)

    model.train(dataset)

    import cv2

    sample_image = cv2.imread("../data/origin/001.jpg")
    target_output, gen_input, gen_output = model.predict(sample_image)
    # print(gen_input)
    # print(gen_output)
    cv2.imshow("Target", target_output)
    cv2.imshow("Input", gen_input)
    cv2.imshow("Predict", gen_output)
    cv2.waitKey(0)
