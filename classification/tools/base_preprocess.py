import numpy as np
from PIL import Image


def get_image_data(image_file, input_shape = [1, 3, 224, 224]):
    """
    image pre_processing function for classfication models.
    Args:
        image_file: input image file path
        imput_shape: model input shape
    Returns: 
        image: image numpy data in NCHW shape
    """

    size = input_shape[2:]
    hints = [256, 256]
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # mean = [123.675, 116.28, 103.53]
    # std = [58.395, 57.12, 57.375]

    # mean = 127.5
    # std = 128.0

    image = Image.open(image_file)
    if image.mode != "RGB":
        image = image.convert("RGB")
    if len(hints) != 0:
        y1 = max(0, int(round((hints[0] - size[0]) / 2.0)))
        x1 = max(0, int(round((hints[1] - size[1]) / 2.0)))
        y2 = min(hints[0], y1 + size[0])
        x2 = min(hints[1], x1 + size[1])
        image = image.resize(hints)
        image = image.crop((x1, y1, x2, y2))
    else:
        image = image.resize((size[1], size[0]))

    image = np.ascontiguousarray(image)
    if mean[0] < 1 and std[0] < 1:
        image = image.astype(np.float32)
        image /= 255.0
        image -= np.array(mean)
        image /= np.array(std)
    else:
        image = image - np.array(mean)  # mean
        image /= np.array(std)  # std

    image = image.transpose((2, 0, 1))
    image = image[np.newaxis, :]

    return image


if __name__ == '__main__':
    img = get_image_data("images/datasets/imagenet.jpg")
    print(img.shape)
