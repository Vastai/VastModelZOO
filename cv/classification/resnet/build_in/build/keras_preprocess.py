import numpy as np
import cv2


def get_image_data(image_file, input_shape = [1, 3, 224, 224]):
    image = cv2.imread(image_file)
    
    resize_shape = [256, 256]
    input_size = input_shape[2:]
    # Resize
    height, width, _ = image.shape
    new_height = height * resize_shape[0] // min(image.shape[:2])
    new_width = width * resize_shape[1] // min(image.shape[:2])

    image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

    # Crop
    height, width, _ = image.shape
    starty = height // 2 - (input_size[0] // 2)
    startx = width // 2 - (input_size[1] // 2)

    image = image[starty:starty + input_size[0], startx:startx + input_size[1]]
    # assert image.shape[0] == input_size[0] and image.shape[1] == input_size[1], (image.shape, height, width)
    
    image = image - [103.939, 116.779, 123.68]
    # image = image[np.newaxis, :]
    
    return image

def get_image_data_v2(image_file, input_shape = [1, 3, 299, 299]):
    image = cv2.imread(image_file)
    
    resize_shape = [342, 342]
    input_size = input_shape[2:]
    # Resize
    height, width, _ = image.shape
    new_height = height * resize_shape[0] // min(image.shape[:2])
    new_width = width * resize_shape[1] // min(image.shape[:2])

    image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

    # Crop
    height, width, _ = image.shape
    starty = height//2 - (input_size[0] // 2)
    startx = width//2 - (input_size[1] // 2)

    image = image[starty:starty + input_size[0], startx:startx + input_size[1]]
    # assert image.shape[0] == input_size[0] and image.shape[1] == input_size[1], (image.shape, height, width)
    
    image = image / 127.5
    image = image - 1.0
    # image = image[np.newaxis, :]
    
    return image


if __name__ == '__main__':
    img = get_image_data("VastModelZOO/images/datasets/imagenet.jpg")
    print(img.shape)