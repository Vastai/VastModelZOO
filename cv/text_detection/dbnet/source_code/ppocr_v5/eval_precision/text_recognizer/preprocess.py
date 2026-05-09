import cv2
import math
import numpy as np

class OCRReisizeNormImg:
    """for ocr image resize and normalization"""

    def __init__(self, rec_image_shape=[3, 48, 320], input_shape=None):
        super().__init__()
        self.rec_image_shape = rec_image_shape
        self.input_shape = input_shape
        self.max_imgW = 3200

    def resize_norm_img(self, img, max_wh_ratio):
        """resize and normalize the img"""
        imgC, imgH, imgW = self.rec_image_shape
        assert imgC == img.shape[2]
        imgW = int((imgH * max_wh_ratio))
        if imgW > self.max_imgW:
            resized_image = cv2.resize(img, (self.max_imgW, imgH))
            resized_w = self.max_imgW
            imgW = self.max_imgW
        else:
            h, w = img.shape[:2]
            ratio = w / float(h)
            if math.ceil(imgH * ratio) > imgW:
                resized_w = imgW
            else:
                resized_w = int(math.ceil(imgH * ratio))
            resized_image = cv2.resize(img, (resized_w, imgH))
        resized_image = resized_image.astype("float32")
        resized_image = resized_image.transpose((2, 0, 1)) / 255
        resized_image -= 0.5
        resized_image /= 0.5
        padding_im = np.zeros((imgC, imgH, imgW), dtype=np.float32)
        padding_im[:, :, 0:resized_w] = resized_image
        return padding_im

    def __call__(self, imgs):
        """apply"""
        if self.input_shape is None:
            return [self.resize(img) for img in imgs]
        else:
            return [self.staticResize(img) for img in imgs]

    def resize(self, img):
        imgC, imgH, imgW = self.rec_image_shape
        max_wh_ratio = imgW / imgH
        h, w = img.shape[:2]
        wh_ratio = w * 1.0 / h
        max_wh_ratio = max(max_wh_ratio, wh_ratio)
        img = self.resize_norm_img(img, max_wh_ratio)
        return img

    def staticResize(self, img):
        imgC, imgH, imgW = self.input_shape
        resized_image = cv2.resize(img, (int(imgW), int(imgH)))
        resized_image = resized_image.transpose((2, 0, 1)) / 255
        resized_image -= 0.5
        resized_image /= 0.5
        return resized_image

