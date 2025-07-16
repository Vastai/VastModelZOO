import cv2
import numpy as np
import vaststreamx as vsx


def load_labels(file):
    with open(file) as fin:
        return [line.strip() for line in fin.readlines()]


def cv_bgr888_to_nv12(bgr888):
    yuv_image = cv2.cvtColor(bgr888, cv2.COLOR_BGR2YUV_I420)
    height, width = bgr888.shape[:2]
    y = yuv_image[:height, :]
    u = yuv_image[height : height + height // 4, :]
    v = yuv_image[height + height // 4 :, :]
    u = np.reshape(u, (height // 2, width // 2))
    v = np.reshape(v, (height // 2, width // 2))
    uv_plane = np.empty((height // 2, width), dtype=np.uint8)
    uv_plane[:, 0::2] = u
    uv_plane[:, 1::2] = v
    yuv_nv12 = np.concatenate((y, uv_plane), axis=0)
    return yuv_nv12


def cv_nv12_to_bgr888(nv12):
    height, width = nv12.shape[0] * 2 // 3, nv12.shape[1]
    y = nv12[:height, :]
    u = nv12[height:, 0::2]
    v = nv12[height:, 1::2]
    u = np.reshape(u, (height // 4, width))
    v = np.reshape(v, (height // 4, width))
    i420 = np.concatenate((y, u, v), axis=0)
    bgr888 = cv2.cvtColor(i420, cv2.COLOR_YUV2BGR_I420)
    return bgr888


def cv_bgr888_to_vsximage(bgr888, vsx_format, device_id):
    h, w = bgr888.shape[:2]
    if vsx_format == vsx.ImageFormat.BGR_INTERLEAVE:
        res = bgr888
    elif vsx_format == vsx.ImageFormat.BGR_PLANAR:
        res = np.array(bgr888).transpose(2, 0, 1)
    elif vsx_format == vsx.ImageFormat.RGB_INTERLEAVE:
        res = cv2.cvtColor(bgr888, cv2.COLOR_BGR2RGB)
    elif vsx_format == vsx.ImageFormat.RGB_PLANAR:
        res = np.array(cv2.cvtColor(bgr888, cv2.COLOR_BGR2RGB)).transpose(2, 0, 1)
    elif vsx_format == vsx.ImageFormat.YUV_NV12:
        res = cv_bgr888_to_nv12(bgr888=bgr888)
    else:
        assert False, f"Unsupport format:{vsx_format}"
    return vsx.create_image(
        res,
        vsx_format,
        w,
        h,
        device_id,
    )


def vsximage_to_cv_bgr888(vsx_image):
    image_np = vsx.as_numpy(vsx_image).squeeze()
    if vsx_image.format == vsx.ImageFormat.BGR_INTERLEAVE:
        return image_np
    elif vsx_image.format == vsx.ImageFormat.RGB_INTERLEAVE:
        return cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    elif vsx_image.format == vsx.ImageFormat.BGR_PLANAR:
        return np.transpose(image_np[[0, 1, 2], :, :], (1, 2, 0))
    elif vsx_image.format == vsx.ImageFormat.RGB_PLANAR:
        return np.transpose(image_np[[2, 1, 0], :, :], (1, 2, 0))
    elif vsx_image.format == vsx.ImageFormat.YUV_NV12:
        return cv_nv12_to_bgr888(image_np)
    elif vsx_image.format == vsx.ImageFormat.GRAY:
        return image_np
    else:
        assert False, f"Unrecognize format:{vsx_image.format}"

def imagetype_to_vsxformat(imagetype):
    if imagetype == 0:
        return vsx.ImageFormat.YUV_NV12
    elif imagetype == 5000:
        return vsx.ImageFormat.RGB_PLANAR
    elif imagetype == 5001:
        return vsx.ImageFormat.BGR_PLANAR
    elif imagetype == 5002:
        return vsx.ImageFormat.RGB_INTERLEAVE
    elif imagetype == 5003:
        return vsx.ImageFormat.BGR_INTERLEAVE
    elif imagetype == 5004:
        return vsx.ImageFormat.GRAY
    else:
        assert False, f"Unrecognize image type {imagetype}"
