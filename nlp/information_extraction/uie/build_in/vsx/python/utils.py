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
    height, width = int(nv12.shape[0] * 2 // 3), nv12.shape[1]
    y = nv12[:height, :]
    u = nv12[height:, 0::2]
    v = nv12[height:, 1::2]
    u = np.reshape(u, (height // 2, width // 2))
    v = np.reshape(v, (height // 2, width // 2))
    uv = np.concatenate((u, v), axis=0)
    uv = np.reshape(uv, (uv.shape[0] // 2, uv.shape[1] * 2))
    i420 = np.concatenate((y, uv), axis=0)
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


def cvtcolorcode_to_vsxformat(cvtcolor_code):
    if cvtcolor_code == 0:
        return (
            vsx.ImageFormat.YUV_NV12,
            vsx.ImageFormat.RGB_PLANAR,
            vsx.ColorCvtCode.YUV2RGB_NV12,
        )
    elif cvtcolor_code == 1:
        return (
            vsx.ImageFormat.YUV_NV12,
            vsx.ImageFormat.BGR_PLANAR,
            vsx.ColorCvtCode.YUV2BGR_NV12,
        )
    elif cvtcolor_code == 3:
        return (
            vsx.ImageFormat.BGR_PLANAR,
            vsx.ImageFormat.RGB_PLANAR,
            vsx.ColorCvtCode.BGR2RGB,
        )
    elif cvtcolor_code == 4:
        return (
            vsx.ImageFormat.BGR_INTERLEAVE,
            vsx.ImageFormat.RGB_PLANAR,
            vsx.ColorCvtCode.BGR2RGB_INTERLEAVE2PLANAR,
        )
    elif cvtcolor_code == 5:
        return (
            vsx.ImageFormat.RGB_INTERLEAVE,
            vsx.ImageFormat.BGR_PLANAR,
            vsx.ColorCvtCode.RGB2BGR_INTERLEAVE2PLANAR,
        )
    elif cvtcolor_code == 6:
        return (
            vsx.ImageFormat.BGR_INTERLEAVE,
            vsx.ImageFormat.BGR_PLANAR,
            vsx.ColorCvtCode.BGR2BGR_INTERLEAVE2PLANAR,
        )
    elif cvtcolor_code == 7:
        return (
            vsx.ImageFormat.RGB_INTERLEAVE,
            vsx.ImageFormat.RGB_PLANAR,
            vsx.ColorCvtCode.RGB2RGB_INTERLEAVE2PLANAR,
        )
    elif cvtcolor_code == 8:
        return (
            vsx.ImageFormat.YUV_NV12,
            vsx.ImageFormat.GRAY,
            vsx.ColorCvtCode.YUV2GRAY_NV12,
        )
    elif cvtcolor_code == 9:
        return (
            vsx.ImageFormat.BGR_INTERLEAVE,
            vsx.ImageFormat.GRAY,
            vsx.ColorCvtCode.BGR2GRAY_INTERLEAVE,
        )
    elif cvtcolor_code == 10:
        return (
            vsx.ImageFormat.BGR_PLANAR,
            vsx.ImageFormat.GRAY,
            vsx.ColorCvtCode.BGR2GRAY_PLANAR,
        )
    elif cvtcolor_code == 11:
        return (
            vsx.ImageFormat.RGB_INTERLEAVE,
            vsx.ImageFormat.GRAY,
            vsx.ColorCvtCode.RGB2GRAY_INTERLEAVE,
        )
    elif cvtcolor_code == 12:
        return (
            vsx.ImageFormat.RGB_PLANAR,
            vsx.ImageFormat.GRAY,
            vsx.ColorCvtCode.RGB2GRAY_PLANAR,
        )
    elif cvtcolor_code == 13:
        return (
            vsx.ImageFormat.RGB_PLANAR,
            vsx.ImageFormat.YUV_NV12,
            vsx.ColorCvtCode.RGB2YUV_NV12_PLANAR,
        )
    elif cvtcolor_code == 14:
        return (
            vsx.ImageFormat.BGR_PLANAR,
            vsx.ImageFormat.YUV_NV12,
            vsx.ColorCvtCode.BGR2YUV_NV12_PLANAR,
        )
    else:
        assert False, f"Unsuport cvtcolor code: {cvtcolor_code}"


def cv_rgb_image_to_vastai(image_cv, device_id):
    assert len(image_cv.shape) >= 2
    h = image_cv.shape[0]
    w = image_cv.shape[1]
    if len(image_cv.shape) == 3:
        return vsx.create_image(
            image_cv, vsx.ImageFormat.BGR_INTERLEAVE, w, h, device_id
        )
    elif len(image_cv.shape) == 2:
        return vsx.create_image(image_cv, vsx.ImageFormat.GRAY, w, h, device_id)
    else:
        raise Exception("unsupported ndarray shape", image_cv.shape)


def get_activation_aligned(
    activation, dtype=np.float16, fc_mode=False, force_int8_layout_to_fp16=False
):  # NCHW
    N = C = H = W = 1
    if len(activation.shape) == 2:
        N, C = activation.shape
        fc_mode = True
    elif len(activation.shape) == 5:
        N, C, H, W, B = activation.shape
    elif len(activation.shape) == 1:
        (C,) = activation.shape
    else:
        N, C, H, W = activation.shape
    h_group = w_group = c_group = 0
    if H == 1 and W == 1 and fc_mode == True:
        if dtype == np.float16:
            h_group, w_group, c_group = 1, 1, 256
        elif dtype == np.int8:
            h_group, w_group, c_group = 1, 1, 512
    else:
        if dtype == np.float16 or force_int8_layout_to_fp16:
            h_group, w_group, c_group = 8, 8, 4
        elif dtype == np.int8:
            h_group, w_group, c_group = 8, 8, 8
    pad_H, pad_W, pad_C = H, W, C
    if H % h_group != 0:
        pad_h = h_group - H % h_group
        pad_H += pad_h
    if W % w_group != 0:
        pad_w = w_group - W % w_group
        pad_W += pad_w
    if C % c_group != 0:
        pad_c = c_group - C % c_group
        pad_C += pad_c
    # tensorize to WHC4c8h8w
    w_num = pad_W // w_group
    h_num = pad_H // h_group
    c_num = pad_C // c_group
    n_num = N
    block_size = w_group * h_group * c_group
    activation = activation.astype(dtype)
    np_arr = np.zeros((n_num, w_num, h_num, c_num, block_size), dtype)
    for n in range(N):
        for c in range(C):
            for h in range(H):
                for w in range(W):
                    addr = (
                        (c % c_group) * h_group * w_group
                        + (h % h_group) * w_group
                        + (w % w_group)
                    )
                    if len(activation.shape) == 2:
                        np_arr[n, w // w_group, h // h_group, c // c_group, addr] = (
                            activation[n, c]
                        )
                    elif len(activation.shape) == 1:
                        np_arr[n, w // w_group, h // h_group, c // c_group, addr] = (
                            activation[n]
                        )
                    else:
                        np_arr[n, w // w_group, h // h_group, c // c_group, addr] = (
                            activation[n, c, h, w]
                        )
    return np_arr


def get_activation_aligned_faster(
    activation,
    dtype=np.float16,
    fc_mode=False,
    force_int8_layout_to_fp16=False,
    is_input=True,
):  # NCHW
    N = C = H = W = 1
    if len(activation.shape) == 2:
        N, C = activation.shape
        fc_mode = True
    elif len(activation.shape) == 5:
        N, C, H, W, B = activation.shape
    elif len(activation.shape) == 1:
        (C,) = activation.shape
    else:
        N, C, H, W = activation.shape
    h_group = w_group = c_group = 0
    if H == 1 and W == 1 and fc_mode == True:
        if dtype == np.float16:
            h_group, w_group, c_group = 1, 1, 256
        elif dtype == np.int8:
            h_group, w_group, c_group = 1, 1, 512
    else:
        if dtype == np.float16 or force_int8_layout_to_fp16:
            h_group, w_group, c_group = 8, 8, 4
        elif dtype == np.int8:
            h_group, w_group, c_group = 8, 8, 8
    pad_H, pad_W, pad_C = H, W, C
    pad_h, pad_w = 0, 0
    if H % h_group != 0:
        pad_h = h_group - H % h_group
        pad_H += pad_h
    if W % w_group != 0:
        pad_w = w_group - W % w_group
        pad_W += pad_w
    if C % c_group != 0:
        pad_c = c_group - C % c_group
        pad_C += pad_c
    # tensorize to WHC4c8h8w
    w_num = pad_W // w_group
    h_num = pad_H // h_group
    c_num = pad_C // c_group
    n_num = N
    block_size = w_group * h_group * c_group
    if activation.dtype != dtype:
        activation = activation.astype(dtype)
    assert len(activation.shape) == 4
    if (pad_h | pad_w) != 0:
        activation = np.pad(activation, ((0, 0), (0, 0), (0, pad_h), (0, pad_w)))
    np_arr = np.zeros((n_num, w_num, h_num, c_num, block_size), dtype)
    block_size_hacked = 3 * 8 * 8 if is_input else 4 * 8 * 8
    c_group_hacked = 3 if is_input else 4
    for n in range(N):
        for c in range(c_num):
            c_index = c * c_group_hacked
            for h in range(h_num):
                h_index = h * h_group
                for w in range(w_num):
                    w_index = w * w_group
                    np_arr[n, w, h, c, :block_size_hacked] = activation[
                        n,
                        c_index : c_index + c_group_hacked,
                        h_index : h_index + h_group,
                        w_index : w_index + w_group,
                    ].flatten()
    return np_arr


def get_activation_aligned_faster_1(activate):
    n, channel, height, width = activate.shape
    # 如果通道数不是4的倍数，则用0填充通道数至4的倍数
    if channel % 4 != 0 or height % 8 != 0 or width % 8 != 0:
        pad_c, pad_h, pad_w = 0, 0, 0
        if channel % 4 != 0:
            pad_c = 4 - channel % 4
        if height % 8 != 0:
            pad_h = 8 - height % 8
        if width % 8 != 0:
            pad_w = 8 - width % 8
        activate = np.pad(
            activate, ((0, 0), (0, pad_c), (0, pad_h), (0, pad_w)), mode="constant"
        )
        n, channel, height, width = activate.shape
    activate = activate.reshape(1, channel // 4, 4, height // 8, 8, width // 8, 8)
    activate = activate.transpose(0, 5, 3, 1, 2, 4, 6)
    activate = activate.reshape(1, width // 8, height // 8, channel // 4, 256)
    return activate


def bert_get_activation_fp16_A(activation, rep_dtype=None):  # NCHW
    # pdb.set_trace()
    if activation.ndim == 2:
        M, K = activation.shape
        activation = activation.reshape((1, M, K))
    N, M, K = activation.shape

    m_group, k_group = 16, 16

    pad_M, pad_K = M, K
    if M % m_group != 0:
        pad_m = m_group - M % m_group
        pad_M += pad_m

    if K % k_group != 0:
        pad_k = k_group - K % k_group
        pad_K += pad_k

    # tensorize to MK16m16k
    n_num = N
    m_num = pad_M // m_group
    k_num = pad_K // k_group
    block_size = m_group * k_group
    activation = activation.astype(np.float16)
    np_arr = np.zeros((n_num, m_num, k_num, block_size), np.float16)

    for n in range(N):
        for m in range(M):
            for k in range(K):
                addr = (m % m_group) * k_group + (k % k_group)
                np_arr[n, m // m_group, k // k_group, addr] = activation[n, m, k]
    return np_arr


def coco80_to_coco91_class():  # converts 80-index (val2014) to 91-index (paper)
    x = [
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        27,
        28,
        31,
        32,
        33,
        34,
        35,
        36,
        37,
        38,
        39,
        40,
        41,
        42,
        43,
        44,
        46,
        47,
        48,
        49,
        50,
        51,
        52,
        53,
        54,
        55,
        56,
        57,
        58,
        59,
        60,
        61,
        62,
        63,
        64,
        65,
        67,
        70,
        72,
        73,
        74,
        75,
        76,
        77,
        78,
        79,
        80,
        81,
        82,
        84,
        85,
        86,
        87,
        88,
        89,
        90,
    ]
    return x
