import numpy as np


def uint16_to_half(x):
    """Convert a uint16 represented number to np.float16
    """
    return np.frombuffer(np.array(x, dtype=np.uint16), dtype=np.float16)[0]


def half_to_uint16(x):
    """Convert a np.float16 number to a uint16 represented
    """
    return int(np.frombuffer(np.array(x, dtype=np.float16), dtype=np.uint16))
