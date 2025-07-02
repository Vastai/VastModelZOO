# Copyright (C) 2022-2023 VASTAI Technologies Co., Ltd. All Rights Reserved.
# coding: utf-8

__all__ = ["err_check"]

from functools import wraps
from ._vaststream_pybind11 import vaml as _vaml


def err_check(func):

    @wraps(func)
    def wrapper(*args, **kwargs):
        ret = func(*args, **kwargs)
        if ret != _vaml.vamlER_SUCCESS:
            raise RuntimeError(f"{func.__name__} error, ret: {ret}.")
        return ret

    return wrapper