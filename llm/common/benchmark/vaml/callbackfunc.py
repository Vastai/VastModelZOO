# # Copyright (C) 2022-2023 VASTAI Technologies Co., Ltd. All Rights Reserved.
# # coding: utf-8

# __all__ = ["registerErrorCallBack", "unRegisterErrorCallBack"]

# from _vaststream_pybind11 import vaml
# from typing import Any
# from .utils import *
# from .common import *

# # =========================== API =============================
# def registerErrorCallBack(errorCallbackPy:Any, userData:str) -> int:
#     """Register error call back function.

#     Args:
#         errorCallbackPy(Any): The callback function of type vamlRegisterErrorCallBack.
#         userData(str): The user defined data to be passed into the callback function.
    
#     Returns:
#         int: The return code. 0 for success, False otherwise.
#     """
#     return vaml.registerErrorCallBack(errorCallbackPy, userData)

# @err_check
# def unRegisterErrorCallBack() -> int:
#     """Unregister error call back function.

#     Returns:
#         int: The return code. 0 for success, False otherwise.
#     """
#     return vaml.unRegisterErrorCallBack()