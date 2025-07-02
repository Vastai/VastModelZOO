# # Copyright (C) 2022-2023 VASTAI Technologies Co., Ltd. All Rights Reserved.
# # coding: utf-8

# __all__ = [
#     "PROF_EXPORT_FILE_TYPE", "ProfConfig", "profStart", "profStop"
# ]

# from _vaststream_pybind11 import vaml as _vaml
# from .utils import *

# # =========================== ENUM ============================
# class PROF_EXPORT_FILE_TYPE():
#     """An enum that defines profiler export file type.

#     Contains CSV_TYPE, TRACEVIEW_TYPE, BOTH_CSV_AND_TRACEVIEW_TYPE, UNSPECIFIED_TYPE.
#     """
#     CSV_TYPE: int = _vaml.profExportFileType.CSV_TYPE
#     TRACEVIEW_TYPE: int = _vaml.profExportFileType.TRACEVIEW_TYPE
#     BOTH_CSV_AND_TRACEVIEW_TYPE: int = _vaml.profExportFileType.BOTH_CSV_AND_TRACEVIEW_TYPE
#     UNSPECIFIED_TYPE: int = _vaml.profExportFileType.UNSPECIFIED_TYPE

# # =========================== STRUCT =============================
# class ProfConfig(_vaml.profConfig):
#     """A struct that defines profiler config.

#     Attributes:
#         fileType(PROF_EXPORT_FILE_TYPE):The file type 0:csv; 1:tracing json; 2:both csv and tracing json
#         retPath(str): The file path.
#         exeCommand(str): The execute command (AI shell script) path.
#     """
#     fileType: PROF_EXPORT_FILE_TYPE
#     retPath: str
#     exeCommand: str

# # =========================== API =============================
# @err_check
# def profStart(profConfig:ProfConfig) -> int:
#     """Start profiler and begin to monitor the profiler.

#     Args:
#         profConfig(ProfConfig): The performance monitoring config.
    
#     Returns:
#         int: The return code. 0 for success, False otherwise.
#     """
#     return _vaml.profStart(profConfig)
    
# @err_check
# def profStop() -> int:
#     """Stop profiler.

#     Returns:
#         int: The return code. 0 for success, False otherwise.
#     """
#     return _vaml.profStop()