# ==============================================================================
#
# Copyright (C) 2025 VastaiTech Technologies Inc.  All rights reserved.
#
# ==============================================================================
#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import sys
import os
import subprocess

python_path = sys.executable

ori_path = os.getcwd()
_cur_file_path = os.path.split(os.path.realpath(__file__))[0]
os.chdir(f'{_cur_file_path}/../../../ppocr/pse_postprocess/pse')
if subprocess.call(
        '{} setup.py build_ext --inplace'.format(python_path), shell=True) != 0:
    raise RuntimeError(
        'Cannot compile pse: {}, if your system is windows, you need to install all the default components of `desktop development using C++` in visual studio 2019+'.
        format(os.path.dirname(os.path.realpath(__file__))))
os.chdir(ori_path)

from .pse import pse
