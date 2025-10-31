# ==============================================================================
#
# Copyright (C) 2025 VastaiTech Technologies Inc.  All rights reserved.
#
# ==============================================================================
#!/usr/bin/env python
# -*- encoding: utf-8 -*-

"""
WiderFace evaluation code
author: wondervictor
mail: tianhengcheng@gmail.com
copyright@wondervictor
"""

from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy

package = Extension('bbox', ['box_overlaps.pyx'], include_dirs=[numpy.get_include()])
setup(ext_modules=cythonize([package]))
