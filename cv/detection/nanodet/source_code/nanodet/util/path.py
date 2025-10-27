# ==============================================================================
#
# Copyright (C) 2025 VastaiTech Technologies Inc.  All rights reserved.
#
# ==============================================================================
#!/usr/bin/env python
# -*- encoding: utf-8 -*-


import os

from .rank_filter import rank_filter


@rank_filter
def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def collect_files(path, exts):
    file_paths = []
    for maindir, subdir, filename_list in os.walk(path):
        for filename in filename_list:
            file_path = os.path.join(maindir, filename)
            ext = os.path.splitext(file_path)[1]
            if ext in exts:
                file_paths.append(file_path)
    return file_paths
