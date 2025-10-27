# ==============================================================================
#
# Copyright (C) 2025 VastaiTech Technologies Inc.  All rights reserved.
#
# ==============================================================================
#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import copy
import warnings

from .coco import CocoDataset
from .xml_dataset import XMLDataset


def build_dataset(cfg, mode):
    dataset_cfg = copy.deepcopy(cfg)
    name = dataset_cfg.pop("name")
    if name == "coco":
        warnings.warn(
            "Dataset name coco has been deprecated. Please use CocoDataset instead."
        )
        return CocoDataset(mode=mode, **dataset_cfg)
    elif name == "xml_dataset":
        warnings.warn(
            "Dataset name xml_dataset has been deprecated. "
            "Please use XMLDataset instead."
        )
        return XMLDataset(mode=mode, **dataset_cfg)
    elif name == "CocoDataset":
        return CocoDataset(mode=mode, **dataset_cfg)
    elif name == "XMLDataset":
        return XMLDataset(mode=mode, **dataset_cfg)
    else:
        raise NotImplementedError("Unknown dataset type!")
