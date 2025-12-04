# ==============================================================================
#
# Copyright (C) 2025 VastaiTech Technologies Inc.  All rights reserved.
#
# ==============================================================================
#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import vaststreamx as vsx


class CustomOpBase:
    def __init__(self, op_name, elf_file, device_id):
        self.device_id_ = device_id
        vsx.set_device(device_id)
        self.custom_op_ = vsx.CustomOperator(op_name=op_name, elf_file_path=elf_file)
