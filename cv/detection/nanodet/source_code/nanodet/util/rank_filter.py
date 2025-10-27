# ==============================================================================
#
# Copyright (C) 2025 VastaiTech Technologies Inc.  All rights reserved.
#
# ==============================================================================
#!/usr/bin/env python
# -*- encoding: utf-8 -*-



def rank_filter(func):
    def func_filter(local_rank=-1, *args, **kwargs):
        if local_rank < 1:
            return func(*args, **kwargs)
        else:
            pass

    return func_filter
