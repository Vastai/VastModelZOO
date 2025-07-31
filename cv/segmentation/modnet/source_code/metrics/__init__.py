# ==============================================================================
#
# Copyright (C) 2025 VastaiTech Technologies Inc.  All rights reserved.
#
# ==============================================================================
#!/usr/bin/env python
# -*- encoding: utf-8 -*-


from .metric import MSE, MAD, Grad, Conn
from .stream_metrics import StreamSegMetrics

metrics_class_dict = {'mad': MAD, 'mse': MSE, 'grad': Grad, 'conn': Conn}
