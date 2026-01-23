# ==============================================================================
#
# Copyright (C) 2025 VastaiTech Technologies Inc.  All rights reserved.
#
# ==============================================================================
#!/usr/bin/env python
# -*- encoding: utf-8 -*-


def _fp8_block_quant_config(qconfig_data):
    from transformers.utils.quantization_config import FineGrainedFP8Config

    fp8_config = FineGrainedFP8Config(
        modules_to_not_convert=qconfig_data.get("ignore", [])
    ).to_dict()

    return {
        "fmt": "e4m3",
        **fp8_config,
    }


quant_config_dict = {"FP8_BLOCK": _fp8_block_quant_config}


def _infer_scheme_from_config(qconfig_data):
    return "FP8_BLOCK"


def get_quant_config(qconfig_data):
    scheme = _infer_scheme_from_config(qconfig_data)
    if scheme in quant_config_dict:
        return quant_config_dict[scheme](qconfig_data)
    else:
        raise ValueError(f"Unsupported quantization scheme: {scheme}")
