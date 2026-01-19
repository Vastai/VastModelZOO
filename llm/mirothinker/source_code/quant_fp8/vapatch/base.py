# ==============================================================================
#
# Copyright (C) 2025 VastaiTech Technologies Inc.  All rights reserved.
#
# ==============================================================================
#!/usr/bin/env python
# -*- encoding: utf-8 -*-


from functools import wraps
from vapatch.quant_config import get_quant_config
import importlib
import sys
from compressed_tensors.utils import remove_dispatch
from accelerate import dispatch_model

def _patch_dispatch_for_generation():
    import llmcompressor.utils.dev as dfg
    original = dfg.dispatch_for_generation

    @wraps(original)
    def wrapper(model):
        device_map_custom = getattr(model, "device_map_custom", None)
        if device_map_custom is not None:
            remove_dispatch(model)
            return dispatch_model(model, device_map=device_map_custom)
        return original(model)

    dfg.dispatch_for_generation = wrapper

    # reload and update existing references
    importlib.reload(dfg)
    for mod_name, module in list(sys.modules.items()):
        if module and hasattr(module, 'dispatch_for_generation'):
            if getattr(module, 'dispatch_for_generation') is original:
                setattr(module, 'dispatch_for_generation', wrapper)


def _patch_quantization_compressor():
    from compressed_tensors.compressors.quantized_compressors.naive_quantized import (
        FloatQuantizationCompressor,
    )

    original = FloatQuantizationCompressor.compress

    @wraps(original)
    def wrapper(self, *args, **kwargs):
        compressed_dict = original(self, *args, **kwargs)
        compressed_dict = {
            key.replace("weight_scale", "weight_scale_inv"): value
            for key, value in compressed_dict.items()
        }
        return compressed_dict

    FloatQuantizationCompressor.compress = wrapper


def _patch_model_compressor():
    from compressed_tensors.compressors.model_compressors import ModelCompressor

    def update_config(self, save_directory: str):
        import os
        import json
        from compressed_tensors.base import (
            QUANTIZATION_CONFIG_NAME,
        )
        from transformers.file_utils import CONFIG_NAME

        # this check is also done in `from_pretrained_model`,
        # but not in `from_pretrained`` or `from_compression_config``
        if not any(
            (self.quantization_config, self.sparsity_config, self.transform_config)
        ):
            return

        # write to config.json file, regardless of whether it exists already
        # overwrite previous config and version if already existing
        config_file_path = os.path.join(save_directory, CONFIG_NAME)
        if os.path.exists(config_file_path):
            with open(config_file_path, "r") as file:
                config_data = json.load(file)
        else:
            config_data = {}

        # serialize configs into json
        qconfig_data = (
            self.quantization_config.model_dump(exclude=["quant_method"])
            if self.quantization_config is not None
            else {}
        )

        # patch quantization_config to adapter vllm_vacc 0.9.2
        config_data[QUANTIZATION_CONFIG_NAME] = get_quant_config(qconfig_data)

        # write results to config.json file
        with open(config_file_path, "w") as config_file:
            json.dump(config_data, config_file, indent=2, sort_keys=True)

    ModelCompressor.update_config = update_config


def patch_compressor():
    print("Patching llmcompressor...")
    _patch_dispatch_for_generation()
    _patch_quantization_compressor()
    _patch_model_compressor()

patch_compressor()
