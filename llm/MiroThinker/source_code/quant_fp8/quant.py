# ==============================================================================
#
# Copyright (C) 2025 VastaiTech Technologies Inc.  All rights reserved.
#
# ==============================================================================
#!/usr/bin/env python
# -*- encoding: utf-8 -*-


# patch llmcompressor for vllm_vacc
import vapatch  # do not remove

import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from accelerate import infer_auto_device_map, init_empty_weights
from llmcompressor.utils.dev import dispatch_for_generation


def parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        default="/models/MiroThinker-v1.5-235B",
        help="Path to the model",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="./",
        help="Path to save the quantized model",
    )
    parser.add_argument(
        "--offload_config",
        type=str,
        help="File path to set the offloading configuration if the GPU memory is not enougt",
    )
    parser.add_argument(
        "--do_sample",
        action="store_true",
        help="Whether to use sampling for generation",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if not os.path.exists(args.model_path):
        raise ValueError(f"Model path {args.model_path} does not exist.")

    model = None
    if args.offload_config:
        import yaml

        # read max memory config from file
        with open(args.offload_config) as f:
            config = yaml.safe_load(f)
        max_memory = config.get("max_memory", None)
        if max_memory is None:
            raise ValueError("offload_config must contain max_memory field.")
        with init_empty_weights():
            dummy_model = AutoModelForCausalLM.from_pretrained(
                args.model_path, torch_dtype="auto"
            )
            device_map = infer_auto_device_map(
                dummy_model,
                max_memory=max_memory,
                no_split_module_classes=dummy_model._no_split_modules,
            )
            del dummy_model

        model = AutoModelForCausalLM.from_pretrained(
            args.model_path, torch_dtype="auto", device_map=device_map
        )
        setattr(model, "device_map_custom", device_map)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path, torch_dtype="auto"
        )

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    # Configure the quantization algorithm and scheme.
    # In this case, we:
    #   * quantize the weights to fp8 with per block 128x128 via ptq
    #   * quantize the input activations to fp8 with dynamic per token per group 128
    recipe = QuantizationModifier(
        targets="Linear",
        scheme="FP8_BLOCK",
        ignore=[
            "lm_head",
            "re:.*mlp.gate$",
            "re:.*input_layernorm$",
            "re:.*post_attention_layernorm$",
        ],
    )

    # Apply quantization.
    oneshot(model=model, recipe=recipe)

    if args.do_sample:
        # Confirm generations of the quantized model look sane.
        from llmcompressor.utils import dispatch_for_generation

        print("========== SAMPLE GENERATION ==============")
        dispatch_for_generation(model)
        input_ids = tokenizer("Hello my name is", return_tensors="pt").input_ids.to(
            model.device
        )
        output = model.generate(input_ids, max_new_tokens=20)
        print(tokenizer.decode(output[0]))
        print("==========================================")

    # Save to disk
    save_name = args.model_path.split("/")[-1] + "-FP8"
    save_dir = os.path.join(args.save_path, save_name)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
