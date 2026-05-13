#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import os
from datasets import load_dataset
from gptqmodel import GPTQModel, QuantizeConfig
from safetensors.torch import safe_open

model_id = "/cx8k/fs101/share/models/Qwen/Qwen3-30B-A3B-Instruct-2507"
quant_path = "./weights/Qwen3-30B-A3B-Instruct-2507-GPTQ-Int4"

calibration_dataset = load_dataset(
    "allenai/c4",
    data_files="en/c4-train.00001-of-01024.json.gz",
    split="train",
).select(range(1024))["text"]

quant_config = QuantizeConfig(
    bits=4,
    group_size=128,
    damp_percent=0.05,
    desc_act=False,
    mse=0.0,
    dynamic={
        r"-:.*\.mlp\.gate$": {},
    },
)

model = GPTQModel.load(model_id, quant_config)


# =========================
# 量化前校验
# =========================

print("\nWill skip these MoE router gate modules:")

gate_count = 0

for name, module in model.model.named_modules():
    if name.endswith(".mlp.gate"):
        print(name, type(module))
        gate_count += 1

print(f"\nTotal mlp.gate modules: {gate_count}")

if gate_count == 0:
    raise RuntimeError("No .mlp.gate modules found. Please check module names.")


# =========================
# 量化并保存
# =========================

model.quantize(calibration_dataset, batch_size=48)
model.save(quant_path)


# =========================
# 量化后校验
# =========================

print("\nChecking saved weights...")

gate_fp_keys = []
gate_quant_keys = []

for filename in os.listdir(quant_path):
    if not filename.endswith(".safetensors"):
        continue

    path = os.path.join(quant_path, filename)

    with safe_open(path, framework="pt", device="cpu") as f:
        for key in f.keys():
            if ".mlp.gate." not in key:
                continue

            if any(x in key for x in [".qweight", ".scales", ".qzeros", ".g_idx"]):
                gate_quant_keys.append(key)
            else:
                gate_fp_keys.append(key)

print("\nSaved fp/bf16/fp16 mlp.gate keys:")
for key in gate_fp_keys[:20]:
    print(key)

print(f"\nTotal fp mlp.gate keys: {len(gate_fp_keys)}")

print("\nSaved quantized mlp.gate keys:")
for key in gate_quant_keys[:20]:
    print(key)

print(f"\nTotal quantized mlp.gate keys: {len(gate_quant_keys)}")

if gate_quant_keys:
    raise RuntimeError("mlp.gate was quantized. Skip rule did not work.")

print("\nOK: mlp.gate was skipped successfully.")
print(f"Saved to: {quant_path}")
