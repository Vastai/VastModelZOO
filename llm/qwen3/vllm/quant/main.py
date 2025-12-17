#!/usr/bin/env python
# -*- encoding: utf-8 -*-


from datasets import load_dataset
from gptqmodel import GPTQModel, QuantizeConfig

model_id = "/models/Qwen/Qwen3-30B-A3B-Instruct-2507"
quant_path = "./weights/Qwen3-30B-A3B-Instruct-2507-GPTQ-Int4"


calibration_dataset = load_dataset("allenai/c4", data_files="en/c4-train.00001-of-01024.json.gz", split="train").select(range(1024))["text"]
quant_config = QuantizeConfig(bits=4, group_size=128, damp_percent=0.05, desc_act=False, mse=0.0)
model = GPTQModel.load(model_id, quant_config)
model.quantize(calibration_dataset, batch_size=64)
model.save(quant_path)
