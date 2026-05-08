#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import onnx
import argparse
from onnx import helper, shape_inference


def get_attr(node, name, default=None):
    for a in node.attribute:
        if a.name == name:
            return list(a.ints) if a.ints else a.i if a.type == 2 else a.s.decode("utf-8")
    return default


def remove_attr(node, name):
    keep = [a for a in node.attribute if a.name != name]
    del node.attribute[:]
    node.attribute.extend(keep)


def build_shape_map(model):
    shape_map = {}

    def collect(vi):
        if not vi.type.HasField("tensor_type"):
            return
        dims = []
        for d in vi.type.tensor_type.shape.dim:
            if d.HasField("dim_value"):
                dims.append(d.dim_value)
            else:
                dims.append(None)  # 动态维度
        shape_map[vi.name] = dims

    for x in model.graph.input:
        collect(x)
    for x in model.graph.value_info:
        collect(x)
    for x in model.graph.output:
        collect(x)
    return shape_map


def compute_pads_1d(in_size, k, s, d, auto_pad_mode):
    # effective kernel
    ek = (k - 1) * d + 1

    if auto_pad_mode == "VALID":
        total = 0
    else:
        # SAME_UPPER / SAME_LOWER
        if in_size is None:
            # 当输入空间维是动态时：
            # - stride=1 可直接得到 total=effective_kernel-1（与输入尺寸无关）
            # - stride>1 时 total 与输入尺寸相关，无法唯一确定
            if s == 1:
                total = ek - 1
            else:
                return None, None
        else:
            out_size = math.ceil(in_size / s)
            total = max((out_size - 1) * s + ek - in_size, 0)

    if auto_pad_mode == "SAME_LOWER":
        pad_head = (total + 1) // 2
        pad_tail = total // 2
    else:  # SAME_UPPER / VALID
        pad_head = total // 2
        pad_tail = total - pad_head

    return pad_head, pad_tail


def convert_auto_pad_to_pads(src, dst):
    model = onnx.load(src)
    model = shape_inference.infer_shapes(model)
    shape_map = build_shape_map(model)

    changed = 0
    target_ops = {"Conv", "MaxPool", "AveragePool"}

    for node in model.graph.node:
        if node.op_type not in target_ops:
            continue

        auto_pad = get_attr(node, "auto_pad", None)
        if not auto_pad or auto_pad == "NOTSET":
            continue
        if auto_pad not in {"SAME_UPPER", "SAME_LOWER", "VALID"}:
            continue

        # 如果已有 pads，不改 pads，但仍移除 auto_pad
        old_pads = get_attr(node, "pads", None)
        if old_pads is not None:
            remove_attr(node, "auto_pad")
            changed += 1
            print(f"[OK] {node.name or '(unnamed)'}: 删除 auto_pad，保留原 pads={old_pads}")
            continue

        if not node.input:
            continue
        in_name = node.input[0]
        in_shape = shape_map.get(in_name, None)
        if not in_shape or len(in_shape) < 3:
            # NCHW/NC... 至少要有空间维
            continue

        # 空间维: Conv/Pool 一般是 N,C,(D),H,W
        spatial_rank = len(in_shape) - 2
        in_spatial = in_shape[2:]
        kernel = get_attr(node, "kernel_shape", None)
        if not kernel:
            print(f"[WARN] 跳过 {node.name or '(unnamed)'}: 缺少 kernel_shape")
            continue

        strides = get_attr(node, "strides", [1] * spatial_rank)
        dilations = get_attr(node, "dilations", [1] * spatial_rank)

        # 长度对齐（防御性处理）
        if len(kernel) != spatial_rank:
            print(f"[WARN] 跳过 {node.name or '(unnamed)'}: kernel rank 不匹配")
            continue
        if len(strides) != spatial_rank:
            strides = (strides + [1] * spatial_rank)[:spatial_rank]
        if len(dilations) != spatial_rank:
            dilations = (dilations + [1] * spatial_rank)[:spatial_rank]

        pad_heads = []
        pad_tails = []
        for i in range(spatial_rank):
            ph, pt = compute_pads_1d(
                in_size=in_spatial[i],
                k=kernel[i],
                s=strides[i],
                d=dilations[i],
                auto_pad_mode=auto_pad,
            )
            if ph is None:
                print(
                    f"[WARN] 跳过 {node.name or '(unnamed)'}: 维度{i}为动态且stride>1，"
                    "无法从 auto_pad 唯一推导 pads；请提供静态输入尺寸后再转换。"
                )
                pad_heads = []
                pad_tails = []
                break
            pad_heads.append(ph)
            pad_tails.append(pt)

        if not pad_heads:
            continue

        new_pads = pad_heads + pad_tails

        remove_attr(node, "auto_pad")
        node.attribute.append(helper.make_attribute("pads", new_pads))
        changed += 1
        print(f"[OK] {node.name or '(unnamed)'}: auto_pad={auto_pad} -> pads={new_pads}")

    onnx.save(model, dst)
    print(f"\n完成: 共修改 {changed} 个节点")
    print(f"输出模型: {dst}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="convert_auto_pad_to_pads")
    parser.add_argument("--input_onnx", default="weights/PP-OCRv5_server_det_infer_inference_sim.onnx", help="input_onnx")
    parser.add_argument("--output_onnx", default="weights/PP-OCRv5_server_det_infer_inference_sim_pads.onnx", help="input_onnx")

    args = parser.parse_args()
    convert_auto_pad_to_pads(args.input_onnx, args.output_onnx)
