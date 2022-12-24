import os
from imagenet import imagenet_info

def is_match_class(line, label_info):
    """
    Verify whether the vacc result is consistent with the label
    Args:
        line: vacc predict text
        label_info: imagenet label info
    Returns:
        match or not 
    """
    line = line.rstrip("\n")
    if len(line) == 0:
        return False
    line_info_path = line.split("/")
    line_info_relay = line.split(":")
    gt_label_dir = line_info_path[-2]

    dt_label_name = line_info_relay[4].strip()
    gt_label_name = label_info[gt_label_dir]

    return gt_label_name == dt_label_name

def imagenet_topk(txt_dir, model_name, topk=5, label_info=imagenet_info):
    """
    Statistics of top1&top5 of single model txt file
    Args:
        txt_dir: vacc predict txt file dir
        model_name: model name
        topk: top N with the highest prediction confidence
        label_info: imagenet_info dict
    Returns:
        top1&top5 value
    """
    total_count = 0
    top1_count = 0
    top5_count = 0
    with open(os.path.join(txt_dir, model_name + ".txt"), "r") as fout:
        lines = fout.readlines()
        for i in range(0, len(lines), topk):
            total_count += 1
            five_lines = lines[i : i + topk]
            matches = [is_match_class(line, label_info) for line in five_lines]
            if matches[0]:
                top1_count += 1
                top5_count += 1
            elif True in matches:
                top5_count += 1
    top1_rate = round(top1_count / total_count * 100, 5)
    top5_rate = round(top5_count / total_count * 100, 5)

    print("[VACC]: ", "top1_rate:", top1_rate, "top5_rate:", top5_rate)

    return top1_rate, top5_rate


if __name__ == "__main__":
    imagenet_topk(txt_dir="./output", model_name="resnet50", topk=5, label_info=imagenet_info)

