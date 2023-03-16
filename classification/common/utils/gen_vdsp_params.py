import copy
import json
import os

import requests
import yaml
from tqdm import tqdm
from util import half_to_uint16

# def yaml_to_json(yaml_path):
#     with open(yaml_path, encoding="utf-8") as f:
#         datas = yaml.load(f, Loader=yaml.FullLoader)  # 将文件的内容转换为字典形式
#     json_datas = json.dumps(datas, indent=5, ensure_ascii=False)  # 将字典的内容转换为json格式的字符串
#     return json_datas

def json_to_yaml(json_path):
    if isinstance(json_path, str):
        with open(json_path, encoding="utf-8") as f:
            datas = json.load(f)  # 将文件的内容转换为字典形式
    elif isinstance(json_path, dict):
        datas = json_path
    else:
        raise ValueError("Make sure the input is a json path or dict")

    yaml_datas = yaml.dump(datas, indent=5, sort_keys=False, allow_unicode=True)  # 将字典的内容转换为yaml格式的字符串

    return yaml_datas


def update_dict_in_list(new_dict: dict = {}, config_list: list = []) -> list:
    new_config_list = copy.deepcopy(config_list)

    if "type" in new_dict.keys():
        for index, config_dict in enumerate(config_list):
            if "type" in config_dict.keys():
                if new_dict["type"] == config_dict["type"]:
                    new_config_list[index] = new_dict

    return new_config_list


def check_link_alive(link: str) -> bool:
    response = requests.get(link)
    if response.status_code == 200:
        return True
    else:
        return False



def update_model_node(json_dict: dict, model_config: dict, model_name: str, model_source: str, ir_type="onnx") -> dict:
    new_model_config = copy.deepcopy(model_config)
    new_model_config["model"]["name"] = model_name

    if model_source == "keras":
        ir_type = "h5"
    ir_dict = json_dict[list(json_datas.keys())[0]][ir_type][model_source]

    candidate_list = [
        model_name + "." + ir_type,
        model_name + "." + ir_type + ".pth",
        model_name + "." + ir_type + ".pt"
    ]

    checkpoint_path = ""
    for candidate_name in candidate_list:
        if candidate_name in ir_dict["file"]:
            checkpoint_path = os.path.join(ir_dict["link"].replace("?download=zip", ""), candidate_name)
            break
    print(ir_dict)
    if check_link_alive(checkpoint_path):
        new_model_config["model"]["checkpoint"] = checkpoint_path
    else:
        raise ValueError("IR model path can not find in: {}".format(checkpoint_path))

    return new_model_config


def updae_engine_node(model_config: dict, deploy_type="vacc", do_quantization=True, quant_mode="percentile") -> dict:
    new_model_config = copy.deepcopy(model_config)
    new_model_config["engine"]["type"] = deploy_type
    new_model_config["engine"]["common"]["do_quantization"] = do_quantization
    new_model_config["engine"]["calibration"]["quant_mode"] = quant_mode

    return new_model_config


def update_vdsp_params(model_config: dict, vdsp_parmas_dict: dict) -> dict:
    new_vdsp_parmas_dict = copy.deepcopy(vdsp_parmas_dict)
    if "dataset" in model_config.keys():
        if "transform_ops" in model_config["dataset"]:
            for update_dict in model_config["dataset"]["transform_ops"]:
                if update_dict["type"] == "Resize":
                    if "size" in update_dict.keys():
                        new_vdsp_parmas_dict["config"][1]["Value"] = update_dict["size"][1]
                        new_vdsp_parmas_dict["config"][2]["Value"] = update_dict["size"][0]
                        new_vdsp_parmas_dict["config"][3]["Value"] = update_dict["size"][1]
                        new_vdsp_parmas_dict["config"][4]["Value"] = update_dict["size"][0]
                        new_vdsp_parmas_dict["config"][5]["Value"] = update_dict["size"][1]
                if update_dict["type"] == "CenterCrop":
                        new_vdsp_parmas_dict["config"][6]["Value"] = update_dict["crop_size"][0]
                if update_dict["type"] == "Normalize":
                    if "div255" in update_dict.keys():
                        if "mean" in update_dict.keys():
                            if update_dict["div255"] == False :
                                new_vdsp_parmas_dict["config"][13]["Value"] = "MINUSMEAN_DIVSTD"
                            else:
                                new_vdsp_parmas_dict["config"][13]["Value"] = "DIV255_MINUSMEAN_DIVSTD"

                            new_vdsp_parmas_dict["config"][7]["Value"] = half_to_uint16(update_dict["mean"][0])
                            new_vdsp_parmas_dict["config"][8]["Value"] = half_to_uint16(update_dict["mean"][1])
                            new_vdsp_parmas_dict["config"][9]["Value"] = half_to_uint16(update_dict["mean"][2])
                            new_vdsp_parmas_dict["config"][10]["Value"] = half_to_uint16(update_dict["std"][0])
                            new_vdsp_parmas_dict["config"][11]["Value"] = half_to_uint16(update_dict["std"][1])
                            new_vdsp_parmas_dict["config"][12]["Value"] = half_to_uint16(update_dict["std"][2])
                        else:
                            new_vdsp_parmas_dict["config"][13]["Value"] = "DIV255"
                    elif "mean" in update_dict.keys():
                        new_vdsp_parmas_dict["config"][13]["Value"] = "DIV255_MINUSMEAN_DIVSTD"

                        new_vdsp_parmas_dict["config"][7]["Value"] = half_to_uint16(update_dict["mean"][0])
                        new_vdsp_parmas_dict["config"][8]["Value"] = half_to_uint16(update_dict["mean"][1])
                        new_vdsp_parmas_dict["config"][9]["Value"] = half_to_uint16(update_dict["mean"][2])
                        new_vdsp_parmas_dict["config"][10]["Value"] = half_to_uint16(update_dict["std"][0])
                        new_vdsp_parmas_dict["config"][11]["Value"] = half_to_uint16(update_dict["std"][1])
                        new_vdsp_parmas_dict["config"][12]["Value"] = half_to_uint16(update_dict["std"][2])

    return new_vdsp_parmas_dict



if __name__ == "__main__":
    pretrained_weights_json_path = "../../shufflenet_v2/pretrained_weights.json"
    convert_type = "onnx"
    yaml_save_dir = "../../shufflenet_v2/vacc_code/vdsp_params"

    # vdsp_params_json
    reference_vdsp_params_json_path = "../../shufflenet_v2/vacc_code/vdsp_params/megvii-shufflenet_v2_x0.5-vdsp_params.json"
    with open(reference_vdsp_params_json_path, encoding="utf-8") as f:
        ref_vdsp_parmas= json.load(f)

    with open(pretrained_weights_json_path, encoding="utf-8") as f:
        json_datas = json.load(f)

    base_model_name = list(json_datas.keys())[0]
    pretrained_data = json_datas[base_model_name]["pretrained"]
    base_config = json_datas[base_model_name]["config"]

    for model_name, model_data in pretrained_data.items():
        for model_source, model_source_data in model_data.items():
            yaml_save_path = os.path.join(yaml_save_dir, model_source + "-" + model_name + "-" + "config.yaml")
            model_config = copy.deepcopy(base_config)

            if "config" in model_source_data.keys():
                update_config = pretrained_data[model_name][model_source]["config"]

                if "model" in update_config.keys():
                    model_config["model"].update(update_config["model"])
                if "engine" in update_config.keys():
                    model_config["engine"].update(update_config["engine"])
                if "dataset" in update_config.keys():
                    if "transform_ops" in update_config["dataset"].keys():
                        update_transform_ops_list = update_config["dataset"]["transform_ops"]
                        for update_dict in update_transform_ops_list:
                            model_config["dataset"]["transform_ops"] = update_dict_in_list(update_dict, model_config["dataset"]["transform_ops"])
                    else:
                        model_config["dataset"].update(update_config["dataset"])

            # update super model config
            model_config = update_model_node(
                json_dict=json_datas,
                model_config=model_config,
                model_name=model_name,
                model_source=model_source,
                ir_type=convert_type
            )

            # update super engine config
            model_config = updae_engine_node(
                model_config=model_config,
                deploy_type="vacc",
                do_quantization=True,
                quant_mode="percentile"
            )

            # update vdsp params
            vdsp_parmas = update_vdsp_params(model_config, ref_vdsp_parmas)
            with open(yaml_save_path.replace("config.yaml", "vdsp_params.json"), 'w') as file:
                # file.write(yaml_data)
                json.dump(vdsp_parmas, file, ensure_ascii=False, indent=2)


            # save model config in yaml file
            # yaml_data = json_to_yaml(model_config)
            # with open(yaml_save_path, 'w') as file:
            #     file.write(yaml_data)

