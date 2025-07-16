import torch
from hubconf import detr_resnet50
import onnx
from onnxsim import simplify

def simplify_onnx(onnx_file):

    onnx_model = onnx.load_model(onnx_file)
    print(f"load onnx model from {onnx_file} to simplify.")

    model_simp, check = simplify(onnx_model)
    assert check, "Simplified ONNX model could not be validated"
    onnx.save(
        model_simp,
        onnx_file.replace(".onnx", "_sim.onnx")
    )
    print("onnx save successed.")

model = detr_resnet50(pretrained=True).eval()
dummy_image = torch.randn(1, 3, 1066, 800)
onnx_file = "detr_res50.onnx"

torch.onnx.export(
    model,
    dummy_image,
    onnx_file,
    opset_version=11,
    do_constant_folding=True,
    input_names=["inputs"],
    output_names=["pred_logits", "pred_boxes"]
)

simplify_onnx(onnx_file)