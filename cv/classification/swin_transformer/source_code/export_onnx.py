import torch
import onnx
import os
from swin_transformer import SwinTransformer
from onnxsim import simplify

# file name
model_path = "./swin_base_patch4_window7_224.pth"
onnx_model_name = "swin_base_patch4_window7_224.onnx"
onnxsim_model_name = "swin_base_patch4_window7_224_sim.onnx"

# torch2onnx
layernorm = torch.nn.LayerNorm
model = SwinTransformer(img_size=224,patch_size=4,in_chans=3,num_classes=1000,embed_dim=128,depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],window_size=7,mlp_ratio=4. ,qkv_bias=True, qk_scale=None, drop_rate=0,
        drop_path_rate=0.5, ape= False, norm_layer=layernorm, patch_norm=True, use_checkpoint=False, fused_window_process=False)

device = torch.device('cpu')
model.to(device)
state_dict = torch.load(model_path, map_location=device)
model.load_state_dict(state_dict["model"])
model.eval()

torch_input = torch.randn(1, 3, 224, 224, dtype=torch.float32)

onnx_model_path = os.path.join(os.path.dirname(model_path), onnx_model_name)

torch.onnx.export(
    model,
    (torch_input),
    onnx_model_path,
    input_names=["input"],
    output_names=["output"],
)
# onnxsim
onnx_model = onnx.load_model(onnx_model_path, load_external_data=True)
model_simp, check = simplify(onnx_model)
onnx.save(
    onnx.shape_inference.infer_shapes(model_simp),
    os.path.join(os.path.dirname(model_path), onnxsim_model_name),
    save_as_external_data=True,
    all_tensors_to_one_file=True
)
assert check, "Simplified ONNX model could not be validated"
print("export successed")