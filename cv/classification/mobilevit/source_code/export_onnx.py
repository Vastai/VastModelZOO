import torch
from cvnets import get_model

from options.opts import get_eval_arguments
from utils.common_utils import device_setup


opts = get_eval_arguments()
opts = device_setup(opts)
onnx_model_path = "mobilevit_s.onnx"

model = get_model(opts)
model.eval()
torch_input = torch.randn(1, 3, 224, 224, dtype=torch.float32)
dynamic_axes = {
    'input': {0: 'batch_size'},
    'output': {0: 'batch_size'}
}
# export
torch.onnx.export(
    model,
    (torch_input),
    onnx_model_path,
    dynamic_axes=dynamic_axes,
    opset_version=12,
    input_names=["input"],
    output_names=["output"],
)
print("success.")