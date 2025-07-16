from model import Net
import torch

torch.manual_seed(1024)
net = Net(reid=True)
batch_size = 1
input_shape = (batch_size, 3, 128, 64)
input = torch.randn(*input_shape)
# input[:] = 1

model_dict = torch.load('/path/to/mot/deepsort/ckpt.t7', map_location=torch.device('cpu'))
net.load_state_dict(model_dict, strict=False)

net.eval()
with torch.no_grad():
    golden_result = net(input)

input_names = ["input"]
output_names = ["output"]

torch.onnx.export(net,
                  input,
                  "fastreid.onnx",
                  export_params=True,
                  input_names=input_names, 
                  output_names=output_names, 
                  opset_version=10)

input_data = torch.randn(*input_shape)
scripted_model = torch.jit.trace(net, input_data)

# scripted_model = torch.jit.script(net)
torch.jit.save(scripted_model, 'fastreid.torchscript.pt')
