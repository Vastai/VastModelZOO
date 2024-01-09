# yolov6 

## onnx导出
1. 将`yolov6/models/effidehead.py`中`Detect`类替换成如下代码
    ```python
    class Detect(nn.Module):
    '''Efficient Decoupled Head
    With hardware-aware degisn, the decoupled head is optimized with
    hybridchannels methods.
    '''
    def __init__(self, num_classes=80, anchors=1, num_layers=3, inplace=True, head_layers=None):  # detection layer
        super().__init__()
        assert head_layers is not None
        self.nc = num_classes  # number of classes
        self.no = num_classes + 5  # number of outputs per anchor
        self.nl = num_layers  # number of detection layers
        if isinstance(anchors, (list, tuple)):
            self.na = len(anchors[0]) // 2
        else:
            self.na = anchors
        self.anchors = anchors
        self.grid = [torch.zeros(1)] * num_layers
        self.prior_prob = 1e-2
        self.inplace = inplace
        stride = [8, 16, 32]  # strides computed during build
        self.stride = torch.tensor(stride)

        # Init decouple head
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.obj_preds = nn.ModuleList()
        self.stems = nn.ModuleList()

        # Efficient decoupled head layers
        for i in range(num_layers):
            idx = i*6
            self.stems.append(head_layers[idx])
            self.cls_convs.append(head_layers[idx+1])
            self.reg_convs.append(head_layers[idx+2])
            self.cls_preds.append(head_layers[idx+3])
            self.reg_preds.append(head_layers[idx+4])
            self.obj_preds.append(head_layers[idx+5])

    def initialize_biases(self):
        for conv in self.cls_preds:
            b = conv.bias.view(self.na, -1)
            b.data.fill_(-math.log((1 - self.prior_prob) / self.prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
        for conv in self.obj_preds:
            b = conv.bias.view(self.na, -1)
            b.data.fill_(-math.log((1 - self.prior_prob) / self.prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def forward(self, x):
        z = []
        out = []
        for i in range(self.nl):
            x[i] = self.stems[i](x[i])
            cls_x = x[i]
            reg_x = x[i]
            cls_feat = self.cls_convs[i](cls_x)
            cls_output = self.cls_preds[i](cls_feat)
            reg_feat = self.reg_convs[i](reg_x)
            reg_output = self.reg_preds[i](reg_feat)
            obj_output = self.obj_preds[i](reg_feat)
            out.append([reg_output, obj_output, cls_output])
        return out[0][0], out[0][1], out[0][2], out[1][0], out[1][1], out[1][2], out[2][0], out[2][1], out[2][2]

    ```
2. 执行`export`

    ```bash
    python deploy/ONNX/export_onnx.py --weights weights/yolov6s.pt
    ```
## appending
```bash
SHA:647796d838a81bb2d270d440035aee76b977154e
```
