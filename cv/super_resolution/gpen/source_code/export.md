## 模型转换

### Code Source
```
link: https://github.com/yangxy/GPEN
branch: master
commit: b611a9f2d05fdc5b82715219ed0f252e42e4d8be
```

### onnx&torchscript
- 拉取代码至`source_code`目录下
- 将[export.py](./export.py)移动至`source_code/GPEN`目录下
- 修改[gpen_model.py#L690](https://github.com/yangxy/GPEN/blob/main/face_model/gpen_model.py#L690)，在return前添加以下代码：
    ```python
    if len(outs) == 2:
        if outs[1] is None:
            outs = outs[0]

    return outs
    ```
- 执行转换脚本，得到`onnx`和`torchscript`：
    ```python
    python super_resolution/gpen/source_code/GPEN/export.py
    ```