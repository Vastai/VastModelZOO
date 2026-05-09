import cv2
import copy
import numpy as np
from PIL import Image
import onnxruntime as ort
from loguru import logger
from typing import List, Union, Optional
from .preprocess import DetResize, NormalizeImage
from .crop_poly import CropPoly
from .postprocess import DBPostProcess

import os
import json
import time
from tqdm import tqdm
from queue import Queue
from threading import Thread, Event
from typing import Dict, Generator, Iterable, List, Union

import vaststreamx as vsx



def get_activation_aligned_faster(activation, dtype=np.float16, fc_mode=False, force_int8_layout_to_fp16=False):  # NCHW
    N = C = H = W = 1
    if len(activation.shape) == 2:
        N, C = activation.shape
        fc_mode = True
    elif len(activation.shape) == 5:
        N, C, H, W, B = activation.shape
    elif len(activation.shape) == 1:
        (C,) = activation.shape
    else:
        N, C, H, W = activation.shape
    h_group = w_group = c_group = 0
    if H == 1 and W == 1 and fc_mode == True:
        if dtype == np.float16:
            h_group, w_group, c_group = 1, 1, 256
        elif dtype == np.int8:
            h_group, w_group, c_group = 1, 1, 512
    else:
        if dtype == np.float16 or force_int8_layout_to_fp16:
            h_group, w_group, c_group = 8, 8, 4
        elif dtype == np.int8:
            h_group, w_group, c_group = 8, 8, 8
    pad_H, pad_W, pad_C = H, W, C
    pad_h, pad_w = 0, 0
    if H % h_group != 0:
        pad_h = h_group - H % h_group
        pad_H += pad_h
    if W % w_group != 0:
        pad_w = w_group - W % w_group
        pad_W += pad_w
    if C % c_group != 0:
        pad_c = c_group - C % c_group
        pad_C += pad_c
    # tensorize to WHC4c8h8w
    w_num = pad_W // w_group
    h_num = pad_H // h_group
    c_num = pad_C // c_group
    n_num = N
    block_size = w_group * h_group * c_group
    activation = activation.astype(dtype)
    assert(len(activation.shape) == 4)
    if (pad_h | pad_w) != 0:
        activation = np.pad(activation, ((0,0),(0,0),(0,pad_h),(0,pad_w)))
    np_arr = np.zeros((n_num, w_num, h_num, c_num, block_size), dtype)
    # for n in range(N):
    #     for c in range(C):
    #         for h in range(H):
    #             for w in range(W):
    #                 addr = (c % c_group) * h_group * w_group + (h % h_group) * w_group + (w % w_group)
    #                 if len(activation.shape) == 2:
    #                     np_arr[n, w // w_group, h // h_group, c // c_group, addr] = activation[n, c]
    #                 elif len(activation.shape) == 1:
    #                     np_arr[n, w // w_group, h // h_group, c // c_group, addr] = activation[n]
    #                 else:
    #                     np_arr[n, w // w_group, h // h_group, c // c_group, addr] = activation[n, c, h, w]
    block_size_hacked = 3 * 8 * 8
    c_group_hacked = 3
    for n in range(N):
        for c in range(c_num):
            c_index = c * c_group_hacked
            for h in range(h_num):
                h_index = h * h_group
                for w in range(w_num):
                    w_index = w * w_group
                    # print(activation[n, c_index:c_index+c_group_hacked, h_index:h_index+h_group, w_index:w_index+w_group].shape)
                    np_arr[n, w, h, c, :block_size_hacked] = activation[n, c_index:c_index+c_group_hacked, h_index:h_index+h_group, w_index:w_index+w_group].flatten()
    return np_arr
    

class VACCDetModel:
    def __init__(self,
        model_prefix_path: Union[str, Dict[str, str]],
        vdsp_params_info: str,
        device_id: int = 0,
        batch_size: int = 1,
        balance_mode: int = 0,
        is_async_infer: bool = False,
        model_output_op_name: str = "", ) -> None:


        if isinstance(vdsp_params_info, str):
            with open(vdsp_params_info) as f:
                vdsp_params_info_dict = json.load(f)
            
        # self.model_size_width = vdsp_params_info_dict["OpConfig"]["OimageWidth"]
        # self.model_size_height = vdsp_params_info_dict["OpConfig"]["OimageHeight"]
        self.device_id = device_id
        self.model_output_op_name = model_output_op_name
        self.input_id = 0

        self.attr = vsx.AttrKey
        assert vsx.set_device(self.device_id)==0
        # 构建model，模型三件套目录
        model_path = model_prefix_path
        self.model = vsx.Model(model_path, batch_size)
        # 输入预处理op
        # RGB_PLANAR
        self.fusion_op = vsx.Operator.load_ops_from_json_file(vdsp_params_info)[0]

        # 构建graph
        self.graph = vsx.Graph()
        self.model_op = vsx.ModelOperator(self.model)
        self.graph.add_operators(self.fusion_op, self.model_op)#

        # 构建stream
        self.infer_stream = vsx.Stream(self.graph, vsx.StreamBalanceMode.ONCE)
        if is_async_infer and len(self.model_output_op_name) > 0:
            self.infer_stream.register_operator_output(self.model_output_op_name, self.model_op)
        else:
            self.infer_stream.register_operator_output(self.model_op)
        
        self.infer_stream.build()
        
        self.current_id = -1
        self.input_dict = {}
        self.event_dict = {}
        self.result_dict = {}
        # 异步
        # self.consumer = Thread(target=self.async_receive_infer)
        # self.consumer.start()

    def async_receive_infer(self, ):
        while 1:
            try:
                result = None
                if len(self.model_output_op_name) > 0:
                    result = self.infer_stream.get_operator_output(self.model_output_op_name)
                else:
                    result = self.infer_stream.get_operator_output(self.model_op)
                if result is not None:
                    # 输出顺序和输入一致 
                    self.current_id += 1
                    input_id,height, width = self.input_dict[self.current_id]
                    model_output_list = [ [vsx.as_numpy(out)[0].astype(np.float32) for out in result[0]] ]
                    self.result_dict[input_id] = []
                    self.post_processing(input_id, model_output_list)
                    self.event_dict[input_id].set()
            except ValueError as e:
                error_message = str(e)
                print(error_message)
                break
    
    def post_processing(self, input_id, stream_output_list):

        preds = stream_output_list[0][0].astype(np.float32)

        self.result_dict[input_id].append(np.expand_dims(np.expand_dims(preds, 0), 0))

    def finish(self):
        self.infer_stream.close_input()
        self.infer_stream.wait_until_done()
        self.consumer.join()

    def _run(self, image:Union[str, np.ndarray]):
        if isinstance(image, str):
            cv_image = cv2.imread(image, cv2.IMREAD_COLOR)
            if cv_image.shape[0]%2 != 0 or cv_image.shape[1]%2 != 0:
                cv_image = cv2.resize(cv_image, (cv_image.shape[1]//2*2, cv_image.shape[0]//2*2))
            image = np.stack(cv2.split(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)))
        assert len(image.shape) == 3
        c, height, width = image.shape
        assert c == 3
        
        nv12_image = vsx.create_image(image, vsx.ImageFormat.RGB_PLANAR, width, height, self.device_id)
        yuv_nv12 = nv12_image#vsx.cvtcolor(nv12_image, vsx.ImageFormat.YUV_NV12)
        
        input_id = self.input_id
        self.input_dict[input_id] = (input_id, height, width)
        self.event_dict[input_id] = Event()


        self.infer_stream.run_async([yuv_nv12])
        
        self.input_id += 1

        return input_id

    def run(self, image:Union[str, np.ndarray]):
        input_id = self._run(image)
        self.event_dict[input_id].wait()
        result = self.result_dict.pop(input_id)
        del self.event_dict[input_id]

        return result

    def run_batch(self, images: Iterable[Union[str, np.ndarray]]) -> Generator[str, None, None]:
        # for img in images:
        #     yield self.run(img)
        
        queue = Queue(20)

        def input_thread():
            for image in tqdm(images):
                input_id = self._run(image)
                queue.put(input_id)
            queue.put(None)

        thread = Thread(target=input_thread)
        thread.start()
        while True:
            input_id = queue.get()
            if input_id is None:
                break
            self.event_dict[input_id].wait()
            result = self.result_dict.pop(input_id)
            del self.event_dict[input_id]
            yield result
    
    def run_sync(self, image:Union[str, np.ndarray]):
        if isinstance(image, str):
            cv_image = cv2.imread(image, cv2.IMREAD_COLOR)
            if cv_image.shape[0]%2 != 0 or cv_image.shape[1]%2 != 0:
                cv_image = cv2.resize(cv_image, (cv_image.shape[1]//2*2, cv_image.shape[0]//2*2))
            image = np.stack(cv2.split(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)))
        assert len(image.shape) == 3
        c, height, width = image.shape
        assert c == 3
        
        nv12_image = vsx.create_image(image, vsx.ImageFormat.RGB_PLANAR, width, height, self.device_id)
        yuv_nv12 = nv12_image#vsx.cvtcolor(nv12_image, vsx.ImageFormat.YUV_NV12)
        # bgr_inter = vsx.create_image(cv_image, vsx.ImageFormat.BGR_INTERLEAVE, cv_image.shape[1], cv_image.shape[0], self.device_id)
        # yuv_nv12 = vsx.cvtcolor(bgr_inter, vsx.ImageFormat.RGB_PLANAR, vsx.ImageColorSpace.COLOR_SPACE_BT709)
        
        output = self.infer_stream.run_sync([yuv_nv12])
        model_output_list = [vsx.as_numpy(out)[0].astype(np.float32) for out in output[0]] 

        return model_output_list
    

class Detector:
    def __init__(self, config, providers: Optional[List[str]] = None):
        self.config = config
        self.providers = providers

        # Get detection config
        self.use_fixed_shape = getattr(config, 'use_fixed_shape', False)

        # Use resize_long for dynamic shape support (same as official inference.yml)
        # Default to 960 as per official PP-OCRv5 config
        self.resize_long = getattr(config, 'resize_long', 960)

        # For backward compatibility: if input_shape is specified, calculate target_size from it
        if hasattr(config, 'input_shape') and config.input_shape is not None:
            det_shape = list(config.input_shape)
            if len(det_shape) == 3:
                _, det_h, det_w = det_shape
                self.target_h = det_h
                self.target_w = det_w
                if not self.use_fixed_shape:
                    self.resize_long = max(int(det_h), int(det_w))
                    logger.warning(
                        f"Using deprecated 'input_shape' config. Please use 'resize_long: {self.resize_long}' instead."
                    )
        else:
            self.target_h = None
            self.target_w = None

        # Pre/Post processors with configurable parameters (aligned with official config)
        if self.use_fixed_shape and self.target_h is not None and self.target_w is not None:
            self.preprocessor = DetResize(input_shape=config.input_shape)
        else:
            self.preprocessor = DetResize(resize_long=self.resize_long)

        self.normalize = NormalizeImage(order="hwc")

        # PostProcess parameters from config (with official defaults)
        thresh = getattr(config, 'thresh', 0.3)
        box_thresh = getattr(config, 'box_thresh', 0.6)
        unclip_ratio = getattr(config, 'unclip_ratio', 1.5)
        max_candidates = getattr(config, 'max_candidates', 1000)

        self.postprocessor = DBPostProcess(
            thresh=thresh,
            box_thresh=box_thresh,
            unclip_ratio=unclip_ratio,
            max_candidates=max_candidates
        )
        self.crop_poly = CropPoly()

        logger.info(
            f"Detector initialized with use_fixed_shape={self.use_fixed_shape}, "
            f"resize_long={self.resize_long}, "
            f"thresh={thresh}, box_thresh={box_thresh}, unclip_ratio={unclip_ratio}"
        )

    def load_onnx_model(self):
        model_path = self.config.path
        if not str(model_path).endswith(".onnx"):
            raise ValueError(f"Model path must end with .onnx: {model_path}")
        self.session = ort.InferenceSession(
            model_path,
            providers=self.providers or ["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def load_vacc_model(self):
        self.model = VACCDetModel(model_prefix_path=self.config.vacc_path,
                               vdsp_params_info=self.config.vdsp_path,
                               device_id=self.config.device_id,
                               batch_size=self.config.batch)

    def load_runmodel_model(self):
        import tvm
        import vacc
        from tvm.contrib import graph_runtime
        import hashlib

        model_path = self.config.vacc_path
        model_name = 'mod'
        hash = hashlib.md5()
        hash.update(model_name.encode())
        md5_model = hash.hexdigest()
        model_key = f"{md5_model}:0:{model_name}"

        kwargs = {"name": model_key}

        ctx = tvm.vacc(0)
        with open(model_path + ".json") as f:
            loaded_json = f.read()
        loaded_lib = tvm.module.load(model_path + ".so")
        with open(model_path + ".params", "rb") as f:
            loaded_params = bytearray(f.read())
        m = graph_runtime.create(loaded_json, loaded_lib, ctx, **kwargs)  # emu
        m.load_param(loaded_params)

        self.model = m

        
    def detect_onnx(self, image: Union[str, np.ndarray, Image.Image]):
        if isinstance(image, str):
            image = cv2.imread(image)
            if image is None:
                raise ValueError(f"Could not read image from path: {image}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif isinstance(image, Image.Image):
            image = np.array(image)
            
        if image is None:
            raise ValueError("Input image is None")
        # logger.info(f"Image shape: {image.shape}")

        # Choose preprocessing strategy based on use_fixed_shape
        if self.use_fixed_shape:
            # Fixed shape mode: resize to exact target shape
            data = self.preprocessor([image])
        else:
            # Dynamic shape mode: resize_long strategy (keep aspect ratio, resize longest side)
            data = self.preprocessor([image], limit_side_len=self.resize_long, limit_type="resize_long")

        input_tensor = data[0]
        input_tensor = self.normalize(input_tensor)
        input_tensor = input_tensor[0].transpose((2, 0, 1))  # HWC -> CHW
        shape_input = data[1]
        input_tensor = np.expand_dims(input_tensor, axis=0)  # Add batch dim

        outputs = self.session.run([self.output_name], {self.input_name: input_tensor})
        boxes = self.postprocessor(outputs, shape_input)
        return boxes

    def detect_vacc(self, image: Union[str, np.ndarray, Image.Image]):
        image_src = np.transpose(image, (2, 0, 1))

        if isinstance(image, str):
            image = cv2.imread(image)
            if image is None:
                raise ValueError(f"Could not read image from path: {image}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif isinstance(image, Image.Image):
            image = np.array(image)
            
        if image is None:
            raise ValueError("Input image is None")
        # logger.info(f"Image shape: {image.shape}")

        # Choose preprocessing strategy based on use_fixed_shape
        if self.use_fixed_shape:
            # Fixed shape mode: resize to exact target shape
            data = self.preprocessor([image])
        else:
            # Dynamic shape mode: resize_long strategy (keep aspect ratio, resize longest side)
            data = self.preprocessor([image], limit_side_len=self.resize_long, limit_type="resize_long")
        
        outputs = self.model.run_sync(image_src)
        outputs = np.expand_dims(outputs[0], axis=0)
        shape_input = data[1]
        boxes = self.postprocessor([outputs], shape_input)
        return boxes


    def detect_runmodel(self, image: Union[str, np.ndarray, Image.Image]):
        import tvm
        import vacc
        from tvm.contrib import graph_runtime

        if isinstance(image, str):
            image = cv2.imread(image)
            if image is None:
                raise ValueError(f"Could not read image from path: {image}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif isinstance(image, Image.Image):
            image = np.array(image)
            
        if image is None:
            raise ValueError("Input image is None")
        # logger.info(f"Image shape: {image.shape}")

        # Choose preprocessing strategy based on use_fixed_shape
        if self.use_fixed_shape:
            # Fixed shape mode: resize to exact target shape
            data = self.preprocessor([image])
        else:
            # Dynamic shape mode: resize_long strategy (keep aspect ratio, resize longest side)
            data = self.preprocessor([image], limit_side_len=self.resize_long, limit_type="resize_long")

        input_tensor = data[0]
        input_tensor = self.normalize(input_tensor)
        input_tensor = input_tensor[0].transpose((2, 0, 1))  # HWC -> CHW
        shape_input = data[1]
        input_tensor = np.expand_dims(input_tensor, axis=0)  # Add batch dim
        
        name = self.model.set_batch_size(1)
        input_image = get_activation_aligned_faster(input_tensor.astype("float16"))
        self.model.set_input(name, 'x', 0, tvm.nd.array(input_image))
        self.model.run(name)

        heatmap = self.model.get_output(name, 0, 0).asnumpy()

        # draw reference to ppocr
        outputs = heatmap.astype(np.float32)
        boxes = self.postprocessor([outputs], shape_input)

        return boxes