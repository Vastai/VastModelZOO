import cv2
import math
import numpy as np
from PIL import Image
import onnxruntime as ort
from typing import List, Optional
from .postprocess import CTCLabelDecode
from typing import List, Union, Tuple, Optional

import os
import json
import time
from tqdm import tqdm
from queue import Queue
from threading import Thread, Event
from typing import Dict, Generator, Iterable, List, Union

import vaststreamx as vsx

from ..text_detector.detector import get_activation_aligned_faster



class VACCRecModel:
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
        
        # self.model_size = vdsp_params_info_dict["OpConfig"]["OimageWidth"]
        self.device_id = device_id
        self.model_output_op_name = model_output_op_name
        self.preprocess_name = "preprocess_res"
        self.input_id = 0
        self.balance_mode = {0:vsx.StreamBalanceMode.ONCE, 1:vsx.StreamBalanceMode.RUN}

        self.attr = vsx.AttrKey
        assert vsx.set_device(self.device_id)==0
        # 构建model，模型三件套目录
        model_path = model_prefix_path
        self.model = vsx.Model(model_path, batch_size)
        # 输入预处理op
        self.fusion_op = vsx.Operator.load_ops_from_json_file(vdsp_params_info)[0]
        
        
        # 构建graph
        self.graph = vsx.Graph()
        self.model_op = vsx.ModelOperator(self.model)
        self.graph.add_operators(self.fusion_op, self.model_op)#

        # 构建stream
        self.infer_stream = vsx.Stream(self.graph, self.balance_mode[balance_mode])
        if is_async_infer and len(self.model_output_op_name) > 0:
            self.infer_stream.register_operator_output(self.model_output_op_name, self.model_op)
        else:
            self.infer_stream.register_operator_output(self.model_op)
        
        # 预处理算子输出
        n,c,h,w = self.model.input_shape[0]
        self.infer_stream.register_operator_output(self.preprocess_name, self.fusion_op, [[(c,h,w), vsx.TypeFlag.FLOAT16]])
        
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
                    #pre_process_tensor = self.infer_stream.get_operator_output(self.preprocess_name)
                    # 输出顺序和输入一致 
                    self.current_id += 1
                    input_id,height, width = self.input_dict[self.current_id]
                    model_output_list = [ [vsx.as_numpy(out).astype(np.float32) for out in result[0]] ]
                    self.result_dict[input_id] = []
                    #print(f"put featuers: input_id-{input_id},{vsx.as_numpy(result[0][0])[0, 0:5]}")
                    self.post_processing(input_id, height, width, model_output_list)
                    self.event_dict[input_id].set()
            except ValueError as e:
                error_message = str(e)
                print(error_message)
                break
    
    def finish(self,):
        self.infer_stream.close_input()
        self.infer_stream.wait_until_done()
        self.consumer.join()

    def post_processing(self, input_id, height, width, stream_output_list):
        output_data = stream_output_list[0][0]
        self.result_dict[input_id].append(
            {
                "features": output_data,
            }
        )
    
    def cv_bgr888_to_vsximage(self, bgr888, vsx_format, device_id):
        h, w = bgr888.shape[:2]
        if vsx_format == vsx.ImageFormat.BGR_INTERLEAVE:
            res = bgr888
        elif vsx_format == vsx.ImageFormat.BGR_PLANAR:
            res = np.array(bgr888).transpose(2, 0, 1)
        elif vsx_format == vsx.ImageFormat.RGB_INTERLEAVE:
            res = cv2.cvtColor(bgr888, cv2.COLOR_BGR2RGB)
        elif vsx_format == vsx.ImageFormat.RGB_PLANAR:
            res = np.array(cv2.cvtColor(bgr888, cv2.COLOR_BGR2RGB)).transpose(2, 0, 1)
        elif vsx_format == vsx.ImageFormat.YUV_NV12:
            res = cv_bgr888_to_nv12(bgr888=bgr888)
        else:
            assert False, f"Unsupport format:{vsx_format}"
        return vsx.create_image(
            res,
            vsx_format,
            w,
            h,
            device_id,
        )
        
    def calculate_padding(self, model_width, model_height, image_width, image_height):
        radio = image_width / image_height
        # print(f"image_width:{image_width} image_height:{image_height} radio:{radio}")
        resize_w = 0
        resize_h = model_height
        # n,c,h,w
        if (model_height * radio > model_width) :
            resize_w = model_width
        else:
            resize_w = int(model_height * radio)
        
        right = model_width - resize_w if model_width - resize_w > 0 else 0
        
        # (resize_width , resize_height , top , bottom ,left, right)
        # return  (model_width, model_height, 0, 0, 0, right)
        return (int(resize_w), resize_h, 0, 0, 0, right)

    def _run(self, image:Union[str, np.ndarray]):
        if isinstance(image, str):
            cv_image = cv2.imread(image, cv2.IMREAD_COLOR)
            if cv_image.shape[0]%2 != 0 or cv_image.shape[1]%2 != 0:
                cv_image = cv2.resize(cv_image, (cv_image.shape[1]//2*2, cv_image.shape[0]//2*2))
            cv_image = np.stack(cv2.split(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)))
        else:
            cv_image = image

        assert len(cv_image.shape) == 3
        c, height, width = cv_image.shape
        assert c == 3

        assert image is not None, f"Failed to read input file: {image}"
        # vsx_image = self.cv_bgr888_to_vsximage(cv_image, vsx.ImageFormat.RGB_PLANAR, self.device_id)
        vsx_image = vsx.create_image(cv_image, vsx.ImageFormat.RGB_PLANAR, width, height, self.device_id)

        ext_op_config = self.calculate_padding(self.model.input_shape[0][3], self.model.input_shape[0][2], width, height)
        # print(ext_op_config)
        # print(vsx.as_numpy(vsx_image))
        
        input_id = self.input_id
        self.input_dict[input_id] = (input_id, height, width)
        self.event_dict[input_id] = Event()


        self.infer_stream.run_async([vsx_image], {
            # "rgb_letterbox_ext" : [(resize_width , resize_height , top , bottom ,left, right)]
            "rgb_letterbox_ext":[ext_op_config] 
        })
        
        self.input_id += 1

        return input_id

    def run(self, image:Union[str, np.ndarray]):
        input_id = self._run(image)
        self.event_dict[input_id].wait()
        result = self.result_dict.pop(input_id)
        del self.event_dict[input_id]

        return result

    def run_batch(self, images: Iterable[Union[str, np.ndarray]]) -> Generator[str, None, None]:
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
            #print(f"====>>>>pop(input_id)={input_id}")
            result = self.result_dict.pop(input_id)
            del self.event_dict[input_id]
            yield result
    
    def run_sync(self, image:Union[str, np.ndarray]):
        if isinstance(image, str):
            cv_image = cv2.imread(image, cv2.IMREAD_COLOR)
            if cv_image.shape[0]%2 != 0 or cv_image.shape[1]%2 != 0:
                cv_image = cv2.resize(cv_image, (cv_image.shape[1]//2*2, cv_image.shape[0]//2*2))
            cv_image = np.stack(cv2.split(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)))
        else:
            cv_image = image

        assert len(cv_image.shape) == 3
        c, height, width = cv_image.shape
        assert c == 3

        assert image is not None, f"Failed to read input file: {image}"
        # vsx_image = self.cv_bgr888_to_vsximage(cv_image, vsx.ImageFormat.RGB_PLANAR, self.device_id)
        vsx_image = vsx.create_image(cv_image, vsx.ImageFormat.RGB_PLANAR, width, height, self.device_id)

        ext_op_config = self.calculate_padding(self.model.input_shape[0][3], self.model.input_shape[0][2], width, height)
        # print(ext_op_config)
        # print(vsx.as_numpy(vsx_image))
        output = self.infer_stream.run_sync([vsx_image], {
                # "rgb_letterbox_ext" : [(resize_width , resize_height , top , bottom ,left, right)]
                "rgb_letterbox_ext":[ext_op_config] 
            })
        model_output_list = [ [vsx.as_numpy(out).astype(np.float32) for out in output[0]] ]
        output_data = model_output_list[0][0]
        
        return output_data
    

class Recognizer:
    def __init__(self, config, providers: Optional[List[str]] = None):
        self.config = config
        self.providers = providers

        charset_path = config.dict_path

        # Rec input shape from config [C, H, W]
        rec_shape = list(config.input_shape)
        if len(rec_shape) != 3:
            raise ValueError(f"rec.input_shape must be [C, H, W], got: {rec_shape}")
        self.imgC, self.imgH, self.imgW = map(int, rec_shape)
        self.use_fixed_shape = getattr(config, 'use_fixed_shape', False)
        self.use_letterbox = getattr(config, 'use_letterbox', False)

        self.processor = CTCLabelDecode(
            character_dict_path=str(charset_path),
            use_space_char=True,
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

    def load_vacc_model(self):
        self.model = VACCRecModel(model_prefix_path=self.config.vacc_path,
                                  vdsp_params_info=self.config.vdsp_path,
                                  device_id=self.config.device_id,
                                  batch_size=self.config.batch)
        
    def resize_norm_img_batch(self, images: List[np.ndarray]) -> Tuple[np.ndarray, List[float], float]:
        """
        Batch preprocessing:
        1. Calculate target width for each image maintaining aspect ratio.
        2. Find the max width in the batch (capped at 3200).
        3. Resize, Normalize, and Pad all images to (B, C, H, Max_W).
        """
        imgC, imgH, imgW = self.imgC, self.imgH, self.imgW
        batch_size = len(images)

        # Calculate aspect ratios and target widths
        ratios = []
        resized_widths = []

        for img in images:
            h, w = img.shape[:2]
            ratio = w / float(h)
            ratios.append(ratio)

            # Logic to ensure minimum width matches config, or expand if image is wider
            max_wh_ratio = max(imgW / float(imgH), ratio)

            # Calculate width required for this image
            current_dyn_width = int(imgH * max_wh_ratio)
            resized_widths.append(current_dyn_width)

        # Determine the maximum width needed for this batch
        # We limit the max width to 3200 to prevent OOM or extremely slow inference on outliers
        max_batch_w = max(resized_widths)
        max_batch_w = min(max_batch_w, 3200)

        # Pre-allocate batch tensor: [B, C, H, W]
        batch_imgs = np.zeros((batch_size, imgC, imgH, max_batch_w), dtype=np.float32)

        for i, img in enumerate(images):
            h, w = img.shape[:2]

            # Calculate actual resize width for this specific image
            # We need to resize strictly by aspect ratio first
            ratio = w / float(h)
            resized_w = int(math.ceil(imgH * ratio))

            # Cap width if it exceeds our batch maximum
            if resized_w > max_batch_w:
                resized_w = max_batch_w

            # Resize
            resized_image = cv2.resize(img, (resized_w, imgH))
            resized_image = resized_image.astype("float32")

            # Normalize: (H, W, C) -> (C, H, W) and scale
            resized_image = resized_image.transpose((2, 0, 1)) / 255.0
            resized_image -= 0.5
            resized_image /= 0.5

            # Pad: Copy the resized image into the pre-allocated zero tensor
            # The rest of the width (max_batch_w - resized_w) remains 0 (padding)
            batch_imgs[i, :, :, 0:resized_w] = resized_image

        # Calculate max_wh_ratio for the whole batch (used for decoding boxes if needed)
        batch_max_wh_ratio = max_batch_w / float(imgH)

        return batch_imgs, ratios, batch_max_wh_ratio

    def resize_norm_img_batch_fixed(self, images: List[np.ndarray]) -> Tuple[np.ndarray, List[float], float]:
        """
        Batch preprocessing with fixed input shape.
        If use_letterbox=True: resizes while maintaining aspect ratio, then pads.
        If use_letterbox=False: directly resizes to target shape without maintaining aspect ratio.
        """
        imgC, imgH, imgW = self.imgC, self.imgH, self.imgW
        batch_size = len(images)

        # Pre-allocate batch tensor: [B, C, H, W]
        batch_imgs = np.zeros((batch_size, imgC, imgH, imgW), dtype=np.float32)
        ratios = []

        for i, img in enumerate(images):
            h, w = img.shape[:2]
            ratio = w / float(h)
            ratios.append(ratio)

            if self.use_letterbox:
                # Letterbox: resize while maintaining aspect ratio
                scale = min(imgH / h, imgW / w)
                new_h, new_w = int(h * scale), int(w * scale)
                resized_image = cv2.resize(img, (new_w, new_h))

                # Calculate padding
                pad_top = (imgH - new_h) // 2
                pad_bottom = imgH - new_h - pad_top
                pad_left = (imgW - new_w) // 2
                pad_right = imgW - new_w - pad_left

                # Add padding with constant value
                resized_image = cv2.copyMakeBorder(
                    resized_image, pad_top, pad_bottom, pad_left, pad_right,
                    cv2.BORDER_CONSTANT, value=(114, 114, 114)
                )
            else:
                # Direct resize to fixed shape
                resized_image = cv2.resize(img, (imgW, imgH))

            resized_image = resized_image.astype("float32")

            # Normalize: (H, W, C) -> (C, H, W) and scale
            resized_image = resized_image.transpose((2, 0, 1)) / 255.0
            resized_image -= 0.5
            resized_image /= 0.5

            batch_imgs[i, :, :, :] = resized_image

        # For fixed shape, max_wh_ratio is simply imgW / imgH
        batch_max_wh_ratio = imgW / float(imgH)

        return batch_imgs, ratios, batch_max_wh_ratio

    def recognize_onnx(self, images: Union[np.ndarray, List[np.ndarray], str, np.ndarray, Image.Image]):
        """
        Run OCR recognition on a batch of images or a single image.
        Args:
            images: Single np.ndarray image or List of np.ndarray images.
        Returns:
            List of results [(text, score), ...]
        """
        if isinstance(images, str):
            images = cv2.imread(images)
            if images is None:
                raise ValueError(f"Could not read image from path: {images}")
            images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB)
        elif isinstance(images, Image.Image):
            images = np.array(images)

        # Handle single image input for backward compatibility
        if isinstance(images, np.ndarray):
            images = [images]

        if not images:
            raise ValueError("Input images list is empty")

        # Preprocess batch - choose mode based on use_fixed_shape
        if self.use_fixed_shape:
            input_tensor, ratios, max_wh_ratio = self.resize_norm_img_batch_fixed(images)
        else:
            input_tensor, ratios, max_wh_ratio = self.resize_norm_img_batch(images)

        # Run Inference
        # ONNX Runtime handles the batch dimension naturally [B, C, H, W]
        outputs = self.session.run([self.output_name], {self.input_name: input_tensor})[0]

        # Post-process (Decode)
        return_word_box = False

        # The processor (CTCLabelDecode) iterates through the batch dimension internally
        text = self.processor(
            outputs,
            return_word_box=return_word_box,
            wh_ratio_list=ratios,
            max_wh_ratio=max_wh_ratio,
        )

        return text

    def recognize_vacc(self, images: Union[str, np.ndarray, Image.Image]):
        image_src = np.transpose(images, (2, 0, 1))

        if isinstance(images, str):
            images = cv2.imread(images)
            if images is None:
                raise ValueError(f"Could not read image from path: {images}")
            images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB)
        elif isinstance(images, Image.Image):
            images = np.array(images)

        # Handle single image input for backward compatibility
        if isinstance(images, np.ndarray):
            images = [images] # pyright: ignore[reportAssignmentType]

        if not images:
            raise ValueError("Input images list is empty")

        # Preprocess batch - choose mode based on use_fixed_shape
        if self.use_fixed_shape:
            input_tensor, ratios, max_wh_ratio = self.resize_norm_img_batch_fixed(images)
        else:
            input_tensor, ratios, max_wh_ratio = self.resize_norm_img_batch(images)

        outputs = self.model.run_sync(image_src)
        outputs = np.expand_dims(outputs, axis=0)

        # Post-process (Decode)
        return_word_box = False

        # The processor (CTCLabelDecode) iterates through the batch dimension internally
        text = self.processor(
            outputs,
            return_word_box=return_word_box,
            wh_ratio_list=ratios,
            max_wh_ratio=max_wh_ratio,
        )
        return text


    def recognize_runmodel(self, images: Union[np.ndarray, List[np.ndarray], str, np.ndarray, Image.Image]):
        """
        Run OCR recognition on a batch of images or a single image.
        Args:
            images: Single np.ndarray image or List of np.ndarray images.
        Returns:
            List of results [(text, score), ...]
        """
        import tvm
        import vacc
        from tvm.contrib import graph_runtime

        if isinstance(images, str):
            images = cv2.imread(images)
            if images is None:
                raise ValueError(f"Could not read image from path: {images}")
            images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB)
        elif isinstance(images, Image.Image):
            images = np.array(images)

        # Handle single image input for backward compatibility
        if isinstance(images, np.ndarray):
            images = [images]

        if not images:
            raise ValueError("Input images list is empty")

        # Preprocess batch - choose mode based on use_fixed_shape
        if self.use_fixed_shape:
            input_tensor, ratios, max_wh_ratio = self.resize_norm_img_batch_fixed(images)
        else:
            input_tensor, ratios, max_wh_ratio = self.resize_norm_img_batch(images)

        # Run Inference
        name = self.model.set_batch_size(1)
        input_image = get_activation_aligned_faster(input_tensor.astype("float16"))
        self.model.set_input(name, 'x', 0, tvm.nd.array(input_image))
        self.model.run(name)

        heatmap = self.model.get_output(name, 0, 0).asnumpy()
        
        # draw reference to ppocr
        outputs = heatmap.astype(np.float32)
        outputs = np.expand_dims(outputs, axis=0)

        # Post-process (Decode)
        return_word_box = False

        # The processor (CTCLabelDecode) iterates through the batch dimension internally
        text = self.processor(
            outputs,
            return_word_box=return_word_box,
            wh_ratio_list=ratios,
            max_wh_ratio=max_wh_ratio,
        )

        return text