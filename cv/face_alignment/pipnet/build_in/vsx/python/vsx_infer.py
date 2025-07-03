import os
import cv2
import json
import torch
import argparse
import numpy as np
from tqdm.contrib import tzip
from queue import Queue
from threading import Thread, Event
from typing import Dict, Generator, Iterable, List, Union

import vaststreamx as vsx
import sys
_cur_file_path = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(_cur_file_path + os.sep + '../../..')
from source_code.utils import (post_process, get_meanface, compute_nme, compute_fr_and_auc,
                   get_label)

class VSXInference:
    def __init__(self,
        model_prefix_path: Union[str, Dict[str, str]],
        vdsp_params_info: str,
        device_id: int = 0,
        batch_size: int = 1,
        is_async_infer: bool = False,
        model_output_op_name: str = "", ) -> None:


        if isinstance(vdsp_params_info, str):
            with open(vdsp_params_info) as f:
                vdsp_params_info_dict = json.load(f)
        
        self.model_size = vdsp_params_info_dict["OpConfig"]["OimageWidth"]
        self.device_id = device_id
        self.model_output_op_name = model_output_op_name
        self.preprocess_name = "preprocess_res"
        self.input_id = 0

        self.attr = vsx.AttrKey
        self.device = vsx.set_device(self.device_id)
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
        
        # 预处理算子输出
        n,c,h,w = self.model.input_shape[0]
        self.infer_stream.register_operator_output(self.preprocess_name, self.fusion_op, [[(c,h,w), vsx.TypeFlag.FLOAT16]])

        self.infer_stream.build()
        
        self.current_id = -1
        self.input_dict = {}
        self.event_dict = {}
        self.result_dict = {}
        # 异步
        self.consumer = Thread(target=self.async_receive_infer)
        self.consumer.start()

    
    def async_receive_infer(self, ):
        while 1:
            try:
                result = None
                if len(self.model_output_op_name) > 0:
                    result = self.infer_stream.get_operator_output(self.model_output_op_name)
                else:
                    result = self.infer_stream.get_operator_output(self.model_op)
                if result is not None:
                    pre_process_tensor = self.infer_stream.get_operator_output(self.preprocess_name)
                    # 输出顺序和输入一致 
                    self.current_id += 1
                    input_id,height, width = self.input_dict[self.current_id]
                    model_output_list = [ [vsx.as_numpy(out)[0].astype(np.float32) for out in result[0]] ]
                    self.result_dict[input_id] = []
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
        output_data = stream_output_list[0]
        self.result_dict[input_id] = output_data

    def _run(self, image:Union[str, np.ndarray]):
        if isinstance(image, str):
            cv_image = cv2.imread(image, cv2.IMREAD_COLOR)
            image = np.stack(cv2.split(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)))
        assert len(image.shape) == 3
        c, height, width = image.shape
        assert c == 3
        
        input_image = vsx.create_image(image, vsx.ImageFormat.RGB_PLANAR, width, height, self.device)

        input_id = self.input_id
        self.input_dict[input_id] = (input_id, height, width)
        self.event_dict[input_id] = Event()


        self.infer_stream.run_async([input_image])
        
        self.input_id += 1

        return input_id

    def run(self, image:Union[str, np.ndarray]):
        input_id = self._run(image)
        self.event_dict[input_id].wait()
        result = self.result_dict.pop(input_id)
        del self.event_dict[input_id]

        return result

    def run_batch(self, images: Iterable[Union[str, np.ndarray]]) -> Generator[str, None, None]:

        queue = Queue(10)
        
        def input_thread():
            for image in images:
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
            image = np.stack(cv2.split(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)))
        assert len(image.shape) == 3
        c, height, width = image.shape
        assert c == 3
        
        input_image = vsx.create_image(image, vsx.ImageFormat.RGB_PLANAR, width, height, self.device)

        output = self.infer_stream.run_sync([input_image])
        model_output_list = [ [vsx.as_numpy(out)[0].astype(np.float32) for out in output[0]] ]
        output_data = model_output_list[0]

        return output_data
        

def parser_args():
    parse = argparse.ArgumentParser(description="RUN WITH VSX")
    parse.add_argument("--data_dir",type=str,default="/path/to/face/face_alignment/wflw/WFLW")
    parse.add_argument("--model_prefix_path",type=str,default="deploy_weights/official_pipnet_run_stream_fp16/mod",help="model info")
    parse.add_argument("--vdsp_params_info",type=str,default="../vacc_code/vdsp_params/official-pip_resnet18-vdsp_params.json",help="vdsp op info",)
    parse.add_argument("--device_id", type=int, default=0, help="device id")
    parse.add_argument("--batch", type=int, default=1, help="bacth size")
    parse.add_argument("--meanface_txt", type=str, default="../source_code/meanface.txt", help="meanface.txt")
    parse.add_argument("--save_dir", type=str, default="./vsx_results", help="save result")
    args = parse.parse_args()

    return args


if __name__ == '__main__':
    args = parser_args()

    vsx_inference = VSXInference(model_prefix_path=args.model_prefix_path,
                        vdsp_params_info=args.vdsp_params_info,
                        device_id=args.device_id,
                        batch_size=args.batch,
                        is_async_infer = False,
                        model_output_op_name = "", 
                    )

    norm_indices = [60, 72]
    num_lms = 98
    num_nb = 10
    net_stride = 32
    meanface_indices, reverse_index1, reverse_index2, max_len = get_meanface(os.path.join(args.meanface_txt), num_nb)
    nmes_merge = []
    image_files = []
    
    # labels = get_label(os.path.join(args.data_dir, "test.txt"), ret_dict=True)
    # for k, v in labels.items():
    #     image_files.append(os.path.join(args.data_dir, "images_test", k))
    # results = vsx_inference.run_batch(image_files)
    # for (image_path, result) in tzip(image_files, results):
    #     lms_gt = labels[os.path.basename(image_path)]
    #     norm = np.linalg.norm(lms_gt.reshape(-1, 2)[norm_indices[0]] - lms_gt.reshape(-1, 2)[norm_indices[1]])
    #     pred0 = np.expand_dims(result[0], axis=0)
    #     pred1 = np.expand_dims(result[1], axis=0)
    #     pred2 = np.expand_dims(result[2], axis=0)
    #     pred3 = np.expand_dims(result[3], axis=0)
    #     pred4 = np.expand_dims(result[4], axis=0)
    #     # print(pred0.shape)
    #     # print(pred4.shape)
    #     pred = (pred0, pred1, pred2, pred3, pred4)
    #     lms_pred_x, lms_pred_y, lms_pred_nb_x, lms_pred_nb_y, outputs_cls, max_cls = post_process(pred, num_nb, net_stride)
        
    #     lms_pred = torch.cat((lms_pred_x, lms_pred_y), dim=1).flatten()
    #     tmp_nb_x = lms_pred_nb_x[reverse_index1, reverse_index2].view(num_lms, max_len)
    #     tmp_nb_y = lms_pred_nb_y[reverse_index1, reverse_index2].view(num_lms, max_len)
    #     tmp_x = torch.mean(torch.cat((lms_pred_x, tmp_nb_x), dim=1), dim=1).view(-1,1)
    #     tmp_y = torch.mean(torch.cat((lms_pred_y, tmp_nb_y), dim=1), dim=1).view(-1,1)
    #     lms_pred_merge = torch.cat((tmp_x, tmp_y), dim=1).flatten()

    #     lms_pred = lms_pred.cpu().numpy()
    #     lms_pred_merge = lms_pred_merge.cpu().numpy()

    #     nme_merge = compute_nme(lms_pred_merge, lms_gt, norm)
    #     nmes_merge.append(nme_merge)
    
    labels = get_label(os.path.join(args.data_dir, "test.txt"))
    from tqdm import tqdm
    for label in tqdm(labels):
        image_name = label[0]
        lms_gt = label[1]
        norm = np.linalg.norm(lms_gt.reshape(-1, 2)[norm_indices[0]] - lms_gt.reshape(-1, 2)[norm_indices[1]])
        image_path = os.path.join(args.data_dir, "test", image_name)
        result = vsx_inference.run_sync(image_path)
        pred0 = np.expand_dims(result[0], axis=0)
        pred1 = np.expand_dims(result[1], axis=0)
        pred2 = np.expand_dims(result[2], axis=0)
        pred3 = np.expand_dims(result[3], axis=0)
        pred4 = np.expand_dims(result[4], axis=0)
        # print(pred0.shape)
        # print(pred4.shape)
        pred = (pred0, pred1, pred2, pred3, pred4)
        lms_pred_x, lms_pred_y, lms_pred_nb_x, lms_pred_nb_y, outputs_cls, max_cls = post_process(pred, num_nb, net_stride)
        
        lms_pred = torch.cat((lms_pred_x, lms_pred_y), dim=1).flatten()
        tmp_nb_x = lms_pred_nb_x[reverse_index1, reverse_index2].view(num_lms, max_len)
        tmp_nb_y = lms_pred_nb_y[reverse_index1, reverse_index2].view(num_lms, max_len)
        tmp_x = torch.mean(torch.cat((lms_pred_x, tmp_nb_x), dim=1), dim=1).view(-1,1)
        tmp_y = torch.mean(torch.cat((lms_pred_y, tmp_nb_y), dim=1), dim=1).view(-1,1)
        lms_pred_merge = torch.cat((tmp_x, tmp_y), dim=1).flatten()

        lms_pred = lms_pred.cpu().numpy()
        lms_pred_merge = lms_pred_merge.cpu().numpy()

        nme_merge = compute_nme(lms_pred_merge, lms_gt, norm)
        nmes_merge.append(nme_merge)
    

    print('Image num:', len(labels))
    print('nme: {}'.format(np.mean(nmes_merge)))
    fr, auc = compute_fr_and_auc(nmes_merge)
    print('fr: {}'.format(fr))
    print('auc: {}'.format(auc))
    vsx_inference.finish()
    

