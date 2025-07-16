import onnx

input_path = "weights/groundingdino_swint_ogc_sim.onnx"


output_path = 'weights/img_encoder_groundingdino_swint_ogc.onnx'
input_names = ['img']
output_names = ['/transformer/encoder/fusion_layers.0/layer_norm_v/Add_1_output_0']
onnx.utils.extract_model(input_path, output_path, input_names, output_names)


output_path = 'weights/text_encoder_groundingdino_swint_ogc.onnx'
input_names = ['input_ids', 'position_ids', 'token_type_ids', 'text_token_mask']
output_names = ['/feat_map/Add_output_0']
onnx.utils.extract_model(input_path, output_path, input_names, output_names)


output_path = 'weights/decoder_groundingdino_swint_ogc.onnx'
input_names = ['/transformer/Concat_8_output_0', '/feat_map/Add_output_0', 'attention_mask', 'position_ids', 'text_token_mask']
output_names = ['logits', 'boxes']
onnx.utils.extract_model(input_path, output_path, input_names, output_names)