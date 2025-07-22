from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import json
import math
import os
import random
import sys
import six
import numpy as np

import tempfile
sys.path.insert(0, "./bert")

import tensorflow as tf

import modeling
import optimization
import tokenization

import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import graph_io
from tensorflow.core.framework import graph_pb2
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.tools import optimize_for_inference_lib, freeze_graph
from tensorflow.python.tools import saved_model_utils






def validate_flags_or_throw(bert_config, args):
  """Validate the input args or throw an exception."""
  tokenization.validate_case_matches_checkpoint(args.do_lower_case,
                                                args.init_checkpoint)

  #if not args.do_train and not args.do_predict:
  #  raise ValueError("At least one of `do_train` or `do_predict` must be True.")

  if args.max_seq_length > bert_config.max_position_embeddings:
    raise ValueError(
        "Cannot use sequence length %d because the BERT model "
        "was only trained up to sequence length %d" %
        (args.max_seq_length, bert_config.max_position_embeddings))

  if args.max_seq_length <= args.max_query_length + 3:
    raise ValueError(
        "The max_seq_length (%d) must be greater than max_query_length "
        "(%d) + 3" % (args.max_seq_length, args.max_query_length))


def model_fn_builder(bert_config, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings):
  """Returns `model_fn` closure for TPUEstimator."""

  def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
    """The `model_fn` for TPUEstimator."""

    tf.logging.info("*** Features ***")
    for name in sorted(features.keys()):
      tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))
    #unique_ids = features["unique_ids"]
    #input_ids = features["input_ids"]
    #input_mask = features["input_mask"]
    #segment_ids = features["segment_ids"]

    input_ids = tf.placeholder(shape=[1, 384], dtype=tf.int32, name="input_ids")
    input_mask = tf.placeholder(shape=[1, 384], dtype=tf.int32, name="input_mask")
    segment_ids = tf.placeholder(shape=[1, 384], dtype=tf.int32, name="segment_ids")

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    (start_logits, end_logits) = create_model(
        bert_config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings)

    tvars = tf.trainable_variables()

    initialized_variable_names = {}
    scaffold_fn = None
    if init_checkpoint:
      (assignment_map, initialized_variable_names
      ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
      if use_tpu:

        def tpu_scaffold():
          tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
          return tf.train.Scaffold()

        scaffold_fn = tpu_scaffold
      else:
        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

    tf.logging.info("**** Trainable Variables ****")
    for var in tvars:
      init_string = ""
      if var.name in initialized_variable_names:
        init_string = ", *INIT_FROM_CKPT*"
      tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                      init_string)
    output_spec = None
    if mode == tf.estimator.ModeKeys.TRAIN:
      seq_length = modeling.get_shape_list(input_ids)[1]

      def compute_loss(logits, positions):
        one_hot_positions = tf.one_hot(
            positions, depth=seq_length, dtype=tf.float32)
        log_probs = tf.nn.log_softmax(logits, axis=-1)
        loss = -tf.reduce_mean(
            tf.reduce_sum(one_hot_positions * log_probs, axis=-1))
        return loss

      start_positions = features["start_positions"]
      end_positions = features["end_positions"]

      start_loss = compute_loss(start_logits, start_positions)
      end_loss = compute_loss(end_logits, end_positions)

      total_loss = (start_loss + end_loss) / 2.0

      train_op = optimization.create_optimizer(
          total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)

      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=total_loss,
          train_op=train_op,
          scaffold_fn=scaffold_fn)
    elif mode == tf.estimator.ModeKeys.PREDICT:
      predictions = {
          #"unique_ids": unique_ids,
          "start_logits": start_logits,
          "end_logits": end_logits,
      }
      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode, predictions=predictions, scaffold_fn=scaffold_fn)
    else:
      raise ValueError(
          "Only TRAIN and PREDICT modes are supported: %s" % (mode))

    return output_spec

  return model_fn


def create_model(bert_config, is_training, input_ids, input_mask, segment_ids,
                 use_one_hot_embeddings):
  """Creates a classification model."""
  model = modeling.BertModel(
      config=bert_config,
      is_training=is_training,
      input_ids=input_ids,
      input_mask=input_mask,
      token_type_ids=segment_ids,
      use_one_hot_embeddings=use_one_hot_embeddings)

  final_hidden = model.get_sequence_output()

  final_hidden_shape = modeling.get_shape_list(final_hidden, expected_rank=3)
  batch_size = final_hidden_shape[0]
  seq_length = final_hidden_shape[1]
  hidden_size = final_hidden_shape[2]

  output_weights = tf.get_variable(
      "cls/squad/output_weights", [2, hidden_size],
      initializer=tf.truncated_normal_initializer(stddev=0.02))

  output_bias = tf.get_variable(
      "cls/squad/output_bias", [2], initializer=tf.zeros_initializer())

  final_hidden_matrix = tf.reshape(final_hidden,
                                   [batch_size * seq_length, hidden_size])
  logits = tf.matmul(final_hidden_matrix, output_weights, transpose_b=True)
  logits = tf.nn.bias_add(logits, output_bias)

  logits = tf.reshape(logits, [batch_size, seq_length, 2])
  logits = tf.transpose(logits, [2, 0, 1])

  unstacked_logits = tf.unstack(logits, axis=0)

  (start_logits, end_logits) = (unstacked_logits[0], unstacked_logits[1])

  return (start_logits, end_logits)


def export_savemodel(args):
    bert_config = modeling.BertConfig.from_json_file(args.bert_config_file)

    validate_flags_or_throw(bert_config, args)

    tf.gfile.MakeDirs(args.output_dir)

    tpu_cluster_resolver = None
    if args.use_tpu and args.tpu_name:
      tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
          args.tpu_name, zone=args.tpu_zone, project=args.gcp_project)

    is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
    run_config = tf.contrib.tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        master=args.master,
        model_dir=args.output_dir,
        save_checkpoints_steps=args.save_checkpoints_steps,
        tpu_config=tf.contrib.tpu.TPUConfig(
            iterations_per_loop=args.iterations_per_loop,
            num_shards=args.num_tpu_cores,
            per_host_input_for_training=is_per_host))

    train_examples = None
    num_train_steps = None
    num_warmup_steps = None

    model_fn = model_fn_builder(
        bert_config=bert_config,
        init_checkpoint=args.init_checkpoint,
        learning_rate=args.learning_rate,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        use_tpu=args.use_tpu,
        use_one_hot_embeddings=args.use_tpu)

    # If TPU is not available, this will fall back to normal Estimator on CPU
    # or GPU.
    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu=args.use_tpu,
        model_fn=model_fn,
        config=run_config,
        train_batch_size=args.train_batch_size,
        predict_batch_size=args.predict_batch_size)

    input_ids = tf.placeholder(shape=[1, args.max_seq_length], dtype=tf.int32, name="input_ids")
    input_mask = tf.placeholder(shape=[1, args.max_seq_length], dtype=tf.int32, name="input_mask")
    segment_ids = tf.placeholder(shape=[1, args.max_seq_length], dtype=tf.int32, name="segment_ids")
    #unique_ids = tf.placeholder(shape=[], dtype=tf.int32, name="unique_ids")

    serving_input_receiver_fn = tf.estimator.export.build_raw_serving_input_receiver_fn({
        "input_ids" : input_ids,
        "input_mask" : input_mask,
        "segment_ids" : segment_ids})
        #"unique_ids" : unique_ids})

    saved_model_dir = args.output_dir #os.path.join(tempfile.gettempdir(), '')
    #os.makedirs(saved_model_dir, exist_ok=True)
    #tf.gfile.MakeDirs(args.output_dir)
    return estimator.export_saved_model(saved_model_dir, serving_input_receiver_fn, None, False, args.init_checkpoint).decode('utf-8')


def get_args(**kwargs):
    tf.logging.set_verbosity(tf.logging.INFO)
    flags = tf.flags

    ## Required parameters
    flags.DEFINE_string(
        "bert_config_file", "",
        "The config json file corresponding to the pre-trained BERT model. "
        "This specifies the model architecture.")

    flags.DEFINE_string("task_name", "", "The name of the task to train.")
    
    flags.DEFINE_string("out_name", "bert_base_squad",
                        "The vocabulary file that the BERT model was trained on.")
    
    flags.DEFINE_string(
        "output_dir", "./output_dir",
        "The output directory where the model checkpoints will be written.")

    flags.DEFINE_string(
        "saved_model_dir", "",
        "The savedmodel directory where the savedmodel is located.")

    ## Other parameters
    flags.DEFINE_string("train_file", None,
                        "SQuAD json for training. E.g., train-v1.1.json")
    
    flags.DEFINE_string(
        "init_checkpoint", "",
        "Initial checkpoint (usually from a pre-trained BERT model).")
    
    flags.DEFINE_bool(
        "do_lower_case", True,
        "Whether to lower case the input text. Should be True for uncased "
        "models and False for cased models.")
    
    flags.DEFINE_integer(
        "max_seq_length", 384,
        "The maximum total input sequence length after WordPiece tokenization. "
        "Sequences longer than this will be truncated, and sequences shorter "
        "than this will be padded.")
    
    flags.DEFINE_integer(
        "doc_stride", 128,
        "When splitting up a long document into chunks, how much stride to "
        "take between chunks.")
    
    flags.DEFINE_integer(
        "max_query_length", 64,
        "The maximum number of tokens for the question. Questions longer than "
        "this will be truncated to this length.")
    
    flags.DEFINE_bool("do_train", False, "Whether to run training.")
    
    flags.DEFINE_bool("do_predict", True, "Whether to run eval on the dev set.")
    
    flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")
    
    flags.DEFINE_integer("predict_batch_size", 8,
                         "Total batch size for predictions.")
    
    flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")
    
    flags.DEFINE_float("num_train_epochs", 3.0,
                       "Total number of training epochs to perform.")
    
    flags.DEFINE_float(
        "warmup_proportion", 0.1,
        "Proportion of training to perform linear learning rate warmup for. "
        "E.g., 0.1 = 10% of training.")
    
    flags.DEFINE_integer("save_checkpoints_steps", 1000,
                         "How often to save the model checkpoint.")
    
    flags.DEFINE_integer("iterations_per_loop", 1000,
                         "How many steps to make in each estimator call.")
    
    flags.DEFINE_integer(
        "n_best_size", 20,
        "The total number of n-best predictions to generate in the "
        "nbest_predictions.json output file.")
    
    flags.DEFINE_integer(
        "max_answer_length", 30,
        "The maximum length of an answer that can be generated. This is needed "
        "because the start and end predictions are not conditioned on one another.")
    
    flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")
    
    tf.flags.DEFINE_string(
        "tpu_name", None,
        "The Cloud TPU to use for training. This should be either the name "
        "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
        "url.")
    
    tf.flags.DEFINE_string(
        "tpu_zone", None,
        "[Optional] GCE zone where the Cloud TPU is located in. If not "
        "specified, we will attempt to automatically detect the GCE project from "
        "metadata.")
    
    tf.flags.DEFINE_string(
        "gcp_project", None,
        "[Optional] Project name for the Cloud TPU-enabled project. If not "
        "specified, we will attempt to automatically detect the GCE project from "
        "metadata.")
    
    tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")
    
    flags.DEFINE_integer(
        "num_tpu_cores", 8,
        "Only used if `use_tpu` is True. Total number of TPU cores to use.")
    
    flags.DEFINE_bool(
        "verbose_logging", False,
        "If true, all of the warnings related to data processing will be printed. "
        "A number of warnings are expected for a normal SQuAD evaluation.")
    
    flags.DEFINE_bool(
        "version_2_with_negative", False,
        "If true, the SQuAD examples contain some that do not have an answer.")
    
    flags.DEFINE_float(
        "null_score_diff_threshold", 0.0,
        "If null_score - best_non_null is greater than the threshold predict null.")

    flags.DEFINE_string(
        "input_pb_file", "",
        "The input protobuf file to convert tvm model. ")

    flags.DEFINE_integer('debug_sample', -1,
                        'the example number in test mode for sanity checks')

    flags.DEFINE_string('dtype', 'float32',
        "The data type for dataset. choices is 'float32' or 'float16'")

    args = flags.FLAGS
    for arg in kwargs:
        if not args([arg]):
            err_str = '"%s" is not among the following parameter list:\n\t' % (arg)
            err_str += '%s' % ('\n\t'.arg)
            raise ValueError(err_str)
        args.__flags[arg].value = kwargs[arg]
    return tf.logging, args


def freeze_savedmodel(saved_model_dir):
    output_graph_filename = os.path.join(str(saved_model_dir), "freezed_output_graph.pb")


    input_graph_def = saved_model_utils.read_saved_model(saved_model_dir)
    for meta_graph_def in input_graph_def.meta_graphs:
        print(meta_graph_def.meta_info_def.tags)
    input_graph_def = saved_model_utils.get_meta_graph_def(
        saved_model_dir, 'serve').graph_def
    #display_nodes(input_graph_def.node)   

    input_saved_model_dir = saved_model_dir
    # output_node_names = "bert/pooler/dense/BiasAdd"
    output_node_names = "BiasAdd"
    input_binary = False
    input_saver_def_path = False
    restore_op_name = None
    filename_tensor_name = None
    clear_devices = False
    input_meta_graph = False
    checkpoint_path = None
    input_graph_filename = None
    saved_model_tags = tag_constants.SERVING

    freeze_graph.freeze_graph(input_graph_filename, input_saver_def_path,
                              input_binary, checkpoint_path, output_node_names,
                              restore_op_name, filename_tensor_name,
                              output_graph_filename, clear_devices, "", "", "",
                              input_meta_graph, input_saved_model_dir,
                              saved_model_tags)
    return output_graph_filename


def optimize_inference(freezed_graph_filename, name):
    if not tf.gfile.Exists(freezed_graph_filename):
        raise ValueError(
            "Input graph file '" + freezed_graph_filename + "' does not exist!")

    input_graph_def = graph_pb2.GraphDef()
    with tf.gfile.Open(freezed_graph_filename, "rb") as f:
        data = f.read()
        input_graph_def.ParseFromString(data)

        #if args.frozen_graph:
        #    input_graph_def.ParseFromString(data)
        #else:
        #    text_format.Merge(data.decode("utf-8"), input_graph_def)

    input_node_names0 = "input_ids_1"
    input_node_names1 = "input_mask_1"
    input_node_names2 = "segment_ids_1"
    # output_node_names = "bert/pooler/dense/BiasAdd"
    output_node_names = "BiasAdd"


    output_graph_def = optimize_for_inference_lib.optimize_for_inference(
      input_graph_def,
      [input_node_names0,input_node_names1,input_node_names2],
      [output_node_names],
      dtypes.int32.as_datatype_enum, False)
   
    #display_nodes(output_graph_def.node)   
    #assert 0
    output_graph_filename = os.path.join(os.path.dirname(freezed_graph_filename), name + ".pb")

    if True:
        f = tf.gfile.FastGFile(output_graph_filename, "wb")
        f.write(output_graph_def.SerializeToString())
    else:
        graph_io.write_graph(output_graph_def,
                           os.path.dirname(output_graph_filename),
                           os.path.basename(output_graph_filename))

    return output_graph_filename


if __name__ == '__main__':
    log, args = get_args()
    save_model = export_savemodel(args)
    freeze_model = freeze_savedmodel(save_model)
    input_pb_file = optimize_inference(freeze_model, name=args.out_name)