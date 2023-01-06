from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import os
import argparse

import tensorflow as tf
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.tools import optimize_for_inference_lib, freeze_graph
from tensorflow.python.framework import dtypes, graph_io
from tensorflow.core.framework import graph_pb2

# from .bert import tokenization
from modeling import BertConfig, get_assignment_map_from_checkpoint, BertModel
from tokenization import validate_case_matches_checkpoint, convert_to_unicode
from optimization import create_optimizer


class InputExample(object):
  """A single training/test example for simple sequence classification."""

  def __init__(self, guid, text_a, text_b=None, label=None):
    """Constructs a InputExample.

    Args:
      guid: Unique id for the example.
      text_a: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
      text_b: (Optional) string. The untokenized text of the second sequence.
        Only must be specified for sequence pair tasks.
      label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """
    self.guid = guid
    self.text_a = text_a
    self.text_b = text_b
    self.label = label


class DataProcessor(object):
  """Base class for data converters for sequence classification data sets."""

  def get_train_examples(self, data_dir):
    """Gets a collection of `InputExample`s for the train set."""
    raise NotImplementedError()

  def get_dev_examples(self, data_dir):
    """Gets a collection of `InputExample`s for the dev set."""
    raise NotImplementedError()

  def get_test_examples(self, data_dir):
    """Gets a collection of `InputExample`s for prediction."""
    raise NotImplementedError()

  def get_labels(self):
    """Gets the list of labels for this data set."""
    raise NotImplementedError()

  @classmethod
  def _read_tsv(cls, input_file, quotechar=None):
    """Reads a tab separated value file."""
    with tf.gfile.Open(input_file, "r") as f:
      reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
      lines = []
      for line in reader:
        lines.append(line)
      return lines


class MrpcProcessor(DataProcessor):
  """Processor for the MRPC data set (GLUE version)."""

  def get_train_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

  def get_dev_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

  def get_test_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

  def get_labels(self):
    """See base class."""
    return ["0", "1"]

  def _create_examples(self, lines, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    for (i, line) in enumerate(lines):
      if i == 0:
        continue
      guid = "%s-%s" % (set_type, i)
      text_a = convert_to_unicode(line[3])
      text_b = convert_to_unicode(line[4])
      if set_type == "test":
        label = "0"
      else:
        label = convert_to_unicode(line[0])
      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    return examples


def create_model(bert_config, is_training, input_ids, input_mask, segment_ids,
                 labels, num_labels, use_one_hot_embeddings):
  """Creates a classification model."""
  model = BertModel(
      config=bert_config,
      is_training=is_training,
      input_ids=input_ids,
      input_mask=input_mask,
      token_type_ids=segment_ids,
      use_one_hot_embeddings=use_one_hot_embeddings)
  # In the demo, we are doing a simple classification task on the entire
  # segment.
  #
  # If you want to use the token-level output, use model.get_sequence_output()
  # instead.
  output_layer = model.get_pooled_output()

  hidden_size = output_layer.shape[-1].value

  output_weights = tf.get_variable(
      "output_weights", [num_labels, hidden_size],
      initializer=tf.truncated_normal_initializer(stddev=0.02))

  output_bias = tf.get_variable(
      "output_bias", [num_labels], initializer=tf.zeros_initializer())

  with tf.variable_scope("loss"):
    if is_training:
      # I.e., 0.1 dropout
      output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

    logits = tf.matmul(output_layer, output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)
    probabilities = tf.nn.softmax(logits, axis=-1)
    log_probs = tf.nn.log_softmax(logits, axis=-1)

    one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)

    per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
    loss = tf.reduce_mean(per_example_loss)

    return (loss, per_example_loss, logits, probabilities)


def model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings):
  """Returns `model_fn` closure for TPUEstimator."""

  def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
    """The `model_fn` for TPUEstimator."""

    tf.logging.info("*** Features ***")
    for name in sorted(features.keys()):
      tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

    #input_ids = features["input_ids"]
    #input_mask = features["input_mask"]
    #segment_ids = features["segment_ids"]
    #label_ids = features["label_ids"]

    input_ids = tf.placeholder(shape=[1, 128], dtype=tf.int32, name="input_ids")
    input_mask = tf.placeholder(shape=[1, 128], dtype=tf.int32, name="input_mask")
    segment_ids = tf.placeholder(shape=[1, 128], dtype=tf.int32, name="segment_ids")
    label_ids = tf.placeholder(shape=[1, 128], dtype=tf.int32, name="label_ids")


    is_real_example = None
    if "is_real_example" in features:
      is_real_example = tf.cast(features["is_real_example"], dtype=tf.float32)
    else:
      is_real_example = tf.ones(tf.shape(label_ids), dtype=tf.float32)

    is_training = False #(mode == tf.estimator.ModeKeys.TRAIN)
    (total_loss, per_example_loss, logits, probabilities) = create_model(
        bert_config, is_training, input_ids, input_mask, segment_ids, label_ids,
        num_labels, use_one_hot_embeddings)

    tvars = tf.trainable_variables()
    initialized_variable_names = {}
    scaffold_fn = None

    if init_checkpoint:
      (assignment_map, initialized_variable_names
      ) = get_assignment_map_from_checkpoint(tvars, init_checkpoint)
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

      train_op = create_optimizer(
          total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)

      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=total_loss,
          train_op=train_op,
          scaffold_fn=scaffold_fn)
    elif mode == tf.estimator.ModeKeys.EVAL:

      def metric_fn(per_example_loss, label_ids, logits, is_real_example):
        predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
        accuracy = tf.metrics.accuracy(
            labels=label_ids, predictions=predictions, weights=is_real_example)
        loss = tf.metrics.mean(values=per_example_loss, weights=is_real_example)
        return {
            "eval_accuracy": accuracy,
            "eval_loss": loss,
        }

      eval_metrics = (metric_fn,
                      [per_example_loss, label_ids, logits, is_real_example])
      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=total_loss,
          eval_metrics=eval_metrics,
          scaffold_fn=scaffold_fn)
    else:
      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          predictions={"probabilities": probabilities},
          scaffold_fn=scaffold_fn)
    return output_spec

  return model_fn


def export_savemodel(args):

    processors = {
        "mrpc": MrpcProcessor,
    }

    validate_case_matches_checkpoint(
        True, 
        args.init_checkpoint)

    bert_config = BertConfig.from_json_file(args.bert_config_file)

    if args.max_seq_length > bert_config.max_position_embeddings:
      raise ValueError(
          "Cannot use sequence length %d because the BERT model "
          "was only trained up to sequence length %d" %
          (args.max_seq_length, bert_config.max_position_embeddings))

    
    tf.gfile.MakeDirs(args.output_dir)

    task_name = args.task_name.lower()

    if task_name not in processors:
      raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name]()

    label_list = processor.get_labels()

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
        num_labels=len(label_list),
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
        eval_batch_size=args.eval_batch_size,
        predict_batch_size=args.predict_batch_size)

    input_ids = tf.placeholder(shape=[1, args.max_seq_length], dtype=tf.int32, name="input_ids")
    input_mask = tf.placeholder(shape=[1, args.max_seq_length], dtype=tf.int32, name="input_mask")
    segment_ids = tf.placeholder(shape=[1, args.max_seq_length], dtype=tf.int32, name="segment_ids")
    label_ids = tf.placeholder(shape=[1, args.max_seq_length], dtype=tf.int32, name="segment_ids")

    serving_input_receiver_fn = tf.estimator.export.build_raw_serving_input_receiver_fn({
        "input_ids" : input_ids,
        "input_mask" : input_mask,
        "segment_ids" : segment_ids,
        "label_ids" : label_ids})

    saved_model_dir = args.output_dir #os.path.join(tempfile.gettempdir(), '')
    #os.makedirs(saved_model_dir, exist_ok=True)
    #tf.gfile.MakeDirs(args.output_dir)
    return estimator.export_saved_model(saved_model_dir, serving_input_receiver_fn, None, False, args.init_checkpoint).decode('utf-8')


def freeze_savedmodel(saved_model_dir):
    output_graph_filename = os.path.join(str(saved_model_dir), "freezed_output_graph.pb")
    input_saved_model_dir = saved_model_dir
    output_node_names = "loss/BiasAdd"
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


def optimize_inference(freezed_graph_filename):
    if not tf.gfile.Exists(freezed_graph_filename):
        raise ValueError(
            "Input graph file '" + freezed_graph_filename + "' does not exist!")

    input_graph_def = graph_pb2.GraphDef()
    with tf.gfile.Open(freezed_graph_filename, "rb") as f:
        data = f.read()
        input_graph_def.ParseFromString(data)

        #display_nodes(input_graph_def.node)   
        #if args.frozen_graph:
        #    input_graph_def.ParseFromString(data)
        #else:
        #    text_format.Merge(data.decode("utf-8"), input_graph_def)

    input_node_names0 = "input_ids_1"
    input_node_names1 = "input_mask_1"
    input_node_names2 = "segment_ids_2"
    output_node_names = "loss/BiasAdd"


    output_graph_def = optimize_for_inference_lib.optimize_for_inference(
      input_graph_def,
      [input_node_names0,input_node_names1,input_node_names2],
      [output_node_names],
      dtypes.int32.as_datatype_enum, False)
   
    #display_nodes(output_graph_def.node)   

    output_graph_filename = os.path.join(os.path.dirname(freezed_graph_filename), "inference_output_graph.pb")

    if True:
        f = tf.gfile.FastGFile(output_graph_filename, "wb")
        f.write(output_graph_def.SerializeToString())
    else:
        graph_io.write_graph(output_graph_def,
                           os.path.dirname(output_graph_filename),
                           os.path.basename(output_graph_filename))

    return output_graph_filename


def get_args(**kwargs):
    parse = argparse.ArgumentParser()
    parse.add_argument("--init_checkpoint", type=str,  default='/home/jies/code/nlp/bert/tmp/bert_large_b64_epoch20/model.ckpt-1146')
    parse.add_argument("--bert_config_file", type=str, default='/home/jies/code/nlp/bert/weights/wwm_uncased_L-24_H-1024_A-16/bert_config.json')
    parse.add_argument("--max_seq_length", type=int, default=128)
    parse.add_argument("--output_dir", type=str, default='/home/jies/code/nlp/bert/tmp/bert_large_b64_epoch20')
    parse.add_argument('--task_name', type=str, default='mrpc') 
    parse.add_argument('--use_tpu', type=bool, default=False) 
    parse.add_argument('--save_checkpoints_steps', type=int, default=1000) 
    parse.add_argument('--iterations_per_loop', type=int, default=1000) 
    parse.add_argument('--train_batch_size', type=int, default=32) 
    parse.add_argument('--eval_batch_size', type=int, default=8) 
    parse.add_argument('--predict_batch_size', type=int, default=8) 
    parse.add_argument('--num_tpu_cores', type=int, default=8) 
    parse.add_argument('--master', type=str, default=None) 
    parse.add_argument('--tpu_name', type=str, default=None)
    parse.add_argument('--tpu_zone', type=str, default=None)
    parse.add_argument('--gcp_project', type=str, default=None)
    parse.add_argument('--learning_rate', type=float, default=5e-5)
    args = parse.parse_args()
    
    return args


if __name__ == "__main__":
    args = get_args()
    save_model = export_savemodel(args)
    out_dir = optimize_inference(freeze_savedmodel(save_model))
    
