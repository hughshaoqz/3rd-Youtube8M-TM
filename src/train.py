# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Binary for training Tensorflow models on the YouTube-8M dataset."""

import json
import os
import time
import string
import random

import eval_util
import export_model
import losses
import frame_level_models
import nextvlad
import video_level_models
import readers
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.lib.io import file_io
from tensorflow import app
from tensorflow import flags
from tensorflow import gfile
from tensorflow import logging
from tensorflow.python.client import device_lib
import utils
import optimization
from radam import RAdamOptimizer

FLAGS = flags.FLAGS

if __name__ == "__main__":
  # Dataset flags.
  flags.DEFINE_string("train_dir", "/tmp/yt8m_model/",
                      "The directory to save the model files in.")
  flags.DEFINE_string(
      "train_data_pattern", "",
      "File glob for the training dataset. If the files refer to Frame Level "
      "features (i.e. tensorflow.SequenceExample), then set --reader_type "
      "format. The (Sequence)Examples are expected to have 'rgb' byte array "
      "sequence feature as well as a 'labels' int64 context feature.")
  flags.DEFINE_string(
      "valid_data_pattern", "",
      "File glob for the valid dataset. Must have the same format as train_data_pattern.")
  flags.DEFINE_string("feature_names", "mean_rgb", "Name of the feature "
                      "to use for training.")
  flags.DEFINE_string("feature_sizes", "1024", "Length of the feature vectors.")

  # Model flags.
  flags.DEFINE_bool(
      "frame_features", False,
      "If set, then --train_data_pattern must be frame-level features. "
      "Otherwise, --train_data_pattern must be aggregated video-level "
      "features. The model must also be set appropriately (i.e. to read 3D "
      "batches VS 4D batches.")
  flags.DEFINE_bool(
      "segment_labels", False,
      "If set, then --train_data_pattern must be frame-level features (but with"
      " segment_labels). Otherwise, --train_data_pattern must be aggregated "
      "video-level features. The model must also be set appropriately (i.e. to "
      "read 3D batches VS 4D batches.")
  flags.DEFINE_bool(
      "valid_segment_labels", False,
      "Same but for validation.")
  flags.DEFINE_string(
      "model", "LogisticModel",
      "Which architecture to use for the model. Models are defined "
      "in models.py.")
  flags.DEFINE_bool(
      "start_new_model", False,
      "If set, this will not resume from a checkpoint and will instead create a"
      " new model instance.")
  flags.DEFINE_bool(
      "google_cloud", False,
      "If set, it means we use eval data for google cloud running.")

  # Training flags.
  flags.DEFINE_integer(
      "num_gpu", 1, "The maximum number of GPU devices to use for training. "
      "Flag only applies if GPUs are installed")
  flags.DEFINE_integer("batch_size", 1024,
                       "How many examples to process per batch for training.")
  flags.DEFINE_string("label_loss", "CrossEntropyLoss",
                      "Which loss function to use for training the model.")
  flags.DEFINE_float(
      "regularization_penalty", 1.0,
      "How much weight to give to the regularization loss (the label loss has "
      "a weight of 1).")
  flags.DEFINE_float("fix_learning_rate", 0.,
                     "Overwrite learning rate with this one if this one is non-zero.")
  flags.DEFINE_float("base_learning_rate", 0.01,
                     "Which learning rate to start with.")
  flags.DEFINE_float(
      "learning_rate_decay", 0.95,
      "Learning rate decay factor to be applied every "
      "learning_rate_decay_examples.")
  flags.DEFINE_float(
      "learning_rate_decay_examples", 4000000,
      "Multiply current learning rate by learning_rate_decay "
      "every learning_rate_decay_examples.")
  flags.DEFINE_integer(
      "num_epochs", 5, "How many passes to make over the dataset before "
      "halting training.")
  flags.DEFINE_integer(
      "max_steps", None,
      "The maximum number of iterations of the training loop.")
  flags.DEFINE_integer(
      "export_model_steps", 1000,
      "The period, in number of steps, with which the model "
      "is exported for batch prediction.")

  # Validation flags.
  flags.DEFINE_integer("valid_top_k", 20,
                       "How many predictions when validation.")
  flags.DEFINE_integer("valid_batch_size", 128,
                       "How many examples to process per batch for validation.")
  flags.DEFINE_integer("valid_checkpoint", 500,
                       "The training step when running validation once.")
  flags.DEFINE_integer("valid_size", 1024,
                       "The size of total validation data you want to use for validation check.")

  # Other flags.
  flags.DEFINE_integer("num_readers", 8,
                       "How many threads to use for reading input files.")
  flags.DEFINE_string("optimizer", "AdamOptimizer",
                      "What optimizer class to use.")
  flags.DEFINE_string(
      "custom_optimizer", '',
      "Whether use bert optimizer.")

  flags.DEFINE_float("clip_gradient_norm", 1.0, "Norm to clip gradients to.")
  flags.DEFINE_bool(
      "log_device_placement", False,
      "Whether to write the device on which every op will run into the "
      "logs on startup.")

  flags.DEFINE_integer("netvlad_cluster_size", 0,
                       "Number of units in the NetVLAD cluster layer.")

  flags.DEFINE_integer("netvlad_hidden_size", 0,
                       "Number of units in the NetVLAD hidden layer.")
  flags.DEFINE_bool(
      "train_all", False,
      "Whether train both train and validate data.")
  flags.DEFINE_string(
      "train_valid_data_pattern", "",
      "File glob for the valid dataset. Must have the same format as train_data_pattern.")
  flags.DEFINE_integer(
      "max_frames", 300,
      "max frames to cap.")
  flags.DEFINE_string("checkpoint_file", "",
                        "e.g. model.ckpt-170820")

def validate_class_name(flag_value, category, modules, expected_superclass):
  """Checks that the given string matches a class of the expected type.

  Args:
    flag_value: A string naming the class to instantiate.
    category: A string used further describe the class in error messages (e.g.
      'model', 'reader', 'loss').
    modules: A list of modules to search for the given class.
    expected_superclass: A class that the given class should inherit from.

  Raises:
    FlagsError: If the given class could not be found or if the first class
    found with that name doesn't inherit from the expected superclass.

  Returns:
    True if a class was found that matches the given constraints.
  """
  candidates = [getattr(module, flag_value, None) for module in modules]
  for candidate in candidates:
    if not candidate:
      continue
    if not issubclass(candidate, expected_superclass):
      raise flags.FlagsError(
          "%s '%s' doesn't inherit from %s." %
          (category, flag_value, expected_superclass.__name__))
    return True
  raise flags.FlagsError("Unable to find %s '%s'." % (category, flag_value))

def get_input_data_tensors(reader,
                           data_pattern,
                           batch_size=1000,
                           num_epochs=None,
                           num_readers=1,
                           train=True):
  """Creates the section of the graph which reads the training data.
  Args:
    reader: A class which parses the training data.
    data_pattern: A 'glob' style path to the data files.
    batch_size: How many examples to process at a time.
    num_epochs: How many passes to make over the training data. Set to 'None' to
      run indefinitely.
    num_readers: How many I/O threads to use.
  Returns:
    A tuple containing the features tensor, labels tensor, and optionally a
    tensor containing the number of frames per video. The exact dimensions
    depend on the reader being used.
  Raises:
    IOError: If no files matching the given pattern were found.
  """
  logging.info("Using batch size of " + str(batch_size) + " for training.")
  with tf.name_scope("train_input"):
    files = gfile.Glob(data_pattern)
    if FLAGS.train_all or FLAGS.segment_labels:

      # gcloud
      if FLAGS.google_cloud:
        a_list = list(string.ascii_lowercase)
        A_list = list(string.ascii_uppercase)

        n_list = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        n_list = n_list + a_list + A_list

        results = []
        for a in n_list:
          for n in n_list:
              results.append(n + a)
        #random.seed(7)
        #random.shuffle(results)
        if train:
            validate_file_nums = results[300:]
        else:
            validate_file_nums = results[:300]

      else:
        # local
        results = []
        for i in range(3844):
            results.append(str(i).zfill(4))
        #random.seed(7)
        #random.shuffle(results)

        if train:
            validate_file_nums = results[300:]
        else:
            validate_file_nums = results[:300]

    if FLAGS.train_all:
      validate_file_list = [FLAGS.train_valid_data_pattern.split('*')[0]\
                               + x +'.tfrecord' for x in validate_file_nums]
      files = files + validate_file_list
    elif FLAGS.segment_labels:
      print('Finetune !')
      files = [FLAGS.train_data_pattern.split('*')[0]\
                               + x +'.tfrecord' for x in validate_file_nums]

    if not files:
      raise IOError("Unable to find training files. data_pattern='" +
                    data_pattern + "'.")
    logging.info("Number of training files: %s.", str(len(files)))
    filename_queue = tf.train.string_input_producer(files,
                                                    num_epochs=num_epochs,
                                                    shuffle=True)
    training_data = [
        reader.prepare_reader(filename_queue) for _ in range(num_readers)
    ]

    return tf.train.shuffle_batch_join(training_data,
                                       batch_size=batch_size,
                                       capacity=batch_size * 5,
                                       min_after_dequeue=batch_size,
                                       allow_smaller_final_batch=True,
                                       enqueue_many=True)
"""
def get_input_data_tensors(reader,
                           data_pattern,
                           batch_size=1000,
                           num_epochs=None,
                           num_readers=1):
  """"""Creates the section of the graph which reads the training data.

  Args:
    reader: A class which parses the training data.
    data_pattern: A 'glob' style path to the data files.
    batch_size: How many examples to process at a time.
    num_epochs: How many passes to make over the training data. Set to 'None' to
      run indefinitely.
    num_readers: How many I/O threads to use.

  Returns:
    A tuple containing the features tensor, labels tensor, and optionally a
    tensor containing the number of frames per video. The exact dimensions
    depend on the reader being used.

  Raises:
    IOError: If no files matching the given pattern were found.
  """"""
  logging.info("Using batch size of " + str(batch_size) + " for training.")
  with tf.name_scope("train_input"):
    files = gfile.Glob(data_pattern)

    # randomly chosen 60 validate files
    # note that validate file names are different on gcloud and locally, due to `curl` download command
    
    # gcloud
    if FLAGS.google_cloud:
      validate_file_nums = ['0t','1Y','2J','45','5K','5u','63','6f','8F','8f','9y','Ap','BN','CH',
                            'CI','Dz','Er','GY','I6','JP','JV','K0','MJ','Mv','Og','Om','PL','QK',
                            'Qh','Ql','T4','UF','Uy','Vo','X6','XX','Zq','aR','cU','fr','hw','k3',
                            'lw','nX','nl','o6','p7','pL','pg','rx','sZ','sd','uS','uf','y1','y5',
                            'yK','yU','z8','zE']
    else:
      # local
      validate_file_nums = [
       '0855', '2284', '3096', '0170', '2846', '0936', '2486', '0817', '0967', '1877', 
       '2876', '3336', '3178', '0675', '3243', '2640', '1167', '3601', '1245', '3570', 
       '2492', '0456', '0926', '1077', '1284', '3554', '0989', '1627', '1524', '3383',
       '2611', '2166', '2377', '3529', '0043', '2211', '1541', '1119', '3725', '1770',
       '3806', '2615', '3087', '1545', '2928', '3651', '1610', '2883', '0704', '1713',
       '2217', '1534', '2579', '1580', '2034', '3751', '1823', '2391', '1769', '0327']

    validate_file_list_60 = [data_pattern.split('*')[0]\
                           + x +'.tfrecord' for x in validate_file_nums]

    train_file_list = [x for x in files if x not in validate_file_list_60]

    if not train_file_list:
      raise IOError("Unable to find training files. data_pattern='" +
                    data_pattern + "'.")
    logging.info("Number of training files: %s.", str(len(train_file_list)))
    filename_queue = tf.train.string_input_producer(
        train_file_list, num_epochs=num_epochs, shuffle=True)
    training_data = [
        reader.prepare_reader(filename_queue) for _ in range(num_readers)
    ]

    return tf.train.shuffle_batch_join(
        training_data,
        batch_size=batch_size,
        capacity=batch_size * 5,
        min_after_dequeue=batch_size,
        allow_smaller_final_batch=True,
        enqueue_many=True)
"""

def find_class_by_name(name, modules):
  """Searches the provided modules for the named class and returns it."""
  modules = [getattr(module, name, None) for module in modules]
  return next(a for a in modules if a)

################################VALIDATION##############################
def get_input_evaluation_tensors(reader,
                                 data_pattern,
                                 batch_size=1024,
                                 num_readers=1):
  """Creates the section of the graph which reads the evaluation data.

  Args:
    reader: A class which parses the training data.
    data_pattern: A 'glob' style path to the data files.
    batch_size: How many examples to process at a time.
    num_readers: How many I/O threads to use.

  Returns:
    A tuple containing the features tensor, labels tensor, and optionally a
    tensor containing the number of frames per video. The exact dimensions
    depend on the reader being used.

  Raises:
    IOError: If no files matching the given pattern were found.
  """
  logging.info("Using batch size of %d for evaluation.", batch_size)
  with tf.name_scope("eval_input"):
    #files = gfile.Glob(data_pattern)
    results = []
    for i in range(3844):
        results.append(str(i).zfill(4))
    random.seed(7)
    random.shuffle(results)

    validate_file_nums = results[:300]

    files = [FLAGS.train_valid_data_pattern.split('*')[0] \
                        + x + '.tfrecord' for x in validate_file_nums]
    if not files:
      raise IOError("Unable to find the evaluation files.")
    logging.info("number of evaluation files: %d", len(files))
    filename_queue = tf.train.string_input_producer(
        files, shuffle=False, num_epochs=1)
    eval_data = [
        reader.prepare_reader(filename_queue) for _ in range(num_readers)
    ]
    return tf.train.batch_join(
        eval_data,
        batch_size=batch_size,
        capacity=3 * batch_size,
        allow_smaller_final_batch=True,
        enqueue_many=True)


def build_graph_valid(reader,
                model,
                eval_data_pattern,
                label_loss_fn,
                batch_size=1024,
                num_readers=1):
  """Creates the Tensorflow graph for evaluation.

  Args:
    reader: The data file reader. It should inherit from BaseReader.
    model: The core model (e.g. logistic or neural net). It should inherit from
      BaseModel.
    eval_data_pattern: glob path to the evaluation data files.
    label_loss_fn: What kind of loss to apply to the model. It should inherit
      from BaseLoss.
    batch_size: How many examples to process at a time.
    num_readers: How many threads to use for I/O operations.
  """

  global_step = tf.Variable(0, trainable=False, name="global_step")
  input_data_dict = get_input_evaluation_tensors(
      reader, eval_data_pattern, batch_size=batch_size, num_readers=num_readers)
  video_id_batch = input_data_dict["video_ids"]
  model_input_raw = input_data_dict["video_matrix"]
  labels_batch = input_data_dict["labels"]
  num_frames = input_data_dict["num_frames"]
  tf.summary.histogram("model_input_raw", model_input_raw)

  feature_dim = len(model_input_raw.get_shape()) - 1

  # Normalize input features.
  model_input = tf.nn.l2_normalize(model_input_raw, feature_dim)

  with tf.variable_scope("tower", reuse=True):
    result = model.create_model(
        model_input,
        num_frames=num_frames,
        vocab_size=reader.num_classes,
        labels=labels_batch,
        is_training=False)
    predictions = result["predictions"]
    tf.summary.histogram("model_activations", predictions)
    if "loss" in result.keys():
      label_loss = result["loss"]
    else:
      label_loss = label_loss_fn.calculate_loss(predictions, labels_batch)

  tf.add_to_collection("valid_global_step", global_step)
  tf.add_to_collection("valid_loss", label_loss)
  tf.add_to_collection("valid_predictions", predictions)
  tf.add_to_collection("valid_input_batch", model_input)
  tf.add_to_collection("valid_input_batch_raw", model_input_raw)
  tf.add_to_collection("valid_video_id_batch", video_id_batch)
  tf.add_to_collection("valid_num_frames", num_frames)
  tf.add_to_collection("valid_labels", tf.cast(labels_batch, tf.float32))
  if FLAGS.valid_segment_labels:
    tf.add_to_collection("valid_label_weights", input_data_dict["label_weights"])
  tf.add_to_collection("valid_summary_op", tf.summary.merge_all())

################################VALIDATION##############################

def build_graph(reader,
                model,
                train_data_pattern,
                eval_reader,
                eval_data_pattern,
                eval_batch_size=1024,
                label_loss_fn=losses.CrossEntropyLoss(),
                batch_size=1000,
                base_learning_rate=0.01,
                fix_learning_rate=0,
                learning_rate_decay_examples=1000000,
                learning_rate_decay=0.95,
                optimizer_class=tf.train.AdamOptimizer,
                clip_gradient_norm=1.0,
                regularization_penalty=1,
                num_readers=1,
                num_epochs=None,
):
  """Creates the Tensorflow graph.

  This will only be called once in the life of
  a training model, because after the graph is created the model will be
  restored from a meta graph file rather than being recreated.

  Args:
    reader: The data file reader. It should inherit from BaseReader.
    model: The core model (e.g. logistic or neural net). It should inherit from
      BaseModel.
    train_data_pattern: glob path to the training data files.
    label_loss_fn: What kind of loss to apply to the model. It should inherit
      from BaseLoss.
    batch_size: How many examples to process at a time.
    base_learning_rate: What learning rate to initialize the optimizer with.
    optimizer_class: Which optimization algorithm to use.
    clip_gradient_norm: Magnitude of the gradient to clip to.
    regularization_penalty: How much weight to give the regularization loss
      compared to the label loss.
    num_readers: How many threads to use for I/O operations.
    num_epochs: How many passes to make over the data. 'None' means an unlimited
      number of passes.
  """

  global_step = tf.Variable(0, trainable=False, name="global_step")

  local_device_protos = device_lib.list_local_devices()
  gpus = [x.name for x in local_device_protos if x.device_type == "GPU"]
  gpus = gpus[:FLAGS.num_gpu]
  num_gpus = len(gpus)

  if num_gpus > 0:
    logging.info("Using the following GPUs to train: " + str(gpus))
    num_towers = num_gpus
    device_string = "/gpu:%d"
  else:
    logging.info("No GPUs found. Training on CPU.")
    num_towers = 1
    device_string = "/cpu:%d"

  #"""
  if FLAGS.custom_optimizer != 'bert':
      ###################ORIGINAL_OPTIMIZER##################
      if fix_learning_rate:
        learning_rate = fix_learning_rate
      else:
        learning_rate = tf.train.exponential_decay(
            base_learning_rate,
            global_step * batch_size * num_towers,
            learning_rate_decay_examples,
            learning_rate_decay,
            staircase=True)
      tf.summary.scalar("learning_rate", learning_rate)

      if FLAGS.custom_optimizer == 'radam':
        logging.info('Use RADAM Optimizer')
        optimizer = RAdamOptimizer(learning_rate=learning_rate)
      else:
        logging.info('Use ADAM Optimizer')
        optimizer = optimizer_class(learning_rate)
      ###################ORIGINAL_OPTIMIZER##################
  #"""
  #"""
  else:
      logging.info('Use Bert Optimizer')
      ###########################BERT OPTIMIZER#######################
      learning_rate = tf.constant(value=base_learning_rate, shape=[], dtype=tf.float32)

      # Implements linear decay of the learning rate.
      learning_rate = tf.train.polynomial_decay(
          learning_rate,
          global_step,
          learning_rate_decay_examples,
          end_learning_rate=0.0,
          power=1.0,
          cycle=False)

      # Implements linear warmup. I.e., if global_step < num_warmup_steps, the
      # learning rate will be `global_step/num_warmup_steps * init_lr`.
      num_warmup_steps = None
      if num_warmup_steps:
        global_steps_int = tf.cast(global_step, tf.int32)
        warmup_steps_int = tf.constant(num_warmup_steps, dtype=tf.int32)

        global_steps_float = tf.cast(global_steps_int, tf.float32)
        warmup_steps_float = tf.cast(warmup_steps_int, tf.float32)

        warmup_percent_done = global_steps_float / warmup_steps_float
        warmup_learning_rate = base_learning_rate * warmup_percent_done

        is_warmup = tf.cast(global_steps_int < warmup_steps_int, tf.float32)
        learning_rate = (
            (1.0 - is_warmup) * learning_rate + is_warmup * warmup_learning_rate)

      # It is recommended that you use this optimizer for fine tuning, since this
      # is how the model was trained (note that the Adam m/v variables are NOT
      # loaded from init_checkpoint.)
      optimizer = optimization.AdamWeightDecayOptimizer(
          learning_rate=learning_rate,
          weight_decay_rate=0.01,
          beta_1=0.9,
          beta_2=0.999,
          epsilon=1e-6,
          exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"])
      #optimizer = RAdamOptimizer(learning_rate=learning_rate)

      tf.summary.scalar("learning_rate", learning_rate)

      #optimizer = optimizer_class(learning_rate)
      ###########################BERT OPTIMIZER#######################
      #"""

  ############# Train Data Processing ############
  input_data_dict = (
      get_input_data_tensors(
          reader,
          train_data_pattern,
          batch_size=batch_size * num_towers,
          num_readers=num_readers,
          num_epochs=num_epochs,
          train=True))
  #breakpoint()
  model_input_raw = input_data_dict["video_matrix"]
  labels_batch = input_data_dict["labels"]
  num_frames = input_data_dict["num_frames"]
  #range_mtx = input_data_dict["range_mtx"]
  video_ids_batch = input_data_dict['video_ids']
  original_setment = input_data_dict['original_setment']
  #mask = input_data_dict['mask']
  print("model_input_shape, ", model_input_raw.shape)
  #print("video_id_batch", video_ids_batch)
  tf.summary.histogram("model/input_raw", model_input_raw)

  feature_dim = len(model_input_raw.get_shape()) - 1

  model_input = tf.nn.l2_normalize(model_input_raw, feature_dim)
  ##breakpoint()
  tower_inputs = tf.split(model_input, num_towers)
  tower_labels = tf.split(labels_batch, num_towers)
  tower_num_frames = tf.split(num_frames, num_towers)
  #tower_range_mtx = tf.split(range_mtx, num_towers)
  tower_gradients = []
  tower_predictions = []
  tower_label_losses = []
  tower_reg_losses = []
  #breakpoint()
  for i in range(num_towers):
    # For some reason these 'with' statements can't be combined onto the same
    # line. They have to be nested.
    with tf.device(device_string % i):
      with (tf.variable_scope(("tower"), reuse=True if i > 0 else None)):
        with (slim.arg_scope([slim.model_variable, slim.variable],
                             device="/cpu:0" if num_gpus != 1 else "/gpu:0")):
          result = model.create_model(
              tower_inputs[i],
              num_frames=tower_num_frames[i],
              vocab_size=reader.num_classes,
              labels=tower_labels[i],
              #range_mtx=tower_range_mtx[i],
          )

          for variable in slim.get_model_variables():
            tf.summary.histogram(variable.op.name, variable)

          predictions = result["predictions"]
          tower_predictions.append(predictions)

          if "loss" in result.keys():
            label_loss = result["loss"]
          else:
            label_loss = label_loss_fn.calculate_loss(predictions,
                                                      tower_labels[i])

          if "regularization_loss" in result.keys():
            reg_loss = result["regularization_loss"]
          else:
            reg_loss = tf.constant(0.0)

          reg_losses = tf.losses.get_regularization_losses()
          if reg_losses:
            reg_loss += tf.add_n(reg_losses)

          tower_reg_losses.append(reg_loss)

          # Adds update_ops (e.g., moving average updates in batch normalization) as
          # a dependency to the train_op.
          update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
          if "update_ops" in result.keys():
            update_ops += result["update_ops"]
          if update_ops:
            with tf.control_dependencies(update_ops):
              barrier = tf.no_op(name="gradient_barrier")
              with tf.control_dependencies([barrier]):
                label_loss = tf.identity(label_loss)

          tower_label_losses.append(label_loss)

          # Incorporate the L2 weight penalties etc.
          final_loss = regularization_penalty * reg_loss + label_loss
          gradients = optimizer.compute_gradients(
              final_loss, colocate_gradients_with_ops=False)
          tower_gradients.append(gradients)
  label_loss = tf.reduce_mean(tf.stack(tower_label_losses))
  tf.summary.scalar("label_loss", label_loss)
  if regularization_penalty != 0:
    reg_loss = tf.reduce_mean(tf.stack(tower_reg_losses))
    tf.summary.scalar("reg_loss", reg_loss)
  merged_gradients = utils.combine_gradients(tower_gradients)

  if clip_gradient_norm > 0:
    with tf.name_scope("clip_grads"):
      merged_gradients = utils.clip_gradient_norms(merged_gradients,
                                                   clip_gradient_norm)
  train_op = optimizer.apply_gradients(
      merged_gradients, global_step=global_step)
  #new_global_step = global_step + 1
  #train_op = tf.group(train_op, [global_step.assign(new_global_step)])

  tf.add_to_collection("global_step", global_step)
  tf.add_to_collection("loss", label_loss)
  tf.add_to_collection("predictions", tf.concat(tower_predictions, 0))
  tf.add_to_collection("input_batch_raw", model_input_raw)
  tf.add_to_collection("input_batch", model_input)
  tf.add_to_collection("num_frames", num_frames)
  tf.add_to_collection("video_ids", video_ids_batch)
  #tf.add_to_collection("range_mtx", range_mtx)
  #tf.add_to_collection("original_setment", original_setment)
  tf.add_to_collection("labels", tf.cast(labels_batch, tf.float32))
  tf.add_to_collection("train_op", train_op)

  ############# Train Data Processing ############

  ############# Valid Data Processing ############
  if eval_data_pattern:
      valid_input_data_dict = (
          get_input_data_tensors(
              eval_reader,
              eval_data_pattern,
              batch_size=eval_batch_size,
              train=False))

      valid_video_id_batch = valid_input_data_dict["video_ids"]
      valid_model_input_raw = valid_input_data_dict["video_matrix"]
      valid_labels_batch = valid_input_data_dict["labels"]
      valid_num_frames = valid_input_data_dict["num_frames"]
      tf.summary.histogram("valid_model_input_raw", valid_model_input_raw)

      valid_feature_dim = len(valid_model_input_raw.get_shape()) - 1

      valid_model_input = tf.nn.l2_normalize(valid_model_input_raw, valid_feature_dim)

      with tf.variable_scope("tower", reuse=True):
          valid_result = model.create_model(
              valid_model_input,
              num_frames=valid_num_frames,
              vocab_size=eval_reader.num_classes,
              labels=valid_labels_batch,
              is_training=False)
          valid_predictions = valid_result["predictions"]
          tf.summary.histogram("valid_model_activations", valid_predictions)
          if "loss" in valid_result.keys():
              valid_label_loss = valid_result["loss"]
          else:
              valid_label_loss = label_loss_fn.calculate_loss(valid_predictions, valid_labels_batch)

      tf.add_to_collection("valid_global_step", global_step)
      tf.add_to_collection("valid_loss", valid_label_loss)
      tf.add_to_collection("valid_predictions", valid_predictions)
      tf.add_to_collection("valid_input_batch", valid_model_input)
      tf.add_to_collection("valid_input_batch_raw", valid_model_input_raw)
      tf.add_to_collection("valid_video_id_batch", valid_video_id_batch)
      tf.add_to_collection("valid_num_frames", valid_num_frames)
      tf.add_to_collection("valid_labels", tf.cast(valid_labels_batch, tf.float32))
      if FLAGS.valid_segment_labels:
          tf.add_to_collection("valid_label_weights", valid_input_data_dict["label_weights"])
      tf.add_to_collection("valid_summary_op", tf.summary.merge_all())

  ############# Valid Data Processing ############
  # """

def get_checkpoint():
  if FLAGS.checkpoint_file != "":
    return os.path.join(FLAGS.train_dir, FLAGS.checkpoint_file)
  else:
    return tf.train.latest_checkpoint(FLAGS.train_dir)

class Trainer(object):
  """A Trainer to train a Tensorflow graph."""

  def __init__(self,
               cluster,
               task,
               train_dir,
               model,
               reader,
               valid_reader,
               model_exporter,
               log_device_placement=True,
               max_steps=None,
               export_model_steps=1000):
    """"Creates a Trainer.

    Args:
      cluster: A tf.train.ClusterSpec if the execution is distributed. None
        otherwise.
      task: A TaskSpec describing the job type and the task index.
    """

    self.cluster = cluster
    self.task = task
    self.is_master = (task.type == "master" and task.index == 0)
    self.train_dir = train_dir
    self.config = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=log_device_placement)
    self.config.gpu_options.allow_growth = True
    self.model = model
    self.reader = reader
    self.valid_reader = valid_reader
    self.model_exporter = model_exporter
    self.max_steps = max_steps
    self.max_steps_reached = False
    self.export_model_steps = export_model_steps
    self.last_model_export_step = 0


#     if self.is_master and self.task.index > 0:
#       raise StandardError("%s: Only one replica of master expected",
#                           task_as_string(self.task))

  def run(self, start_new_model=False):
    """Performs training on the currently defined Tensorflow graph.

    Returns:
      A tuple of the training Hit@1 and the training PERR.
    """
    if self.is_master and start_new_model:
      self.remove_training_directory(self.train_dir)

    if not os.path.exists(self.train_dir):
      os.makedirs(self.train_dir)

    model_flags_dict = {
        "model": FLAGS.model,
        "feature_sizes": FLAGS.feature_sizes,
        "feature_names": FLAGS.feature_names,
        "frame_features": FLAGS.frame_features,
        "label_loss": FLAGS.label_loss,
    }
    flags_json_path = os.path.join(FLAGS.train_dir, "model_flags.json")
    if file_io.file_exists(flags_json_path):
      existing_flags = json.load(file_io.FileIO(flags_json_path, mode="r"))
      if existing_flags != model_flags_dict:
        logging.error(
            "Model flags do not match existing file %s. Please "
            "delete the file, change --train_dir, or pass flag "
            "--start_new_model", flags_json_path)
        logging.error("Ran model with flags: %s", str(model_flags_dict))
        logging.error("Previously ran with flags: %s", str(existing_flags))
        exit(1)
    else:
      # Write the file.
      with file_io.FileIO(flags_json_path, mode="w") as fout:
        fout.write(json.dumps(model_flags_dict))

    target, device_fn = self.start_server_if_distributed()

    meta_filename = self.get_meta_filename(start_new_model, self.train_dir)

    with tf.Graph().as_default() as graph:
      """  
      #if meta_filename:
        input_data_dict = (
             get_input_data_tensors(
                 self.reader,
                 FLAGS.train_data_pattern,
                 batch_size=FLAGS.batch_size,
                 num_readers=FLAGS.num_readers,
                 num_epochs=None))

        model_input_raw_new = input_data_dict["video_matrix"]
        feature_dim = len(model_input_raw_new.get_shape()) - 1
        model_input_new = tf.nn.l2_normalize(model_input_raw_new, feature_dim)
        labels_batch_new = input_data_dict["labels"]
        
        #saver = self.recover_model(meta_filename)
        #logging.info([n.name for n in tf.get_default_graph().as_graph_def().node])
        #breakpoint()
      """
      with tf.device(device_fn):
        if not meta_filename:
          #saver = self.build_model(self.model, self.reader, FLAGS.train_data_pattern)
          #if FLAGS.valid_data_pattern:
          saver = self.build_model(self.model, self.reader, FLAGS.train_data_pattern, self.valid_reader, FLAGS.valid_data_pattern)
        else:
          saver = self.build_model(self.model, self.reader, FLAGS.train_data_pattern, self.valid_reader,
                                     FLAGS.valid_data_pattern)
        global_step = tf.get_collection("global_step")[0]
        loss = tf.get_collection("loss")[0]
        predictions = tf.get_collection("predictions")[0]
        labels = tf.get_collection("labels")[0]
        input_batch = tf.get_collection("input_batch")[0]
        input_batch_raw = tf.get_collection("input_batch_raw")[0]
        train_op = tf.get_collection("train_op")[0]
        video_ids = tf.get_collection("video_ids")[0]


        init_op = tf.global_variables_initializer()

        """
        valid_fetches = {
            "video_id": tf.get_collection("valid_video_id_batch")[0],
            "predictions": tf.get_collection("valid_predictions")[0],
            "labels": tf.get_collection("valid_labels")[0],
            "loss": tf.get_collection("valid_loss")[0],
            "summary": tf.get_collection("valid_summary_op")[0]
        }
        if FLAGS.valid_segment_labels:
            valid_fetches["label_weights"] = tf.get_collection("valid_label_weights")[0]
        """
        if FLAGS.valid_data_pattern:
            evl_metrics = eval_util.EvaluationMetrics(self.valid_reader.num_classes, FLAGS.valid_top_k, None)

            valid_loss = tf.get_collection("valid_loss")[0]
            valid_predictions = tf.get_collection("valid_predictions")[0]
            valid_labels = tf.get_collection("valid_labels")[0]
            valid_summary = tf.get_collection("valid_summary_op")[0]
            if FLAGS.valid_segment_labels:
                valid_label_weights = tf.get_collection("valid_label_weights")[0]

    sv = tf.train.Supervisor(
        graph,
        logdir=self.train_dir,
        init_op=init_op,
        is_chief=self.is_master,
        global_step=global_step,
        save_model_secs=60 * 60,
        save_summaries_secs=600,
        saver=saver)
    #breakpoint()
    logging.info("%s: Starting managed session.", task_as_string(self.task))
    with sv.managed_session(target, config=self.config) as sess:
      try:
        logging.info("%s: Entering training loop.", task_as_string(self.task))
        checkpoint = get_checkpoint()
        if checkpoint:
            print("*" * 20)
            print("*" * 20)
            logging.info("Loading checkpoint for eval: %s", checkpoint)
            # Restores from checkpoint
            saver.restore(sess, checkpoint)
        while (not sv.should_stop()) and (not self.max_steps_reached):
          batch_start_time = time.time()

          #model_input_new_val, labels_val = sess.run(
          #    [model_input_new, labels_batch_new]
          #)
          #breakpoint()
          #_, global_step_val, loss_val, predictions_val = sess.run(
          #    [train_op, global_step, loss, predictions],
          #    feed_dict={input_batch: model_input_new_val,
          #               labels: labels_val}
          #)
          _, global_step_val, loss_val, predictions_val, labels_val, input_batch_raw_val, input_batch_val, video_ids_val = sess.run(
              [train_op, global_step, loss, predictions, labels, input_batch_raw, input_batch, video_ids],
          )
          #breakpoint()
          seconds_per_batch = time.time() - batch_start_time
          examples_per_second = labels_val.shape[0] / seconds_per_batch

          if self.max_steps and self.max_steps <= global_step_val:
            self.max_steps_reached = True

          if FLAGS.valid_data_pattern and self.is_master and global_step_val % FLAGS.valid_checkpoint == 0 and self.train_dir:
            # print("labels_val: ", len(labels_val[0]))
            # print("labels_val: ", np.argmax(labels_val, axis=-1))
            #sess.run([tf.local_variables_initializer()])
            coord = tf.train.Coordinator()
            try:
                threads = []
                for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                    threads.extend(
                        qr.create_threads(sess, coord=coord, daemon=True, start=True))
                logging.info("enter eval_once loop global_step_val = %s. ",
                             global_step_val)
                evl_metrics.clear()
                valid_examples_processed = 0

                while not coord.should_stop():
                    eval_start_time = time.time()
                    if FLAGS.valid_segment_labels:
                        valid_loss_val, valid_predictions_val, valid_labels_val, \
                        valid_label_weights_val, valid_summary_val = sess.run(
                            [valid_loss, valid_predictions, valid_labels, valid_label_weights, valid_summary])
                        valid_predictions_val *= valid_label_weights_val
                    else:
                        valid_loss_val, valid_predictions_val, valid_labels_val, \
                        valid_summary_val = sess.run(
                            [valid_loss, valid_predictions, valid_labels, valid_summary])

                    #"""
                    iteration_info_dict = evl_metrics.accumulate(valid_predictions_val, valid_labels_val,
                                                                 valid_loss_val)
                    seconds_per_batch = time.time() - eval_start_time
                    valid_example_per_second = valid_labels_val.shape[0] / seconds_per_batch
                    iteration_info_dict["examples_per_second"] = valid_example_per_second
                    valid_examples_processed += valid_labels_val.shape[0]
                    #logging.info("validation_examples_processed: %d", valid_examples_processed)
                    iterinfo = utils.AddGlobalStepSummary(
                        sv.summary_writer,
                        global_step_val,
                        iteration_info_dict,
                        summary_scope="Eval")
                    logging.info("valid_examples_processed: %d | %s", valid_examples_processed,
                                 iterinfo)
                    #"""
                    if valid_examples_processed >= FLAGS.valid_size:
                        epoch_info_dict = evl_metrics.get()
                        epoch_info_dict["epoch_id"] = global_step_val
        
                        sv.summary_writer.add_summary(valid_summary_val, global_step_val)
                        epochinfo = utils.AddEpochSummary(
                            sv.summary_writer,
                            global_step_val,
                            epoch_info_dict,
                            summary_scope="Eval")
                        logging.info("Total: examples_processed: %d", valid_examples_processed)
                        logging.info(epochinfo)
                        evl_metrics.clear()
                        coord.request_stop()
                    #"""
                    #"""
                    """
                    hit_at_one = eval_util.calculate_hit_at_one(valid_predictions_val,
                                                                valid_labels_val)
                    perr = eval_util.calculate_precision_at_equal_recall_rate(
                        valid_predictions_val, valid_labels_val)
                    gap = eval_util.calculate_gap(valid_predictions_val, valid_labels_val)
                    eval_end_time = time.time()
                    eval_time = eval_end_time - eval_start_time
        
                    logging.info("training step " + str(global_step_val) + " | Valid Loss: " +
                                 ("%.2f" % valid_loss_val) + " Examples/sec: " +
                                 ("%.2f" % examples_per_second) + " | Hit@1: " +
                                 ("%.2f" % hit_at_one) + " PERR: " + ("%.2f" % perr) +
                                 " GAP: " + ("%.2f" % gap))
        
                    sv.summary_writer.add_summary(
                        utils.MakeSummary("model/Training_Hit@1", hit_at_one),
                        global_step_val)
                    sv.summary_writer.add_summary(
                        utils.MakeSummary("model/Training_Perr", perr), global_step_val)
                    sv.summary_writer.add_summary(
                        utils.MakeSummary("model/Training_GAP", gap), global_step_val)
                    sv.summary_writer.add_summary(
                        utils.MakeSummary("global_step/Examples/Second",
                                          examples_per_second), global_step_val)
                    sv.summary_writer.flush()
                    """

                    #logging.info(
                    #    "Done with batched inference. Now calculating global performance "
                    #    "metrics.")
                    # calculate the metrics for the entire epoch

            except tf.errors.OutOfRangeError as e:
                epoch_info_dict = evl_metrics.get()
                epoch_info_dict["epoch_id"] = global_step_val

                sv.summary_writer.add_summary(valid_summary_val, global_step_val)
                epochinfo = utils.AddEpochSummary(
                    sv.summary_writer,
                    global_step_val,
                    epoch_info_dict,
                    summary_scope="Eval")
                logging.info(epochinfo)
                evl_metrics.clear()

            except Exception as e:  # pylint: disable=broad-except
                logging.info("Unexpected exception: %s", str(e))
                coord.request_stop(e)

            # Exporting the model every x steps

          if self.is_master and self.train_dir and global_step_val % FLAGS.export_model_steps == 0:
            #time_to_export = ((self.last_model_export_step == 0) or
            #                  (global_step_val - self.last_model_export_step >=
            #                   self.export_model_steps))
            #if self.is_master and time_to_export:
              self.export_model(global_step_val, sv.saver, sv.save_path, sess)
              #self.last_model_export_step = global_step_val
        #else:
          logging.info("training step " + str(global_step_val) + " | Loss: " +
                       ("%.2f" % loss_val) + " Examples/sec: " +
                       ("%.2f" % examples_per_second))
      except tf.errors.OutOfRangeError:
        print('+'*8, self.task)
        logging.info("%s: Done training -- epoch limit reached.",
                     task_as_string(self.task))

    logging.info("%s: Exited training loop.", task_as_string(self.task))
    sv.Stop()

  def export_model(self, global_step_val, saver, save_path, session):

    # If the model has already been exported at this step, return.
    if global_step_val == self.last_model_export_step:
      return

    last_checkpoint = saver.save(session, save_path, global_step_val)

    model_dir = "{0}/export/step_{1}".format(self.train_dir, global_step_val)
    logging.info("%s: Exporting the model at step %s to %s.",
                 task_as_string(self.task), global_step_val, model_dir)

    self.model_exporter.export_model(
        model_dir=model_dir,
        global_step_val=global_step_val,
        last_checkpoint=last_checkpoint)

  def start_server_if_distributed(self):
    """Starts a server if the execution is distributed."""

    if self.cluster:
      logging.info("%s: Starting trainer within cluster %s.",
                   task_as_string(self.task), self.cluster.as_dict())
      server = start_server(self.cluster, self.task)
      target = server.target
      device_fn = tf.train.replica_device_setter(
          ps_device="/job:ps",
          worker_device="/job:%s/task:%d" % (self.task.type, self.task.index),
          cluster=self.cluster)
    else:
      target = ""
      device_fn = ""
    return (target, device_fn)

  def remove_training_directory(self, train_dir):
    """Removes the training directory."""
    try:
      logging.info("%s: Removing existing train directory.",
                   task_as_string(self.task))
      gfile.DeleteRecursively(train_dir)
    except:
      logging.error(
          "%s: Failed to delete directory " + train_dir +
          " when starting a new model. Please delete it manually and" +
          " try again.", task_as_string(self.task))

  def get_meta_filename(self, start_new_model, train_dir):
    if start_new_model:
      logging.info("%s: Flag 'start_new_model' is set. Building a new model.",
                   task_as_string(self.task))
      return None

    latest_checkpoint = tf.train.latest_checkpoint(train_dir)
    if not latest_checkpoint:
      logging.info("%s: No checkpoint file found. Building a new model.",
                   task_as_string(self.task))
      return None

    meta_filename = latest_checkpoint + ".meta"
    if not gfile.Exists(meta_filename):
      logging.info("%s: No meta graph file found. Building a new model.",
                   task_as_string(self.task))
      return None
    else:
      return meta_filename

  def recover_model(self, meta_filename):
    logging.info("%s: Restoring from meta graph file %s",
                 task_as_string(self.task), meta_filename)
    return tf.train.import_meta_graph(meta_filename)

  def build_model(self, model, reader, data_pattern, eval_reader=None, eval_data_pattern=None):
    """Find the model and build the graph."""

    label_loss_fn = find_class_by_name(FLAGS.label_loss, [losses])()
    optimizer_class = find_class_by_name(FLAGS.optimizer, [tf.train])

    build_graph(
        reader=reader,
        model=model,
        optimizer_class=optimizer_class,
        clip_gradient_norm=FLAGS.clip_gradient_norm,
        train_data_pattern=data_pattern,
        label_loss_fn=label_loss_fn,
        base_learning_rate=FLAGS.base_learning_rate,
        fix_learning_rate=FLAGS.fix_learning_rate,
        learning_rate_decay=FLAGS.learning_rate_decay,
        learning_rate_decay_examples=FLAGS.learning_rate_decay_examples,
        regularization_penalty=FLAGS.regularization_penalty,
        num_readers=FLAGS.num_readers,
        batch_size=FLAGS.batch_size,
        num_epochs=None,
        eval_reader=eval_reader,
        eval_data_pattern=eval_data_pattern,
        eval_batch_size=FLAGS.valid_batch_size,
    )

    return tf.train.Saver(max_to_keep=0, keep_checkpoint_every_n_hours=0.25)


  def build_model_valid(self, model, reader, data_pattern):
      """Find the model and build the graph."""

      label_loss_fn = find_class_by_name(FLAGS.label_loss, [losses])()

      build_graph_valid(
        reader=reader,
        model=model,
        eval_data_pattern=data_pattern,
        label_loss_fn=label_loss_fn,
        batch_size=FLAGS.valid_batch_size,
        num_readers=FLAGS.num_readers)

def get_reader():
  # Convert feature_names and feature_sizes to lists of values.
  feature_names, feature_sizes = utils.GetListOfFeatureNamesAndSizes(
      FLAGS.feature_names, FLAGS.feature_sizes)

  if FLAGS.frame_features:
    reader = readers.YT8MFrameFeatureReader(
        feature_names=feature_names,
        feature_sizes=feature_sizes,
        segment_labels=FLAGS.segment_labels,
        max_frames=FLAGS.max_frames)
  else:
    reader = readers.YT8MAggregatedFeatureReader(
        feature_names=feature_names, feature_sizes=feature_sizes)

  return reader

def get_reader_valid():
    # Convert feature_names and feature_sizes to lists of values.
    feature_names, feature_sizes = utils.GetListOfFeatureNamesAndSizes(
        FLAGS.feature_names, FLAGS.feature_sizes)

    if FLAGS.frame_features:
        reader = readers.YT8MFrameFeatureReader(
            feature_names=feature_names,
            feature_sizes=feature_sizes,
            segment_labels=FLAGS.valid_segment_labels)
    else:
        reader = readers.YT8MAggregatedFeatureReader(
            feature_names=feature_names, feature_sizes=feature_sizes)

    return reader

class ParameterServer(object):
  """A parameter server to serve variables in a distributed execution."""

  def __init__(self, cluster, task):
    """Creates a ParameterServer.

    Args:
      cluster: A tf.train.ClusterSpec if the execution is distributed. None
        otherwise.
      task: A TaskSpec describing the job type and the task index.
    """

    self.cluster = cluster
    self.task = task

  def run(self):
    """Starts the parameter server."""

    logging.info("%s: Starting parameter server within cluster %s.",
                 task_as_string(self.task), self.cluster.as_dict())
    server = start_server(self.cluster, self.task)
    server.join()


def start_server(cluster, task):
  """Creates a Server.

  Args:
    cluster: A tf.train.ClusterSpec if the execution is distributed. None
      otherwise.
    task: A TaskSpec describing the job type and the task index.
  """

  if not task.type:
    raise ValueError("%s: The task type must be specified." %
                     task_as_string(task))
  if task.index is None:
    raise ValueError("%s: The task index must be specified." %
                     task_as_string(task))

  # Create and start a server.
  return tf.train.Server(
      tf.train.ClusterSpec(cluster),
      protocol="grpc",
      job_name=task.type,
      task_index=task.index)


def task_as_string(task):
  return "/job:%s/task:%s" % (task.type, task.index)


def main(unused_argv):
  # Load the environment.
  env = json.loads(os.environ.get("TF_CONFIG", "{}"))

  # Load the cluster data from the environment.
  cluster_data = env.get("cluster", None)
  cluster = tf.train.ClusterSpec(cluster_data) if cluster_data else None

  # Load the task data from the environment.
  task_data = env.get("task", None) or {"type": "master", "index": 0}
  task = type("TaskSpec", (object,), task_data)

  # Logging the version.
  logging.set_verbosity(tf.logging.INFO)
  logging.info("%s: Tensorflow version: %s.", task_as_string(task),
               tf.__version__)

  # Dispatch to a master, a worker, or a parameter server.
  if not cluster or task.type == "master" or task.type == "worker":
    model = find_class_by_name(FLAGS.model,
                               [frame_level_models, video_level_models, nextvlad])()
    ##breakpoint()
    reader = get_reader()
    if FLAGS.valid_data_pattern:
        valid_reader = get_reader_valid()
    else:
        valid_reader = None
    model_exporter = export_model.ModelExporter(
      frame_features=FLAGS.frame_features, model=model, reader=reader)
    Trainer(cluster, task, FLAGS.train_dir, model, reader, valid_reader, model_exporter,
            FLAGS.log_device_placement, FLAGS.max_steps,
            FLAGS.export_model_steps).run(start_new_model=FLAGS.start_new_model)

  elif task.type == "ps":
    ParameterServer(cluster, task).run()
  else:
    raise ValueError("%s: Invalid task_type: %s." %
                     (task_as_string(task), task.type))


if __name__ == "__main__":
  app.run()
