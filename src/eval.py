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
"""Binary for evaluating Tensorflow models on the YouTube-8M dataset."""

import json
import os
import time

from absl import logging
import eval_util
import frame_level_models
import nextvlad
import losses
import readers
import tensorflow as tf
from tensorflow import app
from tensorflow import flags
from tensorflow import gfile
from tensorflow.python.lib.io import file_io
import utils
import video_level_models
import string
import random

FLAGS = flags.FLAGS

if __name__ == "__main__":
  # Dataset flags.
  flags.DEFINE_string(
      "train_dir", "/tmp/yt8m_model/",
      "The directory to load the model files from. "
      "The tensorboard metrics files are also saved to this "
      "directory.")
  flags.DEFINE_string(
      "eval_data_pattern", "",
      "File glob defining the evaluation dataset in tensorflow.SequenceExample "
      "format. The SequenceExamples are expected to have an 'rgb' byte array "
      "sequence feature as well as a 'labels' int64 context feature.")
  flags.DEFINE_bool(
      "segment_labels", False,
      "If set, then --eval_data_pattern must be frame-level features (but with"
      " segment_labels). Otherwise, --eval_data_pattern must be aggregated "
      "video-level features. The model must also be set appropriately (i.e. to "
      "read 3D batches VS 4D batches.")
  #flags.DEFINE_string("checkpoint_file", "",
  #                      "e.g. model.ckpt-170820")
  flags.DEFINE_integer("checkpoint_file", 0,
                        "170820")
  flags.DEFINE_bool(
      "google_cloud", False,
      "If set, it means we use eval data for google cloud running.")
  # Other flags
  flags.DEFINE_integer("batch_size", 1024,
                       "How many examples to process per batch.")
  flags.DEFINE_integer("num_readers", 8,
                       "How many threads to use for reading input files.")
  flags.DEFINE_boolean("run_once", False, "Whether to run eval only once.")
  flags.DEFINE_integer("top_k", 20, "How many predictions to output per video.")

  flags.DEFINE_integer("netvlad_cluster_size", 0,
                       "Number of units in the NetVLAD cluster layer.")

  flags.DEFINE_integer("netvlad_hidden_size", 0,
                       "Number of units in the NetVLAD hidden layer.")     
  flags.DEFINE_integer(
      "max_frames", 300,
      "max frames to cap.")
def find_class_by_name(name, modules):
  """Searches the provided modules for the named class and returns it."""
  modules = [getattr(module, name, None) for module in modules]
  return next(a for a in modules if a)


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
    # randomly chosen 60 validate files
    # note that validate file names are different on gcloud and locally, due to `curl` download command

    if FLAGS.google_cloud:
        a_list = list(string.ascii_lowercase)
        A_list = list(string.ascii_uppercase)

        n_list = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        n_list = n_list + a_list + A_list

        results = []
        for a in n_list:
            for n in n_list:
                results.append(n + a)
        # random.seed(7)
        # random.shuffle(results)
        validate_file_nums = results[:300]
        #validate_file_nums = results[300:]

    else:
        # local
        results = []
        for i in range(3844):
            results.append(str(i).zfill(4))
        # random.seed(7)
        # random.shuffle(results)
        validate_file_nums = results[:300]
        #validate_file_nums = results[300:]

    validate_file_list = [FLAGS.eval_data_pattern.split('*')[0]\
                           + x +'.tfrecord' for x in validate_file_nums]
    files = validate_file_list

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


def build_graph(reader,
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
  #range_mtx = input_data_dict["range_mtx"]
  tf.summary.histogram("model_input_raw", model_input_raw)

  feature_dim = len(model_input_raw.get_shape()) - 1

  # Normalize input features.
  model_input = tf.nn.l2_normalize(model_input_raw, feature_dim)

  with tf.variable_scope("tower"):
    result = model.create_model(
        model_input,
        num_frames=num_frames,
        vocab_size=reader.num_classes,
        labels=labels_batch,
        is_training=False,
        #range_mtx=range_mtx
        )
    predictions = result["predictions"]
    tf.summary.histogram("model_activations", predictions)
    if "loss" in result.keys():
      label_loss = result["loss"]
    else:
      label_loss = label_loss_fn.calculate_loss(predictions, labels_batch)

  tf.add_to_collection("global_step", global_step)
  tf.add_to_collection("loss", label_loss)
  tf.add_to_collection("predictions", predictions)
  tf.add_to_collection("input_batch", model_input)
  tf.add_to_collection("input_batch_raw", model_input_raw)
  tf.add_to_collection("video_id_batch", video_id_batch)
  tf.add_to_collection("num_frames", num_frames)
  tf.add_to_collection("labels", tf.cast(labels_batch, tf.float32))
  if FLAGS.segment_labels:
    tf.add_to_collection("label_weights", input_data_dict["label_weights"])
  tf.add_to_collection("summary_op", tf.summary.merge_all())

def get_checkpoint(checkpoint_file):
  if FLAGS.checkpoint_file != "":
    return os.path.join(FLAGS.train_dir, checkpoint_file)
  else:
    return tf.train.latest_checkpoint(FLAGS.train_dir)

def evaluation_loop(fetches, saver, summary_writer, evl_metrics, checkpoint,
                    last_global_step_val):
  """Run the evaluation loop once.
  Args:
    fetches: a dict of tensors to be run within Session.
    saver: a tensorflow saver to restore the model.
    summary_writer: a tensorflow summary_writer
    evl_metrics: an EvaluationMetrics object.
    last_global_step_val: the global step used in the previous evaluation.
  Returns:
    The global_step used in the latest model.
  """

  global_step_val = -1
  with tf.Session(
      config=tf.ConfigProto(gpu_options=tf.GPUOptions(
          allow_growth=True))) as sess:
    #checkpoint = get_checkpoint()
    if checkpoint:

      print("*"*20)
      print("*"*20)
      logging.info("Loading checkpoint for eval: %s", checkpoint)
      # Restores from checkpoint
      saver.restore(sess, checkpoint)
      # Assuming model_checkpoint_path looks something like:
      # /my-favorite-path/yt8m_train/model.ckpt-0, extract global_step from it.
      global_step_val = os.path.basename(checkpoint).split("-")[-1]

      # Save model
      saver.save(
          sess,
          os.path.join(FLAGS.train_dir, "inference_model", "inference_model"))
    else:
      logging.info("No checkpoint file found.")
      return global_step_val

    if global_step_val == last_global_step_val:
      logging.info(
          "skip this checkpoint global_step_val=%s "
          "(same as the previous one).", global_step_val)
      return global_step_val

    sess.run([tf.local_variables_initializer()])

    # Start the queue runners.
    coord = tf.train.Coordinator()
    try:
      threads = []
      for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
        threads.extend(
            qr.create_threads(sess, coord=coord, daemon=True, start=True))
      logging.info("enter eval_once loop global_step_val = %s. ",
                   global_step_val)

      evl_metrics.clear()

      examples_processed = 0
      while not coord.should_stop():
        batch_start_time = time.time()
        output_data_dict = sess.run(fetches)
        seconds_per_batch = time.time() - batch_start_time
        labels_val = output_data_dict["labels"]
        summary_val = output_data_dict["summary"]
        example_per_second = labels_val.shape[0] / seconds_per_batch
        examples_processed += labels_val.shape[0]

        predictions = output_data_dict["predictions"]
        #breakpoint()
        if FLAGS.segment_labels:
          # This is a workaround to ignore the unrated labels.
          predictions *= output_data_dict["label_weights"]
        iteration_info_dict = evl_metrics.accumulate(predictions, labels_val,
                                                     output_data_dict["loss"])
        iteration_info_dict["examples_per_second"] = example_per_second
        #breakpoint()
        iterinfo = utils.AddGlobalStepSummary(
            summary_writer,
            global_step_val,
            iteration_info_dict,
            summary_scope="Eval")
        logging.info("examples_processed: %d | %s", examples_processed,
                     iterinfo)

    except tf.errors.OutOfRangeError as e:
      logging.info(
          "Done with batched inference. Now calculating global performance "
          "metrics.")
      # calculate the metrics for the entire epoch
      epoch_info_dict = evl_metrics.get()
      epoch_info_dict["epoch_id"] = global_step_val

      summary_writer.add_summary(summary_val, global_step_val)
      epochinfo = utils.AddEpochSummary(
          summary_writer,
          global_step_val,
          epoch_info_dict,
          summary_scope="Eval")
      logging.info(epochinfo)
      evl_metrics.clear()
    except Exception as e:  # pylint: disable=broad-except
      logging.info("Unexpected exception: %s", str(e))
      coord.request_stop(e)

    coord.request_stop()
    coord.join(threads, stop_grace_period_secs=10)
    logging.info("Total: examples_processed: %d", examples_processed)

    return global_step_val


def evaluate():
  """Starts main evaluation loop."""
  tf.set_random_seed(0)  # for reproducibility

  # Write json of flags
  model_flags_path = os.path.join(FLAGS.train_dir, "model_flags.json")
  if not file_io.file_exists(model_flags_path):
    raise IOError(("Cannot find file %s. Did you run train.py on the same "
                   "--train_dir?") % model_flags_path)
  flags_dict = json.loads(file_io.FileIO(model_flags_path, mode="r").read())

  with tf.Graph().as_default():
    # convert feature_names and feature_sizes to lists of values
    feature_names, feature_sizes = utils.GetListOfFeatureNamesAndSizes(
        flags_dict["feature_names"], flags_dict["feature_sizes"])

    if flags_dict["frame_features"]:
      reader = readers.YT8MFrameFeatureReader(
          feature_names=feature_names,
          feature_sizes=feature_sizes,
          segment_labels=FLAGS.segment_labels)
    else:
      reader = readers.YT8MAggregatedFeatureReader(
          feature_names=feature_names, feature_sizes=feature_sizes)

    model = find_class_by_name(flags_dict["model"],
                               [frame_level_models, video_level_models, nextvlad])()
    label_loss_fn = find_class_by_name(flags_dict["label_loss"], [losses])()

    if not FLAGS.eval_data_pattern:
      raise IOError("'eval_data_pattern' was not specified. Nothing to "
                    "evaluate.")

    build_graph(
        reader=reader,
        model=model,
        eval_data_pattern=FLAGS.eval_data_pattern,
        label_loss_fn=label_loss_fn,
        num_readers=FLAGS.num_readers,
        batch_size=FLAGS.batch_size)
    logging.info("built evaluation graph")

    # A dict of tensors to be run in Session.
    fetches = {
        "video_id": tf.get_collection("video_id_batch")[0],
        "predictions": tf.get_collection("predictions")[0],
        "labels": tf.get_collection("labels")[0],
        "loss": tf.get_collection("loss")[0],
        "summary": tf.get_collection("summary_op")[0]
    }
    if FLAGS.segment_labels:
      fetches["label_weights"] = tf.get_collection("label_weights")[0]

    saver = tf.train.Saver(tf.global_variables())
    summary_writer = tf.summary.FileWriter(
        os.path.join(FLAGS.train_dir, "eval"), graph=tf.get_default_graph())

    evl_metrics = eval_util.EvaluationMetrics(reader.num_classes, FLAGS.top_k, None)

    last_global_step_val = -1
    cur_checkpoint_file = FLAGS.checkpoint_file
    while True:
      checkpoint_file = 'model.ckpt-{}'.format(cur_checkpoint_file)
      checkpoint = get_checkpoint(checkpoint_file)
      try:
          last_global_step_val = evaluation_loop(fetches, saver, summary_writer,
                                                 evl_metrics, checkpoint, last_global_step_val)
          cur_checkpoint_file += 200
      except:
          logging.info('Find invalid checkpoint, try again using checkpoint: {}'.format(cur_checkpoint_file))
          break
      if FLAGS.run_once:
        break


def main(unused_argv):
  logging.set_verbosity(tf.logging.INFO)
  logging.info("tensorflow version: %s", tf.__version__)
  evaluate()


if __name__ == "__main__":
  app.run()