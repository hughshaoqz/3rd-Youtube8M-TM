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
"""Provides readers configured for different datasets."""

import tensorflow as tf
import utils
import numpy as np


def resize_axis(tensor, axis, new_size, fill_value=0):
  """Truncates or pads a tensor to new_size on on a given axis.

  Truncate or extend tensor such that tensor.shape[axis] == new_size. If the
  size increases, the padding will be performed at the end, using fill_value.

  Args:
    tensor: The tensor to be resized.
    axis: An integer representing the dimension to be sliced.
    new_size: An integer or 0d tensor representing the new value for
      tensor.shape[axis].
    fill_value: Value to use to fill any new entries in the tensor. Will be cast
      to the type of tensor.

  Returns:
    The resized tensor.
  """
  tensor = tf.convert_to_tensor(tensor)
  shape = tf.unstack(tf.shape(tensor))
  pad_shape = shape[:]
  pad_shape[axis] = tf.maximum(0, new_size - shape[axis])

  shape[axis] = tf.minimum(shape[axis], new_size)
  shape = tf.stack(shape)
  resized = tf.concat([
      tf.slice(tensor, tf.zeros_like(shape), shape),
      tf.fill(tf.stack(pad_shape), tf.cast(fill_value, tensor.dtype))
  ], axis)
  # Update shape.
  new_shape = tensor.get_shape().as_list()  # A copy is being made.
  new_shape[axis] = new_size
  resized.set_shape(new_shape)
  return resized


class BaseReader(object):
  """Inherit from this class when implementing new readers."""

  def prepare_reader(self, unused_filename_queue):
    """Create a thread for generating prediction and label tensors."""
    raise NotImplementedError()


class YT8MAggregatedFeatureReader(BaseReader):
  """Reads TFRecords of pre-aggregated Examples.

  The TFRecords must contain Examples with a sparse int64 'labels' feature and
  a fixed length float32 feature, obtained from the features in 'feature_name'.
  The float features are assumed to be an average of dequantized values.
  """

  def __init__(  # pylint: disable=dangerous-default-value
      self,
      num_classes=3862,
      feature_sizes=[1024, 128],
      feature_names=["mean_rgb", "mean_audio"]):
    """Construct a YT8MAggregatedFeatureReader.

    Args:
      num_classes: a positive integer for the number of classes.
      feature_sizes: positive integer(s) for the feature dimensions as a list.
      feature_names: the feature name(s) in the tensorflow record as a list.
    """

    assert len(feature_names) == len(feature_sizes), (
        "length of feature_names (={}) != length of feature_sizes (={})".format(
            len(feature_names), len(feature_sizes)))

    self.num_classes = num_classes
    self.feature_sizes = feature_sizes
    self.feature_names = feature_names

  def prepare_reader(self, filename_queue, batch_size=1024):
    """Creates a single reader thread for pre-aggregated YouTube 8M Examples.

    Args:
      filename_queue: A tensorflow queue of filename locations.
      batch_size: batch size used for feature output.

    Returns:
      A tuple of video indexes, features, labels, and padding data.
    """
    reader = tf.TFRecordReader()
    _, serialized_examples = reader.read_up_to(filename_queue, batch_size)

    tf.add_to_collection("serialized_examples", serialized_examples)
    return self.prepare_serialized_examples(serialized_examples)

  def prepare_serialized_examples(self, serialized_examples):
    """Parse a single video-level TF Example."""
    # set the mapping from the fields to data types in the proto
    num_features = len(self.feature_names)
    assert num_features > 0, "self.feature_names is empty!"
    assert len(self.feature_names) == len(self.feature_sizes), \
    "length of feature_names (={}) != length of feature_sizes (={})".format(
        len(self.feature_names), len(self.feature_sizes))

    feature_map = {
        "id": tf.FixedLenFeature([], tf.string),
        "labels": tf.VarLenFeature(tf.int64)
    }
    for feature_index in range(num_features):
      feature_map[self.feature_names[feature_index]] = tf.FixedLenFeature(
          [self.feature_sizes[feature_index]], tf.float32)

    features = tf.parse_example(serialized_examples, features=feature_map)
    labels = tf.sparse_to_indicator(features["labels"], self.num_classes)
    labels.set_shape([None, self.num_classes])
    concatenated_features = tf.concat(
        [features[feature_name] for feature_name in self.feature_names], 1)

    output_dict = {
        "video_ids": features["id"],
        "video_matrix": concatenated_features,
        "labels": labels,
        "num_frames": tf.ones([tf.shape(serialized_examples)[0]])
    }

    return output_dict


class YT8MFrameFeatureReader(BaseReader):
  """Reads TFRecords of SequenceExamples.

  The TFRecords must contain SequenceExamples with the sparse in64 'labels'
  context feature and a fixed length byte-quantized feature vector, obtained
  from the features in 'feature_names'. The quantized features will be mapped
  back into a range between min_quantized_value and max_quantized_value.
  """

  def __init__(  # pylint: disable=dangerous-default-value
      self,
      num_classes=3862,
      feature_sizes=[1024, 128],
      feature_names=["rgb", "audio"],
      max_frames=300,
      segment_labels=False,
      segment_size=5):
    """Construct a YT8MFrameFeatureReader.

    Args:
      num_classes: a positive integer for the number of classes.
      feature_sizes: positive integer(s) for the feature dimensions as a list.
      feature_names: the feature name(s) in the tensorflow record as a list.
      max_frames: the maximum number of frames to process.
      segment_labels: if we read segment labels instead.
      segment_size: the segment_size used for reading segments.
    """

    assert len(feature_names) == len(feature_sizes), (
        "length of feature_names (={}) != length of feature_sizes (={})".format(
            len(feature_names), len(feature_sizes)))

    self.num_classes = num_classes
    self.feature_sizes = feature_sizes
    self.feature_names = feature_names
    self.max_frames = max_frames
    self.segment_labels = segment_labels
    self.segment_size = segment_size

  def get_video_matrix(self, features, feature_size, max_frames,
                       max_quantized_value, min_quantized_value):
    """Decodes features from an input string and quantizes it.

    Args:
      features: raw feature values
      feature_size: length of each frame feature vector
      max_frames: number of frames (rows) in the output feature_matrix
      max_quantized_value: the maximum of the quantized value.
      min_quantized_value: the minimum of the quantized value.

    Returns:
      feature_matrix: matrix of all frame-features
      num_frames: number of frames in the sequence
    """
    decoded_features = tf.reshape(
        tf.cast(tf.decode_raw(features, tf.uint8), tf.float32),
        [-1, feature_size])

    num_frames = tf.minimum(tf.shape(decoded_features)[0], max_frames)
    feature_matrix = utils.Dequantize(decoded_features, max_quantized_value,
                                      min_quantized_value)
    feature_matrix = resize_axis(feature_matrix, 0, max_frames)
    return feature_matrix, num_frames

  def prepare_reader(self,
                     filename_queue,
                     max_quantized_value=2,
                     min_quantized_value=-2):
    """Creates a single reader thread for YouTube8M SequenceExamples.

    Args:
      filename_queue: A tensorflow queue of filename locations.
      max_quantized_value: the maximum of the quantized value.
      min_quantized_value: the minimum of the quantized value.

    Returns:
      A tuple of video indexes, video features, labels, and padding data.
    """
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    return self.prepare_serialized_examples(serialized_example,
                                            max_quantized_value,
                                            min_quantized_value)

  def prepare_serialized_examples(self,
                                  serialized_example,
                                  max_quantized_value=2,
                                  min_quantized_value=-2):
    """Parse single serialized SequenceExample from the TFRecords."""

    # Read/parse frame/segment-level labels.
    context_features = {
        "id": tf.FixedLenFeature([], tf.string),
    }
    if self.segment_labels:
      context_features.update({
          # There is no need to read end-time given we always assume the segment
          # has the same size.
          "segment_labels": tf.VarLenFeature(tf.int64),
          "segment_start_times": tf.VarLenFeature(tf.int64),
          "segment_scores": tf.VarLenFeature(tf.float32)
      })
    else:
      context_features.update({"labels": tf.VarLenFeature(tf.int64)})
    sequence_features = {
        feature_name: tf.FixedLenSequenceFeature([], dtype=tf.string)
        for feature_name in self.feature_names
    }
    contexts, features = tf.parse_single_sequence_example(
        serialized_example,
        context_features=context_features,
        sequence_features=sequence_features)

    # loads (potentially) different types of features and concatenates them
    num_features = len(self.feature_names)
    assert num_features > 0, "No feature selected: feature_names is empty!"

    assert len(self.feature_names) == len(self.feature_sizes), (
        "length of feature_names (={}) != length of feature_sizes (={})".format(
            len(self.feature_names), len(self.feature_sizes)))

    num_frames = -1  # the number of frames in the video
    feature_matrices = [None] * num_features  # an array of different features
    for feature_index in range(num_features):
      feature_matrix, num_frames_in_this_feature = self.get_video_matrix(
          features[self.feature_names[feature_index]],
          self.feature_sizes[feature_index], self.max_frames,
          max_quantized_value, min_quantized_value)
      if num_frames == -1:
        num_frames = num_frames_in_this_feature

      feature_matrices[feature_index] = feature_matrix

    # cap the number of frames at self.max_frames
    num_frames = tf.minimum(num_frames, self.max_frames)
    # concatenate different features
    video_matrix = tf.concat(feature_matrices, 1)

    # Partition frame-level feature matrix to segment-level feature matrix.
    if self.segment_labels:
      start_times = contexts["segment_start_times"].values
      # Here we assume all the segments that started at the same start time has
      # the same segment_size.
      uniq_start_times, seg_idxs = tf.unique(
          start_times, out_idx=tf.dtypes.int64)
      segment_size = self.segment_size
      # Range gather matrix, e.g., [[0,1,2],[1,2,3]] for segment_size == 3.
      range_mtx = tf.expand_dims(
          uniq_start_times, axis=-1) + tf.expand_dims(
              tf.range(0, segment_size, dtype=tf.int64), axis=0)
      # Shape: [num_segment, segment_size, feature_dim].
      batch_video_matrix = tf.gather_nd(video_matrix,
                                        tf.expand_dims(range_mtx, axis=-1))
      num_segment = tf.shape(batch_video_matrix)[0]

      batch_video_ids = tf.reshape(
          tf.tile([contexts["id"]], [num_segment]), (num_segment,))
      batch_frames = tf.reshape(
          tf.tile([segment_size], [num_segment]), (num_segment,))

      # For segment labels, all labels are not exhausively rated. So we only
      # evaluate the rated labels.
      # Label indices for each segment, shape: [num_segment, 2].
      label_indices = tf.stack([seg_idxs, contexts["segment_labels"].values],
                               axis=-1)
      label_values = contexts["segment_scores"].values
      sparse_labels = tf.sparse.SparseTensor(label_indices, label_values,
                                             (num_segment, self.num_classes))
      batch_labels = tf.sparse.to_dense(sparse_labels, validate_indices=False)

      #tf.ones_like(range_mtx, dtype=tf.float32)
      sparse_label_weights = tf.sparse.SparseTensor(
          label_indices, tf.ones_like(label_values, dtype=tf.float32),
          (num_segment, self.num_classes))
      batch_label_weights = tf.sparse.to_dense(
          sparse_label_weights, validate_indices=False)
    else:
      # Process video-level labels.
      label_indices = contexts["labels"].values
      sparse_labels = tf.sparse.SparseTensor(
          tf.expand_dims(label_indices, axis=-1),
          tf.ones_like(contexts["labels"].values, dtype=tf.bool),
          (self.num_classes,))
      labels = tf.sparse.to_dense(
          sparse_labels, default_value=False, validate_indices=False)
      # convert to batch format.
      batch_video_ids = tf.expand_dims(contexts["id"], 0)
      batch_video_matrix = tf.expand_dims(video_matrix, 0)
      batch_labels = tf.expand_dims(labels, 0)
      batch_frames = tf.expand_dims(num_frames, 0)
      batch_label_weights = None
    output_dict = {
        "video_ids": batch_video_ids,
        "video_matrix": batch_video_matrix,
        "labels": batch_labels,
        "num_frames": batch_frames,
        "original_setment": batch_video_matrix,
    }
    if batch_label_weights is not None:
      output_dict["label_weights"] = batch_label_weights
    return output_dict
