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
"""Contains a collection of models which operate on variable-length sequences."""
import math

import models
import video_level_models
import tensorflow as tf
import model_utils as utils
import numpy as np
import tensorflow.contrib.slim as slim
from tensorflow import flags
import modeling
import copy
#from .model.transformer import Transformer
from tensorflow import logging
import pandas as pd

FLAGS = flags.FLAGS

flags.DEFINE_bool("gating_remove_diag", False,
                  "Remove diag for self gating")
flags.DEFINE_bool("lightvlad", False,
                  "Light or full NetVLAD")
flags.DEFINE_bool("vlagd", False,
                  "vlagd of vlad")

flags.DEFINE_integer("iterations", 5,
                     "Number of frames per batch for DBoF.")
flags.DEFINE_bool("dbof_add_batch_norm", True,
                  "Adds batch normalization to the DBoF model.")
flags.DEFINE_bool(
    "sample_random_frames", True,
    "If true samples random frames (for frame level models). If false, a random"
    "sequence of frames is sampled instead.")

flags.DEFINE_integer("dbof_cluster_size", 16384,
                     "Number of units in the DBoF cluster layer.")
flags.DEFINE_integer("dbof_hidden_size", 2048,
                    "Number of units in the DBoF hidden layer.")
flags.DEFINE_bool("dbof_relu", True, 'add ReLU to hidden layer')
flags.DEFINE_integer("dbof_var_features", 0,
                     "Variance features on top of Dbof cluster layer.")

flags.DEFINE_string("dbof_activation", "relu", 'dbof activation')

flags.DEFINE_bool("softdbof_maxpool", False, 'add max pool to soft dbof')


flags.DEFINE_bool("netvlad_relu", True, 'add ReLU to hidden layer')
flags.DEFINE_integer("netvlad_dimred", -1,
                   "NetVLAD output dimension reduction")
flags.DEFINE_integer("gatednetvlad_dimred", 1024,
                   "GatedNetVLAD output dimension reduction")

flags.DEFINE_bool("gating", False,
                   "Gating for NetVLAD")
flags.DEFINE_integer("hidden_size", 1024,
                     "size of hidden layer for BasicStatModel.")


flags.DEFINE_integer("netvlad_hidden_size_video", 1024,
                     "Number of units in the NetVLAD video hidden layer.")

flags.DEFINE_integer("netvlad_hidden_size_audio", 64,
                     "Number of units in the NetVLAD audio hidden layer.")



flags.DEFINE_bool("netvlad_add_batch_norm", True,
                  "Adds batch normalization to the NetVLAD model.")

flags.DEFINE_integer("fv_cluster_size", 64,
                     "Number of units in the NetVLAD cluster layer.")

flags.DEFINE_integer("fv_hidden_size", 2048,
                     "Number of units in the NetVLAD hidden layer.")
flags.DEFINE_bool("fv_relu", True,
                     "ReLU after the NetFV hidden layer.")


flags.DEFINE_bool("fv_couple_weights", True,
                     "Coupling cluster weights or not")
 
flags.DEFINE_float("fv_coupling_factor", 0.01,
                     "Coupling factor")


flags.DEFINE_string("dbof_pooling_method", "max",
                    "The pooling method used in the DBoF cluster layer. "
                    "Choices are 'average' and 'max'.")
flags.DEFINE_string("video_level_classifier_model", "MoeModel",
                    "Some Frame-Level models can be decomposed into a "
                    "generalized pooling operation followed by a "
                    "classifier layer")

flags.DEFINE_integer("lstm_cells", 1024, "Number of LSTM cells.")
flags.DEFINE_integer("lstm_layers", 2, "Number of LSTM layers.")
flags.DEFINE_integer("lstm_cells_video", 1024, "Number of LSTM cells (video).")
flags.DEFINE_integer("lstm_cells_audio", 128, "Number of LSTM cells (audio).")



flags.DEFINE_integer("gru_cells", 1024, "Number of GRU cells.")
flags.DEFINE_integer("gru_cells_video", 1024, "Number of GRU cells (video).")
flags.DEFINE_integer("gru_cells_audio", 128, "Number of GRU cells (audio).")
flags.DEFINE_integer("gru_layers", 2, "Number of GRU layers.")
flags.DEFINE_bool("lstm_random_sequence", False,
                     "Random sequence input for lstm.")
flags.DEFINE_bool("gru_random_sequence", False,
                     "Random sequence input for gru.")
flags.DEFINE_bool("gru_backward", False, "BW reading for GRU")
flags.DEFINE_bool("lstm_backward", False, "BW reading for LSTM")


flags.DEFINE_bool("fc_dimred", True, "Adding FC dimred after pooling")
flags.DEFINE_string('bert_config_file', '', 'configuration file for bert')

flags.DEFINE_integer("bert_hidden_layer", 2,
                     "Number of hidden layer in bert.")
flags.DEFINE_integer("bert_attention_heads", 12,
                     "Number of attention heads in bert.")
flags.DEFINE_float("bert_dropout_prob", 0.1,
                     "Dropout prob in bert.")

flags.DEFINE_bool("add_noise_data_aug", False,
                  "If true, add noise to data.")
flags.DEFINE_bool("shuffle_one_dim_aug", False,
                  "If true, shuffle a random dimension of rgb within batch.")
flags.DEFINE_integer("pad_seq_length", 300, "Number of frames to pad to.")
flags.DEFINE_bool("no_sample", False,
                  "If true, add noise to data.")
flags.DEFINE_string('pooling_strategy', 'mean', 'pooling_strategy for sentence vector')
flags.DEFINE_bool('use_position', False, 'whether use position encoding')
flags.DEFINE_bool('bert_position', False, 'whether use bert position encoding')
flags.DEFINE_bool('pool_layer', False, 'whether use bert fc pooling')

class FrameLevelLogisticModel(models.BaseModel):

  def create_model(self, model_input, vocab_size, num_frames, **unused_params):
    """Creates a model which uses a logistic classifier over the average of the

    frame-level features.

    This class is intended to be an example for implementors of frame level
    models. If you want to train a model over averaged features it is more
    efficient to average them beforehand rather than on the fly.

    Args:
      model_input: A 'batch_size' x 'max_frames' x 'num_features' matrix of
        input features.
      vocab_size: The number of classes in the dataset.
      num_frames: A vector of length 'batch' which indicates the number of
        frames for each video (before padding).

    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      'batch_size' x 'num_classes'.
    """
    iterations = FLAGS.iterations
    num_frames = tf.cast(tf.expand_dims(num_frames, 1), tf.float32)
    if FLAGS.sample_random_frames:
      model_input = utils.SampleRandomFrames(model_input, num_frames,
                                             iterations)
    else:
      model_input = utils.SampleRandomSequence(model_input, num_frames,
                                               iterations)

    if FLAGS.add_noise_data_aug:
      model_input = model_input + 0.1*tf.random.normal(shape=tf.shape(model_input))

    feature_size = model_input.get_shape().as_list()[2]
    if FLAGS.shuffle_one_dim_aug:
      k = np.random.random_integers(feature_size)
      model_input_list = []
      for i in range(k):
        model_input_list.append(model_input[..., i])
      seq_len = model_input[..., k].get_shape().as_list()[1]
      model_input_list.append(tf.reshape(
        tf.random.shuffle(
          tf.reshape(model_input[..., k], (-1, 1))), (-1, seq_len)))
      for i in range(k+1, feature_size):
        model_input_list.append(model_input[..., i])
      model_input = tf.stack(model_input_list, axis=-1)

    denominators = tf.reshape(
        tf.tile(num_frames, [1, feature_size]), [-1, feature_size])
    avg_pooled = tf.reduce_sum(model_input, axis=[1]) / denominators

    output = slim.fully_connected(
        avg_pooled,
        vocab_size,
        activation_fn=tf.nn.sigmoid,
        weights_regularizer=slim.l2_regularizer(1e-8))
    return {"predictions": output}


class FrameLevelLogisticModelMIL(models.BaseModel):

  def create_model(self, model_input, vocab_size, num_frames, **unused_params):
    """Creates a model which uses a logistic classifier over the average of the

    frame-level features.

    This class is intended to be an example for implementors of frame level
    models. If you want to train a model over averaged features it is more
    efficient to average them beforehand rather than on the fly.

    Args:
      model_input: A 'batch_size' x 'max_frames' x 'num_features' matrix of
        input features.
      vocab_size: The number of classes in the dataset.
      num_frames: A vector of length 'batch' which indicates the number of
        frames for each video (before padding).

    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      'batch_size' x 'num_classes'.
    """
    iterations = FLAGS.iterations
    num_frames = tf.cast(tf.expand_dims(num_frames, 1), tf.float32)
    model_input = utils.ReshapeFramesToMIL(model_input) # B x FLAGS.pad_seq_length/5 x 5 x 1132

    feature_size = model_input.get_shape().as_list()[3]

    avg_pooled = tf.reduce_sum(model_input, axis=[2])
    print('avg_pooled shape: ', avg_pooled.get_shape())
    output = slim.fully_connected(
        avg_pooled,
        vocab_size,
        activation_fn=tf.nn.sigmoid,
        weights_regularizer=slim.l2_regularizer(1e-8))
    print('output shape: ', output.get_shape())
    output = tf.reduce_max(output, axis=[1])
    return {"predictions": output}


class FrameLevelLogisticModelMax(models.BaseModel):

  def create_model(self, model_input, vocab_size, num_frames, **unused_params):
    """Creates a model which uses a logistic classifier over the average of the

    frame-level features.

    This class is intended to be an example for implementors of frame level
    models. If you want to train a model over averaged features it is more
    efficient to average them beforehand rather than on the fly.

    Args:
      model_input: A 'batch_size' x 'max_frames' x 'num_features' matrix of
        input features.
      vocab_size: The number of classes in the dataset.
      num_frames: A vector of length 'batch' which indicates the number of
        frames for each video (before padding).

    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      'batch_size' x 'num_classes'.
    """

    iterations = FLAGS.iterations
    num_frames = tf.cast(tf.expand_dims(num_frames, 1), tf.float32)
    if FLAGS.sample_random_frames:
      model_input = utils.SampleRandomFrames(model_input, num_frames,
                                             iterations)
    else:
      model_input = utils.SampleRandomSequence(model_input, num_frames,
                                               iterations)

    feature_size = model_input.get_shape().as_list()[2]

    denominators = tf.reshape(
        tf.tile(num_frames, [1, feature_size]), [-1, feature_size])
    max_pooled = tf.reduce_max(model_input, axis=[1]) / denominators

    output = slim.fully_connected(
        max_pooled,
        vocab_size,
        activation_fn=tf.nn.sigmoid,
        weights_regularizer=slim.l2_regularizer(1e-8))
    return {"predictions": output}


# http://iiis.tsinghua.edu.cn/~weblt/papers/multimodal-keyless-attention.pdf
class FrameLevelLogisticModelKeylessAttention(models.BaseModel):

  def create_model(self, model_input, vocab_size, num_frames, **unused_params):
    """Creates a model which uses a logistic classifier over the average of the

    frame-level features.

    This class is intended to be an example for implementors of frame level
    models. If you want to train a model over averaged features it is more
    efficient to average them beforehand rather than on the fly.

    Args:
      model_input: A 'batch_size' x 'max_frames' x 'num_features' matrix of
        input features.
      vocab_size: The number of classes in the dataset.
      num_frames: A vector of length 'batch' which indicates the number of
        frames for each video (before padding).

    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      'batch_size' x 'num_classes'.
    """
    iterations = FLAGS.iterations
    num_frames = tf.cast(tf.expand_dims(num_frames, 1), tf.float32)
    if FLAGS.sample_random_frames:
      model_input = utils.SampleRandomFrames(model_input, num_frames,
                                             iterations)
    else:
      model_input = utils.SampleRandomSequence(model_input, num_frames,
                                               iterations)

    feature_size = model_input.get_shape().as_list()[2]

    keyless_attention = tf.nn.softmax(slim.fully_connected(model_input, 1))

    attention_out = tf.squeeze(
        tf.matmul(tf.transpose(model_input, perm=[0, 2, 1]), keyless_attention),
        axis=-1)   

    output = slim.fully_connected(
        attention_out,
        vocab_size,
        activation_fn=tf.nn.sigmoid,
        weights_regularizer=slim.l2_regularizer(1e-8))
    return {"predictions": output}


class FrameLevelLogisticModelConcat(models.BaseModel):

  def create_model(self, model_input, vocab_size, num_frames, **unused_params):
    """Creates a model which uses a logistic classifier over the average of the

    frame-level features.

    This class is intended to be an example for implementors of frame level
    models. If you want to train a model over averaged features it is more
    efficient to average them beforehand rather than on the fly.

    Args:
      model_input: A 'batch_size' x 'max_frames' x 'num_features' matrix of
        input features.
      vocab_size: The number of classes in the dataset.
      num_frames: A vector of length 'batch' which indicates the number of
        frames for each video (before padding).

    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      'batch_size' x 'num_classes'.
    """
    iterations = FLAGS.iterations
    if FLAGS.sample_random_frames:
      model_input = utils.SampleRandomFrames(model_input, num_frames,
                                             iterations)
    else:
      model_input = utils.SampleRandomSequence(model_input, num_frames,
                                               iterations)
    feature_size = model_input.get_shape().as_list()[2]

    num_frames_in_seg = model_input.get_shape().as_list()[1]
    # denominators = tf.reshape(
    #     tf.tile(num_frames, [1, feature_size]), [-1, feature_size])
    # avg_pooled = tf.reduce_sum(model_input, axis=[1]) / denominators
    concat = tf.reshape(model_input, [-1, feature_size*num_frames_in_seg])

    output = slim.fully_connected(
        concat,
        vocab_size,
        activation_fn=tf.nn.sigmoid,
        weights_regularizer=slim.l2_regularizer(1e-8))
    return {"predictions": output}


class DbofModel(models.BaseModel):
  """Creates a Deep Bag of Frames model.
  The model projects the features for each frame into a higher dimensional
  'clustering' space, pools across frames in that space, and then
  uses a configurable video-level model to classify the now aggregated features.
  The model will randomly sample either frames or sequences of frames during
  training to speed up convergence.

  Args:
    model_input: A 'batch_size' x 'max_frames' x 'num_features' matrix of input
      features.
    vocab_size: The number of classes in the dataset.
    num_frames: A vector of length 'batch' which indicates the number of frames
      for each video (before padding).

  Returns:
    A dictionary with a tensor containing the probability predictions of the
    model in the 'predictions' key. The dimensions of the tensor are
    'batch_size' x 'num_classes'.
  """

  def create_model(self,
                   model_input,
                   vocab_size,
                   num_frames,
                   iterations=None,
                   add_batch_norm=None,
                   sample_random_frames=None,
                   cluster_size=None,
                   hidden_size=None,
                   is_training=True,
                   **unused_params):
    """See base class.
    Args:
      model_input: A 'batch_size' x 'max_frames' x 'num_features' matrix of
        input features.
      vocab_size: The number of classes in the dataset.
      num_frames: A vector of length 'batch' which indicates the number of
        frames for each video (before padding).
      iterations: the number of frames to be sampled.
      add_batch_norm: whether to add batch norm during training.
      sample_random_frames: whether to sample random frames or random sequences.
      cluster_size: the output neuron number of the cluster layer.
      hidden_size: the output neuron number of the hidden layer.
      is_training: whether to build the graph in training mode.
    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      'batch_size' x 'num_classes'.
    """
    iterations = iterations or FLAGS.iterations
    add_batch_norm = add_batch_norm or FLAGS.dbof_add_batch_norm
    random_frames = sample_random_frames or FLAGS.sample_random_frames
    cluster_size = cluster_size or FLAGS.dbof_cluster_size
    hidden1_size = hidden_size or FLAGS.dbof_hidden_size
    act_fn = self.ACT_FN_MAP.get(FLAGS.dbof_activation)
    assert act_fn is not None, ("dbof_activation is not valid: %s." %
                                FLAGS.dbof_activation)

    num_frames = tf.cast(tf.expand_dims(num_frames, 1), tf.float32)
    if random_frames:
      model_input = utils.SampleRandomFrames(model_input, num_frames,
                                             iterations)
    else:
      model_input = utils.SampleRandomSequence(model_input, num_frames,
                                               iterations)
    max_frames = model_input.get_shape().as_list()[1]
    feature_size = model_input.get_shape().as_list()[2]
    reshaped_input = tf.reshape(model_input, [-1, feature_size])
    tf.compat.v1.summary.histogram("input_hist", reshaped_input)

    if add_batch_norm:
      reshaped_input = slim.batch_norm(reshaped_input,
                                       center=True,
                                       scale=True,
                                       is_training=is_training,
                                       scope="input_bn")

    cluster_weights = tf.compat.v1.get_variable(
        "cluster_weights", [feature_size, cluster_size],
        initializer=tf.random_normal_initializer(stddev=1 /
                                                 math.sqrt(feature_size)))
    tf.compat.v1.summary.histogram("cluster_weights", cluster_weights)
    activation = tf.matmul(reshaped_input, cluster_weights)
    if add_batch_norm:
      activation = slim.batch_norm(activation,
                                   center=True,
                                   scale=True,
                                   is_training=is_training,
                                   scope="cluster_bn")
    else:
      cluster_biases = tf.compat.v1.get_variable(
          "cluster_biases", [cluster_size],
          initializer=tf.random_normal_initializer(stddev=1 /
                                                   math.sqrt(feature_size)))
      tf.compat.v1.summary.histogram("cluster_biases", cluster_biases)
      activation += cluster_biases
    activation = act_fn(activation)
    tf.compat.v1.summary.histogram("cluster_output", activation)

    activation = tf.reshape(activation, [-1, max_frames, cluster_size])
    activation = utils.FramePooling(activation, FLAGS.dbof_pooling_method)

    hidden1_weights = tf.compat.v1.get_variable(
        "hidden1_weights", [cluster_size, hidden1_size],
        initializer=tf.random_normal_initializer(stddev=1 /
                                                 math.sqrt(cluster_size)))
    tf.compat.v1.summary.histogram("hidden1_weights", hidden1_weights)
    activation = tf.matmul(activation, hidden1_weights)
    if add_batch_norm:
      activation = slim.batch_norm(activation,
                                   center=True,
                                   scale=True,
                                   is_training=is_training,
                                   scope="hidden1_bn")
    else:
      hidden1_biases = tf.compat.v1.get_variable(
          "hidden1_biases", [hidden1_size],
          initializer=tf.random_normal_initializer(stddev=0.01))
      tf.compat.v1.summary.histogram("hidden1_biases", hidden1_biases)
      activation += hidden1_biases
    activation = act_fn(activation)
    tf.compat.v1.summary.histogram("hidden1_output", activation)

    aggregated_model = getattr(video_level_models,
                               FLAGS.video_level_classifier_model)
    return aggregated_model().create_model(model_input=activation,
                                           vocab_size=vocab_size,
                                           **unused_params)


class LightVLAD():
    def __init__(self, feature_size,max_frames,cluster_size, add_batch_norm, is_training):
        self.feature_size = feature_size
        self.max_frames = max_frames
        self.is_training = is_training
        self.add_batch_norm = add_batch_norm
        self.cluster_size = int(cluster_size)

    def forward(self,reshaped_input):


        cluster_weights = tf.get_variable("cluster_weights",
              [self.feature_size, self.cluster_size],
              initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(self.feature_size)))
       
        activation = tf.matmul(reshaped_input, cluster_weights)
        
        if self.add_batch_norm:
          activation = slim.batch_norm(
              activation,
              center=True,
              scale=True,
              is_training=self.is_training,
              scope="cluster_bn")
        else:
          cluster_biases = tf.get_variable("cluster_biases",
            [cluster_size],
            initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(self.feature_size)))
          tf.summary.histogram("cluster_biases", cluster_biases)
          activation += cluster_biases
        
        activation = tf.nn.softmax(activation)

        activation = tf.reshape(activation, [-1, self.max_frames, self.cluster_size])
       
        activation = tf.transpose(activation,perm=[0,2,1])
        
        reshaped_input = tf.reshape(reshaped_input,[-1,self.max_frames,self.feature_size])
        vlad = tf.matmul(activation,reshaped_input)
        
        vlad = tf.transpose(vlad,perm=[0,2,1])
        vlad = tf.nn.l2_normalize(vlad,1)

        vlad = tf.reshape(vlad,[-1,self.cluster_size*self.feature_size])
        vlad = tf.nn.l2_normalize(vlad,1)

        return vlad


class NetVLAD():
    def __init__(self, feature_size,max_frames,cluster_size, add_batch_norm, is_training):
        self.feature_size = feature_size
        self.max_frames = max_frames
        self.is_training = is_training
        self.add_batch_norm = add_batch_norm
        self.cluster_size = int(cluster_size)

    def forward(self,reshaped_input):
        dtype = tf.float32
        #breakpoint()
        with tf.variable_scope('conv2') as scope:
          cluster_weights = tf.get_variable("cluster_weights",
                [self.feature_size, self.cluster_size],
                initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(self.feature_size)))
                              
          tf.summary.histogram("cluster_weights", cluster_weights)
          activation = tf.matmul(reshaped_input, cluster_weights)
        
        if self.add_batch_norm:
          activation = slim.batch_norm(
              activation,
              center=True,
              scale=True,
              is_training=self.is_training,
              scope="cluster_bn")
        else:
          cluster_biases = tf.get_variable("cluster_biases",
            [cluster_size],
            initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(self.feature_size)),dtype=dtype)
          tf.summary.histogram("cluster_biases", cluster_biases)
          activation += cluster_biases
        
        activation = tf.nn.softmax(activation)
        tf.summary.histogram("cluster_output", activation)

        activation = tf.reshape(activation, [-1, self.max_frames, self.cluster_size])

        a_sum = tf.reduce_sum(activation,-2,keep_dims=True)
        with tf.variable_scope('cluster_weights2') as scope:
            cluster_weights2 = tf.get_variable("cluster_weights2",
                [1,self.feature_size, self.cluster_size],
                initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(self.feature_size)),dtype=dtype)
            a = tf.multiply(a_sum, cluster_weights2)
        
        activation = tf.transpose(activation,perm=[0,2,1])
        #breakpoint()
        reshaped_input = tf.reshape(reshaped_input,[-1,self.max_frames,self.feature_size])
        vlad = tf.matmul(activation,reshaped_input)
        vlad = tf.transpose(vlad,perm=[0,2,1])
        vlad = tf.subtract(vlad,a)
        vlad = tf.nn.l2_normalize(vlad,1)
        vlad = tf.reshape(vlad,[-1,self.cluster_size*self.feature_size])
        vlad = tf.nn.l2_normalize(vlad,1)
        #breakpoint()
        return vlad


class NetVLAGD():
    def __init__(self, feature_size,max_frames,cluster_size, add_batch_norm, is_training):
        self.feature_size = feature_size
        self.max_frames = max_frames
        self.is_training = is_training
        self.add_batch_norm = add_batch_norm
        self.cluster_size = cluster_size

    def forward(self,reshaped_input):


        cluster_weights = tf.get_variable("cluster_weights",
              [self.feature_size, self.cluster_size],
              initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(self.feature_size)))
       
        activation = tf.matmul(reshaped_input, cluster_weights)
        
        if self.add_batch_norm:
          activation = slim.batch_norm(
              activation,
              center=True,
              scale=True,
              is_training=self.is_training,
              scope="cluster_bn")
        else:
          cluster_biases = tf.get_variable("cluster_biases",
            [cluster_size],
            initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(self.feature_size)))
        
        activation = tf.nn.softmax(activation)

        activation = tf.reshape(activation, [-1, self.max_frames, self.cluster_size])

        gate_weights = tf.get_variable("gate_weights",
            [1, self.cluster_size,self.feature_size],
            initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(self.feature_size)))
        
        gate_weights = tf.sigmoid(gate_weights)

        activation = tf.transpose(activation,perm=[0,2,1])
        
        reshaped_input = tf.reshape(reshaped_input,[-1,self.max_frames,self.feature_size])

        vlagd = tf.matmul(activation,reshaped_input)
        vlagd = tf.multiply(vlagd,gate_weights)

        vlagd = tf.transpose(vlagd,perm=[0,2,1])
        
        vlagd = tf.nn.l2_normalize(vlagd,1)

        vlagd = tf.reshape(vlagd,[-1,self.cluster_size*self.feature_size])
        vlagd = tf.nn.l2_normalize(vlagd,1)

        return vlagd




class GatedDBoF():
    def __init__(self, feature_size,max_frames,cluster_size, max_pool, add_batch_norm, is_training):
        self.feature_size = feature_size
        self.max_frames = max_frames
        self.is_training = is_training
        self.add_batch_norm = add_batch_norm
        self.cluster_size = cluster_size
        self.max_pool = max_pool

    def forward(self, reshaped_input):

        feature_size = self.feature_size
        cluster_size = self.cluster_size
        add_batch_norm = self.add_batch_norm
        max_frames = self.max_frames
        is_training = self.is_training
        max_pool = self.max_pool

        cluster_weights = tf.get_variable("cluster_weights",
          [feature_size, cluster_size],
          initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(feature_size)))
        
        tf.summary.histogram("cluster_weights", cluster_weights)
        activation = tf.matmul(reshaped_input, cluster_weights)
        
        if add_batch_norm:
          activation = slim.batch_norm(
              activation,
              center=True,
              scale=True,
              is_training=is_training,
              scope="cluster_bn")
        else:
          cluster_biases = tf.get_variable("cluster_biases",
            [cluster_size],
            initializer = tf.random_normal(stddev=1 / math.sqrt(feature_size)))
          tf.summary.histogram("cluster_biases", cluster_biases)
          activation += cluster_biases

        activation = tf.nn.softmax(activation)

        activation = tf.reshape(activation, [-1, max_frames, cluster_size])

        activation_sum = tf.reduce_sum(activation,1)
        
        activation_max = tf.reduce_max(activation,1)
        activation_max = tf.nn.l2_normalize(activation_max,1)


        dim_red = tf.get_variable("dim_red",
          [cluster_size, feature_size],
          initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(feature_size)))
 
        cluster_weights_2 = tf.get_variable("cluster_weights_2",
          [feature_size, cluster_size],
          initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(feature_size)))
        
        tf.summary.histogram("cluster_weights_2", cluster_weights_2)
        
        activation = tf.matmul(activation_max, dim_red)
        activation = tf.matmul(activation, cluster_weights_2)
        
        if add_batch_norm:
          activation = slim.batch_norm(
              activation,
              center=True,
              scale=True,
              is_training=is_training,
              scope="cluster_bn_2")
        else:
          cluster_biases = tf.get_variable("cluster_biases_2",
            [cluster_size],
            initializer = tf.random_normal(stddev=1 / math.sqrt(feature_size)))
          tf.summary.histogram("cluster_biases_2", cluster_biases)
          activation += cluster_biases

        activation = tf.sigmoid(activation)

        activation = tf.multiply(activation,activation_sum)
        activation = tf.nn.l2_normalize(activation,1)

        return activation



class SoftDBoF():
    def __init__(self, feature_size,max_frames,cluster_size, max_pool, add_batch_norm, is_training):
        self.feature_size = feature_size
        self.max_frames = max_frames
        self.is_training = is_training
        self.add_batch_norm = add_batch_norm
        self.cluster_size = cluster_size
        self.max_pool = max_pool

    def forward(self, reshaped_input):

        feature_size = self.feature_size
        cluster_size = self.cluster_size
        add_batch_norm = self.add_batch_norm
        max_frames = self.max_frames
        is_training = self.is_training
        max_pool = self.max_pool

        cluster_weights = tf.get_variable("cluster_weights",
          [feature_size, cluster_size],
          initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(feature_size)))
        
        tf.summary.histogram("cluster_weights", cluster_weights)
        activation = tf.matmul(reshaped_input, cluster_weights)
        
        if add_batch_norm:
          activation = slim.batch_norm(
              activation,
              center=True,
              scale=True,
              is_training=is_training,
              scope="cluster_bn")
        else:
          cluster_biases = tf.get_variable("cluster_biases",
            [cluster_size],
            initializer = tf.random_normal(stddev=1 / math.sqrt(feature_size)))
          tf.summary.histogram("cluster_biases", cluster_biases)
          activation += cluster_biases

        activation = tf.nn.softmax(activation)

        activation = tf.reshape(activation, [-1, max_frames, cluster_size])

        activation_sum = tf.reduce_sum(activation,1)
        activation_sum = tf.nn.l2_normalize(activation_sum,1)

        if max_pool:
            activation_max = tf.reduce_max(activation,1)
            activation_max = tf.nn.l2_normalize(activation_max,1)
            activation = tf.concat([activation_sum,activation_max],1)
        else:
            activation = activation_sum
        
        return activation



class DBoF():
    def __init__(self, feature_size,max_frames,cluster_size,activation, add_batch_norm, is_training):
        self.feature_size = feature_size
        self.max_frames = max_frames
        self.is_training = is_training
        self.add_batch_norm = add_batch_norm
        self.cluster_size = cluster_size
        self.activation = activation


    def forward(self, reshaped_input):

        feature_size = self.feature_size
        cluster_size = self.cluster_size
        add_batch_norm = self.add_batch_norm
        max_frames = self.max_frames
        is_training = self.is_training

        cluster_weights = tf.get_variable("cluster_weights",
          [feature_size, cluster_size],
          initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(feature_size)))
        
        tf.summary.histogram("cluster_weights", cluster_weights)
        activation = tf.matmul(reshaped_input, cluster_weights)
        
        if add_batch_norm:
          activation = slim.batch_norm(
              activation,
              center=True,
              scale=True,
              is_training=is_training,
              scope="cluster_bn")
        else:
          cluster_biases = tf.get_variable("cluster_biases",
            [cluster_size],
            initializer = tf.random_normal(stddev=1 / math.sqrt(feature_size)))
          tf.summary.histogram("cluster_biases", cluster_biases)
          activation += cluster_biases

        if activation == 'glu':
            space_ind = range(cluster_size/2)
            gate_ind = range(cluster_size/2,cluster_size)

            gates = tf.sigmoid(activation[:,gate_ind])
            activation = tf.multiply(activation[:,space_ind],gates)

        elif activation == 'relu':
            activation = tf.nn.relu6(activation)
        
        tf.summary.histogram("cluster_output", activation)

        activation = tf.reshape(activation, [-1, max_frames, cluster_size])

        avg_activation = utils.FramePooling(activation, 'average')
        avg_activation = tf.nn.l2_normalize(avg_activation,1)

        max_activation = utils.FramePooling(activation, 'max')
        max_activation = tf.nn.l2_normalize(max_activation,1)
        
        return tf.concat([avg_activation,max_activation],1)

class NetFV():
    def __init__(self, feature_size,max_frames,cluster_size, add_batch_norm, is_training):
        self.feature_size = feature_size
        self.max_frames = max_frames
        self.is_training = is_training
        self.add_batch_norm = add_batch_norm
        self.cluster_size = cluster_size

    def forward(self,reshaped_input):
        cluster_weights = tf.get_variable("cluster_weights",
          [self.feature_size, self.cluster_size],
          initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(self.feature_size)))
     
        covar_weights = tf.get_variable("covar_weights",
          [self.feature_size, self.cluster_size],
          initializer = tf.random_normal_initializer(mean=1.0, stddev=1 /math.sqrt(self.feature_size)))
      
        covar_weights = tf.square(covar_weights)
        eps = tf.constant([1e-6])
        covar_weights = tf.add(covar_weights,eps)

        tf.summary.histogram("cluster_weights", cluster_weights)
        activation = tf.matmul(reshaped_input, cluster_weights)
        if self.add_batch_norm:
          activation = slim.batch_norm(
              activation,
              center=True,
              scale=True,
              is_training=self.is_training,
              scope="cluster_bn")
        else:
          cluster_biases = tf.get_variable("cluster_biases",
            [self.cluster_size],
            initializer = tf.random_normal(stddev=1 / math.sqrt(self.feature_size)))
          tf.summary.histogram("cluster_biases", cluster_biases)
          activation += cluster_biases
        
        activation = tf.nn.softmax(activation)
        tf.summary.histogram("cluster_output", activation)

        activation = tf.reshape(activation, [-1, self.max_frames, self.cluster_size])

        a_sum = tf.reduce_sum(activation,-2,keep_dims=True)

        if not FLAGS.fv_couple_weights:
            cluster_weights2 = tf.get_variable("cluster_weights2",
              [1,self.feature_size, self.cluster_size],
              initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(self.feature_size)))
        else:
            cluster_weights2 = tf.scalar_mul(FLAGS.fv_coupling_factor,cluster_weights)

        a = tf.multiply(a_sum,cluster_weights2)
        
        activation = tf.transpose(activation,perm=[0,2,1])
        
        reshaped_input = tf.reshape(reshaped_input,[-1,self.max_frames,self.feature_size])
        fv1 = tf.matmul(activation,reshaped_input)
        
        fv1 = tf.transpose(fv1,perm=[0,2,1])

        # computing second order FV
        a2 = tf.multiply(a_sum,tf.square(cluster_weights2)) 

        b2 = tf.multiply(fv1,cluster_weights2) 
        fv2 = tf.matmul(activation,tf.square(reshaped_input)) 
     
        fv2 = tf.transpose(fv2,perm=[0,2,1])
        fv2 = tf.add_n([a2,fv2,tf.scalar_mul(-2,b2)])

        fv2 = tf.divide(fv2,tf.square(covar_weights))
        fv2 = tf.subtract(fv2,a_sum)

        fv2 = tf.reshape(fv2,[-1,self.cluster_size*self.feature_size])
      
        fv2 = tf.nn.l2_normalize(fv2,1)
        fv2 = tf.reshape(fv2,[-1,self.cluster_size*self.feature_size])
        fv2 = tf.nn.l2_normalize(fv2,1)

        fv1 = tf.subtract(fv1,a)
        fv1 = tf.divide(fv1,covar_weights) 

        fv1 = tf.nn.l2_normalize(fv1,1)
        fv1 = tf.reshape(fv1,[-1,self.cluster_size*self.feature_size])
        fv1 = tf.nn.l2_normalize(fv1,1)

        return tf.concat([fv1,fv2],1)


class NetVLADModelLF(models.BaseModel):
  """Creates a NetVLAD based model.
  Args:
    model_input: A 'batch_size' x 'max_frames' x 'num_features' matrix of
                 input features.
    vocab_size: The number of classes in the dataset.
    num_frames: A vector of length 'batch' which indicates the number of
         frames for each video (before padding).
  Returns:
    A dictionary with a tensor containing the probability predictions of the
    model in the 'predictions' key. The dimensions of the tensor are
    'batch_size' x 'num_classes'.
  """


  def create_model(self,
                   model_input,
                   vocab_size,
                   num_frames,
                   is_training=True,
                   **unused_params):
    iterations = FLAGS.iterations
    add_batch_norm = FLAGS.netvlad_add_batch_norm
    random_frames = FLAGS.sample_random_frames
    
    hidden1_size = FLAGS.netvlad_hidden_size
    cluster_size = FLAGS.netvlad_cluster_size

    relu = FLAGS.netvlad_relu
    dimred = FLAGS.netvlad_dimred
    gating = FLAGS.gating
    remove_diag = FLAGS.gating_remove_diag
    lightvlad = FLAGS.lightvlad
    vlagd = FLAGS.vlagd

    num_frames = tf.cast(tf.expand_dims(num_frames, 1), tf.float32)

    '''
    ####################################
    n_bag = FLAGS.max_frames // 5
    model_input = utils.ReshapeFramesToMIL(model_input)  # B x FLAGS.pad_seq_length/5 x 5 x 1132
    max_frames = 5
    feature_size = model_input.get_shape().as_list()[3]
    #####################################
    '''
    if random_frames:
      model_input = utils.SampleRandomFrames(model_input, num_frames,
                                             iterations)
    else:
      model_input = utils.SampleRandomSequence(model_input, num_frames,
                                               iterations)
    max_frames = model_input.get_shape().as_list()[1]
    feature_size = model_input.get_shape().as_list()[2]

    reshaped_input = tf.reshape(model_input, [-1, feature_size])

    new_vfeature_size = 1024
    new_afeature_size = 128

    if lightvlad:
      video_NetVLAD = LightVLAD(new_vfeature_size, max_frames,cluster_size, add_batch_norm, is_training)
      audio_NetVLAD = LightVLAD(new_afeature_size, max_frames,cluster_size//2, add_batch_norm, is_training)
    elif vlagd:
      video_NetVLAD = NetVLAGD(new_vfeature_size, max_frames,cluster_size, add_batch_norm, is_training)
      audio_NetVLAD = NetVLAGD(new_afeature_size, max_frames,cluster_size//2, add_batch_norm, is_training)
    else:
      video_NetVLAD = NetVLAD(new_vfeature_size, max_frames,cluster_size, add_batch_norm, is_training)
      audio_NetVLAD = NetVLAD(new_afeature_size, max_frames,cluster_size//2, add_batch_norm, is_training)
    ##breakpoint()
    dtype = tf.float32

    with tf.variable_scope("video_VLAD"):
        vlad_video = video_NetVLAD.forward(reshaped_input[:, 0:1024]) 

    with tf.variable_scope("audio_VLAD"):
        vlad_audio = audio_NetVLAD.forward(reshaped_input[:, 1024:])

    vlad = tf.concat([vlad_video, vlad_audio],1)

    vlad_dim = vlad.get_shape().as_list()[1]

    #
    # 1st hidden layer
    #
    with tf.variable_scope('hidden1_weights') as scope:
      hidden1_weights = tf.get_variable("hidden1_weights",
        [vlad_dim, hidden1_size],
        initializer   = tf.random_normal_initializer(stddev=1/math.sqrt(vlad_dim)),
        dtype=tf.float32)        
    activation = tf.matmul(vlad, hidden1_weights)

    activation2 = slim.batch_norm(
        activation,
        center=True,
        scale=True,
        is_training=is_training,
        scope="hidden1_bn")      
    activation2 = tf.nn.relu6(activation2)

    #
    # 2nd hidden layer
    #
    hidden2_size = 2 * hidden1_size
    with tf.variable_scope('hidden2_weights') as scope:
      hidden2_weights = tf.get_variable("hidden2_weights",
        [hidden1_size, hidden2_size],
        initializer=tf.random_normal_initializer(stddev=1 / math.sqrt(hidden1_size)), dtype=dtype)
    activation2 = tf.matmul(activation2, hidden2_weights)
    #activation2 += activation

    hidden2_biases = tf.get_variable("hidden2_biases", 
      [hidden2_size], 
      initializer = tf.random_normal_initializer(stddev=0.01), dtype=dtype)
    tf.summary.histogram("hidden2_biases", hidden2_biases)
    activation2 += hidden2_biases
    


    if gating:
      with tf.variable_scope('gating_weights') as scope:
          gating_weights = tf.get_variable("gating_weights",
            [hidden2_size, hidden2_size],
            initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(hidden2_size)), dtype=dtype)

      gates = tf.matmul(activation2, gating_weights)
      if add_batch_norm:
        gates = slim.batch_norm(
            gates,
            center=True,
            scale=True,
            is_training=is_training,
            scope="gating_bn")
      else:
        gating_biases = tf.get_variable("gating_biases",
          [hidden2_size],
          initializer = tf.random_normal_initializer(stddev=0.01), dtype=dtype)
        gates += gating_biases

      gates = tf.sigmoid(gates)
      activation2 = tf.multiply(activation2, gates)


    aggregated_model = getattr(video_level_models,
                               FLAGS.video_level_classifier_model)

    ##breakpoint()
    return aggregated_model().create_model(
        model_input=activation2,
        vocab_size=vocab_size,
        is_training=is_training,
        **unused_params)
  

class DbofModelLF(models.BaseModel):
  """Creates a Deep Bag of Frames model.
  The model projects the features for each frame into a higher dimensional
  'clustering' space, pools across frames in that space, and then
  uses a configurable video-level model to classify the now aggregated features.
  The model will randomly sample either frames or sequences of frames during
  training to speed up convergence.
  Args:
    model_input: A 'batch_size' x 'max_frames' x 'num_features' matrix of
                 input features.
    vocab_size: The number of classes in the dataset.
    num_frames: A vector of length 'batch' which indicates the number of
         frames for each video (before padding).
  Returns:
    A dictionary with a tensor containing the probability predictions of the
    model in the 'predictions' key. The dimensions of the tensor are
    'batch_size' x 'num_classes'.
  """

  def create_model(self,
                   model_input,
                   vocab_size,
                   num_frames,
                   iterations=None,
                   add_batch_norm=None,
                   sample_random_frames=None,
                   cluster_size=None,
                   hidden_size=None,
                   is_training=True,
                   **unused_params):
    iterations = iterations or FLAGS.iterations
    add_batch_norm = add_batch_norm or FLAGS.dbof_add_batch_norm
    random_frames = sample_random_frames or FLAGS.sample_random_frames
    cluster_size = cluster_size or FLAGS.dbof_cluster_size
    hidden1_size = hidden_size or FLAGS.dbof_hidden_size
    relu = FLAGS.dbof_relu
    cluster_activation = FLAGS.dbof_activation

    num_frames = tf.cast(tf.expand_dims(num_frames, 1), tf.float32)
    if random_frames:
      model_input = utils.SampleRandomFrames(model_input, num_frames,
                                             iterations)
    else:
      model_input = utils.SampleRandomSequence(model_input, num_frames,
                                               iterations)
    max_frames = model_input.get_shape().as_list()[1]
    feature_size = model_input.get_shape().as_list()[2]
    reshaped_input = tf.reshape(model_input, [-1, feature_size])
    tf.summary.histogram("input_hist", reshaped_input)

    if cluster_activation == 'glu':
        cluster_size = 2*cluster_size

    video_Dbof = DBoF(1024,max_frames,cluster_size, cluster_activation, add_batch_norm, is_training)
    audio_Dbof = DBoF(128,max_frames,cluster_size//8, cluster_activation, add_batch_norm, is_training)


    if add_batch_norm:
      reshaped_input = slim.batch_norm(
          reshaped_input,
          center=True,
          scale=True,
          is_training=is_training,
          scope="input_bn")

    with tf.variable_scope("video_DBOF"):
        dbof_video = video_Dbof.forward(reshaped_input[:,0:1024]) 

    with tf.variable_scope("audio_DBOF"):
        dbof_audio = audio_Dbof.forward(reshaped_input[:,1024:])

    dbof = tf.concat([dbof_video, dbof_audio],1)

    dbof_dim = dbof.get_shape().as_list()[1] 

    hidden1_weights = tf.get_variable("hidden1_weights",
      [dbof_dim, hidden1_size],
      initializer=tf.random_normal_initializer(stddev=1 / math.sqrt(cluster_size)))
    tf.summary.histogram("hidden1_weights", hidden1_weights)
    activation = tf.matmul(dbof, hidden1_weights)

    if add_batch_norm and relu:
      activation = slim.batch_norm(
          activation,
          center=True,
          scale=True,
          is_training=is_training,
          scope="hidden1_bn")
    else:
      hidden1_biases = tf.get_variable("hidden1_biases",
        [hidden1_size],
        initializer = tf.random_normal_initializer(stddev=0.01))
      tf.summary.histogram("hidden1_biases", hidden1_biases)
      activation += hidden1_biases

    if relu:
      activation = tf.nn.relu6(activation)
    tf.summary.histogram("hidden1_output", activation)

    aggregated_model = getattr(video_level_models,
                               FLAGS.video_level_classifier_model)
    
    return aggregated_model().create_model(
        model_input=activation,
        vocab_size=vocab_size,
        is_training=is_training,
        **unused_params)


class GatedDbofModelLF(models.BaseModel):
  """Creates a Gated Deep Bag of Frames model.
  The model projects the features for each frame into a higher dimensional
  'clustering' space, pools across frames in that space, and then
  uses a configurable video-level model to classify the now aggregated features.
  The model will randomly sample either frames or sequences of frames during
  training to speed up convergence.
  Args:
    model_input: A 'batch_size' x 'max_frames' x 'num_features' matrix of
                 input features.
    vocab_size: The number of classes in the dataset.
    num_frames: A vector of length 'batch' which indicates the number of
         frames for each video (before padding).
  Returns:
    A dictionary with a tensor containing the probability predictions of the
    model in the 'predictions' key. The dimensions of the tensor are
    'batch_size' x 'num_classes'.
  """

  def create_model(self,
                   model_input,
                   vocab_size,
                   num_frames,
                   iterations=None,
                   add_batch_norm=None,
                   sample_random_frames=None,
                   cluster_size=None,
                   hidden_size=None,
                   is_training=True,
                   **unused_params):
    iterations = iterations or FLAGS.iterations
    add_batch_norm = add_batch_norm or FLAGS.dbof_add_batch_norm
    random_frames = sample_random_frames or FLAGS.sample_random_frames
    cluster_size = cluster_size or FLAGS.dbof_cluster_size
    hidden1_size = hidden_size or FLAGS.dbof_hidden_size
    fc_dimred = FLAGS.fc_dimred
    relu = FLAGS.dbof_relu
    max_pool = FLAGS.softdbof_maxpool

    num_frames = tf.cast(tf.expand_dims(num_frames, 1), tf.float32)
    if random_frames:
      model_input = utils.SampleRandomFrames(model_input, num_frames,
                                             iterations)
    else:
      model_input = utils.SampleRandomSequence(model_input, num_frames,
                                               iterations)
    max_frames = model_input.get_shape().as_list()[1]
    feature_size = model_input.get_shape().as_list()[2]
    reshaped_input = tf.reshape(model_input, [-1, feature_size])
    tf.summary.histogram("input_hist", reshaped_input)

    video_Dbof = GatedDBoF(1024,max_frames,cluster_size, max_pool, add_batch_norm, is_training)
    audio_Dbof = SoftDBoF(128,max_frames,cluster_size/8, max_pool, add_batch_norm, is_training)


    if add_batch_norm:
      reshaped_input = slim.batch_norm(
          reshaped_input,
          center=True,
          scale=True,
          is_training=is_training,
          scope="input_bn")

    with tf.variable_scope("video_DBOF"):
        dbof_video = video_Dbof.forward(reshaped_input[:,0:1024]) 

    with tf.variable_scope("audio_DBOF"):
        dbof_audio = audio_Dbof.forward(reshaped_input[:,1024:])

    dbof = tf.concat([dbof_video, dbof_audio],1)

    dbof_dim = dbof.get_shape().as_list()[1] 

    if fc_dimred:
        hidden1_weights = tf.get_variable("hidden1_weights",
          [dbof_dim, hidden1_size],
          initializer=tf.random_normal_initializer(stddev=1 / math.sqrt(cluster_size)))
        tf.summary.histogram("hidden1_weights", hidden1_weights)
        activation = tf.matmul(dbof, hidden1_weights)

        if add_batch_norm and relu:
          activation = slim.batch_norm(
              activation,
              center=True,
              scale=True,
              is_training=is_training,
              scope="hidden1_bn")
        else:
          hidden1_biases = tf.get_variable("hidden1_biases",
            [hidden1_size],
            initializer = tf.random_normal_initializer(stddev=0.01))
          tf.summary.histogram("hidden1_biases", hidden1_biases)
          activation += hidden1_biases

        if relu:
          activation = tf.nn.relu6(activation)
        tf.summary.histogram("hidden1_output", activation)
    else:
        activation = dbof

    aggregated_model = getattr(video_level_models,
                               FLAGS.video_level_classifier_model)

    
    return aggregated_model().create_model(
        model_input=activation,
        vocab_size=vocab_size,
        is_training=is_training,
        **unused_params)


class SoftDbofModelLF(models.BaseModel):
  """Creates a Soft Deep Bag of Frames model.
  The model projects the features for each frame into a higher dimensional
  'clustering' space, pools across frames in that space, and then
  uses a configurable video-level model to classify the now aggregated features.
  The model will randomly sample either frames or sequences of frames during
  training to speed up convergence.
  Args:
    model_input: A 'batch_size' x 'max_frames' x 'num_features' matrix of
                 input features.
    vocab_size: The number of classes in the dataset.
    num_frames: A vector of length 'batch' which indicates the number of
         frames for each video (before padding).
  Returns:
    A dictionary with a tensor containing the probability predictions of the
    model in the 'predictions' key. The dimensions of the tensor are
    'batch_size' x 'num_classes'.
  """

  def create_model(self,
                   model_input,
                   vocab_size,
                   num_frames,
                   iterations=None,
                   add_batch_norm=None,
                   sample_random_frames=None,
                   cluster_size=None,
                   hidden_size=None,
                   is_training=True,
                   **unused_params):
    iterations = iterations or FLAGS.iterations
    add_batch_norm = add_batch_norm or FLAGS.dbof_add_batch_norm
    random_frames = sample_random_frames or FLAGS.sample_random_frames
    cluster_size = cluster_size or FLAGS.dbof_cluster_size
    hidden1_size = hidden_size or FLAGS.dbof_hidden_size
    fc_dimred = FLAGS.fc_dimred
    relu = FLAGS.dbof_relu
    max_pool = FLAGS.softdbof_maxpool

    num_frames = tf.cast(tf.expand_dims(num_frames, 1), tf.float32)
    if random_frames:
      model_input = utils.SampleRandomFrames(model_input, num_frames,
                                             iterations)
    else:
      model_input = utils.SampleRandomSequence(model_input, num_frames,
                                               iterations)
    max_frames = model_input.get_shape().as_list()[1]
    feature_size = model_input.get_shape().as_list()[2]
    reshaped_input = tf.reshape(model_input, [-1, feature_size])
    tf.summary.histogram("input_hist", reshaped_input)

    video_Dbof = SoftDBoF(1024,max_frames,cluster_size, max_pool, add_batch_norm, is_training)
    audio_Dbof = SoftDBoF(128,max_frames,cluster_size/8, max_pool, add_batch_norm, is_training)


    if add_batch_norm:
      reshaped_input = slim.batch_norm(
          reshaped_input,
          center=True,
          scale=True,
          is_training=is_training,
          scope="input_bn")

    with tf.variable_scope("video_DBOF"):
        dbof_video = video_Dbof.forward(reshaped_input[:,0:1024]) 

    with tf.variable_scope("audio_DBOF"):
        dbof_audio = audio_Dbof.forward(reshaped_input[:,1024:])

    dbof = tf.concat([dbof_video, dbof_audio],1)

    dbof_dim = dbof.get_shape().as_list()[1] 

    if fc_dimred:
        hidden1_weights = tf.get_variable("hidden1_weights",
          [dbof_dim, hidden1_size],
          initializer=tf.random_normal_initializer(stddev=1 / math.sqrt(cluster_size)))
        tf.summary.histogram("hidden1_weights", hidden1_weights)
        activation = tf.matmul(dbof, hidden1_weights)

        if add_batch_norm and relu:
          activation = slim.batch_norm(
              activation,
              center=True,
              scale=True,
              is_training=is_training,
              scope="hidden1_bn")
        else:
          hidden1_biases = tf.get_variable("hidden1_biases",
            [hidden1_size],
            initializer = tf.random_normal_initializer(stddev=0.01))
          tf.summary.histogram("hidden1_biases", hidden1_biases)
          activation += hidden1_biases

        if relu:
          activation = tf.nn.relu6(activation)
        tf.summary.histogram("hidden1_output", activation)
    else:
        activation = dbof

    aggregated_model = getattr(video_level_models,
                               FLAGS.video_level_classifier_model)

    
    return aggregated_model().create_model(
        model_input=activation,
        vocab_size=vocab_size,
        is_training=is_training,
        **unused_params)

    
class NetFVModelLF(models.BaseModel):
  """Creates a NetFV based model.
     It emulates a Gaussian Mixture Fisher Vector pooling operations
  Args:
    model_input: A 'batch_size' x 'max_frames' x 'num_features' matrix of
                 input features.
    vocab_size: The number of classes in the dataset.
    num_frames: A vector of length 'batch' which indicates the number of
         frames for each video (before padding).
  Returns:
    A dictionary with a tensor containing the probability predictions of the
    model in the 'predictions' key. The dimensions of the tensor are
    'batch_size' x 'num_classes'.
  """


  def create_model(self,
                   model_input,
                   vocab_size,
                   num_frames,
                   is_training=True,
                   **unused_params):
    iterations = FLAGS.iterations
    add_batch_norm = FLAGS.netvlad_add_batch_norm
    random_frames = FLAGS.sample_random_frames
    cluster_size = FLAGS.fv_cluster_size
    hidden1_size = FLAGS.fv_hidden_size
    relu = FLAGS.fv_relu
    gating = FLAGS.gating

    num_frames = tf.cast(tf.expand_dims(num_frames, 1), tf.float32)
    if is_training:
        if random_frames:
          model_input = utils.SampleRandomFrames(model_input, num_frames,
                                                 iterations)
        else:
          model_input = utils.SampleRandomSequence(model_input, num_frames,
                                                   iterations)
    max_frames = model_input.get_shape().as_list()[1]
    feature_size = model_input.get_shape().as_list()[2]
    reshaped_input = tf.reshape(model_input, [-1, feature_size])
    tf.summary.histogram("input_hist", reshaped_input)

    video_NetFV = NetFV(1024,max_frames,cluster_size, add_batch_norm, is_training)
    audio_NetFV = NetFV(128,max_frames,cluster_size/2, add_batch_norm, is_training)


    if add_batch_norm:
      reshaped_input = slim.batch_norm(
          reshaped_input,
          center=True,
          scale=True,
          is_training=is_training,
          scope="input_bn")

    with tf.variable_scope("video_FV"):
        fv_video = video_NetFV.forward(reshaped_input[:,0:1024]) 

    with tf.variable_scope("audio_FV"):
        fv_audio = audio_NetFV.forward(reshaped_input[:,1024:])

    fv = tf.concat([fv_video, fv_audio],1)

    fv_dim = fv.get_shape().as_list()[1] 
    hidden1_weights = tf.get_variable("hidden1_weights",
      [fv_dim, hidden1_size],
      initializer=tf.random_normal_initializer(stddev=1 / math.sqrt(cluster_size)))
    
    activation = tf.matmul(fv, hidden1_weights)

    if add_batch_norm and relu:
      activation = slim.batch_norm(
          activation,
          center=True,
          scale=True,
          is_training=is_training,
          scope="hidden1_bn")
    else:
      hidden1_biases = tf.get_variable("hidden1_biases",
        [hidden1_size],
        initializer = tf.random_normal_initializer(stddev=0.01))
      tf.summary.histogram("hidden1_biases", hidden1_biases)
      activation += hidden1_biases
   
    if relu:
      activation = tf.nn.relu6(activation)

    if gating:
        gating_weights = tf.get_variable("gating_weights_2",
          [hidden1_size, hidden1_size],
          initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(hidden1_size)))
        
        gates = tf.matmul(activation, gating_weights)
        
        if add_batch_norm:
          gates = slim.batch_norm(
              gates,
              center=True,
              scale=True,
              is_training=is_training,
              scope="gating_bn")
        else:
          gating_biases = tf.get_variable("gating_biases",
            [cluster_size],
            initializer = tf.random_normal(stddev=1 / math.sqrt(feature_size)))
          gates += gating_biases

        gates = tf.sigmoid(gates)

        activation = tf.multiply(activation,gates)


    aggregated_model = getattr(video_level_models,
                               FLAGS.video_level_classifier_model)

    return aggregated_model().create_model(
        model_input=activation,
        vocab_size=vocab_size,
        is_training=is_training,
        **unused_params)


class LstmModel(models.BaseModel):

  def create_model(self, model_input, vocab_size, num_frames, is_training=True, **unused_params):
    """Creates a model which uses a stack of LSTMs to represent the video.
    Args:
      model_input: A 'batch_size' x 'max_frames' x 'num_features' matrix of
                   input features.
      vocab_size: The number of classes in the dataset.
      num_frames: A vector of length 'batch' which indicates the number of
           frames for each video (before padding).
    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      'batch_size' x 'num_classes'.
    """
    lstm_size = FLAGS.lstm_cells
    number_of_layers = FLAGS.lstm_layers
    random_frames = FLAGS.lstm_random_sequence
    iterations = FLAGS.iterations
    backward = FLAGS.lstm_backward

    if random_frames:
      num_frames_2 = tf.cast(tf.expand_dims(num_frames, 1), tf.float32)
      model_input = utils.SampleRandomFrames(model_input, num_frames_2,
                                             iterations)
    if backward:
      model_input = tf.reverse_sequence(model_input, num_frames, seq_axis=1) 
 
    stacked_lstm = tf.contrib.rnn.MultiRNNCell(
            [
                tf.contrib.rnn.BasicLSTMCell(
                    lstm_size, forget_bias=1.0, state_is_tuple=False)
                for _ in range(number_of_layers)
                ], state_is_tuple=False)

    loss = 0.0
    with tf.variable_scope("RNN"):
      outputs, state = tf.nn.dynamic_rnn(stacked_lstm, model_input,
                                         sequence_length=num_frames,
                                         dtype=tf.float32)

    aggregated_model = getattr(video_level_models,
                               FLAGS.video_level_classifier_model)

    return aggregated_model().create_model(
        model_input=state,
        vocab_size=vocab_size,
        is_training=is_training,
        **unused_params)


class GruModel(models.BaseModel):

  def create_model(self, model_input, vocab_size, num_frames, is_training=True, **unused_params):
    """Creates a model which uses a stack of GRUs to represent the video.
    Args:
      model_input: A 'batch_size' x 'max_frames' x 'num_features' matrix of
                   input features.
      vocab_size: The number of classes in the dataset.
      num_frames: A vector of length 'batch' which indicates the number of
           frames for each video (before padding).
    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      'batch_size' x 'num_classes'.
    """
    gru_size = FLAGS.gru_cells
    number_of_layers = FLAGS.gru_layers
    backward = FLAGS.gru_backward
    random_frames = FLAGS.gru_random_sequence
    iterations = FLAGS.iterations
    
    if random_frames:
      num_frames_2 = tf.cast(tf.expand_dims(num_frames, 1), tf.float32)
      model_input = utils.SampleRandomFrames(model_input, num_frames_2,
                                             iterations)
 
    if backward:
        model_input = tf.reverse_sequence(model_input, num_frames, seq_axis=1) 
    
    stacked_GRU = tf.contrib.rnn.MultiRNNCell(
            [
                tf.contrib.rnn.GRUCell(gru_size)
                for _ in range(number_of_layers)
                ], state_is_tuple=False)

    loss = 0.0
    with tf.variable_scope("RNN"):
      outputs, state = tf.nn.dynamic_rnn(stacked_GRU, model_input,
                                         sequence_length=num_frames,
                                         dtype=tf.float32)

    aggregated_model = getattr(video_level_models,
                               FLAGS.video_level_classifier_model)
    return aggregated_model().create_model(
        model_input=state,
        vocab_size=vocab_size,
        is_training=is_training,
        **unused_params)

#################################BERT###########################################

class BertTransformer():
    def __init__(self, config, use_position=False, bert_position=False):
        self.config = config
        self.use_position = use_position
        self.bert_position = bert_position

    def forward(self, model_input, num_hidden_layers, hidden_size, is_training=True, name=''):
        if self.use_position:
            with tf.variable_scope("{}preprocess".format(name)):
                if self.bert_position:
                    # Add positional embeddings and token type embeddings, then layer
                    # normalize and perform dropout.
                    self.preprocess_output = modeling.embedding_postprocessor(
                        input_tensor=model_input,
                        use_token_type=False,
                        token_type_vocab_size=self.config.type_vocab_size,
                        token_type_embedding_name="token_type_embeddings",
                        use_position_embeddings=True,
                        position_embedding_name="position_embeddings",
                        initializer_range=self.config.initializer_range,
                        max_position_embeddings=self.config.max_position_embeddings,
                        dropout_prob=self.config.hidden_dropout_prob)
                else:
                    model_input += modeling.positional_encoding(model_input, self.config.max_position_embeddings)
                    self.preprocess_output = tf.layers.dropout(model_input, self.config.hidden_dropout_prob, training=is_training)
        else:
            self.preprocess_output = model_input

        with tf.variable_scope("{}encoder".format(name)):
            self.all_encoder_layers = modeling.transformer_model(
                #query_tensor=self.preprocess_output,#query_input,
                input_tensor=self.preprocess_output,
                attention_mask=None,
                hidden_size=hidden_size,
                num_hidden_layers=num_hidden_layers,
                num_attention_heads=self.config.num_attention_heads,
                intermediate_size=self.config.intermediate_size,
                intermediate_act_fn=modeling.get_activation(self.config.hidden_act),
                hidden_dropout_prob=self.config.hidden_dropout_prob,
                attention_probs_dropout_prob=self.config.attention_probs_dropout_prob,
                initializer_range=self.config.initializer_range,
                do_return_all_layers=True)

        return self.all_encoder_layers

    def pool(self, sequence_output, pooling_strategy):
        with tf.variable_scope("pooler"):
            # We "pool" the model by simply taking the hidden state corresponding
            # to the first token. We assume that this has been pre-trained

            if pooling_strategy == 'mean':
                pooled_output = tf.reduce_mean(sequence_output, axis=1)
            elif pooling_strategy == 'first':
                pooled_output = tf.squeeze(sequence_output[:, -1:, :], axis=1)
            elif pooling_strategy == 'attn':
                #B = sequence_output.get_shape()[0].value
                B = tf.shape(sequence_output)[0]
                #if B is None:
                #    B = FLAGS.batch_size
                D = sequence_output.get_shape()[-1].value
                query = tf.get_variable('query', shape=[1, D, 1])
                query = tf.tile(query, [B, 1, 1])
                score = tf.nn.softmax(tf.matmul(sequence_output, query) / (D ** 0.5), axis=1)  # [B,L,1]
                pooled_output = tf.reduce_sum(score * sequence_output, axis=1)  # [B, D]
            elif pooling_strategy == 'none':
                dim1, dim2 = sequence_output.shape[1], sequence_output.shape[2]
                pooled_output = tf.reshape(sequence_output, [-1, dim1*dim2])

        return pooled_output


class Bert(models.BaseModel):

    def create_model(self, model_input, vocab_size, num_frames, is_training=True, **unused_params):
        """Creates a model which uses a stack of GRUs to represent the video.
        Args:
          model_input: A 'batch_size' x 'max_frames' x 'num_features' matrix of
                       input features.
          vocab_size: The number of classes in the dataset.
          num_frames: A vector of length 'batch' which indicates the number of
               frames for each video (before padding).
        Returns:
          A dictionary with a tensor containing the probability predictions of the
          model in the 'predictions' key. The dimensions of the tensor are
          'batch_size' x 'num_classes'.
        """

        iterations = FLAGS.iterations
        random_frames = FLAGS.sample_random_frames
        no_sample = FLAGS.no_sample
        num_frames = tf.cast(tf.expand_dims(num_frames, 1), tf.float32)

        config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)
        config = copy.deepcopy(config)

        config.num_attention_heads = FLAGS.bert_attention_heads
        config.hidden_dropout_prob = FLAGS.bert_dropout_prob
        config.attention_probs_dropout_prob = FLAGS.bert_dropout_prob
        new_vfeature_size = 1024
        new_afeature_size = 128

        if is_training:
            if not no_sample:
                if random_frames:
                    model_input = utils.SampleRandomFrames(model_input, num_frames, iterations)
                else:
                    model_input = utils.SampleRandomSequence(model_input, num_frames, iterations)
        else:
            config.hidden_dropout_prob = 0.0
            config.attention_probs_dropout_prob = 0.0

        bert_transformer = BertTransformer(config, FLAGS.use_position, FLAGS.bert_position)

        num_hidden_layers = FLAGS.bert_hidden_layer
        hidden_size = new_vfeature_size + new_afeature_size

        # Transformer Encoder
        self.all_encoder_layers = bert_transformer.forward(model_input, num_hidden_layers, hidden_size, is_training)
        self.sequence_output = self.all_encoder_layers[-1]

        # Pool the sequence level representation
        self.pooled_output = bert_transformer.pool(self.sequence_output, FLAGS.pooling_strategy)

        aggregated_model = getattr(video_level_models,
                                   FLAGS.video_level_classifier_model)
        return aggregated_model().create_model(
            model_input=self.pooled_output,
            vocab_size=vocab_size,
            is_training=is_training,
            **unused_params)

class BertCross(models.BaseModel):

    def create_model(self, model_input, vocab_size, num_frames, is_training=True, **unused_params):
        """Creates a model which uses a stack of GRUs to represent the video.
        Args:
          model_input: A 'batch_size' x 'max_frames' x 'num_features' matrix of
                       input features.
          vocab_size: The number of classes in the dataset.
          num_frames: A vector of length 'batch' which indicates the number of
               frames for each video (before padding).
        Returns:
          A dictionary with a tensor containing the probability predictions of the
          model in the 'predictions' key. The dimensions of the tensor are
          'batch_size' x 'num_classes'.
        """
        iterations = FLAGS.iterations
        random_frames = FLAGS.sample_random_frames
        no_sample = FLAGS.no_sample
        num_frames = tf.cast(tf.expand_dims(num_frames, 1), tf.float32)

        config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)
        config = copy.deepcopy(config)

        config.num_attention_heads = FLAGS.bert_attention_heads
        config.hidden_dropout_prob = FLAGS.bert_dropout_prob
        config.attention_probs_dropout_prob = FLAGS.bert_dropout_prob

        new_vfeature_size = 1024
        new_afeature_size = 128

        if is_training:
            if not no_sample:
                if random_frames:
                    model_input = utils.SampleRandomFrames(model_input, num_frames,
                                                           iterations)
                else:
                    model_input = utils.SampleRandomSequence(model_input, num_frames, iterations)
        else:
            config.hidden_dropout_prob = 0.0
            config.attention_probs_dropout_prob = 0.0

        bert_transformer = BertTransformer(config, FLAGS.use_position, FLAGS.bert_position)

        # Frame Representation
        frame_num_hidden_layers = FLAGS.bert_hidden_layer - 1
        frame_hidden_size = new_vfeature_size

        frame_all_encoder_layers = bert_transformer.forward(model_input[:, :, 0:1024], frame_num_hidden_layers,
                                                           frame_hidden_size, is_training, name='frame_')
        frame_sequence_output = frame_all_encoder_layers[-1]

        # Audio Representation
        audio_num_hidden_layers = FLAGS.bert_hidden_layer - 1
        audio_hidden_size = new_afeature_size

        audio_all_encoder_layers = bert_transformer.forward(model_input[:, :, 1024:], audio_num_hidden_layers,
                                                           audio_hidden_size, is_training, name='audio_')

        audio_sequence_output = audio_all_encoder_layers[-1]

        # Cross-Modal
        all_sequence_output = tf.concat([frame_sequence_output, audio_sequence_output], 2)
        all_num_hidden_layers = 1
        all_hidden_size = new_vfeature_size + new_afeature_size
        all_encoder_layers = bert_transformer.forward(all_sequence_output, all_num_hidden_layers,
                                                           all_hidden_size, is_training, name='all_')
        self.sequence_output = all_encoder_layers[-1]

        # Pool the sequence level representation
        self.pooled_output = bert_transformer.pool(self.sequence_output, FLAGS.pooling_strategy)

        aggregated_model = getattr(video_level_models,
                                   FLAGS.video_level_classifier_model)
        return aggregated_model().create_model(
            model_input=self.pooled_output,
            vocab_size=vocab_size,
            is_training=is_training,
            **unused_params)