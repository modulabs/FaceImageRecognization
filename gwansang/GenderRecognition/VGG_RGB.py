
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
from math import ceil
import sys

import numpy as np
import tensorflow as tf

VGG_MEAN = [103.939, 116.779, 123.68]


class VGG_RGB:
    
    weight_list = []
    
    def __init__(self, vgg16_npy_path=None):
        if vgg16_npy_path is None:
            path = sys.modules[self.__class__.__module__].__file__
            # print path
            path = os.path.abspath(os.path.join(path, os.pardir))
           
            # print path
            vgg16_npy_path = os.path.join(path, "vgg16.npy")
            logging.info("Load npy file from '%s'.", vgg16_npy_path)
        if not os.path.isfile(vgg16_npy_path):
            logging.error(("File '%s' not found. Download it from "
                           "https://dl.dropboxusercontent.com/u/"
                           "50333326/vgg16.npy"), vgg16_npy_path)
            sys.exit(1)

        self.data_dict = np.load(vgg16_npy_path, encoding='latin1').item()
        print("npy file loaded")
        self.wd = 5e-4
        self.learning_rate = 1e-6
        print("construction FCN32VGG_RGB")

    def he_init_uniform(self, n_inputs, variable_name):
        init_range = tf.sqrt(6.0 / (n_inputs))
        return tf.random_uniform_initializer(-init_range, init_range, name=variable_name)

    def xavier_init_uniform(self, n_inputs, n_outputs, variable_name):
        init_range = tf.sqrt(6.0 / (n_inputs + n_outputs))
        return tf.random_uniform_initializer(-init_range, init_range, name=variable_name)

    def he_init_normal(self, shape, n_inputs, variable_name):
        init_range = tf.sqrt(2.0 / (n_inputs))
        return tf.get_variable(name=variable_name, shape=shape,
                               initializer=tf.random_normal_initializer(stddev=init_range))

    def xavier_init_normal(self, shape, n_inputs, n_outputs, variable_name):
        init_range = tf.sqrt(2.0 / (n_inputs + n_outputs))
        return tf.get_variable(name=variable_name, shape=shape,
                               initializer=tf.random_normal_initializer(stddev=init_range))

    def init_normal(self, shape, variable_name):
        return tf.get_variable(name=variable_name, shape=shape,
                               initializer=tf.random_normal_initializer(stddev=0.01))

    def init_bias(self, shape, variable_name):
        return tf.get_variable(name=variable_name, shape=shape, initializer=tf.constant_initializer(0.0))

    def _variable_with_weight_decay(self, shape, stddev, wd):
        """Helper to create an initialized Variable with weight decay.

        Note that the Variable is initialized with a truncated normal
        distribution.
        A weight decay is added only if one is specified.

        Args:
          name: name of the variable
          shape: list of ints
          stddev: standard deviation of a truncated Gaussian
          wd: add L2Loss weight decay multiplied by this float. If None, weight
              decay is not added for this Variable.

        Returns:
          Variable Tensor
        """

        initializer = tf.truncated_normal_initializer(stddev=stddev)
        var = tf.get_variable('weights', shape=shape, initializer=initializer)
        if wd and (not tf.get_variable_scope().reuse):
            weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
            tf.add_to_collection('losses', weight_decay)
        return var

    def _conv_layer(self, input, w, b, name, relu=True):
        with tf.variable_scope(name) as scope:
            conv = tf.nn.conv2d(input, w, [1, 1, 1, 1], padding="SAME")
            bias = tf.nn.bias_add(conv, b)

            if relu:
                relu = tf.nn.relu(bias)

            self._activation_summary(relu)
            return relu

    def _score_layer(self, input, conv_biases, name, num_classes):
        with tf.variable_scope(name) as scope:
            in_features = input.get_shape()[3].value
            shape = [1, 1, in_features, num_classes]

            num_input = in_features
            stddev = (2 / num_input) ** 0.5

            w_decay = self.wd
            weights = self._variable_with_weight_decay(shape, stddev, w_decay)
            conv = tf.nn.conv2d(input, weights, [1, 1, 1, 1], padding="SAME")

            bias = tf.nn.bias_add(conv, conv_biases)

            self._activation_summary(bias)

            return bias

    def _fc_layer(self, input, w, b, name, relu=True, debug=False):
        with tf.variable_scope(name) as scope:
            # shape = input.get_shape().as_list()

            conv = tf.nn.conv2d(input, w, [1, 1, 1, 1], padding='SAME')
            bias = tf.nn.bias_add(conv, b)

            if relu:
                bias = tf.nn.relu(bias)
            self._activation_summary(bias)

            if debug:
                bias = tf.Print(bias, [tf.shape(bias)],
                                message='Shape of %s' % name,
                                summarize=4, first_n=1)
            return bias

    def _fc_layer2(self, input,w,b, name,relu=True):
        with tf.variable_scope(name) as scope:
            shape = input.get_shape().as_list()
            dim = 1
            for d in shape[1:]:
                dim *= d
            x = tf.reshape(input, [-1, dim])

            # Fully connected layer. Note that the '+' operation automatically
            # broadcasts the biases.
            fc = tf.nn.bias_add(tf.matmul(x, w), b)
            if relu:
                fc = tf.nn.relu(fc)

            return fc

    def _max_pool(self, input, name, debug=False):
        pool = tf.nn.max_pool(input, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                              padding='SAME', name=name)

        if debug:
            pool = tf.Print(pool, [tf.shape(pool)],
                            message='Shape of %s' % name,
                            summarize=4, first_n=1)
        return pool


    def loss2(self, logits, labels, num_classes):

        logits = tf.reshape(logits, [-1, num_classes])
        # epsilon = tf.constant(value=1e-4)
        # logits = logits + epsilon

        labels = tf.to_int64(tf.reshape(labels, [-1]))
        print('shape of logits: %s' % str(logits.get_shape()))
        print('shape of labels: %s' % str(labels.get_shape()))

        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels, name='Cross_Entropy')
        cross_entropy_mean = tf.reduce_mean(cross_entropy, name='xentropy_mean')
        tf.add_to_collection('losses', cross_entropy_mean)

        loss = tf.add_n(tf.get_collection('losses'), name='total_loss')
        return loss

    def loss(self, logits, labels, num_classes, head=None):
        """Calculate the loss from the logits and the labels.

        Args:
          logits: tensor, float - [batch_size, width, height, num_classes].
              Use vgg_fcn.up as logits.
          labels: Labels tensor, int32 - [batch_size, width, height, num_classes].
              The ground truth of your data.
          head: numpy array - [num_classes]
              Weighting the loss of each class
              Optional: Prioritize some classes

        Returns:
          loss: Loss tensor of type float.
        """
        with tf.name_scope('loss'):
            logits = tf.reshape(logits, [-1, num_classes])
            epsilon = tf.constant(value=1e-4)
            logits = logits + epsilon
            labels = tf.to_float(tf.reshape(labels, (-1, num_classes)))
            # labels = tf.reshape(labels, [-1, num_classes])

            # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, labels))


            softmax = tf.nn.softmax(logits)

            if head is not None:
                cross_entropy = -tf.reduce_sum(tf.mul(labels * tf.log(softmax),
                                                      head), reduction_indices=[1])
            else:
                cross_entropy = -tf.reduce_sum(
                    labels * tf.log(softmax), reduction_indices=[1])

            cross_entropy_mean = tf.reduce_mean(cross_entropy,
                                                name='xentropy_mean')
            tf.add_to_collection('losses', cross_entropy_mean)

            loss = tf.add_n(tf.get_collection('losses'), name='total_loss')

        return loss

    def get_conv_filter(self, name):
        with tf.variable_scope(name) as scope:
            init = tf.constant_initializer(value=self.data_dict[name][0],
                                           dtype=tf.float32)
            shape = self.data_dict[name][0].shape
            print('Layer name: %s' % name)
            print('Layer shape: %s' % str(shape))
            var = tf.get_variable(name="filter", initializer=init, shape=shape)
            if not tf.get_variable_scope().reuse:
                weight_decay = tf.mul(tf.nn.l2_loss(var), self.wd,
                                      name='weight_loss')
                tf.add_to_collection('losses', weight_decay)
            return var

    def get_bias(self, name, num_classes=None):
        with tf.variable_scope(name) as scope:
            bias_wights = self.data_dict[name][1]
            shape = self.data_dict[name][1].shape
            if name == 'fc8':
                bias_wights = self._bias_reshape(bias_wights, shape[0],
                                                 num_classes)
                shape = [num_classes]
            init = tf.constant_initializer(value=bias_wights,
                                           dtype=tf.float32)
            return tf.get_variable(name="biases", initializer=init, shape=shape)

    def _bias_reshape(self, bweight, num_orig, num_new):
        """ Build bias weights for filter produces with `_summary_reshape`
        """
        n_averaged_elements = num_orig // num_new
        avg_bweight = np.zeros(num_new)
        for i in range(0, num_orig, n_averaged_elements):
            start_idx = i
            end_idx = start_idx + n_averaged_elements
            avg_idx = start_idx // n_averaged_elements
            if avg_idx == num_new:
                break
            avg_bweight[avg_idx] = np.mean(bweight[start_idx:end_idx])
        return avg_bweight

    def get_fc_weight_reshape(self, name, shape, num_classes=None):
        with tf.variable_scope(name) as scope:
            print('Layer name: %s' % name)
            print('Layer shape: %s' % shape)
            weights = self.data_dict[name][0]
            weights = weights.reshape(shape)
            if num_classes is not None:
                weights = self._summary_reshape(weights, shape, num_new=num_classes)
            init = tf.constant_initializer(value=weights,
                                           dtype=tf.float32)
            return tf.get_variable(name="weights", initializer=init, shape=shape)

    def get_fc_weight_reshape2(self, name, num_classes=None):
        with tf.variable_scope(name) as scope:
            weights = self.data_dict[name][0]
            shape = np.shape(weights)
            print('Layer name: %s' % name)
            #print('Layer shape: %s' % shape)

            if num_classes is not None:
                weights = self._summary_reshape(weights, shape, num_new=num_classes)
            init = tf.constant_initializer(value=weights,
                                           dtype=tf.float32)
            return tf.get_variable(name="weights", initializer=init, shape=shape)

    def _summary_reshape(self, fweight, shape, num_new):
        """ Produce weights for a reduced fully-connected layer.
        FC8 of VGG produces 1000 classes. Most semantic segmentation
        task require much less classes. This reshapes the original weights
        to be used in a fully-convolutional layer which produces num_new
        classes. To archive this the average (mean) of n adjanced classes is
        taken.
        Consider reordering fweight, to perserve semantic meaning of the
        weights.
        Args:
          fweight: original weights
          shape: shape of the desired fully-convolutional layer
          num_new: number of new classes
        Returns:
          Filter weights for `num_new` classes.
        """
        num_orig = shape[1]
        shape[1] = num_new
        assert (num_new < num_orig)
        n_averaged_elements = num_orig // num_new
        avg_fweight = np.zeros(shape)
        for i in range(0, num_orig, n_averaged_elements):
            start_idx = i
            end_idx = start_idx + n_averaged_elements
            avg_idx = start_idx // n_averaged_elements
            if avg_idx == num_new:
                break
            avg_fweight[:, avg_idx] = np.mean(
                fweight[:, start_idx:end_idx], axis=1)
        return avg_fweight

    def build(self, X, Y, train, scope, num_classes=2, random_init_fc8=False, debug=False):
        """
        ----------
        img: image batch tensor
        train: bool
            Whether to build train or inference graph
        num_classes: int
            How many classes should be predicted (by fc8)
        random_init_fc8 : bool
            Whether to initialize fc8 layer randomly.
            Finetuning is required in this case.
        debug: bool
            Whether to print additional Debug Information.
        """
        with tf.name_scope('Processing'):

            red, green, blue = tf.split(3, 3, X)
            # assert red.get_shape().as_list()[1:] == [224, 224, 1]
            # assert green.get_shape().as_list()[1:] == [224, 224, 1]
            # assert blue.get_shape().as_list()[1:] == [224, 224, 1]
            bgr = tf.concat(3, [
                blue - VGG_MEAN[0],
                green - VGG_MEAN[1],
                red - VGG_MEAN[2],
            ])

            if debug:
                bgr = tf.Print(bgr, [tf.shape(bgr)],
                               message='Shape of input image: ',
                               summarize=4, first_n=1)

        with tf.variable_scope(scope):
            w1_1 = self.get_conv_filter("conv1_1")
            b1_1 = self.get_bias("conv1_1")
            w1_2 = self.get_conv_filter("conv1_2")
            b1_2 = self.get_bias("conv1_2")

            self.conv1_1 = self._conv_layer(bgr, w1_1, b1_1, "conv1_1")
            self.conv1_2 = self._conv_layer(self.conv1_1, w1_2, b1_2, "conv1_2")
            self.pool1 = self._max_pool(self.conv1_2, 'pool1', debug)

            w2_1 = self.get_conv_filter("conv2_1")
            b2_1 = self.get_bias("conv2_1")
            w2_2 = self.get_conv_filter("conv2_2")
            b2_2 = self.get_bias("conv2_2")

            self.conv2_1 = self._conv_layer(self.pool1, w2_1, b2_1, "conv2_1")
            self.conv2_2 = self._conv_layer(self.conv2_1, w2_2, b2_2, "conv2_2")
            self.pool2 = self._max_pool(self.conv2_2, 'pool2', debug)

            w3_1 = self.get_conv_filter("conv3_1")
            b3_1 = self.get_bias("conv3_1")
            w3_2 = self.get_conv_filter("conv3_2")
            b3_2 = self.get_bias("conv3_2")
            w3_3 = self.get_conv_filter("conv3_3")
            b3_3 = self.get_bias("conv3_3")

            self.conv3_1 = self._conv_layer(self.pool2, w3_1, b3_1, "conv3_1")
            self.conv3_2 = self._conv_layer(self.conv3_1, w3_2, b3_2, "conv3_2")
            self.conv3_3 = self._conv_layer(self.conv3_2, w3_3, b3_3, "conv3_3")
            self.pool3 = self._max_pool(self.conv3_3, 'pool3', debug)

            w4_1 = self.get_conv_filter("conv4_1")
            b4_1 = self.get_bias("conv4_1")
            w4_2 = self.get_conv_filter("conv4_2")
            b4_2 = self.get_bias("conv4_2")
            w4_3 = self.get_conv_filter("conv4_3")
            b4_3 = self.get_bias("conv4_3")

            self.conv4_1 = self._conv_layer(self.pool3, w4_1, b4_1, "conv4_1")
            self.conv4_2 = self._conv_layer(self.conv4_1, w4_2, b4_2, "conv4_2")
            self.conv4_3 = self._conv_layer(self.conv4_2, w4_3, b4_3, "conv4_3")
            self.pool4 = self._max_pool(self.conv4_3, 'pool4', debug)

            w5_1 = self.get_conv_filter("conv5_1")
            b5_1 = self.get_bias("conv5_1")
            w5_2 = self.get_conv_filter("conv5_2")
            b5_2 = self.get_bias("conv5_2")
            w5_3 = self.get_conv_filter("conv5_3")
            b5_3 = self.get_bias("conv5_3")

            self.conv5_1 = self._conv_layer(self.pool4, w5_1, b5_1, "conv5_1")
            self.conv5_2 = self._conv_layer(self.conv5_1, w5_2, b5_2, "conv5_2")
            self.conv5_3 = self._conv_layer(self.conv5_2, w5_3, b5_3, "conv5_3")
            self.pool5 = self._max_pool(self.conv5_3, 'pool5', debug)

            fc6_w = self.get_fc_weight_reshape("fc6",[7*7*512,4096])
            fc6_b = self.get_bias("fc6")
            fc7_w = self.get_fc_weight_reshape2("fc7")
            fc7_b = self.get_bias("fc7")
            fc8_w = self.get_fc_weight_reshape("fc8",[4096,1000], num_classes=num_classes)
            fc8_b = self.get_bias("fc8", num_classes=num_classes)

            self.fc6 = self._fc_layer2(self.pool5, fc6_w, fc6_b, "fc6")

            if train == True:
                self.fc6 = tf.nn.dropout(self.fc6, 0.5)

            self.fc7 = self._fc_layer2(self.fc6, fc7_w, fc7_b, "fc7")
            if train == True:
                self.fc7 = tf.nn.dropout(self.fc7, 0.5)

            if random_init_fc8:
                self.score_fr = self._score_layer(self.fc7, fc8_b, "score_fr", num_classes)
            else:
                self.score_fr = self._fc_layer2(self.fc7, fc8_w, fc8_b, "score_fr", relu=False)

            self.pred = tf.argmax(self.score_fr, dimension=1)

            lossValue = self.loss2(self.score_fr, Y, num_classes)

            return lossValue

    def _activation_summary(self, x):
        """Helper to create summaries for activations.

        Creates a summary that provides a histogram of activations.
        Creates a summary that measure the sparsity of activations.

        Args:
          x: Tensor
        Returns:
          nothing
        """
        # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
        # session. This helps the clarity of presentation on tensorboard.
        tensor_name = x.op.name
        # tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
        tf.histogram_summary(tensor_name + '/activations', x)
        tf.scalar_summary(tensor_name + '/sparsity', tf.nn.zero_fraction(x))