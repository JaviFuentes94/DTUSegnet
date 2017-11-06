import inspect
import os

import numpy as np
import tensorflow as tf
import time

from utils import rgb2bgr

class SegNet(object):
    """Builds the SegNet model"""
    def __init__(self, segnet_npy_path=None):

        #Loads the weights from the model
        if segnet_npy_path is None:
            path = inspect.getfile(SegNet)
            path = os.path.abspath(os.path.join(path, os.pardir))
            path = os.path.join(path,"tensorflow-vgg\\vgg16.npy")
            segnet_npy_path = path
            print(path)

        self.data_dict = np.load(segnet_npy_path, encoding='latin1').item()
        print("npy file loaded")

    def build_without_decoder(self, im_rgb):
        """
        load variable from npy to build  SegNet without the decoder (i.e. VGG16 without dense layers)

        :param rgb: rgb image [batch, height, width, 3] values scaled [0, 1]
        """

        start_time = time.time()
        print("build model started")
       
        im_bgr=rgb2bgr(im_rgb)
        

        self.convE1_1 = self.conv_layer(im_bgr, "convE1_1")
        self.convE1_2 = self.conv_layer(self.convE1_1, "convE1_2")
        self.pool1 = self.max_pool(self.convE1_2, 'pool1')

        self.convE2_1 = self.conv_layer(self.pool1, "convE2_1")
        self.convE2_2 = self.conv_layer(self.convE2_1, "convE2_2")
        self.pool2 = self.max_pool(self.convE2_2, 'pool2')

        self.convE3_1 = self.conv_layer(self.pool2, "convE3_1")
        self.convE3_2 = self.conv_layer(self.convE3_1, "convE3_2")
        self.convE3_3 = self.conv_layer(self.convE3_2, "convE3_3")
        self.pool3 = self.max_pool(self.convE3_3, 'pool3')

        self.convE4_1 = self.conv_layer(self.pool3, "convE4_1")
        self.convE4_2 = self.conv_layer(self.convE4_1, "convE4_2")
        self.convE4_3 = self.conv_layer(self.convE4_2, "convE4_3")
        self.pool4 = self.max_pool(self.convE4_3, 'pool4')

        self.convE5_1 = self.conv_layer(self.pool4, "convE5_1")
        self.convE5_2 = self.conv_layer(self.convE5_1, "convE5_2")
        self.convE5_3 = self.conv_layer(self.convE5_2, "convE5_3")
        self.pool5 = self.max_pool(self.convE5_3, 'pool5')

        self.data_dict = None
        print(("build model finished: %ds" % (time.time() - start_time)))

    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def conv_layer(self, bottom, name):
        with tf.variable_scope(name):
            filt = self.get_conv_filter(name)

            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')

            conv_biases = self.get_bias(name)
            bias = tf.nn.bias_add(conv, conv_biases)

            relu = tf.nn.relu(bias)
            return relu


    def get_conv_filter(self, name):
        return tf.constant(self.data_dict[name][0], name="filter")

    def get_bias(self, name):
        return tf.constant(self.data_dict[name][1], name="biases")





