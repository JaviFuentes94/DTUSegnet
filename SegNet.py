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

        self.encoderbuilt = False
        self.decoderbuilt = False

    def build(self, im_rgb):
        self.build_encoder(im_rgb)
        self.build_decoder()

    def build_encoder(self, im_rgb):
        """
        load variable from npy to build  SegNet without the decoder (i.e. VGG16 without dense layers)

        :param rgb: rgb image [batch, height, width, 3] values scaled [0, 1]
        """

        start_time = time.time()
        print("build model started")
       
        im_bgr=rgb2bgr(im_rgb)
        
        self.convE1_1 = self.conv_layer(im_bgr, "conv1_1")
        self.convE1_2 = self.conv_layer(self.convE1_1, "conv1_2")
        self.pool1, self.pool1_indices = self.max_pool(self.convE1_2, 'pool1')

        self.convE2_1 = self.conv_layer(self.pool1, "conv2_1")
        self.convE2_2 = self.conv_layer(self.convE2_1, "conv2_2")
        self.pool2, self.pool2_indices = self.max_pool(self.convE2_2, 'pool2')

        self.convE3_1 = self.conv_layer(self.pool2, "conv3_1")
        self.convE3_2 = self.conv_layer(self.convE3_1, "conv3_2")
        self.convE3_3 = self.conv_layer(self.convE3_2, "conv3_3")
        self.pool3, self.pool3_indices = self.max_pool(self.convE3_3, 'pool3')

        self.convE4_1 = self.conv_layer(self.pool3, "conv4_1")
        self.convE4_2 = self.conv_layer(self.convE4_1, "conv4_2")
        self.convE4_3 = self.conv_layer(self.convE4_2, "conv4_3")
        self.pool4, self.pool4_indices = self.max_pool(self.convE4_3, 'pool4')

        self.convE5_1 = self.conv_layer(self.pool4, "conv5_1")
        self.convE5_2 = self.conv_layer(self.convE5_1, "conv5_2")
        self.convE5_3 = self.conv_layer(self.convE5_2, "conv5_3")
        self.pool5, self.pool5_indices = self.max_pool(self.convE5_3, 'pool5')

        self.data_dict = None
        print(("build model finished: %ds" % (time.time() - start_time)))

        self.encoderbuilt = True

    def build_decoder(self):
        """
        load variable from npy to build encoder of SegNet 
        """

        if self.encoderbuilt == False:
            print("Error: Encoder has to be built prior to the decoder")
            return

        start_time = time.time()

        print("build decoder started")
        
        self.upsample1 = self.upsample_layer(self.pool5, "upsample_1")
        self.convD1_1 = self.conv_layer_decoder(self.upsample1, "convD1_1", 512)
        self.convD1_2 = self.conv_layer_decoder(self.convD1_1, "convD1_2", 512)
        self.convD1_3 = self.conv_layer_decoder(self.convD1_2, "convD1_3", 512)

        self.upsample2 = self.upsample_layer(self.convD1_3, "upsample_2")
        self.convD2_1 = self.conv_layer_decoder(self.upsample2, "convD2_1", 512)
        self.convD2_2 = self.conv_layer_decoder(self.convD2_1, "convD2_2", 512)
        self.convD2_3 = self.conv_layer_decoder(self.convD2_2, "convD2_3", 256)

        self.upsample3 = self.upsample_layer(self.convD2_3, "upsample_3")
        self.convD3_1 = self.conv_layer_decoder(self.upsample3, "convD3_1", 256)
        self.convD3_2 = self.conv_layer_decoder(self.convD3_1, "convD3_2", 256)
        self.convD3_3 = self.conv_layer_decoder(self.convD3_2, "convD3_3", 128)

        self.upsample4 = self.upsample_layer(self.convD3_3, "upsample_4")
        self.convD4_1 = self.conv_layer_decoder(self.upsample4, "convD4_1", 128)
        self.convD4_2 = self.conv_layer_decoder(self.convD4_1, "convD4_2", 64)

        self.upsample5 = self.upsample_layer(self.convD4_2, "upsample_5")
        self.convD5_1 = self.conv_layer_decoder(self.upsample5, "convD5_1", 64)
        self.convD5_2 = self.conv_layer_decoder(self.convD5_1, "convD5_2", 3)

        #Calculate softmax - this might be not used as we use softmax_cross_entropy_with_logits
        #for loss calculation
        softmax = tf.nn.softmax(self.convD5_2)

        #Create an image with classicifactions might be not neccessary as well
        argmax = tf.argmax(softmax, 3)

        print(("build Decoder finished: %ds" % (time.time() - start_time)))

    def max_pool(self, bottom, name):
        return tf.nn.max_pool_with_argmax(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def conv_layer(self, bottom, name):
        with tf.variable_scope(name):
            filt = self.get_conv_filter(name)

            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')

            print(name)
            print(filt.shape)

            conv_biases = self.get_bias(name)
            bias = tf.nn.bias_add(conv, conv_biases)

            relu = tf.nn.relu(bias)
            return relu

    def get_conv_filter(self, name):
        return tf.constant(self.data_dict[name][0], name="filter")

    def get_bias(self, name):
        return tf.constant(self.data_dict[name][1], name="biases")

    def upsample_layer(self, bottom, name):
        print("To be done")
        return bottom
 
    def conv_layer_decoder(self, bottom, name, size_out):
        conv = tf.layers.conv2d(
            inputs=bottom,
            filters=size_out,
            kernel_size=[3, 3],
            padding="same",
            use_bias=True,
            bias_initializer=tf.zeros_initializer(),
            activation=tf.nn.relu)
        return conv