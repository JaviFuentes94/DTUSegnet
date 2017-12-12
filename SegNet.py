import inspect
import os

import numpy as np
import tensorflow as tf
import time

from utils import rgb2bgr, gray_to_RGB
import PoolingProcedure as custompool

FLAGS = tf.app.flags.FLAGS

class SegNet(object):
    """Builds the SegNet model"""
    def __init__(self, num_class, segnet_npy_path=None, depthIncluded=0):

        #Loads the weights from the model
        if segnet_npy_path is None:
            path = inspect.getfile(SegNet)
            path = os.path.abspath(os.path.join(path, os.pardir))
            path = os.path.join(path,"tensorflow-vgg/vgg16.npy")
            segnet_npy_path = path
            print(path)

        self.data_dict = np.load(segnet_npy_path, encoding='latin1').item()
        print("npy file loaded")

        self.num_class = num_class
        self.depthIncluded = depthIncluded

        self.encoderbuilt = False
        self.decoderbuilt = False

        self.pool = custompool.PoolingProcedure()

    def build(self, im_rgb, phase):
        self.phase = phase
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
        im_bgr= tf.nn.local_response_normalization(im_bgr)

        self.convE1_1 = self.conv_layer(im_bgr, "conv1_1")
        self.convE1_2 = self.conv_layer(self.convE1_1, "conv1_2")
        self.pool1, self.argmax1 = self.pool.max_pool(self.convE1_2, 'pool1')

        self.convE2_1 = self.conv_layer(self.pool1, "conv2_1")
        self.convE2_2 = self.conv_layer(self.convE2_1, "conv2_2")
        self.pool2, self.argmax2 = self.pool.max_pool(self.convE2_2, 'pool2')

        self.convE3_1 = self.conv_layer(self.pool2, "conv3_1")
        self.convE3_2 = self.conv_layer(self.convE3_1, "conv3_2")
        self.convE3_3 = self.conv_layer(self.convE3_2, "conv3_3")
        self.pool3, self.argmax3 = self.pool.max_pool(self.convE3_3, 'pool3')

        self.convE4_1 = self.conv_layer(self.pool3, "conv4_1")
        self.convE4_2 = self.conv_layer(self.convE4_1, "conv4_2")
        self.convE4_3 = self.conv_layer(self.convE4_2, "conv4_3")
        self.pool4, self.argmax4 = self.pool.max_pool(self.convE4_3, 'pool4')

        self.convE5_1 = self.conv_layer(self.pool4, "conv5_1")
        self.convE5_2 = self.conv_layer(self.convE5_1, "conv5_2")
        self.convE5_3 = self.conv_layer(self.convE5_2, "conv5_3")
        self.pool5, self.argmax5 = self.pool.max_pool(self.convE5_3, 'pool5')

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

        self.upsample1 = self.pool.unpool(self.pool5,'pool5',"upsample_1", self.argmax5)
        self.convD1_1 = self.conv_layer_decoder(self.upsample1, "convD1_1", 512)
        self.convD1_2 = self.conv_layer_decoder(self.convD1_1, "convD1_2", 512)
        self.convD1_3 = self.conv_layer_decoder(self.convD1_2, "convD1_3", 512)

        self.upsample2 = self.pool.unpool(self.convD1_3,'pool4', "upsample_2", self.argmax4)
        self.convD2_1 = self.conv_layer_decoder(self.upsample2, "convD2_1", 512)
        self.convD2_2 = self.conv_layer_decoder(self.convD2_1, "convD2_2", 512)
        self.convD2_3 = self.conv_layer_decoder(self.convD2_2, "convD2_3", 256)

        self.upsample3 = self.pool.unpool(self.convD2_3,'pool3', "upsample_3", self.argmax3)
        self.convD3_1 = self.conv_layer_decoder(self.upsample3, "convD3_1", 256)
        self.convD3_2 = self.conv_layer_decoder(self.convD3_1, "convD3_2", 256)
        self.convD3_3 = self.conv_layer_decoder(self.convD3_2, "convD3_3", 128)

        self.upsample4 = self.pool.unpool(self.convD3_3,'pool2', "upsample_4", self.argmax2)
        self.convD4_1 = self.conv_layer_decoder(self.upsample4, "convD4_1", 128)
        self.convD4_2 = self.conv_layer_decoder(self.convD4_1, "convD4_2", 64)

        self.upsample5 = self.pool.unpool(self.convD4_2,'pool1', "upsample_5", self.argmax1)
        self.convD5_1 = self.conv_layer_decoder(self.upsample5, "convD5_1", 64)
        self.convD5_2 = self.conv_layer_decoder(self.convD5_1, "convD5_2", self.num_class)

        #Calculate softmax - this might be not used as we use softmax_cross_entropy_with_logits
        #for loss calculation
        #this for each pixel goes through num_class classes
        with tf.name_scope("softmax"):
            self.softmax_layer = tf.nn.softmax(self.convD5_2)
        #print("softmax_layer", self.softmax_layer.shape)
        #Create an image with classicifactions might be not neccessary as well
        #this for each pixel returns the class with biggest probability
        with tf.name_scope("argmax"):
            self.argmax_layer = tf.argmax(self.softmax_layer, 3)
            img_summary = tf.summary.image('Result_image', tf.expand_dims(tf.to_float(self.argmax_layer), 3))
            #print("armax_layer", self.argmax_layer.shape)
            #img = gray_to_RGB(tf.to_float(self.argmax_layer[0]))
            #img_summary = tf.summary.image('Result image',img)
        print(("build Decoder finished: %ds" % (time.time() - start_time)))

    def conv_layer(self, bottom, name):
        with tf.variable_scope(name):

            filt = self.get_conv_filter(name)
            conv_biases = self.get_bias(name)

            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')
            bias = tf.nn.bias_add(conv, conv_biases)
            relu = tf.nn.relu(self.batch_norm_layer(bias))

            #print(name)
            #print(conv.shape)
            #print(filt.shape)

            return relu

    def get_conv_filter(self, name):

        filt = self.data_dict[name][0]

        # If this is a network working on input with width channel, calc 1st layer wieghts
        # As the average of RGB weights, then multiply by 32 to change the scale from 0-255 to 0-8m
        if (name == "conv1_1") and (self.depthIncluded == 1):
            averaged = np.average(filt, axis=2)
            averaged = averaged.reshape(3,3,1,64)
            averaged = averaged * 32
            filt = np.append(filt, averaged, 2)

        return tf.Variable(filt, name="filter")

    def get_bias(self, name):
        return tf.Variable(self.data_dict[name][1], name="biases")

    def conv_layer_decoder(self, bottom, name, size_out):
        with tf.variable_scope(name):
            #Added weight initialization as described in the paper
            conv = tf.layers.conv2d(
                inputs=bottom,
                filters=size_out,
                kernel_size=[3, 3],
                padding="same",
                use_bias=True,
                kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                bias_initializer=tf.zeros_initializer(),
                activation=None,
                name = name)
            #print(name)
            #print(conv.shape)

            conv = tf.nn.relu(self.batch_norm_layer(conv))

            return conv

    def batch_norm_layer(self, BNinput):
        #return input
        return tf.contrib.layers.batch_norm(BNinput, is_training = self.phase)
