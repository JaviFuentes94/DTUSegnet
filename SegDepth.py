import inspect
import os

import numpy as np
import tensorflow as tf
import time

from utils import rgb2bgr, gray_to_RGB
import layers

FLAGS = tf.app.flags.FLAGS

class SegNet(object):
    """Builds the SegNet model"""
    def __init__(self, im_rgbd, phase):
        """
        Builds the SegNet model including depth information

        Args:
            im_rgbd: rgb+depth image placeholder [batch, height, width, 4] values scaled [0, 1]
            num_class: number of segmentation classes
            vgg16_npy_path: path to the weights of the vgg16 model
        """

        #Loads the weights from the model

        self.data_dict = np.load(FLAGS.vgg16_npy_path, encoding='latin1').item()
        print("Vgg16 weights loaded")

        self.num_class = FLAGS.num_class
        self.phase = phase

        start_time = time.time()
        print("build SegNet started")
        im_bgr = rgb2bgr(im_rgbd[:,:,:,0:3])
        im_bgrd = tf.concat([im_bgr, tf.expand_dims(im_rgbd[:,:,:,3],3)],axis=3)
        im_bgrd= tf.nn.local_response_normalization(im_bgrd)

        #ENCODER

        self.convE1_1 = layers.first_depth_conv_layer(im_bgrd, "conv1_1", self.data_dict, phase)
        self.convE1_2 = layers.conv_layer(self.convE1_1, "conv1_2", self.data_dict, phase)
        self.pool1, self.argmax1 = layers.max_pool(self.convE1_2, 'pool1')

        self.convE2_1 = layers.conv_layer(self.pool1, "conv2_1", self.data_dict, phase)
        self.convE2_2 = layers.conv_layer(self.convE2_1, "conv2_2", self.data_dict, phase)
        self.pool2, self.argmax2 = layers.max_pool(self.convE2_2, 'pool2')

        self.convE3_1 = layers.conv_layer(self.pool2, "conv3_1", self.data_dict, phase)
        self.convE3_2 = layers.conv_layer(self.convE3_1, "conv3_2", self.data_dict, phase)
        self.convE3_3 = layers.conv_layer(self.convE3_2, "conv3_3", self.data_dict, phase)
        self.pool3, self.argmax3 = layers.max_pool(self.convE3_3, 'pool3')

        self.convE4_1 = layers.conv_layer(self.pool3, "conv4_1", self.data_dict, phase)
        self.convE4_2 = layers.conv_layer(self.convE4_1, "conv4_2", self.data_dict, phase)
        self.convE4_3 = layers.conv_layer(self.convE4_2, "conv4_3", self.data_dict, phase)
        self.pool4, self.argmax4 = layers.max_pool(self.convE4_3, 'pool4')

        self.convE5_1 = layers.conv_layer(self.pool4, "conv5_1", self.data_dict, phase)
        self.convE5_2 = layers.conv_layer(self.convE5_1, "conv5_2", self.data_dict, phase)
        self.convE5_3 = layers.conv_layer(self.convE5_2, "conv5_3", self.data_dict, phase)
        self.pool5, self.argmax5 = layers.max_pool(self.convE5_3, 'pool5')

        #DECODER

        self.upsample1 = layers.unpool(self.pool5,'pool5',"upsample_1", self.argmax5)
        self.convD1_1 = layers.conv_layer_decoder(self.upsample1, "convD1_1", 512, phase)
        self.convD1_2 = layers.conv_layer_decoder(self.convD1_1, "convD1_2", 512, phase)
        self.convD1_3 = layers.conv_layer_decoder(self.convD1_2, "convD1_3", 512, phase)

        self.upsample2 = layers.unpool(self.convD1_3,'pool4', "upsample_2", self.argmax4)
        self.convD2_1 = layers.conv_layer_decoder(self.upsample2, "convD2_1", 512, phase)
        self.convD2_2 = layers.conv_layer_decoder(self.convD2_1, "convD2_2", 512, phase)
        self.convD2_3 = layers.conv_layer_decoder(self.convD2_2, "convD2_3", 256, phase)

        self.upsample3 = layers.unpool(self.convD2_3,'pool3', "upsample_3", self.argmax3)
        self.convD3_1 = layers.conv_layer_decoder(self.upsample3, "convD3_1", 256, phase)
        self.convD3_2 = layers.conv_layer_decoder(self.convD3_1, "convD3_2", 256, phase)
        self.convD3_3 = layers.conv_layer_decoder(self.convD3_2, "convD3_3", 128, phase)

        self.upsample4 = layers.unpool(self.convD3_3,'pool2', "upsample_4", self.argmax2)
        self.convD4_1 = layers.conv_layer_decoder(self.upsample4, "convD4_1", 128, phase)
        self.convD4_2 = layers.conv_layer_decoder(self.convD4_1, "convD4_2", 64, phase)

        self.upsample5 = layers.unpool(self.convD4_2,'pool1', "upsample_5", self.argmax1)
        self.convD5_1 = layers.conv_layer_decoder(self.upsample5, "convD5_1", 64, phase)
        self.convD5_2 = layers.conv_layer_decoder(self.convD5_1, "convD5_2", self.num_class, phase)

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
        print(("build SegNet finished: %ds" % (time.time() - start_time)))

    def load_model(self,saver,sess):
        modelPath= FLAGS.model_path
        saver.restore(sess,modelPath)
