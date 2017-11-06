# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 22:17:52 2017

@author: Szymon
"""

import tensorflow as tf
import numpy as np

import os
import sys

script_dir = os.path.dirname(__file__) #<-- absolute dir the script is in
rel_path = "tensorflow-vgg"
abs_file_path = os.path.join(script_dir, rel_path)
sys.path.insert(0, abs_file_path)

import SegNet as sn
import utils

images = tf.placeholder("float", [None, 224, 224, 3])

segnet = sn.SegNet()
segnet.build_without_decoder(images)

print("\nConvolution 1_1 output")
print(segnet.convE1_1.shape)
print("Convolution 1_2 output")
print(segnet.convE1_2.shape)
print("Pooling 1 output")
print(segnet.pool1.shape)

print("\nConvolution 2_1 output")
print(segnet.convE2_1.shape)
print("Convolution 2_2 output")
print(segnet.convE2_2.shape)
print("Pooling 2 output")
print(segnet.pool2.shape)

print("\nConvolution 3_1 output")
print(segnet.convE3_1.shape)
print("Convolution 3_2 output")
print(segnet.convE3_2.shape)
print("Convolution 3_3 output")
print(segnet.convE3_3.shape)
print("Pooling 3 output")
print(segnet.pool3.shape)

print("\nConvolution 4_1 output")
print(segnet.convE4_1.shape)
print("Convolution 4_2 output")
print(segnet.convE4_2.shape)
print("Convolution 4_3 output")
print(segnet.convE4_3.shape)
print("Pooling 4 output")
print(segnet.pool4.shape)

print("\nConvolution 5_1 output")
print(segnet.convE5_1.shape)
print("Convolution 5_2 output")
print(segnet.convE5_2.shape)
print("Convolution 5_3 output")
print(segnet.convE5_3.shape)
print("Pooling 5 output")
print(segnet.pool5.shape)

print("Pooling 5 indices")
print(segnet.pool5_indices.shape)




#Load images
#img1 = utils.load_image("./tensorflow-vgg/test_data/tiger.jpeg")
#img2 = utils.load_image("./tensorflow-vgg/test_data/puzzle.jpeg")

#Reshape the images
#batch1 = img1.reshape((1, 224, 224, 3))
#batch2 = img2.reshape((1, 224, 224, 3))
#
#batch = np.concatenate((batch1, batch2), 0)

#with tf.Session(config=tf.ConfigProto(gpu_options=(tf.GPUOptions(per_process_gpu_memory_fraction=0.7)))) as sess:
##with tf.device('/cpu:0'):
##    with tf.Session() as sess:
#    images = tf.placeholder("float", [2, 224, 224, 3])
#    feed_dict = {images: batch}
#
#    vgg = vgg16.Vgg16()
#    with tf.name_scope("content_vgg"):
#        vgg.build(images)
#
#    prob = sess.run(vgg.prob, feed_dict=feed_dict)
#    print(prob)
#    utils.print_prob(prob[0], './synset.txt')
#    utils.print_prob(prob[1], './synset.txt')
