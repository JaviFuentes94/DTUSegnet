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

import vgg16
import utils

images = tf.placeholder("float", [None, 224, 224, 3])

vgg = vgg16.Vgg16()
vgg.build(images)

#Load images
img1 = utils.load_image("./tensorflow-vgg/test_data/tiger.jpeg")
img2 = utils.load_image("./tensorflow-vgg/test_data/puzzle.jpeg")

#Reshape the images
batch1 = img1.reshape((1, 224, 224, 3))
batch2 = img2.reshape((1, 224, 224, 3))

batch = np.concatenate((batch1, batch2), 0)

with tf.Session(config=tf.ConfigProto(gpu_options=(tf.GPUOptions(per_process_gpu_memory_fraction=0.7)))) as sess:
#with tf.device('/cpu:0'):
#    with tf.Session() as sess:
    images = tf.placeholder("float", [2, 224, 224, 3])
    feed_dict = {images: batch}

    vgg = vgg16.Vgg16()
    with tf.name_scope("content_vgg"):
        vgg.build(images)

    prob = sess.run(vgg.prob, feed_dict=feed_dict)
    print(prob)
    utils.print_prob(prob[0], './synset.txt')
    utils.print_prob(prob[1], './synset.txt')
