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

sess = tf.InteractiveSession()

images = tf.placeholder("float", [None, 224, 224, 3])

segnet = sn.SegNet(num_class = 10)
segnet.build(images)

with tf.Session(config=tf.ConfigProto(gpu_options=(tf.GPUOptions(per_process_gpu_memory_fraction=0.5)))) as sess:
#with tf.device('/cpu:0'):
#    with tf.Session() as sess:    
    #images = tf.placeholder("float", [2, 224, 224, 3])

    writer = tf.summary.FileWriter('./Tensorboard', sess.graph) #Saves the graph in the Tensorboard folder 

    #feed_dict = {images: batch}

    segnet = sn.SegNet()
    with tf.name_scope("SegNet"):
        segnet.build(images)
    