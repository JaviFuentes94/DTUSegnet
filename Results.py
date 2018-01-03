import tensorflow as tf
import numpy as np
import time
import os
import sys
import argparse

import SegNet as sn
import utils
import batch
import SegNetFlags
#os.environ["CUDA_VISIBLE_DEVICES"] = '1' ## <--- TO SELECT HPC GPU

### SUNRGBD ### 

#Reset
tf.reset_default_graph()
### DEFINING THE FLAGS ###
SegNetFlags.define_FLAGS(False,False,False)
FLAGS = tf.app.flags.FLAGS

### DEFINING THE PLACEHOLDERS ###
images_ph = tf.placeholder(tf.float32, [None, FLAGS.inputImX, FLAGS.inputImY, FLAGS.n_channels])
labels_ph= tf.placeholder(tf.int32, [None, FLAGS.inputImX, FLAGS.inputImY])
phase_ph = tf.placeholder(tf.bool, name='phase')

### BUILDING THE NETWORK ###
segnet = sn.SegNet(im_rgb = images_ph, phase=phase_ph)
### LOAD DATASET ###
batch = batch.batch()

saver = tf.train.Saver()
with tf.Session(config=tf.ConfigProto(gpu_options=(tf.GPUOptions(per_process_gpu_memory_fraction=0.75)))) as sess:
    utils.show_comparison("Train", saver, sess, batch, segnet, images_ph, labels_ph, phase_ph)

with tf.Session(config=tf.ConfigProto(gpu_options=(tf.GPUOptions(per_process_gpu_memory_fraction=0.75)))) as sess:
    utils.show_comparison("Train", saver, sess, batch, segnet, images_ph, labels_ph, phase_ph)
