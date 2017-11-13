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
import training_ops
from Data_utils import gray_to_RGB

sess = tf.InteractiveSession()

images_ph = tf.placeholder(tf.float32, [None, 224, 224, 3])
imgIn = utils.load_image(".\\Data\\images\\0001TP_006720.png")
imgIn = imgIn.reshape((1, 224, 224, 3))

labels_ph= tf.placeholder(tf.int32, [None, 224, 224])
imgLabel = utils.load_image(".\\Data\\labels\\Gray_labels2\\0001TP_006720.png")
imgLabel = imgLabel.reshape((1, 224, 224))
#gray_to_RGB(skimage.img_as_int(imgLabel),"Label.png")

segnet = sn.SegNet(num_class = 12)
segnet.build(images_ph)

predictions = segnet.convD5_2
loss_op = training_ops.calc_loss(predictions, labels_ph)
train_op = training_ops.train_network(loss_op)


#init = tf.initialize_all_variables()
init =  tf.global_variables_initializer()
with tf.Session(config=tf.ConfigProto(gpu_options=(tf.GPUOptions(per_process_gpu_memory_fraction=0.5)))) as sess:
#with tf.device('/cpu:0'):
#    with tf.Session() as sess:
    #images = tf.placeholder("float", [2, 224, 224, 3])
    
    sess.run(init)
    for i in range(100):
        feed_dict = {images_ph: imgIn, labels_ph: imgLabel}
        fetches_train = [train_op, loss_op]
        #writer = tf.summary.FileWriter('./Tensorboard', sess.graph) #Saves the graph in the Tensorboard folder
    
        res = sess.run(fetches = fetches_train, feed_dict=feed_dict)
    
        #print("Train WTF "+res[0])
        print(res[1])

        feed_test = {images_ph: imgIn}
        img = sess.run(segnet.argmax, feed_dict=feed_test)

        gray_to_RGB(img[0])
        
