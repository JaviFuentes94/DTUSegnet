# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 22:17:52 2017

@author: Szymon
"""

import tensorflow as tf
import numpy as np
import time
import os
import sys

script_dir = os.path.dirname(__file__) #<-- absolute dir the script is in
rel_path = "tensorflow-vgg"
abs_file_path = os.path.join(script_dir, rel_path)
sys.path.insert(0, abs_file_path)

import SegNet as sn
import utils
import training_ops


timestr = time.strftime("%Y%m%d-%H%M%S")
tensorboard_path=os.path.join("./Tensorboard", timestr)
#sess = tf.InteractiveSession()

images_ph = tf.placeholder(tf.float32, [None, 224, 224, 3])
imgIn = utils.load_image_input(".\\Data\\images\\0001TP_007140.png")
imgIn = imgIn.reshape((1, 224, 224, 3))

labels_ph= tf.placeholder(tf.int32, [None, 224, 224])
imgLabel = utils.load_image_labels(".\\Data\\labels\\0001TP_007140.png")
imgLabel = imgLabel.reshape((1, 224, 224))

#utils.gray_to_RGB(imgLabel,"Label.png")

segnet = sn.SegNet(num_class = 12 )
segnet.build(images_ph)

loss_op = training_ops.calc_loss(segnet.convD5_2, labels_ph)
train_op = training_ops.train_network(loss_op)
acc_op = training_ops.calc_accuracy(segnet.argmax_layer, labels_ph)

init =  tf.global_variables_initializer()
with tf.Session(config=tf.ConfigProto(gpu_options=(tf.GPUOptions(per_process_gpu_memory_fraction=0.5)))) as sess:
#with tf.device('/cpu:0'):
#    with tf.Session() as sess:
    #images = tf.placeholder("float", [2, 224, 224, 3])
    merged = tf.summary.merge_all()
    tensorboard_writer = tf.summary.FileWriter(tensorboard_path, sess.graph) #Saves the graph in the Tensorboard folder
    sess.run(init)

    for i in range(5000):

        feed_test = {images_ph: imgIn}
        img = sess.run(segnet.argmax_layer, feed_dict=feed_test)

        #if (i%10)==0:
            #utils.show_image(img[0])

        #DEBUG
        fetches_test = [loss_op, acc_op]
        feed_test2 = {images_ph: imgIn, labels_ph: imgLabel}
        res = sess.run(fetches_test, feed_dict=feed_test2)
        print("Test loss")
        print(res[0])
        print("Test accuracy")
        print(res[1])

        #DEBUG

        feed_dict = {images_ph: imgIn, labels_ph: imgLabel}
        fetches_train = [merged, train_op, loss_op, acc_op]
        summary, _ , _, _ = sess.run(fetches = fetches_train, feed_dict=feed_dict)
        tensorboard_writer.add_summary(summary,i)
        #print("Train WTF "+res[0])
        # print("Loss")
        # print(res[1])
        # print("Accuracy")
        # print(res[2])
        #utils.gray_to_RGB(img[0])

    utils.show_image(img[0])
