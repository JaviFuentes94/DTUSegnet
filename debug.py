# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 22:17:52 2017

@author: Szymon
"""

import tensorflow as tf
import numpy as np
import os
import sys
import SegNet as sn
import utils


script_dir = os.path.dirname(__file__) #<-- absolute dir the script is in
rel_path = "tensorflow-vgg/vgg16.npy"
abs_file_path = os.path.join(script_dir, rel_path)

data_dict = np.load(abs_file_path, encoding='latin1').item()

print(data_dict["conv1_1"][0].shape)
print(data_dict["conv1_1"][1].shape)
print(type(data_dict["conv1_1"]))
print(len(data_dict["conv1_1"]))


print("Start")

toaverage  = data_dict["conv1_1"][0]
averaged = np.average(toaverage, axis=2)



print(toaverage.shape)
r = toaverage[0][0][0][0]
g = toaverage[0][0][1][0]
b = toaverage[0][0][2][0]
print(r)
print(g)
print(b)
print( (r+b+g)/3 )

averaged = averaged.reshape(3,3,1,64)
appended = np.append(toaverage, averaged, 2)


print(appended.shape)
r = appended[0][0][0][0]
g = appended[0][0][1][0]
b = appended[0][0][2][0]
a = appended[0][0][3][0]
print(r)
print(g)
print(b)
print(a)

# for output in range(0,63):
# 	for x in range(0,2):
# 		for y in range(0,2):
			

print(type(data_dict))
print(type(toaverage))
print(type(data_dict["conv1_1"][0]))
print(type(data_dict["conv1_1"][1]))








# with tf.Session(config=tf.ConfigProto(gpu_options=(tf.GPUOptions(per_process_gpu_memory_fraction=0.7)))) as sess:
#     images = tf.placeholder("float", [2, 224, 224, 3])
#     feed_dict = {images: batch}

#     vgg = vgg16.Vgg16()
#     with tf.name_scope("content_vgg"):
#         vgg.build(images)

#     prob = sess.run(vgg.prob, feed_dict=feed_dict)
#     print(prob)
#     utils.print_prob(prob[0], './synset.txt')
#     utils.print_prob(prob[1], './synset.txt')