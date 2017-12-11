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



print("Start filt")

# filt  = data_dict["conv1_1"][0]

# print(filt.shape)
# r = filt[0][0][0][0]
# g = filt[0][0][1][0]
# b = filt[0][0][2][0]
# print(r)
# print(g)
# print(b)
# print( (r+b+g)/3 )

# averaged = np.average(filt, axis=2)
# averaged = averaged.reshape(3,3,1,64)
# averaged = averaged * 32
# filt = np.append(filt, averaged, 2)

# print(filt.shape)
# r = filt[0][0][0][0]
# g = filt[0][0][1][0]
# b = filt[0][0][2][0]
# a = filt[0][0][3][0]
# print(r)
# print(g)
# print(b)
# print(a)

print("Start bias")
bias  = data_dict["conv1_1"][1]
print(bias.shape)
print(bias)
