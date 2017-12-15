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

#os.environ["CUDA_VISIBLE_DEVICES"] = '3' ## <--- TO SELECT HPC GPU
script_dir = os.path.dirname(__file__) #<-- absolute dir the script is in
rel_path = "tensorflow-vgg"
abs_file_path = os.path.join(script_dir, rel_path)
sys.path.insert(0, abs_file_path)

import SegNet as sn
import utils
import training_ops
import batch
### DEFINING THE RUNNING OPTIONS ###
isCamVid = 1
if isCamVid:
    num_class = 12
else:
    num_class = 37
depthIncluded  = 0
inRAM = 1

#Reset
tf.reset_default_graph()
### DEFINING THE FLAGS ###
FLAGS = tf.app.flags.FLAGS
#224,224 // 360,480
tf.app.flags.DEFINE_integer('inputImX',352, 'Size of the x axis of the input image')
tf.app.flags.DEFINE_integer('inputImY',480, 'Size of the y axis of the input image')
tf.app.flags.DEFINE_bool('isTraining',True, 'Size of the y axis of the input image')

#tf.app.flags.DEFINE_string('CamVid_test_images_path', '.\\CamVid\\test_images\\*.png', 'Path for the test images')
#tf.app.flags.DEFINE_string('CamVid_train_images_path', '.\\CamVid\\train_images\\*.png', 'Path for the train images')
#tf.app.flags.DEFINE_string('CamVid_test_labels_path', '.\\CamVid\\test_labels\\*.png', 'Path for the test labels')
#tf.app.flags.DEFINE_string('CamVid_train_labels_path', '.\\CamVid\\train_labels\\*.png', 'Path for the train labels')
#tf.app.flags.DEFINE_string('MBF_weights_path','Data\\labels\\class_weights.txt','path to the MBF weights')

#tf.app.flags.DEFINE_string('SUNRGBD_test_images_path', '.\\SUNRGBD\\test_images\\*.jpg', 'Path for the test images')
#tf.app.flags.DEFINE_string('SUNRGBD_train_images_path', '.\\SUNRGBD\\train_images\\*.jpg', 'Path for the train images')
#tf.app.flags.DEFINE_string('SUNRGBD_test_labels_path', '.\\SUNRGBD\\test_labels\\*.png', 'Path for the test labels')
#tf.app.flags.DEFINE_string('SUNRGBD_train_labels_path', '.\\SUNRGBD\\train_labels\\*.png', 'Path for the train labels')
#tf.app.flags.DEFINE_string('SUNRGBD_test_depth_path', '.\\SUNRGBD\\test_depth\\*.png', 'Path for the test depths')
#tf.app.flags.DEFINE_string('SUNRGBD_train_depth_path', '.\\SUNRGBD\\train_depth\\*.png', 'Path for the train depths')

tf.app.flags.DEFINE_string('CamVid_test_images_path', './CamVid/test_images/*png', 'Path for the test images')
tf.app.flags.DEFINE_string('CamVid_train_images_path', './CamVid/train_images/*png', 'Path for the train images')
tf.app.flags.DEFINE_string('CamVid_test_labels_path', './CamVid/test_labels/*png', 'Path for the test labels')
tf.app.flags.DEFINE_string('CamVid_train_labels_path', './CamVid/train_labels/*png', 'Path for the train labels')
tf.app.flags.DEFINE_string('MBF_weights_path','CamVid/class_weights.txt','path to the MBF weights')

tf.app.flags.DEFINE_string('SUNRGBD_test_images_path', './SUNRGBD/test_images/*.jpg', 'Path for the test images')
tf.app.flags.DEFINE_string('SUNRGBD_train_images_path', './SUNRGBD/train_images/*.jpg', 'Path for the train images')
tf.app.flags.DEFINE_string('SUNRGBD_test_labels_path', './SUNRGBD/test_labels/*.png', 'Path for the test labels')
tf.app.flags.DEFINE_string('SUNRGBD_train_labels_path', './SUNRGBD/train_labels/*.png', 'Path for the train labels')
tf.app.flags.DEFINE_string('SUNRGBD_test_depth_path', './SUNRGBD/test_depth/*.png', 'Path for the test depths')
tf.app.flags.DEFINE_string('SUNRGBD_train_depth_path', './SUNRGBD/train_depth/*.png', 'Path for the train depths')
timestr = time.strftime("%Y%m%d-%H%M%S")

tensorboard_path=os.path.join("./Tensorboard", timestr)

### DEFINING THE PLACEHOLDERS ###
images_ph = tf.placeholder(tf.float32, [None, FLAGS.inputImX, FLAGS.inputImY, 3])
labels_ph= tf.placeholder(tf.int32, [None, FLAGS.inputImX, FLAGS.inputImY])
phase_ph = tf.placeholder(tf.bool, name='phase')


### BUILDING THE NETWORK ###
segnet = sn.SegNet(num_class = num_class, depthIncluded = depthIncluded)
segnet.build(images_ph, phase_ph)

### DEFINING THE OPERATIONS ###
loss_op = training_ops.calc_loss(segnet.convD5_2, labels_ph, num_class)
MFB_loss_op = training_ops.calc_MFB_loss(segnet.convD5_2, labels_ph, num_class,FLAGS)
#MFB_loss_op = loss_op
train_op = training_ops.train_network(loss_op)
G_acc_op, C_acc_opp, G_accs_op, C_accs_opp  = training_ops.calc_accuracy(segnet.argmax_layer, labels_ph,num_class, phase_ph)

### BUILDING BATCH and TEST ###
batch = batch.batch(FLAGS,isCamVid,depthIncluded,inRAM)
test_im, test_lab = batch.get_test()
test_len = test_im.shape[0]
fetches_test = [G_accs_op, C_accs_opp]
chunk_size = 10
list_feed_test = []
for s in range(0,test_len,chunk_size):
    if s+chunk_size-1 >= test_len:
        e = test_im.shape[0]
    else:
        e = s+chunk_size
    list_feed_test.append({images_ph: test_im[s:e], labels_ph: test_lab[s:e], phase_ph: 0})

init =  tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
saver = tf.train.Saver(tf.global_variables())
print("running the train loop")
with tf.Session(config=tf.ConfigProto(gpu_options=(tf.GPUOptions(per_process_gpu_memory_fraction=0.9)))) as sess:
    merged = tf.summary.merge_all()
    tensorboard_writer = tf.summary.FileWriter(tensorboard_path, sess.graph) #Saves the graph in the Tensorboard folder
    sess.run(init)
    current_epoch = 0

    print("number of trainable parameters : ",np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))
    for i in range(500000):

        imgIn, imgLabel = batch.get_train(5)

        feed_dict = {images_ph: imgIn, labels_ph: imgLabel, phase_ph: 1}
        fetches_train = [segnet.argmax_layer, merged, train_op, loss_op, MFB_loss_op]
        img, summary, _ , loss, MFB_loss = sess.run(fetches = fetches_train, feed_dict=feed_dict)
        tensorboard_writer.add_summary(summary,i)

        if (i%10)==0:
            #utils.show_image(img[0])
            #utils.show_image(imgIn[0])
            #utils.show_image(imgLabel[0])
            G_acc, C_acc = sess.run(fetches = [G_acc_op, C_acc_opp], feed_dict=feed_dict)
            print(i,"	Train loss",loss,"	MFB loss", MFB_loss,"	G_acc", G_acc, "	C_acc", C_acc)

        if batch.get_epoch() > current_epoch:
            print("new epoch")
            current_epoch= batch.get_epoch()
            C_acc_test = []
            G_acc_test = []
            for feed_test in list_feed_test:
                res = sess.run(fetches_test, feed_dict=feed_test)
                G_acc_test.append(res[0])
                C_acc_test.append(res[1])
            G_acc_test = tf.concat(G_acc_test,axis = 0)
            C_acc_test = tf.concat(C_acc_test,axis = 0)
            G_acc = tf.reduce_mean(G_acc_test).eval()
            C_acc = tf.reduce_mean(C_acc_test).eval()
            print("EPOCHS: ", current_epoch,"	Test G_acc", G_acc, "C_acc", C_acc)
            #After 5 epoch save the model
            if (current_epoch%5)==0:
                save_path = saver.save(sess, "./Models/model.ckpt", global_step = current_epoch)
                print("Model saved in file: %s" % save_path)


    utils.show_image(img[0])
