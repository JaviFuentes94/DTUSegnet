import tensorflow as tf
import numpy as np
import time
import os
import sys

modelPath= "./Models/model.ckpt-105"

import SegNet as sn
import utils
import training_ops
import batch

import matplotlib.pyplot as plt
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
tf.app.flags.DEFINE_string('CamVid_validation_images_path', './CamVid/validation_images/*png', 'Path for the train images')
tf.app.flags.DEFINE_string('CamVid_test_labels_path', './CamVid/test_labels/*png', 'Path for the test labels')
tf.app.flags.DEFINE_string('CamVid_train_labels_path', './CamVid/train_labels/*png', 'Path for the train labels')
tf.app.flags.DEFINE_string('CamVid_validation_labels_path', './CamVid/validation_labels/*png', 'Path for the test labels')
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
#MFB_loss_op = training_ops.calc_MFB_loss(segnet.convD5_2, labels_ph, num_class,FLAGS)
MFB_loss_op = loss_op
train_op = training_ops.train_network(loss_op)
G_acc_op, C_acc_opp, G_accs_op, C_accs_opp  = training_ops.calc_accuracy(segnet.argmax_layer, labels_ph,num_class, phase_ph)

### BUILDING BATCH and TEST ###
batch = batch.batch(FLAGS,isCamVid,depthIncluded,inRAM)
test_im, test_lab = batch.get_test()
#print("test_im",test_im, " shape: ",test_im.shape)
test_len = test_im.shape[0]
#print("test_len",test_len)
fetches_test = [G_accs_op, C_accs_opp, loss_op]
chunk_size = 10
list_feed_test = []
list_sizes = []
for s in range(0,test_len,chunk_size):
    if s+chunk_size-1 >= test_len:
        e = test_im.shape[0]-1
    else:
        e = s+chunk_size-1
    list_feed_test.append({images_ph: test_im[s:e], labels_ph: test_lab[s:e], phase_ph: 0})
    list_sizes.append(e-s+1)

### BUILDING TRAINING BATCHES ###
train_im, train_lab = batch.get_train_all()
#print("train_im",train_im,"shape", train_im.shape )
train_len = train_im.shape[0]
#print("train_im",train_im)
fetches_train = [G_accs_op, C_accs_opp, loss_op]
chunk_size = 10
list_feed_train = []
list_sizes_train = []
for s in range(0,train_len,chunk_size):
    if s+chunk_size-1 >= train_len:
        e = train_im.shape[0]-1
    else:
        e = s+chunk_size-1
    list_feed_train.append({images_ph: train_im[s:e], labels_ph: train_lab[s:e], phase_ph: 0})
    list_sizes_train.append(e-s+1)

## BUILD VALIDATION SET ##
fetches_val = [G_accs_op, C_accs_opp, loss_op]
val_images, val_labels = batch.get_validation()

val_len = val_images.shape[0]
print("Validation images", val_len, "Validation labels", val_labels.shape[0])
chunk_size = 10
list_feed_val = []
list_sizes_val = []
for s in range(0,val_len,chunk_size):
    if s+chunk_size-1 >= val_len:
        e = val_images.shape[0]-1
    else:
        e = s+chunk_size-1
    list_feed_val.append({images_ph: val_images[s:e], labels_ph: val_labels[s:e], phase_ph: 0})
    list_sizes_val.append(e-s+1)

#init =  tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

saver = tf.train.Saver()
print("running the train loop")
with tf.Session(config=tf.ConfigProto(gpu_options=(tf.GPUOptions(per_process_gpu_memory_fraction=0.75)))) as sess:
   
    saver.restore(sess, modelPath)

    #Take some images and show them
    n_images=2
    im_visual, label_visual = batch.get_visualization_images(nImages=n_images)
    fetches_visualization = [segnet.argmax_layer]
    feed_dict = {images_ph: im_visual, labels_ph: label_visual, phase_ph: 0}
    im_result = sess.run(fetches_visualization, feed_dict=feed_dict)
    im_result_arr=np.array(im_result).squeeze()
    print("im_result. shape", np.array(im_result).shape, "First shape ")

    f, axarr = plt.subplots(n_images, 3, sharex='col', sharey='row')
    axarr[0,0].set_title('Test images')
    axarr[0,1].set_title('Ground truth labels')
    axarr[0,2].set_title('Our implementation')
    for i in range(0,n_images):
        axarr[i,0].imshow(im_visual[i])
        axarr[i,0].axis('off')
        axarr[i,1].imshow(label_visual[i])
        axarr[i,1].axis('off')
        axarr[i,2].imshow(im_result_arr[i])
        axarr[i,2].axis('off')
    plt.show()

    #Check training loss and acc (It takes a lot of time)
    C_acc_train = []
    G_acc_train = []
    loss_train=[]
    for feed_train in list_feed_train:
        res = sess.run(fetches_train, feed_dict=feed_train)
        G_acc_train.append(res[0])
        C_acc_train.append(res[1])
        #loss_train.append(res[2])
    G_acc_t = tf.reduce_mean(tf.concat(G_acc_train,axis = 0)).eval()
    C_acc_t = tf.reduce_mean(tf.concat(C_acc_train,axis = 0)).eval()
    #loss_train_mean = tf.reduce_mean(tf.concat(loss_train,axis = 0)).eval()
    print("TRAIN G_acc", G_acc_t, "C_acc", C_acc_t)#, "Loss", train_loss_mean)


    #Check test accuracies
    C_acc_test = []
    G_acc_test = []
    loss_test = []
    for feed_test in list_feed_test:
        res = sess.run(fetches_test, feed_dict=feed_test)
        G_acc_test.append(res[0])
        C_acc_test.append(res[1])
        #loss_test.append(res[2])
    G_acc = tf.reduce_mean(tf.concat(G_acc_test,axis = 0)).eval()
    C_acc = tf.reduce_mean(tf.concat(C_acc_test,axis = 0)).eval()
    #loss_test_mean = tf.reduce_mean(tf.concat(loss_test,axis = 0)).eval()
    
    print("TEST G_acc", G_acc, "C_acc", C_acc)#, "Loss", loss_test_mean)

    #Check validation
    C_acc_val = []
    G_acc_val = []
    loss_val = []
    for feed_val in list_feed_val:
        res = sess.run(fetches_val, feed_dict=feed_val)
        G_acc_val.append(res[0])
        C_acc_val.append(res[1])
        #loss_val.append(res[2])
    G_acc_val = tf.reduce_mean(tf.concat(G_acc_val,axis = 0)).eval()
    C_acc_val = tf.reduce_mean(tf.concat(C_acc_val,axis = 0)).eval()
    #loss_val_mean = tf.reduce_mean(tf.concat(loss_val,axis = 0)).eval()
    
    print("VALIDATION G_acc", G_acc_val, "C_acc", C_acc_val)#, "Loss", loss_val_mean)

