import tensorflow as tf
import time
import os
import inspect

FLAGS = tf.app.flags.FLAGS
#224,224 // 360,480 // 352,480
isCamVid=True
tf.app.flags.DEFINE_bool('isCamVid',isCamVid, 'Defines if we are using CamVid or SUNRGBD')
if isCamVid:
    tf.app.flags.DEFINE_integer('num_class',12, 'Number of classes in the dataset')
    data_dir = './CamVid/'
    labels_format = 'png'
    image_format = 'png'
else:
    tf.app.flags.DEFINE_integer('num_class',38, 'Number of classes in the dataset')
    data_dir = './SUNRGBD/'
    labels_format = 'png'
    image_format = 'jpg'

tf.app.flags.DEFINE_bool('inRAM',True, 'Defines if we are loading all the datset in RAM')
tf.app.flags.DEFINE_bool('depthIncluded',False, 'Defines if we are using depth')

tf.app.flags.DEFINE_integer('inputImX',224, 'Size of the x axis of the input image')
tf.app.flags.DEFINE_integer('inputImY',224, 'Size of the y axis of the input image')


tf.app.flags.DEFINE_bool('isTraining',True, 'Defines if we are training the model or testing it ')


#Paths to datasets
tf.app.flags.DEFINE_string('test_images_path', data_dir + 'test_images/*' + image_format, 'Path for the test images')
tf.app.flags.DEFINE_string('test_labels_path', data_dir + 'test_labels/*' + labels_format, 'Path for the test labels')

tf.app.flags.DEFINE_string('train_images_path', data_dir + 'train_images/*' + image_format, 'Path for the train images')
tf.app.flags.DEFINE_string('train_labels_path', data_dir + 'train_labels/*' + labels_format, 'Path for the train labels')

tf.app.flags.DEFINE_string('validation_images_path', data_dir + 'validation_images/*png', 'Path for the validation images')
tf.app.flags.DEFINE_string('validation_labels_path', data_dir + 'validation_labels/*png', 'Path for the validation labels')

tf.app.flags.DEFINE_string('MBF_weights_path',data_dir + 'class_weights.txt','path to the MBF weights')

tf.app.flags.DEFINE_string('test_depth_path', data_dir +'test_depth/*.png', 'Path for the test depths')
tf.app.flags.DEFINE_string('train_depth_path', data_dir +'train_depth/*.png', 'Path for the train depths')

#Tensorboard
timestr = time.strftime("%Y%m%d-%H%M%S")
tensorboard_path=os.path.join("./Tensorboard", timestr)
tf.app.flags.DEFINE_string('tensorboard_path', tensorboard_path, 'path to the Tensorboard file')

path = inspect.getfile(inspect.currentframe())
path = os.path.abspath(os.path.join(path, os.pardir))
path = os.path.join(path,"vgg16.npy")

tf.app.flags.DEFINE_string('vgg16_npy_path', path, 'path to the vgg16 weights file')