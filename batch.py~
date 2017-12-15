import glob
import random
import utils
import numpy as np
import tensorflow as tf
import time


FLAGS = tf.app.flags.FLAGS

class batch:
    def __init__(self,FLAGS,isCamVid = 1,depthIncluded = 0,inRAM = 1):
        start_time = time.time()
        if isCamVid:
            self.train_images_filenames = glob.glob(FLAGS.CamVid_train_images_path)
            self.train_labels_filenames = glob.glob(FLAGS.CamVid_train_labels_path)
            self.test_images_filenames = glob.glob(FLAGS.CamVid_test_images_path)
            self.test_labels_filenames = glob.glob(FLAGS.CamVid_test_labels_path)
        else:
            self.train_images_filenames = glob.glob(FLAGS.SUNRGBD_train_images_path)
            self.train_labels_filenames = glob.glob(FLAGS.SUNRGBD_train_labels_path)
            self.train_depths_filenames = glob.glob(FLAGS.SUNRGBD_train_depth_path)
            self.test_images_filenames = glob.glob(FLAGS.SUNRGBD_test_images_path)
            self.test_labels_filenames = glob.glob(FLAGS.SUNRGBD_test_labels_path)
            self.test_depths_filenames = glob.glob(FLAGS.SUNRGBD_test_depth_path)

        self.test_size = len(self.test_images_filenames)
        self.train_size = len(self.train_images_filenames)
        self.train_rand_idx = list(range(0,self.train_size))

        self.epoch = 0
        self.current_batch_train=0
        self.current_batch_test=0
        self.depthIncluded  = depthIncluded
        self.inRAM = inRAM
        if inRAM:
            self.train_images = [utils.load_image_input(i) for i in self.train_images_filenames]
            self.train_labels = [utils.load_image_labels(i) for i in self.train_labels_filenames]
            self.test_images = np.asarray([utils.load_image_input(i) for i in self.test_images_filenames])
            self.test_labels = np.asarray([utils.load_image_labels(i) for i in self.test_labels_filenames])

            #print(self.train_images[0].shape,self.train_labels[0].shape,self.test_images[0].shape,self.test_labels[0].shape)
            if depthIncluded:
                self.train_depths = [utils.load_image_input(i) for i in self.train_depths_filenames]
                self.test_depths = np.asarray([utils.load_image_input(i) for i in self.test_depths_filenames])
        print(("build batch finished: %ds" % (time.time() - start_time)))
        print("train size: ",self.train_size,"  test size: ",self.test_size)

    def get_train(self,size):
        if self.current_batch_train + size >= self.train_size:
            self.epoch+=1
            self.current_batch_train=0
            random.shuffle(self.train_rand_idx)
        b_im = np.zeros((size, FLAGS.inputImX, FLAGS.inputImY, 3))
        b_lab = np.zeros((size, FLAGS.inputImX, FLAGS.inputImY))
        if self.inRAM:
            for i in range(size):
                idx = self.train_rand_idx[self.current_batch+i]
                b_im[i] = self.train_images[idx]
                b_lab[i] = self.train_labels[idx]
        else:
            for i in range(size):
                idx = self.train_rand_idx[self.current_batch_train+i]
                b_im[i] = utils.load_image_input(self.train_images_filenames[idx])
                b_lab[i] = utils.load_image_labels(self.train_labels_filenames[idx])
        self.current_batch_train += size
        return b_im, b_lab

    def get_epoch(self):
        return self.epoch

    def get_test(self):
        if self.inRAM:
            return self.test_images, self.test_labels
        else:
            size = 5
            b_im = np.zeros((size, FLAGS.inputImX, FLAGS.inputImY, 3))
            b_lab = np.zeros((size, FLAGS.inputImX, FLAGS.inputImY))
            for i in range(size):
                b_im[i] = utils.load_image_input(self.test_images_filenames[i])
                b_lab[i] = utils.load_image_labels(self.test_labels_filenames[i])
            return b_im, b_lab
