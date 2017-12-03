import glob
import random
import utils
import numpy as np
import tensorflow as tf

class batch:
    def __init__(self,FLAGS):
        images_filenames = glob.glob(FLAGS.images_path)
        labels_filenames = glob.glob(FLAGS.labels_path)
        self.val_size = 100
        rand_idx =  list(range(0,len(images_filenames)))
        random.shuffle(rand_idx)
        self.train_images_filenames = [images_filenames[i] for i in rand_idx[self.val_size+1:]]
        self.train_labels_filenames = [labels_filenames[i] for i in rand_idx[self.val_size+1:]]
        self.val_images_filenames = [images_filenames[i] for i in rand_idx[:self.val_size]]
        self.val_labels_filenames = [labels_filenames[i] for i in rand_idx[:self.val_size]]
        self.train_rand_idx = list(range(0,len(self.train_images_filenames)))

    def get_train(self,size):
        b_im = np.zeros((size, 224, 224, 3))
        b_lab = np.zeros((size, 224, 224))
        random.shuffle(self.train_rand_idx)
        for i in range(size):
            idx = self.train_rand_idx[i]
            b_im[i] = utils.load_image_input(self.train_images_filenames[idx])
            b_lab[i] = utils.load_image_labels(self.train_labels_filenames[idx])
        return b_im, b_lab

    def get_validation(self):

        v_im = np.zeros((self.val_size, 224, 224, 3))
        v_lab = np.zeros((self.val_size, 224, 224))
        for idx in range(self.val_size):
            v_im[idx] = utils.load_image_input(self.val_images_filenames[idx])
            v_lab[idx] = utils.load_image_labels(self.val_labels_filenames[idx])
        return v_im, v_lab
