from __future__ import division
from scipy import misc
from collections import Counter
from multiprocessing import Pool
import statistics
import math
import glob
import numpy as np


def build_RGB_to_label_dict():
    RGB_to_label_dict = {}
    with open("labels/class.txt") as class_file:
        for c in class_file.readlines():
            l = c.split()
            RGB = l[0]+ ' ' +l[1]+ ' ' +l[2]
            RGB_to_label_dict[RGB] = l[3]
    return RGB_to_label_dict
def build_empty_counter():
    class_counter = Counter()
    with open("labels/class.txt") as class_file:
        for c in class_file.readlines():
            l = c.split()
            RGB = l[0]+ ' ' +l[1]+ ' ' +l[2]
            class_counter[RGB] = 0
    return class_counter
def class_count(image_paths):
    class_counter = build_empty_counter()
    for image_path in image_paths:
        print(image_path)
        img = misc.imread(image_path)
        for row in img:
            for pix in row:
                RGB =  str(pix[0])+ ' ' + str(pix[1])+ ' ' + str(pix[2])
                if RGB in class_counter:
                    class_counter[RGB] += 1
    return class_counter

def get_RGB_img_paths():
    image_paths = glob.glob('./labels/RGB_labels/*.png')
    n = int(math.ceil(len(image_paths)/4))
    chunks = [image_paths[i:i+n] for i in range(0, len(image_paths), n)]
    for chunk in chunks:
        yield chunk

def get_class_count():
    final_counter = Counter()
    p = Pool(4)
    Counter_list = p.imap_unordered(class_count, get_RGB_img_paths(), chunksize=1)
    for counter in Counter_list:
        final_counter += counter
    RGB_to_label_dict = build_RGB_to_label_dict()
    with open("labels/class_count.txt","w") as file:
        for RGB in RGB_to_label_dict:
            file.write(RGB_to_label_dict[RGB]+ ' ' +str(final_counter[RGB]) + '\n')
    return final_counter

def get_class_frequencies():
    ## uncomment if "class_count.txt" doesn't exist yet
    get_class_count()
    with open("labels/class_count.txt") as class_count:
        with open("labels/class_frequencies.txt","w") as class_frequencies:
            total =0
            labels = []
            counts = []
            for line in class_count.readlines():
                list = line.split()
                labels.append((list[0],int(list[1])))
                counts.append(int(list[1]))
                total += int(list[1])
            freq = [i/total for i in counts]
            class_frequencies.write(str(statistics.median(freq)))
            for c in labels:
                class_frequencies.write('\n'+c[0]+ ' ' + str(c[1]/total))

def build_RGB_to_gray_dict():
    RGB_to_gray_dict = {}
    cur_class = 0
    with open("labels/class.txt") as class_file:
        for c in class_file.readlines():
            l = c.split()
            RGB = l[0]+ ' ' +l[1]+ ' ' +l[2]
            RGB_to_gray_dict[RGB] = cur_class
            cur_class += 1
    return RGB_to_gray_dict

def RGB_to_gray(image_paths):
    for image_path in image_paths:
        RGB_to_gray_dict = build_RGB_to_gray_dict()
        filename = image_path[20:]
        RGB_img = misc.imread(image_path)
        shape = RGB_img.shape
        gray_img = np.zeros((shape[0],shape[1]))
        for i,row in enumerate(RGB_img):
            for j,pix in enumerate(row):
                RGB =  str(pix[0])+ ' ' + str(pix[1])+ ' ' + str(pix[2])
                if RGB in RGB_to_gray_dict:
                    gray_img[i,j] = RGB_to_gray_dict[RGB]
        misc.imsave('./labels/Gray_labels/'+filename, gray_img)
    return gray_img
p = Pool(4)
result = p.imap_unordered(RGB_to_gray, get_RGB_img_paths(), chunksize=1)
for r in result:
    i = result
#RGB_to_gray_dict = build_RGB_to_gray_dict()
#RGB_to_gray("./labels/RGB_labels/0001TP_006690_L.png",RGB_to_gray_dict)
