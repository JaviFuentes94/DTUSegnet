from __future__ import division
from scipy import misc
from collections import Counter
from multiprocessing import Pool
import statistics
import math
import glob
import numpy as np

def class_count(image_paths):
    class_counter = Counter()
    for image_path in image_paths:
        print(image_path)
        img = misc.imread(image_path)
        for row in img:
            for label in row:
                class_counter[str(label)] += 1
    return class_counter

def get_img_paths():
    image_paths = glob.glob('./labels/*.png')
    n = int(math.ceil(len(image_paths)/4))
    chunks = [image_paths[i:i+n] for i in range(0, len(image_paths), n)]
    for chunk in chunks:
        yield chunk

def get_class_count():
    final_counter = Counter()
    p = Pool(4)
    Counter_list = p.imap_unordered(class_count, get_img_paths(), chunksize=1)
    for counter in Counter_list:
        final_counter += counter
    with open("labels/class_count.txt","w") as file:
        for label in final_counter:
            file.write(label + ' ' +str(final_counter[label]) + '\n')
    return final_counter

def get_class_frequencies_and_weights():
    ## uncomment if "class_count.txt" doesn't exist yet
    #get_class_count()
    with open("labels/class_count.txt") as class_count:
        with open("labels/class_frequencies.txt","w") as class_frequencies:
            with open("labels/class_weights.txt","w") as class_weights:
                total =0
                labels = []
                counts = []
                for line in class_count.readlines():
                    list = line.split()
                    labels.append((list[0],int(list[1])))
                    counts.append(int(list[1]))
                    total += int(list[1])
                freq = [i/total for i in counts]
                median = statistics.median(freq)
                weights = [median/f for f in freq]
                #class_frequencies.write(median)
                for i,c in enumerate(labels):
                    class_frequencies.write(c[0]+ ' ' + str(freq[i])+'\n')
                    class_weights.write(c[0]+ ' ' + str(weights[i])+'\n')
def gray_to_RGB(image_path):
    with open("class.txt") as file:
        colors = []
        for line in file.readlines():
            l = line.split()
            colors.append((l[0],l[1],l[2]))
    gray_img = misc.imread(image_path)
    shape = gray_img.shape
    RGB_img = np.zeros((shape[0],shape[1],3))
    for i,row in enumerate(gray_img):
        for j,label in enumerate(row):
            RGB_img[i,j,] = colors[label]
    misc.imsave("test.png", RGB_img)

gray_to_RGB("./labels/0001TP_006690.png")
#get_class_frequencies_and_weights()
