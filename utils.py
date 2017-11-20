import skimage
import skimage.io
import skimage.transform
import numpy as np
import tensorflow as tf
from scipy import misc
import matplotlib.pyplot as plt


VGG_MEAN = [103.939, 116.779, 123.68]
# synset = [l.strip() for l in open('synset.txt').readlines()]


# returns image of shape [224, 224, 3]
# [height, width, depth]
def load_image_input(path):
    # load image
    img = skimage.io.imread(path)
    #Why do we need to scale it?
    img = img / 255.0
    assert (0 <= img).all() and (img <= 1.0).all()
    # print "Original Image Shape: ", img.shape
    # we crop image from center
    short_edge = min(img.shape[:2])
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy: yy + short_edge, xx: xx + short_edge]
    # resize to 224, 224
    resized_img = skimage.transform.resize(crop_img, (224, 224))
    #show_image(resized_img)

    return resized_img

def load_image_labels(path):
    '''Load the labels image without scaling'''
    img = skimage.io.imread(path)

    short_edge = min(img.shape[:2])
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy: yy + short_edge, xx: xx + short_edge]
    # resize to 224, 224
    resized_img = skimage.transform.resize(crop_img, (224, 224))
    resized_img = skimage.transform.resize(crop_img, (224, 224))
    show_image(resized_img)

    resized_img = resized_img * 255

    return resized_img


def show_image(img):

    print('image shape')
    print(img.shape)
    print('Image type')
    print(img.dtype)
    print('Image content')
    print(img)
    plt.imshow(img)
    plt.show()
    return img


def gray_to_RGB(img, name="test.png"):
    img=skimage.img_as_int(img)[0]
    with open("Data\\class.txt") as file:
        colors = []
        for line in file.readlines():
            l = line.split()
            colors.append((l[0],l[1],l[2]))

    #img = Image.fromarray(img, "L")
    gray_img = img
    shape = gray_img.shape
    RGB_img = np.zeros((shape[0],shape[1],3))
    for i,row in enumerate(gray_img):
        for j,label in enumerate(row):
            RGB_img[i,j,] = colors[label]
    misc.imsave(name, RGB_img)
    
# returns the top1 string
def print_prob(prob, file_path):
    synset = [l.strip() for l in open(file_path).readlines()]

    # print prob
    pred = np.argsort(prob)[::-1]

    # Get top1 label
    top1 = synset[pred[0]]
    print(("Top1: ", top1, prob[pred[0]]))
    # Get top5 label
    top5 = [(synset[pred[i]], prob[pred[i]]) for i in range(5)]
    print(("Top5: ", top5))
    return top1


def load_image2(path, height=None, width=None):
    # load image
    img = skimage.io.imread(path)
    img = img / 255.0
    if height is not None and width is not None:
        ny = height
        nx = width
    elif height is not None:
        ny = height
        nx = img.shape[1] * ny / img.shape[0]
    elif width is not None:
        nx = width
        ny = img.shape[0] * nx / img.shape[1]
    else:
        ny = img.shape[0]
        nx = img.shape[1]
    return skimage.transform.resize(img, (ny, nx))


def test():
    img = skimage.io.imread("./test_data/starry_night.jpg")
    ny = 300
    nx = img.shape[1] * ny / img.shape[0]
    img = skimage.transform.resize(img, (ny, nx))
    skimage.io.imsave("./test_data/test/output.jpg", img)

def rgb2bgr(rgb):
    ''' It converts from rgb to bgr. Not sure why is it necesary tho '''
    with tf.variable_scope("rgb2bgr"):
        rgb_scaled = rgb * 255.0
        # Convert RGB to BGR
        red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=rgb_scaled)
        assert red.get_shape().as_list()[1:] == [224, 224, 1]
        assert green.get_shape().as_list()[1:] == [224, 224, 1]
        assert blue.get_shape().as_list()[1:] == [224, 224, 1]
        #It normalizes the values of the image based on the means of the vgg
        bgr = tf.concat(axis=3, values=[
            blue - VGG_MEAN[0],
            green - VGG_MEAN[1],
            red - VGG_MEAN[2],
        ])
        assert bgr.get_shape().as_list()[1:] == [224, 224, 3]
        return bgr


if __name__ == "__main__":
    test()
