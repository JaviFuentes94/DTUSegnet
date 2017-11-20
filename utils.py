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
    #show_image(resized_img)

    resized_img = resized_img * 255
    #show_image(resized_img)
    return resized_img


def show_image(img):

    # print('image shape')
    # print(img.shape)
    # print('Image type')
    # print(img.dtype)
    # print('Image content')
    # print(img)
    img = gray_to_RGB(img)
    plt.imshow(img)
    plt.show()
    return img


def gray_to_RGB(img):
    with open("Data\\colors.txt") as file:
        colors = []
        for line in file.readlines():
            l = line.split()
            colors.append((l[0],l[1],l[2]))

    #img = Image.fromarray(img, "L")
    gray_img = img
    shape = gray_img.shape
    RGB_img = np.zeros((shape[0],shape[1],3),dtype=np.uint8)
    for i,row in enumerate(gray_img):
        for j,label in enumerate(row):
            label = int(label)
            RGB_img[i,j,] = colors[label]
    return RGB_img

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
    img = skimage.io.imread(".\\Data\\labels\\0006R0_f02070.png")
    show_image(img)
    #img = skimage.io.imread("./Data/0001TP_006720_L.png")
    #plt.imshow(img)
    # ny = 300
    # nx = img.shape[1] * ny / img.shape[0]
    # img = skimage.transform.resize(img, (ny, nx))
    # skimage.io.imsave("./test_data/test/output.jpg", img)

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

def variable_summaries(var):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)

if __name__ == "__main__":
    test()
