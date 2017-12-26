import tensorflow as tf
FLAGS = tf.app.flags.FLAGS

"""Implements all the functions for the different layers of SegNet """
ksize=[1, 2, 2, 1]

def max_pool(input_layer, name):
    """
       Regular max pooling that also saves the argmax indices for the unpooling.
       Args:
           input_layer:   previous layer
           name:     name of the layer
       Return:
           pool:    max pool layer
           argmax:  argmax argument out of the pool layer
    """
    with tf.variable_scope(name):
        layer, argmax = tf.nn.max_pool_with_argmax(input_layer, ksize, strides=[1, 2, 2, 1], padding='SAME', name=name)
        #DEBUG
        #print(name)
        #print(layer.shape)
        return layer, argmax

def unpool(input_layer, encoder_layer_name, decoder_layer_name, argmax):
    """
       Unpooling layer after a previous max pooling layer.
       Args:
           input_layer:             previous layer to be unpooled
           encoder_layer_name:      the name from the max_pooling layer that we want to take the indeces from
           decoder_layer_name:      name of the layer
           argmax:                  argmax parameter out of max_pool_with_argmax
       Return:
           unpool:    unpooling tensor

       Inspired by the implementation discussed in https://github.com/tensorflow/tensorflow/issues/2169
    """
    with tf.variable_scope(decoder_layer_name):

        input_shape = tf.shape(input_layer)
        output_shape = [input_shape[0], input_shape[1] * ksize[1], input_shape[2]*ksize[2], input_shape[3]]

        flat_input_size = tf.reduce_prod(input_shape)
        flat_output_shape = [output_shape[0], output_shape[1] * output_shape[2] * output_shape[3]]

        ind = argmax

        pool_ = tf.reshape(input_layer, [flat_input_size])
        batch_range = tf.reshape(tf.range(tf.cast(output_shape[0], tf.int64), dtype=ind.dtype),
                                          shape=[input_shape[0], 1, 1, 1])

        b = tf.ones_like(ind) * batch_range
        b1 = tf.reshape(b, [flat_input_size, 1])
        ind_ = tf.reshape(ind, [flat_input_size, 1])
        ind_ = tf.concat([b1, ind_], 1)

        ret = tf.scatter_nd(ind_, pool_, shape=tf.cast(flat_output_shape, tf.int64))
        ret = tf.reshape(ret, output_shape)

        set_input_shape = input_layer.get_shape()
        set_output_shape = [set_input_shape[0],set_input_shape[1] * ksize[1], set_input_shape[2] * ksize[2], set_input_shape[3]]
        ret.set_shape(set_output_shape)

        #DEBUG
        #print(decoder_layer_name)
        #print(ret.shape)
        return ret

def conv_layer(input_layer, name, data_dict, phase):
    """
       Regular convolutional layer with the weights out of vgg16.
       Args:
           input_layer:   previous layer
           name:     name of the layer
           data_dict:        dictionaty with the vgg16 weights
       Return:
           relu:    output of the layer
    """
    with tf.variable_scope(name):

        filt = get_conv_filter(name, data_dict)
        conv_biases = get_bias(name, data_dict)

        conv = tf.nn.conv2d(input_layer, filt, [1, 1, 1, 1], padding='SAME')
        bias = tf.nn.bias_add(conv, conv_biases)
        relu = tf.nn.relu(batch_norm_layer(bias, phase))

        #print(name)
        #print(conv.shape)
        #print(filt.shape)

        return relu

def get_conv_filter(name, data_dict):
    """
       Gets the filter weights of vgg16
       Args:
           name:     name of the layer
           data_dict:        dictionaty with the vgg16 weights
       Return:
           filt:    filter 
    """
    return tf.Variable(data_dict[name][0], name="filter")

def first_depth_conv_layer(input_layer, name, data_dict, phase):
    """
       Regular convolutional layer with the weights out of vgg16 and depth weights added.
       Args:
           input_layer:   previous layer
           name:     name of the layer
           data_dict:        dictionaty with the vgg16 weights
       Return:
           relu:    output of the layer
    """
    with tf.variable_scope(name):

        filt = get_first_depth_conv_filter(name, data_dict)
        conv_biases = get_bias(name, data_dict)

        conv = tf.nn.conv2d(input_layer, filt, [1, 1, 1, 1], padding='SAME')
        bias = tf.nn.bias_add(conv, conv_biases)
        relu = tf.nn.relu(batch_norm_layer(bias, phase))

        #print(name)
        #print(conv.shape)
        #print(filt.shape)

        return relu

def get_first_depth_conv_filter(name, data_dict):
    """
       Gets the filter weights of vgg16
       Args:
           name:     name of the layer
           data_dict:        dictionaty with the vgg16 weights
       Return:
           filt:    filter 
    """
    filt = data_dict[name][0]

    # If this is a network working on input with width channel, calc 1st layer wieghts
    # As the average of RGB weights, then multiply by 32 to change the scale from 0-255 to 0-8m
 
    averaged = np.average(filt, axis=2)
    averaged = averaged.reshape(3,3,1,64)
    averaged = averaged * 32
    filt = np.append(filt, averaged, 2)

    return tf.Variable(filt, name="filter")



def get_bias(name, data_dict):
    """
       Gets the biases weights of vgg16
       Args:
           name:     name of the layer
           data_dict:        dictionaty with the vgg16 weights
       Return:
           bias:    biases
    """
    return tf.Variable(data_dict[name][1], name="biases")

def conv_layer_decoder(input_layer, name, size_out, phase):
    """
       Regular convolutional layer of the decoder as described in the SegNet paper
       Args:
           name:     name of the layer
           input_layer:      input layer
           size_out:        output size for the filters
       Return:
           relu:    output of the layer
    """
    with tf.variable_scope(name):
        #Added weight initialization as described in the paper
        conv = tf.layers.conv2d(
            inputs=input_layer,
            filters=size_out,
            kernel_size=[3, 3],
            padding="same",
            use_bias=True,
            kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
            bias_initializer=tf.zeros_initializer(),
            activation=None,
            name = name)
        #print(name)
        #print(conv.shape)

        relu = tf.nn.relu(batch_norm_layer(conv, phase))

        return relu


def batch_norm_layer(input_layer, phase):
    """
       Batch normalization layer 
       Args:
           phase:     phase of the network (training or testing)
           input_layer:      input layer
       Return:
           layer:    output of the layer
    """
    return tf.contrib.layers.batch_norm(input_layer, is_training = phase)
