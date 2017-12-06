import tensorflow as tf

class PoolingProcedure(object):
    """Implements all the functions for the pooling-unpooling operation"""
    def __init__(self):
        self.upsample_idx_dict={}
        self.ksize=[1, 2, 2, 1]

    def max_pool(self, input_layer, name):
        """
           Regular max pooling that also saves the argmax indices for the unpooling.
           Args:
               input_layer:   previous layer
               name:     name of the layer
           Return:
               pool:    max pool layer
        """
        with tf.variable_scope(name):
            layer, argmax = tf.nn.max_pool_with_argmax(input_layer, self.ksize, strides=[1, 2, 2, 1], padding='SAME', name=name)
            self.upsample_idx_dict[name]=argmax
            #DEBUG
            print(name)
            print(layer.shape)
            return layer, argmax

    def unpool(self, input_layer, encoder_layer_name, decoder_layer_name, argmax):
        """
           Unpooling layer after a previous max pooling layer.
           Args:
               input_layer:             previous layer to be unpooled
               encoder_layer_name:      the name from the max_pooling layer that we want to take the indeces from
               decoder_layer_name:      name of the layer
           Return:
               unpool:    unpooling tensor

           Inspired by the implementation discussed in https://github.com/tensorflow/tensorflow/issues/2169
        """
        with tf.variable_scope(decoder_layer_name):
            
            input_shape = tf.shape(input_layer)
            output_shape = [input_shape[0], input_shape[1] * self.ksize[1], input_shape[2]*self.ksize[2], input_shape[3]]

            flat_input_size = tf.reduce_prod(input_shape)
            flat_output_shape = [output_shape[0], output_shape[1] * output_shape[2] * output_shape[3]]

            #Access indixes from the encoder layer through a dictionary
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
            set_output_shape = [set_input_shape[0],set_input_shape[1] * self.ksize[1], set_input_shape[2] * self.ksize[2], set_input_shape[3]]
            ret.set_shape(set_output_shape)

            #DEBUG
            print(decoder_layer_name)
            print(ret.shape)
            return ret


