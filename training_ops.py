### This file implements training operations
# calc_loss       - to calculate the cross-entropy
# calc_accuracy   - to calculate the accuracy
# train_network   - to backpropagate the error through the network (1 training step)
import tensorflow as tf
import utils

# 1) Define cross entropy loss
def calc_loss(predictions, labels):
    """computing cross entropy per sample, use softmax_cross_entropy_with_logits to avoid problems with log(0)
   	    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels, predictions)
   	    I believe using that one the labels will have to be in [widthxheight] shape
   	    instead of 3d [widthxheightxclassnum]"""
    with tf.variable_scope("Loss"):
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=predictions)
   	    # Sum over all pixels
        cross_entropy = tf.reduce_sum(cross_entropy, [1, 2])
   	    # Average over samples
   	    # Averaging makes the loss invariant to batch size, which is very nice.
        cross_entropy = tf.reduce_mean(cross_entropy)
        #Show cross entropy in tensorboard
        tf.summary.scalar("Cross entropy", cross_entropy)
        return cross_entropy

# 2) Define accuracy
def calc_accuracy(predictions, labels):
    with tf.variable_scope("Accuracy"):    
        # Calculate the number of pixels with the same value in pred and lab
        predictions = tf.cast(predictions, tf.int32)
        equal_elements = tf.equal(predictions, labels)
        num_equal_elements = tf.reduce_sum(tf.cast(equal_elements, tf.int32))
        #Calculate global accuracy as fraction of matching pixels 
        accuracy = num_equal_elements/tf.size(labels)
        tf.summary.scalar("Accuracy", accuracy)
        return accuracy

# 3) Define the training op
def train_network(loss):
    with tf.variable_scope("TrainOp"):
        # Defining our optimizer
        optimizer = tf.train.MomentumOptimizer(learning_rate=0.001, momentum=0.9)
        #optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.000001)
        # Computing our gradients
        grads_and_vars = optimizer.compute_gradients(loss)
        #utils.variable_summaries(grads_and_vars)
        #tf.summary.scalar("grads_and_vars", grads_and_vars)
        
		# Applying the gradients
        capped_gvs = [(tf.clip_by_norm(grad, 50), var) for grad, var in grads_and_vars]
        for grad, var in capped_gvs:
            tf.summary.histogram(var.name + '/gradient', grad)
        # Applying the gradients
        train_op = optimizer.apply_gradients(capped_gvs)
        return train_op