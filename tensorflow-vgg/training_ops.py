### This file implements training operations
# calc_loss       - to calculate the cross-entropy
# calc_accuracy   - to calculate the accuracy
# train_network   - to backpropagate the error through the network (1 training step)

# 1) Define cross entropy loss
with tf.variable_scope('loss'):
    def calc_loss(predictions, labels)
    	# computing cross entropy per sample, use softmax_cross_entropy_with_logits to avoid problems with log(0) 
    	#cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels, predictions)
    	
    	#I believe using that one the labels will have to be in [widthxheight] shape
    	#instead of 3d [widthxheightxclassnum]
    	cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=predictions)

    	# Sum over all pixels
		cross_entropy = tf.reduce_sum(cross_entropy, [1, 2])

    	# Average over samples
    	# Averaging makes the loss invariant to batch size, which is very nice.
    	cross_entropy = tf.reduce_mean(cross_entropy)
    	return cross_entropy

# 2) Define accuracy
with tf.variable_scope('performance'):
    def calc_accuracy(predictions, labels)
    	#to be implemented
    	return accuracy

# 3) Define the training op
with tf.variable_scope('trainOP'):
    def train_network(loss)
    	# Defining our optimizer
    	optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
    	# Computing our gradients
    	grads_and_vars = optimizer.compute_gradients(loss)
    	# Applying the gradients
    	train_op = optimizer.apply_gradients(grads_and_vars)
    	return train_op