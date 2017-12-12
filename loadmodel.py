import tensorflow as tf

modelPath= "./Models/model.ckpt-5"
saver = tf.train.Saver()

with tf.Session(config=tf.ConfigProto(gpu_options=(tf.GPUOptions(per_process_gpu_memory_fraction=0.9)))) as sess:
   
    saver.restore(modelPath)

    print("Test Loss : %s" % testLoss.eval())
    print("MFBLoss : %s" % MFBLoss.eval())