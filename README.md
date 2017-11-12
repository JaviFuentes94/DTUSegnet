# DTUSegnet
Implementation of SegNet for the Deep Learning course in DTU. 
Changes

## Instructions Tensorboard

In the py code, inside of the session:

 writer = tf.summary.FileWriter('./Tensorboard', sess.graph) 

Once the code is executed and that file is saved in the 
command prompt just write: 

 tensorboard --logdir="./Tensorboard"

If it doesn't open a window directly in the browser just go to: 

 http://localhost:6006/#graphs