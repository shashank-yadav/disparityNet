import tensorflow as tf
import numpy as np
from hypercolumn import hypercolumn

batch_size = 4
height = 32
width = 256
channels = 3
num_steps = 1000000
learning_rate = 1e-6
logs_path = "/tmp/disparityNet/"

X = tf.placeholder(tf.float32, shape=[batch_size, height, width, channels])

img = np.ones( (batch_size, height, width, channels), np.float32 )

net = hypercolumn({})
emb = net(X, is_train=True)

init = tf.global_variables_initializer()

with tf.Session() as sess:

	sess.run( init )

	out = sess.run( emb, feed_dict={X:img} )

	print(out.shape)

