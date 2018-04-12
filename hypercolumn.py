import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim

from vgg import *

class hypercolumn(object):
	
	"""docstring for model"""
	def __init__(self, arg):
		self.arg = arg
		self.net = {}

	def __call__(self, image, is_train):

		self.shape = image.get_shape().as_list() 
		self.B, self.H, self.W, self.C = self.shape

		with slim.arg_scope( vgg_arg_scope() ):
			embeddings = vgg_16( image, is_training=False )

		with tf.variable_scope('Decoder', reuse=tf.AUTO_REUSE) as scope:
			emb = embeddings['conv5']
			self.net['conv5'] = tf.image.resize_images( emb, (self.H, self.W) )
			
			emb = embeddings['conv4']
			self.net['conv4'] = tf.image.resize_images( emb, (self.H, self.W) )
			
			emb = embeddings['conv3']
			self.net['conv3'] = tf.image.resize_images( emb, (self.H, self.W) )
			
			emb = embeddings['conv2']
			self.net['conv2'] = tf.image.resize_images( emb, (self.H, self.W) )
			
			emb = embeddings['conv1']
			self.net['conv1'] = tf.image.resize_images( emb, (self.H, self.W) )
			
			self.out = tf.concat( [self.net['conv1'], self.net['conv2'], self.net['conv3'], self.net['conv4'], self.net['conv5']], axis=3 )

		return( self.out )
