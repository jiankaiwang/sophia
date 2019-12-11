#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 18:02:45 2018
@author: jiankaiwang
"""

import os
import tensorflow as tf

print("TF Version: {}".format(tf.__version__))

# In[]

tf.reset_default_graph()

with tf.Graph().as_default():

  # define your own NN here
  x = tf.placeholder(tf.int32, name='x')
  y = tf.placeholder(tf.int32, name='y')
  b = tf.Variable(1, name='b')
  xy = tf.multiply(x, y)
  op = tf.add(xy, b, name='op_to_store')

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # run with a simple example
    feed_dict = {x: 10, y: 3}
    print(sess.run(op, feed_dict))

    # write out a frozen model
    constant_graph = tf.graph_util.convert_variables_to_constants(
      sess, sess.graph_def, ['op_to_store'])
    with tf.gfile.FastGFile(os.path.join( 'model.pb'), mode='wb') as f:
      f.write(constant_graph.SerializeToString())