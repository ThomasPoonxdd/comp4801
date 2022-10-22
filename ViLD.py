import tensorflow as tf
# import tensorflow.compat.v1 as v1
# import sys
# import numpy as np

# from contextlib import contextmanager
# saved_model_dir = './image_path_v2'
# model = tf.saved_model.load(saved_model_dir, tags=['serve'])
# import os
# checkpoint_path = "resnet152_vild/checkpoint"
with tf.Session() as sess:
  new_saver = tf.train.import_meta_graph('resnet152_vild/model.ckpt-180000.meta')
  new_saver.restore(sess, tf.train.latest_checkpoint('./resnet152_vild/'))