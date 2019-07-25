# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen
# --------------------------------------------------------
"""
Convert depreciated VGG16 snapshots to the ones that support tensorflow format

It will check the specific snapshot at the vgg16_depre folder, and copy it to the same location at vgg16 folder
See experimental/scripts/convert_vgg16.sh for how to use it.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
from model.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.train_val import filter_roidb, get_training_roidb
from datasets.factory import get_imdb
import datasets.imdb
import argparse
import pprint
import numpy as np
import sys
import os
import os.path as osp
import shutil

try:
  import cPickle as pickle
except ImportError:
  import pickle

import tensorflow as tf
from tensorflow.python import pywrap_tensorflow

from nets.vgg16 import vgg16

def parse_args():
  """
  Parse input arguments
  """
  parser = argparse.ArgumentParser(description='Convert an old VGG16 snapshot to new format')
  parser.add_argument('--cfg', dest='cfg_file',
                      help='optional config file',
                      default=None, type=str)
  parser.add_argument('--snapshot', dest='snapshot',
                      help='vgg snapshot prefix',
                      type=str)
  parser.add_argument('--imdb', dest='imdb_name',
                      help='dataset to train on',
                      default='voc_2007_trainval', type=str)
  parser.add_argument('--iters', dest='max_iters',
                      help='number of iterations to train',
                      default=70000, type=int)
  parser.add_argument('--tag', dest='tag',
                      help='tag of the model',
                      default=None, type=str)
  parser.add_argument('--set', dest='set_cfgs',
                      help='set config keys', default=None,
                      nargs=argparse.REMAINDER)

  if len(sys.argv) == 1:
    parser.print_help()
    sys.exit(1)

  args = parser.parse_args()
  return args

def combined_roidb(imdb_names):
  """
  Combine multiple roidbs
  """

  def get_roidb(imdb_name):
    imdb = get_imdb(imdb_name)
    print('Loaded dataset `{:s}` for training'.format(imdb.name))
    imdb.set_proposal_method(cfg.TRAIN.PROPOSAL_METHOD)
    print('Set proposal method: {:s}'.format(cfg.TRAIN.PROPOSAL_METHOD))
    roidb = get_training_roidb(imdb)
    return roidb

  roidbs = [get_roidb(s) for s in imdb_names.split('+')]
  roidb = roidbs[0]
  if len(roidbs) > 1:
    for r in roidbs[1:]:
      roidb.extend(r)
    tmp = get_imdb(imdb_names.split('+')[1])
    imdb = datasets.imdb.imdb(imdb_names, tmp.classes)
  else:
    imdb = get_imdb(imdb_names)
  return imdb, roidb

def get_variables_in_checkpoint_file(file_name):
  try:
    reader = pywrap_tensorflow.NewCheckpointReader(file_name)
    var_to_shape_map = reader.get_variable_to_shape_map()
    return var_to_shape_map 
  except Exception as e:  # pylint: disable=broad-except
    print(str(e))
    if "corrupted compressed block contents" in str(e):
      print("It's likely that your checkpoint file has been compressed "
            "with SNAPPY.")

def convert_names(name):
  # removing :0
  name = name.replace(':0', '')
  # replace
  name = name.replace('vgg_16/', 'vgg16_default/')
  name = name.replace('/biases', '/bias')
  name = name.replace('/weights', '/weight')
  name = name.replace('/conv1/', '/')
  name = name.replace('/conv2/', '/')
  name = name.replace('/conv3/', '/')
  name = name.replace('/conv4/', '/')
  name = name.replace('/conv5/', '/')

  return name

# Just build the graph, load the weights/statistics, and save them
def convert_from_depre(net, imdb, input_dir, output_dir, snapshot, max_iters):
  if not osp.exists(output_dir):
    os.makedirs(output_dir)

  tfconfig = tf.ConfigProto(allow_soft_placement=True)
  tfconfig.gpu_options.allow_growth = True
  sess = tf.Session(config=tfconfig)

  num_classes = imdb.num_classes
  with sess.graph.as_default():
    tf.set_random_seed(cfg.RNG_SEED)
    layers = net.create_architecture(sess, 'TRAIN', num_classes, tag='default',
                                            anchor_scales=cfg.ANCHOR_SCALES,
                                            anchor_ratios=cfg.ANCHOR_RATIOS)
    loss = layers['total_loss']
    # Learning rate should be reduced already
    lr = tf.Variable(cfg.TRAIN.LEARNING_RATE * cfg.TRAIN.GAMMA, trainable=False)
    momentum = cfg.TRAIN.MOMENTUM
    optimizer = tf.train.MomentumOptimizer(lr, momentum)
    gvs = optimizer.compute_gradients(loss)
    if cfg.TRAIN.DOUBLE_BIAS:
      final_gvs = []
      with tf.variable_scope('Gradient_Mult') as scope:
        for grad, var in gvs:
          scale = 1.
          if cfg.TRAIN.DOUBLE_BIAS and '/biases:' in var.name:
            scale *= 2.
          if not np.allclose(scale, 1.0):
            grad = tf.multiply(grad, scale)
          final_gvs.append((grad, var))
      train_op = optimizer.apply_gradients(final_gvs)
    else:
      train_op = optimizer.apply_gradients(gvs)

    checkpoint = osp.join(input_dir, snapshot + '.ckpt')
    variables = tf.global_variables()
    name2var = {convert_names(v.name): v for v in variables}
    target_names = get_variables_in_checkpoint_file(checkpoint)
    restorer = tf.train.Saver(name2var)
    saver = tf.train.Saver()

    print('Importing...')
    restorer.restore(sess, checkpoint)
    checkpoint = osp.join(output_dir, snapshot + '.ckpt')
    print('Exporting...')
    saver.save(sess, checkpoint)

    # also copy the pkl file
    index = osp.join(input_dir, snapshot + '.pkl')
    outdex = osp.join(output_dir, snapshot + '.pkl')
    shutil.copy(index, outdex)

  sess.close()


if __name__ == '__main__':
  args = parse_args()

  print('Called with args:')
  print(args)

  if args.cfg_file is not None:
    cfg_from_file(args.cfg_file)
  if args.set_cfgs is not None:
    cfg_from_list(args.set_cfgs)

  print('Using config:')
  pprint.pprint(cfg)

  np.random.seed(cfg.RNG_SEED)

  # train set
  imdb, _ = combined_roidb(args.imdb_name)

  # output directory where the snapshot will be exported
  output_dir = get_output_dir(imdb, args.tag)
  print('Output will be exported to `{:s}`'.format(output_dir))

  # input directory where the snapshot will be imported
  input_dir = output_dir.replace('/vgg16/', '/vgg16_depre/')
  print('Input will be imported from `{:s}`'.format(input_dir))

  net = vgg16()

  convert_from_depre(net, imdb, input_dir, output_dir, args.snapshot, args.max_iters)
