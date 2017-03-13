from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time

import tensorflow as tf


def get_log_dir(log_base,suffix):
    """
    For each run create a direction
    :param log_base:
    :param suffix:
    :return:
    """
    run = time.strftime('%Y-%m-%d_%H-%M-%S')
    log_dir = '%s/%s_%s'%(log_base,run,suffix)
    os.mkdir(log_dir)
    return log_dir


def leaky_relu(x):
    return tf.maximum(0.2*x,x)