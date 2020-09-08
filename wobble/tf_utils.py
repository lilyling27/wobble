# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["load_op_library"]

import os
import sysconfig
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


def load_op_library(module_file, name):
    suffix = sysconfig.get_config_var("EXT_SUFFIX")
    dirname = os.path.dirname(os.path.abspath(module_file))
    libfile = os.path.join(dirname, name)
    if suffix is not None:
        libfile += suffix
    else:
        libfile += ".so"
    return tf.load_op_library(libfile)
