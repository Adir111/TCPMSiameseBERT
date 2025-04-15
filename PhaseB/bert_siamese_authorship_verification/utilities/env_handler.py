import os
import tensorflow as tf


def is_colab():
    return 'COLAB_GPU' in os.environ


def is_tf_2_10():
    # Colab uses 2.15.0 <=
    return tf.__version__.startswith("2.10")
