import tensorflow as tf


def is_tf_2_10():
    """
    Check if the current TensorFlow version is 2.10.x.
    Assumes that if isn't 2.10, then it's 2.16.1 or higher.
    """
    return tf.__version__.startswith('2.10.')
