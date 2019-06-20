import numpy as np
import random as rn
import tensorflow as tf


def seed_randomness(random_seed):
    """
    seed random everywhere
    there is still problem with GPU
    https://keras.io/getting-started/faq/#how-can-i-obtain-reproducible-results-using-keras-during-development

    there is way to disable GPU, use before python script
    CUDA_VISIBLE_DEVICES="" PYTHONHASHSEED=0

    PYTHONHASHSEED - make python hash function reproducible
    (it's used in set and dict)


    this one also could be helpful, but I haven't used it yet

    Force TensorFlow to use single thread.
    Multiple threads are a potential source of non-reproducible results.
    For further details, see: https://stackoverflow.com/questions/42022950/

    session_conf = tf.ConfigProto(intra_op_parallelism_threads=1,
                                  inter_op_parallelism_threads=1)

    :param random_seed:
    :return:
    """
    np.random.seed(random_seed)
    rn.seed(random_seed)
    tf.random.set_random_seed(random_seed)
