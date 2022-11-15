""" Create models using TensorFlow
"""
import tensorflow as tf


def baseline(inputs):
    """ Baseline model
    """
    energy = tf.reduce_sum(
        inputs,
        axis=1,
        keepdims=True)
    logits = tf.layers.dense(
        energy,
        units=1,
        activation=None)
    return logits


def linear_reg(inputs):
    """ Linear regression
    """
    logits = tf.layers.dense(
        inputs,
        units=1,
        activation=None)
    return logits


def nn(inputs):
    """ Shallow neural network
    """
    dense1 = tf.layers.dense(
        inputs,
        units=10,
        activation=tf.nn.relu)
    dense2 = tf.layers.dense(
        dense1,
        units=10,
        activation=tf.nn.relu)
    logits = tf.layers.dense(
        dense2,
        units=1,
        activation=None)
    return logits


def cnn(inputs):
    """ Convolutional neural network
    """
    conv1 = tf.layers.conv2d(
        tf.reshape(inputs, [-1, 28, 28, 1]),
        filters=32,
        kernel_size=5,
        padding='same',
        activation=tf.nn.relu)
    pool1 = tf.layers.max_pooling2d(
        conv1,
        pool_size=2,
        strides=2)
    conv2 = tf.layers.conv2d(
        pool1,
        filters=64,
        kernel_size=5,
        padding='same',
        activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(
        conv2,
        pool_size=2,
        strides=2)
    dense = tf.layers.dense(
        tf.reshape(pool2, [-1, 7*7*64]),
        units=64,
        activation=tf.nn.relu)
    logits = tf.layers.dense(
        dense,
        units=1,
        activation=None)
    return logits


if __name__ == '__main__':
    """ Profile model parameters
    """
    inputs = tf.random_uniform([1, 28*28])
    # logits = baseline(inputs)
    # logits = linear_reg(inputs)
    # logits = nn(inputs)
    logits = cnn(inputs)
    param_stats = tf.profiler.profile(
            tf.get_default_graph(),
            options=tf.profiler.ProfileOptionBuilder
            .trainable_variables_parameter())
    print(f'\ntotal_params: {param_stats.total_parameters}\n')
