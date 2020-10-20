import tensorflow as tf


def dice_loss(y_true, y_pred, epsilon=1e-15):
    numerator = 2 * tf.reduce_sum(y_true * y_pred, axis=(1, 2, 3))
    denominator = tf.reduce_sum(y_true + y_pred, axis=(1, 2, 3))
    loss = tf.squeeze(tf.reshape(1 - numerator / denominator, (-1, 1, 1)))
    return loss

