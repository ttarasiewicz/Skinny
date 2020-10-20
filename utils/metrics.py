import tensorflow as tf
from tensorflow.keras import backend as K


def recall(y_true, y_pred):
    y_pred = K.round(y_pred)
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall_score = true_positives / (possible_positives + K.epsilon())
    return recall_score


def precision(y_true, y_pred):
    y_pred = K.round(y_pred)
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision_score = true_positives / (predicted_positives + K.epsilon())
    return precision_score


def f1(y_true, y_pred):
    precision_score = precision(y_true, y_pred)
    recall_score = recall(y_true, y_pred)
    return 2 * ((precision_score * recall_score) / (precision_score + recall_score + K.epsilon()))


def iou(y_true, y_pred):
    y_pred = K.round(y_pred)
    intersection = y_true * y_pred
    not_true = 1 - y_true
    union = y_true + (not_true * y_pred)

    return (K.sum(intersection, axis=-1) + K.epsilon()) / (K.sum(union, axis=-1) + K.epsilon())
