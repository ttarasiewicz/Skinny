import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
from tensorflow.keras.utils import plot_model
import os
from abc import ABC, abstractmethod


def inception_module(prev_layer, filters: int, activation=layers.LeakyReLU):
    filters = filters // 4
    conv_1 = layers.Conv2D(filters, (1, 1), padding='same')(prev_layer)
    conv_1 = activation()(conv_1)
    conv_3 = layers.Conv2D(filters, (1, 1), padding='same')(prev_layer)
    conv_3 = layers.Conv2D(filters, (3, 3), padding='same')(conv_3)
    conv_3 = activation()(conv_3)
    conv_5 = layers.Conv2D(filters, (1, 1), padding='same')(prev_layer)
    conv_5 = layers.Conv2D(filters, (5, 5), padding='same')(conv_5)
    conv_5 = activation()(conv_5)
    max_pool = layers.MaxPool2D(padding='same', strides=(1, 1))(prev_layer)
    max_pool = layers.Conv2D(filters, (1, 1), padding='same')(max_pool)
    max_pool = activation()(max_pool)
    return tf.concat([conv_1, conv_3, conv_5, max_pool], axis=-1)


def dense_block(prev_layer, filters: int, kernel_size: int or tuple, activation=layers.LeakyReLU):
    dense_1 = layers.Conv2D(filters // 2, kernel_size, padding='same')(prev_layer)
    dense_1 = layers.BatchNormalization()(dense_1)
    dense_1 = activation()(dense_1)
    dense_2 = layers.Conv2D(filters // 4, kernel_size, padding='same')(dense_1)
    dense_2 = layers.BatchNormalization()(dense_2)
    dense_2 = activation()(dense_2)
    dense_3 = layers.Conv2D(filters // 8, kernel_size, padding='same')(dense_2)
    dense_3 = layers.BatchNormalization()(dense_3)
    dense_3 = activation()(dense_3)
    return tf.concat([dense_1, dense_2, dense_3, prev_layer], axis=-1)


def get_filters_count(level: int, initial_filters: int) -> int:
    return 2**(level-1)*initial_filters


class Model(ABC):
    def __init__(self, levels: int, initial_filters: int,
                 image_channels: int = 3, checkpoint_path: str = None) -> None:
        self.levels = levels
        self.initial_filters = initial_filters
        self.image_channels = image_channels
        self.checkpoint_path = checkpoint_path
        self.model = None

    def load_weights(self) -> keras.Model:
        self.model.load_weights()
        return self.model

    def get_model(self) -> keras.Model:
        self.model = self.create_model()
        if self.checkpoint_path is not None:
            try:
                path = os.path.join(self.checkpoint_path, 'checkpoint')
                self.model.load_weights(path)
            except Exception as e:
                raise e
        return self.model

    @abstractmethod
    def create_model(self) -> keras.Model:
        pass


class Skinny(Model):
    name = "Skinny"

    def create_model(self) -> keras.Model:
        self.levels += 1
        kernel_size = (3, 3)
        layers_list = [None for _ in range(self.levels)]
        layers_list[0] = layers.Input(shape=(None, None, self.image_channels), name='feature')
        activation = layers.LeakyReLU

        for i in range(1, self.levels):
            filters = get_filters_count(i, self.initial_filters)
            prev = i-1
            if i != 1:
                layers_list[i] = layers.MaxPool2D()(layers_list[prev])
                prev = i

            layers_list[i] = layers.Conv2D(filters, kernel_size, padding='same')(layers_list[prev])
            layers_list[i] = layers.BatchNormalization()(layers_list[i])
            layers_list[i] = activation()(layers_list[i])
            layers_list[i] = inception_module(layers_list[i], filters, activation)

        for i in range(self.levels-2, 0, -1):
            filters = get_filters_count(i, self.initial_filters)
            layers_list[i+1] = layers.UpSampling2D()(layers_list[i+1])
            layers_list[i+1] = layers.Conv2D(filters, kernel_size, padding='same')(layers_list[i+1])
            layers_list[i+1] = layers.BatchNormalization()(layers_list[i+1])
            layers_list[i+1] = activation()(layers_list[i+1])

            layers_list[i] = dense_block(layers_list[i], filters, kernel_size, activation)

            layers_list[i] = tf.concat([layers_list[i+1], layers_list[i]], axis=-1)
            layers_list[i] = inception_module(layers_list[i], filters, activation)

        layers_list[1] = layers.Conv2D(self.initial_filters, kernel_size, padding='same')(layers_list[1])
        layers_list[1] = activation()(layers_list[1])
        layers_list[1] = layers.Conv2D(self.initial_filters//2, kernel_size, padding='same')(layers_list[1])
        layers_list[1] = activation()(layers_list[1])
        layers_list[1] = layers.Conv2D(1, kernel_size, padding='same',
                                       activation='sigmoid', name='label')(layers_list[1])

        model = keras.Model(inputs=[layers_list[0]], outputs=[layers_list[1]], name=self.name)
        plot_model(model, to_file=f'{self.name}.png', show_shapes=True)
        return model


class SkinnyNOD(Model):
    name = 'Skinny_NoD'

    def create_model(self) -> keras.Model:
        self.levels += 1
        kernel_size = (3, 3)
        layers_list = [None for _ in range(self.levels)]
        layers_list[0] = layers.Input(shape=(None, None, self.image_channels), name='feature')
        activation = layers.LeakyReLU

        for i in range(1, self.levels):
            filters = get_filters_count(i, self.initial_filters)
            prev = i-1
            if i != 1:
                layers_list[i] = layers.MaxPool2D()(layers_list[prev])
                prev = i

            layers_list[i] = layers.Conv2D(filters, kernel_size, padding='same')(layers_list[prev])
            layers_list[i] = layers.BatchNormalization()(layers_list[i])
            layers_list[i] = activation()(layers_list[i])
            layers_list[i] = inception_module(layers_list[i], filters, activation)

        for i in range(self.levels-2, 0, -1):
            filters = get_filters_count(i, self.initial_filters)
            layers_list[i+1] = layers.UpSampling2D()(layers_list[i+1])
            layers_list[i+1] = layers.Conv2D(filters, kernel_size, padding='same')(layers_list[i+1])
            layers_list[i+1] = layers.BatchNormalization()(layers_list[i+1])
            layers_list[i+1] = activation()(layers_list[i+1])

            layers_list[i] = tf.concat([layers_list[i+1], layers_list[i]], axis=-1)
            layers_list[i] = inception_module(layers_list[i], filters, activation)

        layers_list[1] = layers.Conv2D(self.initial_filters, kernel_size, padding='same')(layers_list[1])
        layers_list[1] = activation()(layers_list[1])
        layers_list[1] = layers.Conv2D(self.initial_filters//2, kernel_size, padding='same')(layers_list[1])
        layers_list[1] = activation()(layers_list[1])
        layers_list[1] = layers.Conv2D(1, kernel_size, padding='same',
                                       activation='sigmoid', name='label')(layers_list[1])

        model = keras.Model(inputs=[layers_list[0]], outputs=[layers_list[1]], name=self.name)
        plot_model(model, to_file=f'{self.name}.png', show_shapes=True)
        return model


class SkinnyNOID(Model):
    name = 'Skinny_NoID'

    def create_model(self) -> keras.Model:
        self.levels += 1
        kernel_size = (3, 3)
        layers_list = [None for _ in range(self.levels)]
        layers_list[0] = layers.Input(shape=(None, None, self.image_channels), name='feature')
        activation = layers.LeakyReLU

        for i in range(1, self.levels):
            filters = get_filters_count(i, self.initial_filters)
            prev = i-1
            if i != 1:
                layers_list[i] = layers.MaxPool2D()(layers_list[prev])
                prev = i

            layers_list[i] = layers.Conv2D(filters, kernel_size, padding='same')(layers_list[prev])
            layers_list[i] = layers.BatchNormalization()(layers_list[i])
            layers_list[i] = activation()(layers_list[i])
            layers_list[i] = layers.Conv2D(filters, kernel_size, padding='same')(layers_list[i])
            layers_list[i] = layers.BatchNormalization()(layers_list[i])
            layers_list[i] = activation()(layers_list[i])

        for i in range(self.levels-2, 0, -1):
            filters = get_filters_count(i, self.initial_filters)
            layers_list[i+1] = layers.UpSampling2D()(layers_list[i+1])
            layers_list[i+1] = layers.Conv2D(filters, kernel_size, padding='same')(layers_list[i+1])
            layers_list[i+1] = layers.BatchNormalization()(layers_list[i+1])
            layers_list[i+1] = activation()(layers_list[i+1])

            layers_list[i] = tf.concat([layers_list[i+1], layers_list[i]], axis=-1)
            layers_list[i] = layers.Conv2D(filters, kernel_size, padding='same')(layers_list[i])
            layers_list[i] = layers.BatchNormalization()(layers_list[i])
            layers_list[i] = activation()(layers_list[i])

        layers_list[1] = layers.Conv2D(self.initial_filters, kernel_size, padding='same')(layers_list[1])
        layers_list[1] = activation()(layers_list[1])
        layers_list[1] = layers.Conv2D(self.initial_filters//2, kernel_size, padding='same')(layers_list[1])
        layers_list[1] = activation()(layers_list[1])
        layers_list[1] = layers.Conv2D(1, kernel_size, padding='same',
                                       activation='sigmoid', name='label')(layers_list[1])

        model = keras.Model(inputs=[layers_list[0]], outputs=[layers_list[1]], name=self.name)
        plot_model(model, to_file=f'{self.name}.png', show_shapes=True)
        return model
