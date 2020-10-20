import os
import tensorflow as tf
import time
from utils.DataLoader import DataLoader
import utils


class Trainer:

    def __init__(self, data_loader: DataLoader, model: tf.keras.models.Model, log_dir: str = './logs'):
        self.data_loader = data_loader
        self.model = model
        self.metrics = []
        self.losses = []
        self.callbacks = []
        self.log_dir = os.path.join(log_dir, model.name)
        self.timelog = None

    @property
    def model(self) -> tf.keras.models.Model:
        return self.__model

    @model.setter
    def model(self, model: tf.keras.models.Model):
        self.__model = model

    @property
    def data_loader(self) -> DataLoader:
        return self.__data_loader

    @data_loader.setter
    def data_loader(self, data_loader: DataLoader):
        self.__data_loader = data_loader

    def add_metrics(self, metrics):
        if type(metrics) is not list:
            metrics = [metrics]
        for metric in metrics:
            self.metrics.append(metric)

    def add_losses(self, losses) -> None:
        if type(losses) is not list:
            losses = [losses]
        for loss in losses:
            self.losses.append(loss)

    def add_callbacks(self, callbacks) -> None:
        if type(callbacks) is not list:
            callbacks = [callbacks]
        for callback in callbacks:
            self.callbacks.append(callback)

    def __configure_callbacks(self, **args):
        callback: utils.callbacks.CustomCallback
        for callback in self.callbacks:
            callback.set_timelog(**args)

    def __combined_loss(self):
        def loss(y_true, y_pred):
            result = None
            for i, v in enumerate(self.losses):
                if i == 0:
                    result = v(y_true, y_pred)
                else:
                    result += v(y_true, y_pred)
            return result
        return loss

    def train(self, epochs, optimizer, initial_epoch=0, verbose=1):
        assert self.model is not None, "Model hasn't been set for the trainer."
        assert self.data_loader is not None, "DataLoader hasn't been set for the trainer."
        self.timelog = time.strftime("%Y%m%d-%H%M%S")
        self.__configure_callbacks(timelog=self.timelog)
        self.model.compile(optimizer=optimizer, loss=self.__combined_loss(), metrics=self.metrics)

        self.__model.fit(self.data_loader.train_dataset, validation_data=self.data_loader.val_dataset, epochs=epochs,
                         verbose=verbose, initial_epoch=initial_epoch, callbacks=self.callbacks, shuffle=True)
