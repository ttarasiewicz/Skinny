import os
from utils.DataLoader import DataLoader
from xml.etree import ElementTree as ET
from utils import models


class Trainer:

    def __init__(self, data_loader: DataLoader, model: models.Model,
                 log_dir: str = './logs', evaluate_test_data=False):
        self.data_loader = data_loader
        self.model = model
        self.metrics = []
        self.losses = []
        self.callbacks = []
        self.log_dir = os.path.join(log_dir, model.name)
        self.timelog = None
        self.evaluate_test_data = evaluate_test_data

    @property
    def model(self) -> models.Model:
        return self.__model

    @model.setter
    def model(self, model: models.Model):
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

    def combined_loss(self):
        def loss(y_true, y_pred):
            result = None
            for i, v in enumerate(self.losses):
                if i == 0:
                    result = v(y_true, y_pred)
                else:
                    result += v(y_true, y_pred)
            return result
        return loss

    def __log_evaluation_metrics(self, metrics: dict):
        root = ET.Element('metrics')
        tree = ET.ElementTree(root)
        for name, value in metrics.items():
            metric_element = ET.SubElement(root, name)
            metric_element.text = str(value)
        tree.write(open(os.path.join(self.model.get_logdir(), 'test_metrics.xml'), 'w'), encoding='unicode')

    def train(self, epochs, optimizer, initial_epoch=0, verbose=1):
        assert self.model is not None, "Model hasn't been set for the trainer."
        assert self.data_loader is not None, "DataLoader hasn't been set for the trainer."
        os.makedirs(self.model.get_logdir(), exist_ok=True)
        model = self.model.get_model()
        model.compile(optimizer=optimizer, loss=self.combined_loss(), metrics=self.metrics)

        model.fit(self.data_loader.train_dataset, validation_data=self.data_loader.val_dataset,
                  epochs=epochs, verbose=verbose, initial_epoch=initial_epoch,
                  callbacks=self.callbacks, shuffle=True)
        if self.evaluate_test_data:
            evaluation_metrics = model.evaluate(self.data_loader.test_dataset, verbose=1)
            evaluation_metrics = dict(zip(model.metrics_names, evaluation_metrics))
            self.__log_evaluation_metrics(evaluation_metrics)
