import tensorflow as tf
from utils import models
from typing import Callable, Any


class TrainingScheduler:
    def __init__(self) -> None:
        self.data = []

    def add_training_data(self, model: models.Model,
                          training_func: Callable[[tf.keras.Model, Any], None],
                          **kwargs) -> None:
        self.data.append((model, training_func, kwargs))

    def train(self) -> None:
        for model, func, kwargs in self.data:
            tf.keras.backend.clear_session()
            tf.compat.v1.reset_default_graph()
            model = model.get_model()
            func(model=model, **kwargs)

