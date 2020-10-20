import os
from tensorflow import keras


class CustomCallback(keras.callbacks.Callback):
    timelog = None

    def set_timelog(self, timelog, **kwargs):
        self.timelog = timelog


class ModelCheckpoint(keras.callbacks.ModelCheckpoint, CustomCallback):
    def set_model(self, model):
        assert self.timelog is not None, "Starting time of the training hasn't been logged!"
        self.filepath = os.path.join(self.filepath, model.name, self.timelog, 'checkpoint')
        os.makedirs(self.filepath, exist_ok=True)
        super().set_model(model)


class ReduceLROnPlateau(keras.callbacks.ReduceLROnPlateau, CustomCallback):
    pass


class ProgbarLogger(keras.callbacks.ProgbarLogger, CustomCallback):
    pass


class EarlyStopping(keras.callbacks.EarlyStopping, CustomCallback):
    pass


class TensorBoard(keras.callbacks.TensorBoard, CustomCallback):
    def set_model(self, model):
        assert self.timelog is not None, "Starting time of the training hasn't been logged!"
        self.log_dir = os.path.join(self.log_dir, model.name, self.timelog, 'tensorboard')
        os.makedirs(self.log_dir, exist_ok=True)
        super().set_model(model)
