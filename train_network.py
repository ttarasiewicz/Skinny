from utils import models
from utils.Preprocessor import Preprocessor
from utils.DataLoader import DataLoader
from utils.Trainer import Trainer
from utils.WorkScheduler import WorkScheduler
from utils import losses, metrics, callbacks
import tensorflow as tf
from tensorflow.keras import backend as K

levels = 6
log_dir = 'logs'
max_epochs = 200
initial_lr = 1e-4

preprocessor = Preprocessor()
preprocessor.cast(dtype=tf.float32).normalize().downscale(max_pixel_count=512**2).pad(network_levels=levels)


def train_function(model: models.Model, batch_size: int) -> None:
    data_loader = DataLoader(dataset_dir=r'dataset', batch_size=batch_size, preprocessor=preprocessor)
    trainer = Trainer(data_loader=data_loader, model=model, evaluate_test_data=True)
    trainer.add_losses([K.binary_crossentropy, losses.dice_loss])
    trainer.add_metrics([
        metrics.f1,
        metrics.iou,
        metrics.precision,
        metrics.recall
    ])
    trainer.add_callbacks([
        callbacks.ModelCheckpoint(filepath=model.get_logdir(), verbose=1, save_best_only=True,
                                  monitor='val_f1', mode='max', save_weights_only=True),
        callbacks.ReduceLROnPlateau(monitor='val_f1', factor=0.5, verbose=1, mode='max', min_lr=1e-6, patience=5),
        callbacks.EarlyStopping(monitor='val_f1', mode='max', patience=10, verbose=1),
        callbacks.TensorBoard(log_dir=model.get_logdir(), histogram_freq=5)
    ])

    trainer.train(max_epochs, tf.keras.optimizers.Adam(learning_rate=initial_lr), verbose=1)


scheduler = WorkScheduler()
scheduler.add_data(models.SkinnyNOID(levels, 15, 3, log_dir), train_function, batch_size=5)
scheduler.add_data(models.SkinnyNOD(levels, 20, 3, log_dir), train_function, batch_size=3)
scheduler.add_data(models.Skinny(levels, 19, 3, log_dir), train_function, batch_size=3)
scheduler.do_work()
