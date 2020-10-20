from utils import Models
from utils.Preprocessor import Preprocessor
from utils.DataLoader import DataLoader
from utils.Trainer import Trainer
from utils.TrainingScheduler import TrainingScheduler
from utils import losses, metrics, callbacks
import tensorflow as tf
from tensorflow.keras import backend as K

levels = 6
log_dir = 'logs'
max_epochs = 200
initial_lr = 1e-4

preprocessor = Preprocessor()
preprocessor.cast(dtype=tf.float32).normalize().downscale(max_pixel_count=512**2).pad(network_levels=levels)


def train_function(model: tf.keras.Model, batch_size) -> None:
    data_loader = DataLoader(dataset_dir=r'dataset', batch_size=batch_size, preprocessor=preprocessor)
    trainer = Trainer(data_loader=data_loader, model=model)
    trainer.add_losses([K.binary_crossentropy, losses.dice_loss])
    trainer.add_metrics([
        metrics.f1,
        metrics.iou,
        metrics.precision,
        metrics.recall
    ])
    trainer.add_callbacks([
        callbacks.ModelCheckpoint(log_dir, verbose=1, save_best_only=True,
                                  monitor='val_f1', mode='max', save_weights_only=True),
        callbacks.ReduceLROnPlateau(monitor='val_f1', factor=0.5, verbose=1, mode='max', min_lr=1e-6, patience=5),
        callbacks.EarlyStopping(monitor='val_f1', mode='max', patience=10, verbose=1),
        callbacks.TensorBoard(log_dir=log_dir, histogram_freq=5)
    ])

    trainer.train(max_epochs, tf.keras.optimizers.Adam(learning_rate=initial_lr), verbose=1)


scheduler = TrainingScheduler()
scheduler.add_training_data(Models.SkinnyNOID(levels, 15),
                            train_function, batch_size=5)
scheduler.add_training_data(Models.SkinnyNOD(levels, 20), train_function, batch_size=3)
scheduler.add_training_data(Models.Skinny(levels, 19), train_function, batch_size=3)
scheduler.train()
