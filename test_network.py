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


def train_function(model: models.Model) -> None:
    data_loader = DataLoader(dataset_dir=r'dataset', batch_size=1, preprocessor=preprocessor)
    trainer = Trainer(data_loader=data_loader, model=model, evaluate_test_data=True)
    trainer.add_losses([K.binary_crossentropy, losses.dice_loss])
    trainer.add_metrics([
        metrics.f1,
        metrics.iou,
        metrics.precision,
        metrics.recall
    ])

    model.get_model()
    optimizer = tf.keras.optimizers.Adam(learning_rate=initial_lr)
    model.keras_model.compile(optimizer=optimizer, loss=trainer.combined_loss(), metrics=trainer.metrics)
    evaluation_metrics = model.keras_model.evaluate(data_loader.test_dataset, verbose=1)
    evaluation_metrics = dict(zip(model.keras_model.metrics_names, evaluation_metrics))
    print(evaluation_metrics)


scheduler = WorkScheduler()
scheduler.add_data(models.Model(load_checkpoint=True, checkpoint_extension='h5',
                                model_name='Skinny_NoID'), train_function)
scheduler.add_data(models.Model(load_checkpoint=True, checkpoint_extension='h5',
                                model_name='Skinny_NoD'), train_function)
scheduler.add_data(models.Model(load_checkpoint=True, checkpoint_extension='h5',
                                model_name='Skinny'), train_function)
scheduler.do_work()
