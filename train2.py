#!/usr/bin/env python3
import os

import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0 and tf.config.experimental.get_device_details(physical_devices[0]).get('device_name') != 'METAL':
    tf.keras.mixed_precision.set_global_policy('mixed_float16')

    physical_devices = tf.config.list_physical_devices('GPU')
    try:
        for gpu in physical_devices:
            tf.config.experimental.set_memory_growth(gpu, True)
    except:
        # Invalid device or cannot modify virtual devices once initialized.
        pass

from net.transformer_trainer import TransformerDecoderModel
from dataset.data_transformer import generate_data, train_data, test_data

save_target = 'result2'
batchsize = 256

class GenerateCallback(tf.keras.callbacks.Callback):
    def __init__(self, log_dir) -> None:
        super().__init__()

        self.summary_writer_test = tf.summary.create_file_writer(
                os.path.join(log_dir, "predict"))
        self.ds = generate_data()

    def on_epoch_end(self, epoch, logs=None):
        result_text = self.model.generate(self.ds)

        with self.summary_writer_test.as_default():
            tf.summary.text("predict", result_text, step=epoch)

def train2():
    model = TransformerDecoderModel()
    #opt1 = tf.keras.optimizers.Adam(learning_rate=1e-4)
    boundaries = [100, 1000*50, 1000*200]
    values = [1e-5, 2e-4, 1e-4, 1e-5]
    lr = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries, values)
    opt1 = tf.keras.optimizers.AdamW(learning_rate=lr, weight_decay=1e-2)
    opt1.exclude_from_weight_decay(var_names=['layer_normalization','/bias'])
    model.compile(optimizer=opt1)

    callbacks = [
        tf.keras.callbacks.TerminateOnNaN(),
        GenerateCallback(log_dir=os.path.join(save_target,'log')),
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(save_target,'ckpt2','ckpt'), 
            save_weights_only=True),
        tf.keras.callbacks.BackupAndRestore(os.path.join(save_target,'backup')),
        tf.keras.callbacks.TensorBoard(
            log_dir=os.path.join(save_target,'log'),
            write_graph=False),
        tf.keras.callbacks.CSVLogger(os.path.join(save_target,'training.csv'), append=True),
    ]

    model.fit(
        train_data(batchsize),
        epochs=2000,
        steps_per_epoch=1000,
        validation_data=test_data(batchsize),
        validation_steps=200,
        callbacks=callbacks,
        )


if __name__ == '__main__':
    train2()
