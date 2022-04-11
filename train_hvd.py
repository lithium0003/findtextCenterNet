#!/usr/bin/env python3
import os
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

import tensorflow as tf
tf.keras.mixed_precision.set_global_policy('mixed_float16')

import horovod.tensorflow.keras as hvd
hvd.init()

import tensorflow_addons as tfa

# Pin GPU to be used to process local rank (one GPU per process)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    tf.config.set_visible_devices(gpus[hvd.local_rank()], 'GPU')

import os

save_target = 'result/'
#batchsize = 10
#syncBn = True
batchsize = 14
syncBn = False

import net
import dataset

def train():
    data = dataset.FontData()

    model = net.TextDetectorModel(syncbn=syncBn)
    #opt = tf.keras.optimizers.Adam(learning_rate=1e-4)
    scaled_lr = 1e-4 * hvd.size()
    opt = tfa.optimizers.AdamW(learning_rate=scaled_lr, weight_decay=1e-6)
    opt = hvd.DistributedOptimizer(opt, compression=hvd.Compression.fp16)
    model.compile(optimizer=opt)

    patience = 20 // hvd.size()
    if patience < 4:
        patience = 4

    callbacks = [
        hvd.callbacks.BroadcastGlobalVariablesCallback(0),
        hvd.callbacks.MetricAverageCallback(),
        tf.keras.callbacks.TerminateOnNaN(),
        hvd.callbacks.LearningRateWarmupCallback(initial_lr=scaled_lr, warmup_epochs=5, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_id_acc',
            factor=0.5,
            patience=patience,
            min_lr=1e-7),
    ]

    if hvd.rank() == 0:
        callbacks += [
            tf.keras.callbacks.ModelCheckpoint(
                os.path.join(save_target,'ckpt','ckpt'), 
                save_weights_only=True,
                monitor='val_id_acc',
                save_best_only=True),
            tf.keras.callbacks.BackupAndRestore(os.path.join(save_target,'backup')),
            tf.keras.callbacks.TensorBoard(
                log_dir=os.path.join(save_target,'log'),
                histogram_freq=1,
                write_graph=False),
            tf.keras.callbacks.CSVLogger(os.path.join(save_target,'training.csv'), append=True),
        ]

    model.fit(
        data.train_data(batchsize), epochs=4000 // (batchsize * hvd.size()), steps_per_epoch=1000,
        validation_data=data.test_data(batchsize), validation_steps=200,
        verbose=1 if hvd.rank() == 0 else 0,
        callbacks=callbacks)

if __name__ == '__main__':
    train()
