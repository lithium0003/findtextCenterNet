#!/usr/bin/env python3
import os
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

import tensorflow as tf
import tensorflow_addons as tfa

save_target = 'result/'
batchsize = 1

import net
import dataset

def train():
    data = dataset.FontData()

    model = net.TextDetectorModel(renorm=True)
    #opt = tf.keras.optimizers.Adam(learning_rate=1e-4)
    opt = tfa.optimizers.AdamW(learning_rate=2e-5, weight_decay=1e-6)
    model.compile(optimizer=opt)

    save_dir = os.path.join(save_target,'ckpt','ckpt')
    backup_dir = os.path.join(save_target,'backup')
    tb_dir = os.path.join(save_target,'log')
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        save_dir, 
        save_weights_only=True,
        monitor='val_id_acc',
        save_best_only=True)
    backup_callback = tf.keras.callbacks.BackupAndRestore(backup_dir)
    tb_callcack = tf.keras.callbacks.TensorBoard(
        log_dir=tb_dir,
        histogram_freq=1,
        write_graph=False)
    reduce_callback = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_id_acc',
        factor=0.5,
        patience=50,
        min_lr=1e-7)
    csv_callback = tf.keras.callbacks.CSVLogger(os.path.join(save_target,'training.csv'), append=True)

    model.fit(
        data.train_data(batchsize), epochs=4000, steps_per_epoch=1000,
        validation_data=data.test_data(batchsize), validation_steps=200,
        callbacks=[cp_callback, backup_callback, tb_callcack, reduce_callback, csv_callback])

if __name__ == '__main__':
    train()
