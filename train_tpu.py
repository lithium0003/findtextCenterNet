#!/usr/bin/env python3
import tensorflow as tf
tf.keras.mixed_precision.set_global_policy('mixed_bfloat16')

import tensorflow_addons as tfa

#cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='local')
cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='ocr')

tf.config.experimental_connect_to_cluster(cluster_resolver)
tf.tpu.experimental.initialize_tpu_system(cluster_resolver)
strategy = tf.distribute.TPUStrategy(cluster_resolver)
print("All devices: ", tf.config.list_logical_devices('TPU'))

gspath='gs://your-bucket-name/'
save_target = 'gs://your-bucket-name-result/'
batchsize = 4 * 8

import net
import dataset

def train():
    with strategy.scope():
        model = net.TextDetectorModel()
        #opt = tf.keras.optimizers.Adam(learning_rate=1e-4)
        opt = tfa.optimizers.AdamW(learning_rate=1e-4, weight_decay=1e-6)
        model.compile(optimizer=opt, steps_per_execution=4)

    ds1 = dataset.train_data(gspath, batchsize)
    ds2 = dataset.test_data(gspath, batchsize)

    save_dir = tf.io.gfile.join(save_target,'ckpt','ckpt')
    backup_dir = tf.io.gfile.join(save_target,'backup')
    tb_dir = tf.io.gfile.join(save_target,'log')
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
        patience=10,
        min_lr=1e-7)
    csv_callback = tf.keras.callbacks.CSVLogger(tf.io.gfile.join(save_target,'training.csv'), append=True)

    model.fit(
        ds1, epochs=4000 // batchsize, steps_per_epoch=1000,
        validation_data=ds2, validation_steps=200,
        callbacks=[cp_callback, backup_callback, tb_callcack, reduce_callback, csv_callback])

if __name__ == '__main__':
    train()
