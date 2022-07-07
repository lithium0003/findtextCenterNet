#!/usr/bin/env python3
import os

os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

import tensorflow as tf
tf.keras.mixed_precision.set_global_policy('mixed_float16')
import tensorflow_addons as tfa

save_target = 'result'
batchsize = 16

import net
import dataset

class SimpleTextDetectorModel(tf.keras.models.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.detector = net.CenterNetDetectionBlock(pre_weight=False)
        self.decoder = net.SimpleDecoderBlock()

        inputs = tf.keras.Input(shape=(net.height,net.width,3))
        self.detector(inputs)

def copy_layers(src, dest):
    for srclayer, destlayer in zip(src, dest):
        if hasattr(srclayer, 'layers'):
            copy_layers(srclayer.layers, destlayer.layers)
        else:
            dest_names = [v.name for v in destlayer.weights]
            for src_value in srclayer.weights:
                if src_value.name in dest_names:
                    i = dest_names.index(src_value.name)
                    destlayer.weights[i].assign(src_value)
                else:
                    print('skip', src_value)
            destlayer.finalize_state()

def load_weights(model, path):
    model1 = SimpleTextDetectorModel()
    last = tf.train.latest_checkpoint(path)
    print(last)
    model1.load_weights(last).expect_partial()

    copy_layers(src=model1.detector.layers, dest=model.detector.layers)
    copy_layers(src=model1.decoder.layers, dest=model.decoder.layers)

def train(pretrain=None):
    data = dataset.FontData()

    model = net.TextDetectorModel(logdir=os.path.join(save_target,'log'))
    #opt1 = tf.keras.optimizers.Adam(learning_rate=1e-4)
    opt1 = tfa.optimizers.AdamW(learning_rate=1e-4, weight_decay=1e-6, exclude_from_weight_decay=['_bn/','/bias'])
    opt2 = tf.keras.optimizers.Adam(learning_rate=1e-4)
    model.compile(detector_optimizer=opt1, decoder_optimizer=opt2, weighted_metrics=[])

    if pretrain:
        load_weights(model, pretrain)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(save_target,'ckpt','ckpt'), 
            save_weights_only=True),
        tf.keras.callbacks.BackupAndRestore(os.path.join(save_target,'backup')),
        tf.keras.callbacks.TensorBoard(
            log_dir=os.path.join(save_target,'log'),
            histogram_freq=1,
            write_graph=False),
        tf.keras.callbacks.CSVLogger(os.path.join(save_target,'training.csv'), append=True),
    ]

    model.fit(
        data.train_data(batchsize), epochs=1000, steps_per_epoch=1000,
        validation_data=data.test_data(batchsize), validation_steps=50,
        callbacks=callbacks)

if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1:
        batchsize = int(sys.argv[1])
    train()
