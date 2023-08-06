#!/usr/bin/env python3
import os
import tensorflow as tf
import numpy as np
tf.keras.mixed_precision.set_global_policy('mixed_float16')

physical_devices = tf.config.list_physical_devices('GPU')
try:
    for gpu in physical_devices:
        tf.config.experimental.set_memory_growth(gpu, True)
except:
    # Invalid device or cannot modify virtual devices once initialized.
    pass

save_target = 'result1'
batchsize = 8

import net
from dataset import data_detector

class SimpleTextDetectorModel(tf.keras.models.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.detector = net.CenterNetDetectionBlock(pre_weight=False)
        self.decoder = net.SimpleDecoderBlock()

    def call(self, inputs, **kwargs):
        maps, feature = self.detector(inputs, **kwargs)
        mask1 = maps[...,0] > 0
        feature = tf.boolean_mask(feature, mask1)
        return self.decoder(feature)

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
    model1.build(input_shape=[None, net.height, net.width, 3])
    last = tf.train.latest_checkpoint(path)
    print(last)
    model1.load_weights(last).expect_partial()

    copy_layers(src=model1.detector.layers, dest=model.detector.layers)
    copy_layers(src=model1.decoder.layers, dest=model.decoder.layers)

class LearningRateReducer(tf.keras.callbacks.Callback):
    def __init__(self, monitor="val_loss", patience=0, reduce_rate=0.5, min_lr=1e-6, significant_change=0.1, momentum=0.9):
        super().__init__()
        self.monitor = monitor
        self.patience = patience
        self.reduce_rate = reduce_rate
        self.min_lr = min_lr
        self.significant_change = significant_change
        self.momentum = momentum
        self.wait = 0

    def on_train_begin(self, logs=None):
        self.last_loss = np.Inf

    def on_epoch_begin(self, epoch, logs=None):
        if self.wait > self.patience:
            self.wait = 0
            # reduce lr
            lr = float(tf.keras.backend.get_value(self.model.optimizer.lr))
            if lr < self.min_lr:
                return
            lr *= self.reduce_rate
            tf.keras.backend.set_value(self.model.optimizer.lr, tf.keras.backend.get_value(lr))

            if hasattr(self.model, 'backbone_optimizer') and self.model.backbone_optimizer is not None:
                lr = float(tf.keras.backend.get_value(self.model.backbone_optimizer.lr))
                lr *= self.reduce_rate
                tf.keras.backend.set_value(self.model.backbone_optimizer.lr, tf.keras.backend.get_value(lr))

            if hasattr(self.model, 'decoder_optimizer') and self.model.decoder_optimizer is not None:
                lr = float(tf.keras.backend.get_value(self.model.decoder_optimizer.lr))
                lr *= self.reduce_rate
                tf.keras.backend.set_value(self.model.decoder_optimizer.lr, tf.keras.backend.get_value(lr))

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        monitor_value = logs.get(self.monitor)
        logs["lr"] = tf.keras.backend.get_value(self.model.optimizer.lr)

        if monitor_value is None:
            return
        if tf.math.is_finite(self.last_loss) and tf.math.is_finite(monitor_value):
            self.last_loss = self.momentum * self.last_loss + (1 - self.momentum) * monitor_value
        else:
            if tf.math.is_finite(monitor_value):
                self.last_loss = monitor_value
        logs["lastvalue"] = tf.keras.backend.get_value(self.last_loss)

        if (self.last_loss - monitor_value) / monitor_value > self.significant_change:
            self.wait = 0
        else:
            self.wait += 1


def train(pretrain=None):
    model = net.TextDetectorModel(pre_weight=not pretrain)
    opt1 = tf.keras.optimizers.Adam(learning_rate=3e-4)
    opt2 = tf.keras.optimizers.Adam(learning_rate=1e-4)
    opt3 = tf.keras.optimizers.Adam(learning_rate=4e-4)
    model.compile(optimizer=opt1, backbone_optimizer=opt2, decoder_optimizer=opt3)

    if pretrain:
        load_weights(model, pretrain)

    callbacks = [
        tf.keras.callbacks.TerminateOnNaN(),
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(save_target,'ckpt1','ckpt'), 
            save_best_only=True,
            save_weights_only=True),
        tf.keras.callbacks.BackupAndRestore(os.path.join(save_target,'backup')),
        LearningRateReducer(
            monitor='val_loss', 
            patience=3,
            reduce_rate=0.5,
            min_lr=1e-4,
            significant_change=0.03,
            momentum=0.9),
        tf.keras.callbacks.CSVLogger(
            os.path.join(save_target,'resultlog.csv'),
            append = True
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=os.path.join(save_target,'log'),
            write_graph=False),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=250),
    ]

    model.fit(
        data_detector.train_data(batchsize),
        epochs=1000,
        steps_per_epoch=1000,
        validation_data=data_detector.test_data(batchsize),
        validation_steps=200,
        callbacks=callbacks,
        )

if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1:
        batchsize = int(sys.argv[1])

    if os.path.exists('pretrain'):
        train(pretrain='pretrain')
    else:
        train()
