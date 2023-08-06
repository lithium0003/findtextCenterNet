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

import net

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
    checkpoint = tf.train.Checkpoint(model=model1)
    checkpoint.restore(last).expect_partial()

    copy_layers(src=model1.detector.layers, dest=model.detector.layers)
    copy_layers(src=model1.decoder.layers, dest=model.decoder.layers)

def convert():
    model = net.TextDetectorModel(pre_weight=False)

    if os.path.exists('backup'):
        load_weights(model, os.path.join('backup','chief'))
    elif os.path.exists('chief'):
        load_weights(model, 'chief')
    else:
        print('backup not found.')
        return

    model.save_weights(os.path.join('ckpt1','ckpt'))

if __name__ == '__main__':
    convert()