#!/usr/bin/env python3

import tensorflow as tf
my_devices = tf.config.list_physical_devices(device_type='CPU')
tf.config.set_visible_devices(devices= my_devices, device_type='CPU')

import numpy as np
from PIL import Image
import os
import glob
from multiprocessing import Pool
import concurrent.futures

from render_font.generate_random_txt import get_random_text
from render_font.get_aozora import decode_ruby
from const import samples_per_file

tfdata_path = 'train_data1'

def count_prevfile(train=True):
    if train:
        prev_files = sorted(glob.glob(os.path.join(tfdata_path,'train*.tfrecords')))
        if len(prev_files) > 0:
            k = int(os.path.splitext(os.path.basename(prev_files[-1]))[0][-8:]) + 1
        else:
            k = 0
    else:
        prev_files = sorted(glob.glob(os.path.join(tfdata_path,'test*.tfrecords')))
        if len(prev_files) > 0:
            k = int(os.path.splitext(os.path.basename(prev_files[-1]))[0][-8:]) + 1
        else:
            k = 0
    return k

def get_filepath(k, train=True):
    os.makedirs(tfdata_path, exist_ok=True)

    if train:
        filename = os.path.join(tfdata_path,'train%08d.tfrecords'%k)
    else:
        filename = os.path.join(tfdata_path,'test%08d.tfrecords'%k)
    return filename

def process(i):
    global rng
    print(i)
    while True:
        try:
            d = get_random_text(rng)
            if np.count_nonzero(d['image']) == 0:
                continue
            if d['image'].shape[0] >= (1 << 27) or d['image'].shape[1] >= (1 << 27) or d['image'].shape[0] * d['image'].shape[1] >= (1 << 29):
                continue
            print(i,'done')
            return d
        except Exception as e:
            print(e,i,'error')
            continue

def write_data(k, p, train=True):
    with tf.io.TFRecordWriter(get_filepath(k, train=train)) as file_writer:
        value = [i for i in range(samples_per_file)]
        for d in p.imap_unordered(process, value):
            w = d['image'].shape[1]
            h = d['image'].shape[0]
            sep_image = np.asarray(Image.fromarray(d['sep_image']).resize((w // 2, h // 2)))
            textline_image = np.asarray(Image.fromarray(d['textline_image']).resize((w // 2, h // 2)))
            image = tf.io.encode_png(d['image'][...,None]).numpy()
            sep_image = tf.io.encode_png(sep_image[...,None]).numpy()
            textline_image = tf.io.encode_png(textline_image[...,None]).numpy()
            example_proto = tf.train.Example(features=tf.train.Features(feature={
                'str': tf.train.Feature(bytes_list=tf.train.BytesList(value=[d['str'].encode('utf-8')])),
                'width': tf.train.Feature(int64_list=tf.train.Int64List(value=[w])),
                'height': tf.train.Feature(int64_list=tf.train.Int64List(value=[h])),
                'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),
                'sep_image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[sep_image])),
                'textline_image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[textline_image])),
                'position': tf.train.Feature(bytes_list=tf.train.BytesList(value=[d['position'].astype(np.float32).tobytes()])),
                'code_list': tf.train.Feature(bytes_list=tf.train.BytesList(value=[d['code_list'].astype(np.int32).tobytes()])),
            }))
            record_bytes = example_proto.SerializeToString()
            file_writer.write(record_bytes)
            print(decode_ruby(d['str']))
            print()

def init():
    global rng
    rng = np.random.default_rng()

def create_data(train=True, count=1):
    k = count_prevfile(train=train)
    if k < count:
        with Pool(initializer=init) as p:
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                for i in range(k, count):
                    executor.submit(write_data, i, p, train)

if __name__=="__main__":
    import sys
    import multiprocessing
    multiprocessing.set_start_method('spawn')

    if len(sys.argv) < 3:
        test_count = 2
        train_count = 200
    else:
        test_count = int(sys.argv[1])
        train_count = int(sys.argv[2])

    create_data(train=False, count=test_count)
    create_data(train=True, count=train_count)
