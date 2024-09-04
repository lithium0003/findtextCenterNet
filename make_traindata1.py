#!/usr/bin/env python3

# pip install -U pillow webdataset matplotlib
import numpy as np
from PIL import Image
import webdataset as wds
from multiprocessing import Pool, Manager
from pathlib import Path
from functools import partial

from render_font.generate_random_txt import get_random_text
from const import samples_per_file

data_path = Path('train_data1')
data_path.mkdir(exist_ok=True)

def get_filepath(train=True):
    if train:
        return str(data_path / 'train%08d.tar')
    else:
        return str(data_path / 'test%08d.tar')

def process(i, semaphore):
    rng = np.random.default_rng()
    semaphore.acquire()
    while True:
        try:
            d = get_random_text(rng)
            if np.count_nonzero(d['image']) == 0:
                continue
            if d['image'].shape[0] >= (1 << 27) or d['image'].shape[1] >= (1 << 27) or d['image'].shape[0] * d['image'].shape[1] >= (1 << 29):
                continue
            d['i'] = i
            return d
        except Exception as e:
            print(e,i,'error')
            continue

def create_data(train=True, count=1):
    if count < 1:
        return
    with Manager() as manager:
        semaphore = manager.Semaphore(1000)
        with wds.ShardWriter(get_filepath(train=train), maxcount=samples_per_file) as sink:
            with Pool() as p:
                for d in p.imap_unordered(partial(process, semaphore=semaphore), range(samples_per_file * count)):
                    print(d['i'],samples_per_file * count)
                    w = d['image'].shape[1] // 2 * 2
                    h = d['image'].shape[0] // 2 * 2
                    d['image'] = d['image'][:h,:w]
                    sep_image = np.asarray(Image.fromarray(d['sep_image']).resize((w // 2, h // 2)))
                    textline_image = np.asarray(Image.fromarray(d['textline_image']).resize((w // 2, h // 2)))
                    image = d['image']
                    sep_image = sep_image
                    textline_image = textline_image
                    position = d['position'].astype(np.float32)
                    code_list = d['code_list'].astype(np.int32)

                    sink.write({
                        "__key__": '%014d'%d['i'],
                        "txt": d['str'],
                        "image.png": image,
                        "sepline.png": sep_image,
                        "textline.png": textline_image,
                        "position.npy": position,
                        "code_list.npy": code_list,
                    })
                    semaphore.release()

if __name__=="__main__":
    import sys
    import multiprocessing
    multiprocessing.set_start_method('spawn')

    if len(sys.argv) < 3:
        test_count = 1
        train_count = 1
    else:
        test_count = int(sys.argv[1])
        train_count = int(sys.argv[2])

    create_data(train=False, count=test_count)
    create_data(train=True, count=train_count)
