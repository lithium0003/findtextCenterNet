import torch
import webdataset as wds
from torch.utils.data import DataLoader
import glob
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

from .processer import random_background, random_single, random_double, process

Image.MAX_IMAGE_PIXELS = 1000000000

rng = np.random.default_rng()
imagelist = glob.glob('data/background/*', recursive=True)


def random_distortion(im):
    if rng.random() < 0.3:
        im += 0.2 * rng.random() * rng.normal(size=im.shape)
        im = np.clip(im, 0, 1)
    if rng.random() < 0.3:
        im = gaussian_filter(im, sigma=1.5*rng.random())
        im = np.clip(im, 0, 1)
    elif rng.random() < 0.3:
        blurred = gaussian_filter(im, sigma=1.)
        im = im + 10. * rng.random() * (im - blurred)
        im = np.clip(im, 0, 1)
    return im

def transforms3(x):
    if rng.random() < 0.3:
        bgimage = rng.choice(imagelist)
        bgimg = np.asarray(Image.open(bgimage).convert("RGBA"))[:,:,:3]
        im = random_background(x, bgimg)
    elif rng.random() < 0.5:
        im = random_single(x)
    else:
        im = random_double(x)
    return random_distortion(im)

def identity(x):
    return x

def get_dataset(train=True):
    if train:
        shard_pattern = 'train_data1/train{00000000..00000399}.tar'
        shard_pattern = 'pipe:wget -O - -q --tries=0 --retry-on-http-error=500 --retry-connrefused --waitretry=1 --read-timeout=20 --timeout=15 -t 0 --continue https://huggingface.co/datasets/lithium0003/findtextCenterNet_dataset/resolve/main/train_data1/train{00000000..00000399}.tar'
        num_sample = 400 * 100
    else:
        shard_pattern = 'train_data1/test{00000000..00000015}.tar'
        shard_pattern = 'pipe:wget -O - -q --tries=0 --retry-on-http-error=500 --retry-connrefused --waitretry=1 --read-timeout=20 --timeout=15 -t 0 --continue https://huggingface.co/datasets/lithium0003/findtextCenterNet_dataset/resolve/main/train_data1/test{00000000..00000015}.tar'
        num_sample = 16 * 100
    dataset = (
        wds.WebDataset(shard_pattern, shardshuffle=True)
        .decode('l8')
        .rename(
            image='image.png', 
            position='position.npy', 
            textline='textline.png', 
            sepline='sepline.png',
            codelist='code_list.npy',
            )
        .to_tuple('image','textline','sepline','position','codelist')
        .map(process)
        .map_tuple(transforms3,identity,identity)
    )
    return dataset, num_sample

if __name__=='__main__':
    import matplotlib.pylab as plt
    import time
    from dataset.multi import MultiLoader

    dataset, count = get_dataset(train=False)
    dataloader = MultiLoader(dataset.batched(1))

    st = time.time()
    for sample in dataloader:
        print((time.time() - st) * 1000)
        image, labelmap, idmap = sample

        st = time.time()
        # continue

        plt.figure()
        if len(image[0].shape) > 2:
            plt.imshow(image[0].transpose(1,2,0))
        else:
            plt.imshow(image[0])

        plt.figure()
        plt.subplot(2,4,1)
        if len(image[0].shape) > 2:
            plt.imshow(image[0].transpose(1,2,0))
        else:
            plt.imshow(image[0])
        for i in range(5):
            plt.subplot(2,4,2+i)
            plt.imshow(labelmap[0,i])
        plt.subplot(2,4,7)
        plt.imshow(idmap[0,0])
        plt.subplot(2,4,8)
        plt.imshow(idmap[0,1])
        plt.show()
        st = time.time()
