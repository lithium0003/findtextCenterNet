import torch
import webdataset as wds
from torch.utils.data import DataLoader
import glob
import os
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

from .processer import random_background, random_mono, random_single, random_double, process

Image.MAX_IMAGE_PIXELS = 1000000000

rng = np.random.default_rng()
imagelist = glob.glob('data/background/*', recursive=True)

def random_salt(x, s, prob=0.05):
    sizex = x.shape[1]
    sizey = x.shape[0]
    s = min(max(1, int(s / 4)), rng.integers(1, 16))
    shape = ((sizey + s)//s, (sizex + s)//s)
    noise = rng.choice([0,1,np.nan], p=[prob / 2, 1 - prob, prob / 2], size=shape).astype(x.dtype)
    noise = np.repeat(noise, s, axis=0)
    noise = np.repeat(noise, s, axis=1)
    noise = noise[:sizey, :sizex]
    return np.nan_to_num(x * noise, nan=1) 

def random_distortion(im, s):
    if rng.random() < 0.3:
        alpha = min(0.4 * rng.random(), 20 / max(1,s))
        im += alpha * rng.normal(size=im.shape)
        im = np.clip(im, 0, 1)
    if rng.random() < 0.3:
        sigma = min(s / 8, 1.5*rng.random())
        im = gaussian_filter(im, sigma=sigma)
        im = np.clip(im, 0, 1)
    elif rng.random() < 0.3:
        blurred = gaussian_filter(im, sigma=5.)
        im = im + 10. * rng.random() * (im - blurred)
        im = np.clip(im, 0, 1)
    return im

def transforms3(x):
    x1,x2,x3,x4 = x
    if rng.random() < 0.2:
        x1 = random_salt(x1, x4, prob=0.2 * rng.random())

    if rng.random() < 0.3:
        bgimage = rng.choice(imagelist)
        bgimg = np.asarray(Image.open(bgimage).convert("RGBA"))[:,:,:3]
        im = random_background(x1, bgimg)
    elif rng.random() < 0.5:
        im = random_mono(x1)
    elif rng.random() < 0.5:
        im = random_single(x1)
    else:
        im = random_double(x1)
    return random_distortion(im, x4), x2, x3


def get_dataset(train=True, calib=False):
    local_disk = False
    downloader = os.path.join(os.path.dirname(__file__), 'downloader')
    if calib:
        if local_disk:
            shard_pattern = 'train_data1/test00000000.tar'
        else:
            shard_pattern = 'pipe:%s https://huggingface.co/datasets/lithium0003/findtextCenterNet_dataset/resolve/main/train_data1/test00000000.tar'
            shard_pattern = shard_pattern%(downloader)
    else:
        if train:
            if local_disk:
                shard_pattern = 'train_data1/train{00000000..00001023}.tar'
            else:
                shard_pattern = 'pipe:%s https://huggingface.co/datasets/lithium0003/findtextCenterNet_dataset/resolve/main/train_data1/train{00000000..00001023}.tar'
                shard_pattern = shard_pattern%(downloader)
        else:
            if local_disk:
                shard_pattern = 'train_data1/test{00000000..00000063}.tar'
            else:
                shard_pattern = 'pipe:%s https://huggingface.co/datasets/lithium0003/findtextCenterNet_dataset/resolve/main/train_data1/test{00000000..00000063}.tar'
                shard_pattern = shard_pattern%(downloader)
    dataset = (
        wds.WebDataset(shard_pattern, shardshuffle=True)
        .shuffle(1000)
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
        .map(transforms3)
    )
    return dataset

if __name__=='__main__':
    import matplotlib.pylab as plt
    import time
    from dataset.multi import MultiLoader

    dataset = get_dataset(train=False)
    # dataloader = DataLoader(dataset, batch_size=1, num_workers=4)
    dataloader = MultiLoader(dataset.batched(1))

    st = time.time()
    for sample in dataloader:
        print((time.time() - st) * 1000)
        image, labelmap, idmap = sample
        image = torch.tensor(image, dtype=torch.float)
        labelmap = torch.tensor(labelmap, dtype=torch.float)
        idmap = torch.tensor(idmap, dtype=torch.long)

        st = time.time()
        # continue

        plt.figure()
        if len(image[0].shape) > 2:
            plt.imshow(image[0].permute(1,2,0))
        else:
            plt.imshow(image[0])

        plt.figure()
        plt.subplot(2,4,1)
        if len(image[0].shape) > 2:
            plt.imshow(image[0].permute(1,2,0))
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
