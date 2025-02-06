#!/usr/bin/env python3

import numpy as np
import glob
from scipy.ndimage import gaussian_filter
from PIL import Image
import os

from render_font.generate_random_txt import get_random_text, get_random_text2
from processer3 import random_background, random_mono, random_single, random_double, process3

if os.path.exists('TextDetector.mlpackage'):
    print('coreml')
    from process_coreml import call_model
else:
    print('torch')
    from process_torch import call_model

Image.MAX_IMAGE_PIXELS = 1000000000

imagelist = glob.glob('data/background/*', recursive=True)
rng = np.random.default_rng()

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

def transforms3(x1,minsize):
    if rng.random() < 0.2:
        im = random_salt(x1, minsize, prob=0.2 * rng.random())

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
    return random_distortion(im, minsize)

def save_value(code, value, vert):
    value = np.expand_dims(value.astype(np.float16), axis=0)
    base_path = 'code_features'
    os.makedirs(base_path, exist_ok=True)
    if vert == 0:
        filename = os.path.join(base_path,'h%08x.npy'%code)
    else:
        filename = os.path.join(base_path,'v%08x.npy'%code)
    if os.path.exists(filename):
        data = np.load(filename)
        value = np.concatenate([data,value], axis=0)
    np.save(filename, value)

def proc():
    while True:
        try:
            if rng.uniform() < 0.2:
                d = get_random_text(rng)
            else:
                d = get_random_text2(rng)
        except Exception as e:
            print(e,'error')
            continue
        if np.count_nonzero(d['image']) == 0:
            continue
        if d['image'].shape[0] >= (1 << 27) or d['image'].shape[1] >= (1 << 27) or d['image'].shape[0] * d['image'].shape[1] >= (1 << 29):
            continue
        
        image, position, minsize = process3(d['image'], d['position'].astype(np.float32))
        image = transforms3(image, minsize)

        locations, glyphfeatures, vert = call_model(image)

        for i, loc in enumerate(locations):
            cx = loc[1]
            cy = loc[2]
            w = loc[3]
            h = loc[4]

            for j, (pcx, pcy, pw, ph) in enumerate(position):
                dist = np.sqrt((cx - pcx) ** 2 + (cy - pcy) ** 2)
                if dist < min(w/2, h/2):
                    code = d['code_list'][j,0]
                    v = vert[i]
                    print(code, cx, cy, w, h, v)
                    save_value(code, glyphfeatures[i], v)
                    break

if __name__=='__main__':
    proc()