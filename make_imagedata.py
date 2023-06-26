#!/usr/bin/env python3
from PIL import Image
import os

from dataset import data_detector

if __name__ == "__main__":
    os.makedirs('img')
    outfile = os.path.join('img','img%05d.png')

    ds = data_detector.train_data(1)

    for i,d in enumerate(ds.take(1000)):
        print(i)
        image = d['image']
        image = image[0,...].numpy().astype('uint8')
        im = Image.fromarray(image)

        im.save(outfile%i)