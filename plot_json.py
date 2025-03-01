#!/usr/bin/env python3

import matplotlib
matplotlib.use("Agg")
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import json

try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass

def plot1(target_file):
    im0 = Image.open(target_file).convert('RGB')
    with open(target_file+'.json', 'r', encoding='utf-8') as file:
        data = json.load(file)

    fig = plt.figure(dpi=100, figsize=(im0.width / 100, im0.height / 100))
    plt.imshow(im0)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    fprop = FontProperties(fname='data/jpfont/NotoSerifJP-Regular.otf')
    for box in data['box']:
        cx = box['cx']
        cy = box['cy']
        w = box['w']
        h = box['h']
        text = box['text']
        blockidx = box['blockidx']
        lineidx = box['lineidx']
        subidx = box['subidx']
        ruby = box['ruby']
        rubybase = box['rubybase']
        emphasis = box['emphasis']
        vertical = box['vertical']

        points = [
            [cx - w / 2, cy - h / 2],
            [cx + w / 2, cy - h / 2],
            [cx + w / 2, cy + h / 2],
            [cx - w / 2, cy + h / 2],
            [cx - w / 2, cy - h / 2],
        ]
        points = np.array(points)
        if vertical == 0:
            plt.plot(points[:,0], points[:,1],color='cyan')
        else:
            plt.plot(points[:,0], points[:,1],color='magenta')

        points = [
            [cx - w / 2 - 1, cy - h / 2 - 1],
            [cx + w / 2 + 1, cy - h / 2 - 1],
            [cx + w / 2 + 1, cy + h / 2 + 1],
            [cx - w / 2 - 1, cy + h / 2 + 1],
            [cx - w / 2 - 1, cy - h / 2 - 1],
        ]
        points = np.array(points)
        if ruby == 1:
            plt.plot(points[:,0], points[:,1],color='green')
        elif rubybase == 1:
            plt.plot(points[:,0], points[:,1],color='yellow')

        points = [
            [cx - w / 2 + 1, cy - h / 2 + 1],
            [cx + w / 2 - 1, cy - h / 2 + 1],
            [cx + w / 2 - 1, cy + h / 2 - 1],
            [cx - w / 2 + 1, cy + h / 2 - 1],
            [cx - w / 2 + 1, cy - h / 2 + 1],
        ]
        points = np.array(points)
        if emphasis == 1:
            plt.plot(points[:,0], points[:,1],color='blue')

        plt.gca().text(cx - w/2, cy - h/2, text, fontsize=max(w,h)*0.5, color='blue', fontproperties=fprop, ha='left', va='top')

    plt.savefig(target_file+'.boxplot.png')
    plt.close('all')

def plot2(target_file):
    im0 = Image.open(target_file).convert('RGB')
    with open(target_file+'.json', 'r', encoding='utf-8') as file:
        data = json.load(file)

    fig = plt.figure(dpi=100, figsize=(im0.width / 100, im0.height / 100))
    plt.imshow(im0)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    fprop = FontProperties(fname='data/jpfont/NotoSerifJP-Regular.otf')
    for line in data['line']:
        x1 = line['x1']
        y1 = line['y1']
        x2 = line['x2']
        y2 = line['y2']
        text = line['text']
        blockidx = line['blockidx']
        lineidx = line['lineidx']

        size = 0
        for box in data['box']:
            if blockidx == box['blockidx'] and lineidx == box['lineidx']:
                vertical = box['vertical']
                size = max(size, max(box['w'], box['h'])*0.5)

        points = [
            [x1, y1],
            [x2, y1],
            [x2, y2],
            [x1, y2],
            [x1, y1],
        ]
        points = np.array(points)
        if vertical == 0:
            plt.plot(points[:,0], points[:,1],color='cyan')
            rotation = 0
            plt.gca().text(x1, y2, text, fontsize=size, color='blue', fontproperties=fprop, rotation=rotation, ha='left', va='top')
        else:
            plt.plot(points[:,0], points[:,1],color='magenta')
            rotation = 270
            plt.gca().text(x1, y1, text, fontsize=size, color='blue', fontproperties=fprop, rotation=rotation, ha='right', va='top')


    plt.savefig(target_file+'.lineplot.png')
    plt.close('all')

if __name__=='__main__':
    import sys

    plot1(sys.argv[1])
    plot2(sys.argv[1])