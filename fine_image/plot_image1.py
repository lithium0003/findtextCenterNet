#!/usr/bin/env python3

import numpy as np
import sys
from PIL import Image
import json
import os

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

if len(sys.argv) < 2:
    print(sys.argv[0],'target.png')
    exit(1)

fprop = FontProperties(fname='data/jpfont/NotoSerifJP-Regular.otf')

target_file = sys.argv[1]
if len(sys.argv) > 2:
    for arg in sys.argv[2:]:
        if arg == 'kr':
            fprop = FontProperties(fname='data/krfont/NotoSerifKR-Regular.otf')
            print('kr font')

im0 = Image.open(target_file).convert('RGB')

linesfile = target_file + '.lines.png'
if os.path.exists(linesfile):
    lines_all = Image.open(linesfile)
    lines_all = lines_all.resize((lines_all.width * 4, lines_all.height * 4), resample=Image.Resampling.BILINEAR)

    fig = plt.figure()
    plt.imshow(im0)
    plt.imshow(lines_all, cmap='gray', alpha=0.5)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

sepsfile = target_file + '.seps.png'
if os.path.exists(sepsfile):
    seps_all = Image.open(sepsfile)
    seps_all = seps_all.resize((seps_all.width * 4, seps_all.height * 4), resample=Image.Resampling.BILINEAR)

    fig = plt.figure()
    plt.imshow(im0)
    plt.imshow(seps_all, cmap='gray', alpha=0.5)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

fig = plt.figure()
plt.imshow(im0)
fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

with open(target_file+'.json', 'r', encoding='utf-8') as file:
    outdict = json.load(file)

for i, pos in enumerate(outdict['textbox']):
    cx = pos['cx']
    cy = pos['cy']
    w = pos['w']
    h = pos['h']

    points = [
        [cx - w / 2, cy - h / 2],
        [cx + w / 2, cy - h / 2],
        [cx + w / 2, cy + h / 2],
        [cx - w / 2, cy + h / 2],
        [cx - w / 2, cy - h / 2],
    ]
    points = np.array(points)
    if pos['p_code8'] > 0.5:
        c = 'red'
    else:
        c = 'cyan'
    plt.plot(points[:,0], points[:,1],color=c)
    if pos['p_code2'] > 0.5:
        points = [
            [cx - w / 2 - 1, cy - h / 2 - 1],
            [cx + w / 2 + 1, cy - h / 2 - 1],
            [cx + w / 2 + 1, cy + h / 2 + 1],
            [cx - w / 2 - 1, cy + h / 2 + 1],
            [cx - w / 2 - 1, cy - h / 2 - 1],
        ]
        points = np.array(points)
        plt.plot(points[:,0], points[:,1],color='yellow')
    if pos['p_code1'] > 0.5:
        points = [
            [cx - w / 2 + 1, cy - h / 2 + 1],
            [cx + w / 2 - 1, cy - h / 2 + 1],
            [cx + w / 2 - 1, cy + h / 2 - 1],
            [cx - w / 2 + 1, cy + h / 2 - 1],
            [cx - w / 2 + 1, cy - h / 2 + 1],
        ]
        points = np.array(points)
        plt.plot(points[:,0], points[:,1],color='magenta')
    if pos['p_code4'] > 0.5:
        points = [
            [cx - w / 2 + 2, cy - h / 2 + 2],
            [cx + w / 2 - 2, cy - h / 2 + 2],
            [cx + w / 2 - 2, cy + h / 2 - 2],
            [cx - w / 2 + 2, cy + h / 2 - 2],
            [cx - w / 2 + 2, cy - h / 2 + 2],
        ]
        points = np.array(points)
        plt.plot(points[:,0], points[:,1],color='blue')

    if pos['text']:
        if pos['p_code1'] > 0.5:
            c = 'green'
        else:
            c = 'blue'
        plt.gca().text(cx, cy, pos['text'], fontsize=28, color=c, fontproperties=fprop)

plt.show()
