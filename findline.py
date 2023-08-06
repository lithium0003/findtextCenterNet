#!/usr/bin/env python3
import tensorflow as tf
if len(tf.config.list_physical_devices('GPU')) > 0:
    tf.keras.mixed_precision.set_global_policy('mixed_float16')

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

import sys
import subprocess
import os

import net
from makedata.process import TextDetectorModel, calcHist

npzfile = 'params.npz'

if len(sys.argv) < 2 and os.path.exists(npzfile):
    print('loading params')
    with np.load(npzfile, mmap_mode='r') as params:
        locations = params['locations']
        glyphfeatures = params['glyphfeatures']
        lines = params['lines']
        seps = params['seps']
        im0 = params['im0']
else:
    im0 = Image.open(sys.argv[1]).convert('RGB')
    im0 = np.asarray(im0)

    model = TextDetectorModel()

    stepx = net.width * 3 // 4
    stepy = net.height * 3 // 4

    padx = max(0, stepx - (im0.shape[1] - net.width) % stepx, net.width - im0.shape[1])
    pady = max(0, stepy - (im0.shape[0] - net.height) % stepy, net.height - im0.shape[0])
    im = np.pad(im0, [[0,pady],[0,padx],[0,0]], 'constant', constant_values=((255,255),(255,255),(255,255)))

    im1 = tf.image.convert_image_dtype(im, dtype=tf.float32)
    im1 = im1 * 255.

    yi = tf.data.Dataset.range(0, im.shape[0] - net.height + 1, stepy)
    xi = tf.data.Dataset.range(0, im.shape[1] - net.width + 1, stepx)
    ds0 = yi.flat_map(lambda y: xi.map(lambda x: (x, y)))
    ds0 = ds0.map(lambda x,y: {
        'input': im1[y:y+net.height,x:x+net.width,:],
        'offsetx': x,
        'offsety': y,
        })
    ds0 = ds0.batch(8)
    ds0 = ds0.prefetch(tf.data.AUTOTUNE)

    locations, glyphfeatures, lines, seps = model.eval(ds0, im1, cut_off=0.5)

    valid_locations = []
    for i, (p, x, y, w, h, c1, c2, c4) in enumerate(locations):
        x1 = np.clip(int(x - w/2), 0, im.shape[1])
        y1 = np.clip(int(y - h/2), 0, im.shape[0])
        x2 = np.clip(int(x + w/2) + 1, 0, im.shape[1])
        y2 = np.clip(int(y + h/2) + 1, 0, im.shape[0])
        if calcHist(im[y1:y2,x1:x2,:]) < 50:
            continue
        valid_locations.append(i)
    locations = locations[valid_locations,:]
    glyphfeatures = glyphfeatures[valid_locations,:]

    np.savez_compressed(npzfile, locations=locations, glyphfeatures=glyphfeatures, lines=lines, seps=seps, im0=im0)

# plt.imshow(im0)
# for p, cx, cy, w, h, c1, c2 in locations:
#     points = [
#         [cx - w / 2, cy - h / 2],
#         [cx + w / 2, cy - h / 2],
#         [cx + w / 2, cy + h / 2],
#         [cx - w / 2, cy + h / 2],
#         [cx - w / 2, cy - h / 2],
#     ]
#     points = np.array(points)
#     plt.plot(points[:,0], points[:,1])
# plt.show()

print('construct data')
h, w = lines.shape
input_binary = int(0).to_bytes(4, 'little')
input_binary += int(w).to_bytes(4, 'little')
input_binary += int(h).to_bytes(4, 'little')
input_binary += lines.tobytes()
input_binary += seps.tobytes()
input_binary += int(locations.shape[0]).to_bytes(4, 'little')
input_binary += locations[:,1:].tobytes()
input_binary += int(im0.shape[1] // 2).to_bytes(4, 'little')
input_binary += int(im0.shape[0] // 2).to_bytes(4, 'little')

print('run')
result = subprocess.run('./linedetect', input=input_binary, stdout=subprocess.PIPE).stdout
detected_boxes = []
p = 0
max_block = 0
count = int.from_bytes(result[p:p+4], byteorder='little')
p += 4
for i in range(count):
    id = int.from_bytes(result[p:p+4], byteorder='little', signed=True)
    p += 4
    block = int.from_bytes(result[p:p+4], byteorder='little', signed=True)
    max_block = max(max_block, block)
    p += 4
    idx = int.from_bytes(result[p:p+4], byteorder='little', signed=True)
    p += 4
    subidx = int.from_bytes(result[p:p+4], byteorder='little', signed=True)
    p += 4
    subtype = int.from_bytes(result[p:p+4], byteorder='little', signed=True)
    p += 4
    detected_boxes.append((id,block,idx,subidx,subtype))

print(detected_boxes)

plt.imshow(im0)
cmap = plt.get_cmap('rainbow', max_block+1)
for id, block, idx, subidx, subtype in detected_boxes:
    if id < 0:
        continue
    cx = locations[id, 1]
    cy = locations[id, 2]
    w = locations[id, 3]
    h = locations[id, 4]

    points = [
        [cx - w / 2, cy - h / 2],
        [cx + w / 2, cy - h / 2],
        [cx + w / 2, cy + h / 2],
        [cx - w / 2, cy + h / 2],
        [cx - w / 2, cy - h / 2],
    ]
    points = np.array(points)
    plt.plot(points[:,0], points[:,1], color=cmap(block))
    if idx < 0:
        t = '*'
    else:
        if subtype & 2+4 == 2+4:
            points = [
                [cx - w / 2 + 1, cy - h / 2 + 1],
                [cx + w / 2 - 1, cy - h / 2 + 1],
                [cx + w / 2 - 1, cy + h / 2 - 1],
                [cx - w / 2 + 1, cy + h / 2 - 1],
                [cx - w / 2 + 1, cy - h / 2 + 1],
            ]
            points = np.array(points)
            plt.plot(points[:,0], points[:,1], color='yellow')
            t = '%d-r%d-%d'%(block, idx, subidx)
        elif subtype & 2+4 == 2:
            points = [
                [cx - w / 2 + 1, cy - h / 2 + 1],
                [cx + w / 2 - 1, cy - h / 2 + 1],
                [cx + w / 2 - 1, cy + h / 2 - 1],
                [cx - w / 2 + 1, cy + h / 2 - 1],
                [cx - w / 2 + 1, cy - h / 2 + 1],
            ]
            points = np.array(points)
            plt.plot(points[:,0], points[:,1], color='blue')
            t = '%d-b%d-%d'%(block, idx, subidx)
        else:
            t = '%d-%d-%d'%(block, idx, subidx)
    if subtype & 8 == 8:
        t += '+'
    plt.text(cx, cy, t, color='black')

# plt.figure()
# plt.imshow(lines)

# plt.figure()
# plt.imshow(seps)

# linemap = np.loadtxt('linemap.txt')
# plt.figure()
# plt.imshow(linemap)

# angle = np.loadtxt('angle.txt')
# plt.figure()
# plt.imshow(angle)

plt.show()

