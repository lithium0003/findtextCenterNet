#!/usr/bin/env python3

import torch
import numpy as np
import sys
from PIL import Image
import json
import glob
import subprocess

import matplotlib.pyplot as plt

try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass

from util_func import width, height, scale, feature_dim, modulo_list, sigmoid
from models.detector import TextDetectorModel, CenterNetDetector

if len(sys.argv) < 2:
    print(sys.argv[0],'target.png')
    exit(1)

target_files = []
model_size = 'xl'
resize = 1.0
cutoff = 0.4
for arg in sys.argv[1:]:
    if arg.startswith('--cutoff='):
        cutoff = float(arg.split('=')[1])
        print('cutoff: ', cutoff)
    elif arg.startswith('--resize='):
        resize = float(arg.split('=')[1])
        print('resize: ', resize)
    elif arg.startswith('--model='):
        model_size = arg.split('=')[1]
        print('model_size: ', model_size)
        if model_size == 's':
            print('model s')
        elif model_size == 'm':
            print('model m')
        elif model_size == 'l':
            print('model l')
        elif model_size == 'xl':
            print('model xl')
        else:
            exit(1)
    else:
        target_files += glob.glob(arg)

target_files = sorted(target_files)

print('load')
model = TextDetectorModel(model_size=model_size)
data = torch.load('model.pt', map_location="cpu", weights_only=True)
model.load_state_dict(data['model_state_dict'])
detector = CenterNetDetector(model.detector)
if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'
device = torch.device(device)
detector.to(device=device)
detector.eval()

def eval(ds, org_img, centers):
    print(org_img.shape)
    print("test")

    glyphfeatures = np.zeros([centers.shape[0], feature_dim], dtype=np.float32)

    for n, inputs in enumerate(ds):
        print(n, '/', len(ds))
        x_i = inputs['offsetx']
        y_i = inputs['offsety']
        x_s = width // scale
        y_s = height // scale

        images = torch.from_numpy(inputs['input'] / 255.).permute(0,3,1,2).to(device=device)
        with torch.no_grad():
            heatmap, features = detector(images)
            features = features.cpu().numpy()

        x_min = int(x_s * 1 / 8) if x_i > 0 else 0
        x_max = int(x_s * 7 / 8) + 1 if x_i + width < org_img.shape[1] else x_s
        y_min = int(y_s * 1 / 8) if y_i > 0 else 0
        y_max = int(y_s * 7 / 8) + 1 if y_i + height < org_img.shape[0] else y_s

        target = np.where(np.logical_and(np.logical_and(x_i + x_min * scale < centers[:,0], centers[:,0] < x_i + x_max * scale),
                                         np.logical_and(y_i + y_min * scale < centers[:,1], centers[:,1] < y_i + y_max * scale)))[0]
        for i in target:
            xi = int((centers[i,0] - x_i) / scale)
            yi = int((centers[i,1] - y_i) / scale)
            glyphfeatures[i,:] = features[0,:,yi,xi]

    return glyphfeatures.astype(np.float16)

stepx = width * 3 // 4
stepy = height * 3 // 4

for target_file in target_files:
    print(target_file)

    lines = np.asarray(Image.open(target_file+'.lines.png')).astype(np.float32) / 255
    seps = np.asarray(Image.open(target_file+'.seps.png')).astype(np.float32) / 255

    with open(target_file+'.json', 'r', encoding='utf-8') as file:
        data = json.load(file)
    textbox = data['textbox']
    if len(textbox) == 0:
        print('empty')
        continue

    locations = []
    for box in textbox:
        cx = box['cx']
        cy = box['cy']
        w = box['w']
        h = box['h']
        code1 = box['p_code1']
        code2 = box['p_code2']
        code4 = box['p_code4']
        code8 = box['p_code8']
        locations.append([cx,cy,w,h,code1,code2,code4,code8])
    locations = np.array(locations, dtype=np.float32)

    print('construct data')
    h, w = lines.shape
    input_binary = int(0).to_bytes(4, 'little')
    input_binary += int(w).to_bytes(4, 'little')
    input_binary += int(h).to_bytes(4, 'little')
    input_binary += lines.tobytes()
    input_binary += seps.tobytes()
    input_binary += int(locations.shape[0]).to_bytes(4, 'little')
    input_binary += locations.tobytes()

    print('run')
    result = subprocess.run('textline_detect/linedetect', input=input_binary, stdout=subprocess.PIPE).stdout
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

    # im = Image.open(target_file).convert('RGB')

    # fig = plt.figure()
    # plt.imshow(im)
    # fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    # cmap = plt.get_cmap('rainbow', max_block+1)
    # for id, block, idx, subidx, subtype in detected_boxes:
    #     if id < 0:
    #         continue
    #     cx = locations[id, 0]
    #     cy = locations[id, 1]
    #     w = locations[id, 2]
    #     h = locations[id, 3]

    #     points = [
    #         [cx - w / 2, cy - h / 2],
    #         [cx + w / 2, cy - h / 2],
    #         [cx + w / 2, cy + h / 2],
    #         [cx - w / 2, cy + h / 2],
    #         [cx - w / 2, cy - h / 2],
    #     ]
    #     points = np.array(points)
    #     plt.plot(points[:,0], points[:,1], color=cmap(block))
    #     if idx < 0:
    #         t = '*'
    #     else:
    #         if subtype & 2+4 == 2+4:
    #             points = [
    #                 [cx - w / 2 + 1, cy - h / 2 + 1],
    #                 [cx + w / 2 - 1, cy - h / 2 + 1],
    #                 [cx + w / 2 - 1, cy + h / 2 - 1],
    #                 [cx - w / 2 + 1, cy + h / 2 - 1],
    #                 [cx - w / 2 + 1, cy - h / 2 + 1],
    #             ]
    #             points = np.array(points)
    #             plt.plot(points[:,0], points[:,1], color='yellow')
    #             t = '%d-r%d-%d'%(block, idx, subidx)
    #         elif subtype & 2+4 == 2:
    #             points = [
    #                 [cx - w / 2 + 1, cy - h / 2 + 1],
    #                 [cx + w / 2 - 1, cy - h / 2 + 1],
    #                 [cx + w / 2 - 1, cy + h / 2 - 1],
    #                 [cx - w / 2 + 1, cy + h / 2 - 1],
    #                 [cx - w / 2 + 1, cy - h / 2 + 1],
    #             ]
    #             points = np.array(points)
    #             plt.plot(points[:,0], points[:,1], color='blue')
    #             t = '%d-b%d-%d'%(block, idx, subidx)
    #         else:
    #             t = '%d-%d-%d'%(block, idx, subidx)
    #     if subtype & 8 == 8:
    #         t += '+'
    #     plt.text(cx - w/2, cy - h/2, t, color='black')
    # plt.show()
    # continue

    centers = []
    boxlist = []
    for id, block, idx, subidx, subtype in detected_boxes:
        if id < 0:
            continue
        boxlist.append({
            'boxid': len(centers),
            'blockid': block,
            'lineid': idx,
            'subidx': subidx,
            'subtype': subtype,
            'text': textbox[id].get('text', None),
        })
        centers.append([locations[id,0], locations[id,1]])
    centers = np.array(centers, dtype=np.float32)

    im0 = Image.open(target_file).convert('RGB')
    if resize != 1.0:
        im0 = im0.resize((int(im0.width * resize), int(im0.height * resize)), resample=Image.Resampling.BILINEAR)
    im0 = np.asarray(im0)

    padx = max(0, (width - im0.shape[1]) % stepx, width - im0.shape[1])
    pady = max(0, (height - im0.shape[0]) % stepy, height - im0.shape[0])
    im0 = np.pad(im0, [[0,pady],[0,padx],[0,0]], 'constant', constant_values=((255,255),(255,255),(255,255)))

    im = im0

    ds0 = []
    for y in range(0, im0.shape[0] - height + 1, stepy):
        for x in range(0, im0.shape[1] - width + 1, stepx):
            ds0.append({
                'input': im[y:y+height,x:x+width,:],
                'offsetx': x,
                'offsety': y,
            })

    glyph = eval(ds0, im, centers)
    np.save(target_file+'.npy', glyph)

    data['boxlist'] = boxlist
    with open(target_file+'.json', 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=2, ensure_ascii=False)
