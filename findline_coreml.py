#!/usr/bin/env python3

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

import sys
import subprocess
import os

from net.const import width, scale, height, feature_dim
from util_funcs import calcHist

npzfile = 'params.npz'

def eval(ds, org_img, cut_off = 0.5):
    import coremltools as ct

    mlmodel_detector = ct.models.MLModel('TextDetector.mlpackage')

    print(org_img.shape)
    print("test")

    locations = [np.zeros(5+4, dtype=np.float32)]
    glyphfeatures = [np.zeros(feature_dim, dtype=np.float32)]
    keymap_all = np.zeros([org_img.shape[0] // scale, org_img.shape[1] // scale], dtype=np.float32)
    lines_all = np.zeros([org_img.shape[0] // scale, org_img.shape[1] // scale], dtype=np.float32)
    seps_all = np.zeros([org_img.shape[0] // scale, org_img.shape[1] // scale], dtype=np.float32)
    code_all = []
    for _ in range(4):
        code_all.append(np.zeros([org_img.shape[0] // scale, org_img.shape[1] // scale], dtype=np.float32))

    for n, inputs in enumerate(ds):
        print(n)
        x_i = inputs['offsetx']
        y_i = inputs['offsety']
        x_is = x_i // scale
        y_is = y_i // scale
        x_s = width // scale
        y_s = height // scale

        input_image = Image.fromarray(inputs['input'], mode="RGB")
        output = mlmodel_detector.predict({'Image': input_image})
        maps = output['Output_heatmap']
        feature = output['Output_feature']

        mask = np.zeros([y_s, x_s], dtype=bool)
        x_min = int(x_s * 1 / 6) if x_i > 0 else 0
        x_max = int(x_s * 5 / 6) if x_i + width < org_img.shape[1] else x_s
        y_min = int(y_s * 1 / 6) if y_i > 0 else 0
        y_max = int(y_s * 5 / 6) if y_i + height < org_img.shape[0] else y_s
        mask[y_min:y_max, x_min:x_max] = True

        keymap_p = maps[0,:,:,0]
        line_p = maps[0,:,:,6]
        seps_p = maps[0,:,:,7]
        code_p = []
        for k in range(4):
            code_p.append(maps[0,:,:,8+k])

        keymap_all[y_is:y_is+y_s,x_is:x_is+x_s] = np.maximum(keymap_p * mask, keymap_all[y_is:y_is+y_s,x_is:x_is+x_s])
        lines_all[y_is:y_is+y_s,x_is:x_is+x_s] = np.maximum(line_p * mask, lines_all[y_is:y_is+y_s,x_is:x_is+x_s])
        seps_all[y_is:y_is+y_s,x_is:x_is+x_s] = np.maximum(seps_p * mask, seps_all[y_is:y_is+y_s,x_is:x_is+x_s])
        for k in range(4):
            code_all[k][y_is:y_is+y_s,x_is:x_is+x_s] = np.maximum(code_p[k] * mask, code_all[k][y_is:y_is+y_s,x_is:x_is+x_s])

        peak = maps[0,:,:,1]
        idxy, idxx  = np.unravel_index(np.argsort(-peak.ravel()), peak.shape)

        for y, x in zip(idxy, idxx):
            if peak[y,x] < cut_off:
                break
            w = maps[0,y,x,2]
            h = maps[0,y,x,3]
            dx = maps[0,y,x,4]
            dy = maps[0,y,x,5]
            if w * h <= 0:
                continue
            ix = x * scale + dx + x_i
            iy = y * scale + dy + y_i

            codes = []
            for k in range(4):
                codes.append(code_p[k][y,x])

            locations.append(np.array([peak[y,x], ix, iy, w, h, *codes]))
            glyphfeatures.append(feature[0, y, x, :])

    locations = np.array(locations, dtype=np.float32)
    glyphfeatures = np.array(glyphfeatures, dtype=np.float32)

    idx = np.argsort(-locations[:,0])
    done_area = np.zeros([0,4], dtype=np.float32)
    selected_idx = []
    for i in idx:
        p = locations[i,0]
        if p < cut_off:
            break
        cx = locations[i,1]
        cy = locations[i,2]
        w = locations[i,3]
        h = locations[i,4]
        area0_vol = w * h
        if done_area.size > 0:
            area1_vol = done_area[:,2] * done_area[:,3]
            inter_xmin = np.maximum(cx - w / 2, done_area[:,0] - done_area[:,2] / 2)
            inter_ymin = np.maximum(cy - h / 2, done_area[:,1] - done_area[:,3] / 2)
            inter_xmax = np.minimum(cx + w / 2, done_area[:,0] + done_area[:,2] / 2)
            inter_ymax = np.minimum(cy + h / 2, done_area[:,1] + done_area[:,3] / 2)
            inter_w = np.maximum(inter_xmax - inter_xmin, 0.)
            inter_h = np.maximum(inter_ymax - inter_ymin, 0.)
            inter_vol = inter_w * inter_h
            union_vol = area0_vol + area1_vol - inter_vol
            iou = np.where(union_vol > 0., inter_vol / union_vol, 0.)
            if iou.max() > 0.5:
                continue
            if inter_vol.max() > area0_vol * 0.5:
                continue
        done_area = np.vstack([done_area, np.array([cx, cy, w, h])])
        selected_idx.append(i)
    
    if len(selected_idx) > 0:
        selected_idx = np.array(selected_idx)

        locations = locations[selected_idx,:]
        glyphfeatures = glyphfeatures[selected_idx,:]
    else:
        locations = np.zeros([0,5+4], dtype=np.float32)
        glyphfeatures = np.zeros([0,feature_dim], dtype=np.float32)

    for i in range(locations.shape[0]):
        cx = locations[i,1]
        cy = locations[i,2]
        x = int(cx / scale)
        y = int(cy / scale)
        if x >= 0 and x < org_img.shape[1] // scale and y >= 0 and y < org_img.shape[0] // scale:
            for k in range(4):
                locations[i,5+k] = max(code_all[k][y,x], locations[i,5+k])

    return locations, glyphfeatures, lines_all, seps_all


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

    stepx = width * 1 // 2
    stepy = height * 1 // 2

    padx = max(0, stepx - (im0.shape[1] - width) % stepx, width - im0.shape[1])
    pady = max(0, stepy - (im0.shape[0] - height) % stepy, height - im0.shape[0])
    im0 = np.pad(im0, [[0,pady],[0,padx],[0,0]], 'constant', constant_values=((255,255),(255,255),(255,255)))

    ds0 = []
    for y in range(0, im0.shape[0] - height + 1, stepy):
        for x in range(0, im0.shape[1] - width + 1, stepx):
            ds0.append({
                'input': im0[y:y+height,x:x+width,:],
                'offsetx': x,
                'offsety': y,
            })

    locations, glyphfeatures, lines, seps = eval(ds0, im0, cut_off=0.35)

    valid_locations = []
    for i, (p, x, y, w, h, c1, c2, c4, c8) in enumerate(locations):
        x1 = np.clip(int(x - w/2), 0, im0.shape[1])
        y1 = np.clip(int(y - h/2), 0, im0.shape[0])
        x2 = np.clip(int(x + w/2) + 1, 0, im0.shape[1])
        y2 = np.clip(int(y + h/2) + 1, 0, im0.shape[0])
        if calcHist(im0[y1:y2,x1:x2,:]) < 35:
            continue
        valid_locations.append(i)
    locations = locations[valid_locations,:]
    glyphfeatures = glyphfeatures[valid_locations,:]

    np.savez_compressed(npzfile, locations=locations, glyphfeatures=glyphfeatures, lines=lines, seps=seps, im0=im0)

# plt.imshow(im0)
# for p, cx, cy, w, h, c1, c2, c4, c8 in locations:
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

