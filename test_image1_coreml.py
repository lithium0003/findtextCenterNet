#!/usr/bin/env python3

import coremltools as ct

import numpy as np
import sys
import os
from PIL import Image, ImageFilter

from matplotlib import rcParams
rcParams['font.serif'] = ['Noto Serif CJK JP', 'IPAexMincho', 'Hiragino Mincho ProN']

import matplotlib.pyplot as plt

if len(sys.argv) < 2:
    print(sys.argv[0],'target.png','(twopass)')
    exit(1)

target_file = sys.argv[1]
twopass = False
if len(sys.argv) > 2:
    if sys.argv[2] == 'twopass':
        twopass = True

mlmodel_detector = ct.models.MLModel('TextDetector.mlpackage')
mlmodel_decoder = ct.models.MLModel('CodeDecoder.mlpackage')

from util_funcs import calc_predid, width, height, scale, feature_dim

def eval(ds, org_img, cut_off = 0.5, locations0 = None, glyphfeatures0 = None):
    print(org_img.shape)
    print("test")

    locations = [np.zeros(5+4)]
    glyphfeatures = [np.zeros(feature_dim, dtype=np.float32)]
    keymap_all = np.zeros([org_img.shape[0] // scale, org_img.shape[1] // scale])
    lines_all = np.zeros([org_img.shape[0] // scale, org_img.shape[1] // scale])
    seps_all = np.zeros([org_img.shape[0] // scale, org_img.shape[1] // scale])
    code_all = []
    for _ in range(4):
        code_all.append(np.zeros([org_img.shape[0] // scale, org_img.shape[1] // scale]))

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

    locations = np.array(locations)
    if locations0 is not None:
        locations = np.concatenate([locations, locations0])
    glyphfeatures = np.array(glyphfeatures)
    if glyphfeatures0 is not None:
        glyphfeatures = np.concatenate([glyphfeatures, glyphfeatures0])

    idx = np.argsort(-locations[:,0])
    done_area = np.zeros([0,4])
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
            if iou.max() > 0.75:
                continue
            if inter_vol.max() > area0_vol * 0.75:
                continue
        done_area = np.vstack([done_area, np.array([cx, cy, w, h])])
        selected_idx.append(i)
    
    if len(selected_idx) > 0:
        selected_idx = np.array(selected_idx)

        locations = locations[selected_idx,:]
        glyphfeatures = glyphfeatures[selected_idx,:]
    else:
        locations = np.zeros([0,5+4])
        glyphfeatures = np.zeros([0,feature_dim], dtype=np.float32)

    for i in range(locations.shape[0]):
        cx = locations[i,1]
        cy = locations[i,2]
        x = int(cx / scale)
        y = int(cy / scale)
        if x >= 0 and x < org_img.shape[1] // scale and y >= 0 and y < org_img.shape[0] // scale:
            for k in range(4):
                locations[i,5+k] = max(code_all[k][y,x], locations[i,5+k])

    plt.figure()
    plt.imshow(keymap_all,interpolation='none',vmin=0.,vmax=1.)
    plt.title('keymap')

    plt.figure()
    plt.imshow(lines_all,interpolation='none',vmin=0.,vmax=1.)
    plt.title('textline')
    
    plt.figure()
    plt.imshow(seps_all,interpolation='none',vmin=0.,vmax=1.)
    plt.title('separator')
    
    title_str = [
        'ruby',
        'rubybase',
        'emphasis',
        'space',
    ]
    for k in range(4):
        plt.figure()
        plt.imshow(code_all[k],interpolation='none',vmin=0.,vmax=1.)
        plt.title('code%d '%(2**k) + title_str[k])

    return locations, glyphfeatures

def decode(glyphfeatures):
    print("decode")
    glyphids = []
    glyphprobs = []
    for data in glyphfeatures:
        decode_output = mlmodel_decoder.predict({'Input': np.expand_dims(data,0)})
        p = decode_output['Output_p'][0]
        ids = list(decode_output['Output_id'][0].astype(int))
        i = calc_predid(*ids)
        glyphids.append(i)
        glyphprobs.append(p)

    glyphids = np.stack(glyphids)
    glyphprobs = np.stack(glyphprobs)
    
    return  glyphids, glyphprobs

stepx = width * 1 // 2
stepy = height * 1 // 2

im0 = Image.open(target_file).convert('RGB')
#im0 = im0.filter(ImageFilter.SHARPEN)
im0 = np.asarray(im0)

padx = max(0, stepx - (im0.shape[1] - width) % stepx, width - im0.shape[1])
pady = max(0, stepy - (im0.shape[0] - height) % stepy, height - im0.shape[0])
im0 = np.pad(im0, [[0,pady],[0,padx],[0,0]], 'constant', constant_values=((255,255),(255,255),(255,255)))

if twopass and (im0.shape[1] / stepx > 2 or im0.shape[0] / stepy > 2):
    print('two-pass')
    s = max(im0.shape[1], im0.shape[0]) / max(width, height)
    im1 = Image.fromarray(im0).resize((int(im0.shape[1] / s), int(im0.shape[0] / s)), resample=Image.BILINEAR)
    im1 = np.asarray(im1)
    padx = max(0, width - im1.shape[1])
    pady = max(0, height - im1.shape[0])
    im1 = np.pad(im1, [[0,pady],[0,padx],[0,0]], 'constant', constant_values=((255,255),(255,255),(255,255)))

    ds1 = []
    ds1.append({
        'input': im1,
        'offsetx': 0,
        'offsety': 0,
        })

    locations0, glyphfeatures0 = eval(ds1, im1, cut_off=0.5)
    locations0[:,1:] = locations0[:,1:] * s
else:
    locations0, glyphfeatures0 = None, None

ds0 = []
for y in range(0, im0.shape[0] - height + 1, stepy):
    for x in range(0, im0.shape[1] - width + 1, stepx):
        ds0.append({
            'input': im0[y:y+height,x:x+width,:],
            'offsetx': x,
            'offsety': y,
        })
locations, glyphfeatures = eval(ds0, im0, cut_off=0.5,
        locations0=locations0, glyphfeatures0=glyphfeatures0)
glyphids, glyphprobs = decode(glyphfeatures)

plt.figure()
plt.hist(np.reshape(glyphfeatures,[-1]), bins=50)
plt.title('features')

plt.figure()
plt.imshow(im0)

for i, loc in enumerate(locations):
    cx = loc[1]
    cy = loc[2]
    w = loc[3]
    h = loc[4]
    codes = loc[5:]
    cid = glyphids[i]
    p = glyphprobs[i]
    g = glyphfeatures[i]

    points = [
        [cx - w / 2, cy - h / 2],
        [cx + w / 2, cy - h / 2],
        [cx + w / 2, cy + h / 2],
        [cx - w / 2, cy + h / 2],
        [cx - w / 2, cy - h / 2],
    ]
    points = np.array(points)
    if codes[3] > 0.5:
        c = 'red'
    else:
        c = 'cyan'
    plt.plot(points[:,0], points[:,1],color=c)
    if codes[1] > 0.5:
        points = [
            [cx - w / 2 - 1, cy - h / 2 - 1],
            [cx + w / 2 + 1, cy - h / 2 - 1],
            [cx + w / 2 + 1, cy + h / 2 + 1],
            [cx - w / 2 - 1, cy + h / 2 + 1],
            [cx - w / 2 - 1, cy - h / 2 - 1],
        ]
        points = np.array(points)
        plt.plot(points[:,0], points[:,1],color='yellow')
    if codes[0] > 0.5:
        points = [
            [cx - w / 2 + 1, cy - h / 2 + 1],
            [cx + w / 2 - 1, cy - h / 2 + 1],
            [cx + w / 2 - 1, cy + h / 2 - 1],
            [cx - w / 2 + 1, cy + h / 2 - 1],
            [cx - w / 2 + 1, cy - h / 2 + 1],
        ]
        points = np.array(points)
        plt.plot(points[:,0], points[:,1],color='magenta')
    if codes[2] > 0.5:
        points = [
            [cx - w / 2 + 2, cy - h / 2 + 2],
            [cx + w / 2 - 2, cy - h / 2 + 2],
            [cx + w / 2 - 2, cy + h / 2 - 2],
            [cx - w / 2 + 2, cy + h / 2 - 2],
            [cx - w / 2 + 2, cy - h / 2 + 2],
        ]
        points = np.array(points)
        plt.plot(points[:,0], points[:,1],color='blue')

    if cid < 0x10FFFF:
        pred_char = chr(cid)
    else:
        pred_char = None
    if pred_char:
        if codes[0] > 0.5:
            c = 'green'
        else:
            c = 'blue'
        plt.gca().text(cx, cy, pred_char, fontsize=28, color=c, family='serif')
    plt.gca().text(cx - w/2, cy + h/2, '%.2f'%(p*100), color='green')
    #print(pred_char,cx,cy,w,h,p)

plt.show()
