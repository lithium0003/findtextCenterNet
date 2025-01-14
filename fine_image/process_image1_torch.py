#!/usr/bin/env python3

import torch

import numpy as np
import sys
from PIL import Image
import itertools
import json

from util_func import calc_predid, width, height, scale, feature_dim, sigmoid
from models.detector import TextDetectorModel, CenterNetDetector, CodeDecoder

if len(sys.argv) < 2:
    print(sys.argv[0],'target.png','(twopass)')
    exit(1)

target_file = sys.argv[1]
model_size = 'xl'
resize = 1.0
cutoff = 0.4
if len(sys.argv) > 2:
    for arg in sys.argv[2:]:
        if arg.startswith('cutoff='):
            cutoff = float(arg.split('=')[1])
            print('cutoff: ', cutoff)
        elif arg == 's':
            model_size = 's'
            print('model s')
        elif arg == 'm':
            model_size = 'm'
            print('model m')
        elif arg == 'l':
            model_size = 'l'
            print('model l')
        elif arg == 'xl':
            model_size = 'xl'
            print('model xl')
        elif arg.startswith('x'):
            resize = float(arg[1:])
            print('resize: ', resize)

model = TextDetectorModel(model_size=model_size)
data = torch.load('model.pt', map_location="cpu", weights_only=True)
model.load_state_dict(data['model_state_dict'])
detector = CenterNetDetector(model.detector)
decoder = CodeDecoder(model.decoder)
if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'
device = torch.device(device)
detector.to(device=device)
decoder.to(device=device)
detector.eval()
decoder.eval()

def eval(ds, org_img, cut_off = 0.5):
    print(org_img.shape)
    print("test")

    locations = [np.zeros(5+4)]
    glyphfeatures = [np.zeros(feature_dim, dtype=np.float32)]
    lines_all = np.zeros([org_img.shape[0] // scale, org_img.shape[1] // scale])
    seps_all = np.zeros([org_img.shape[0] // scale, org_img.shape[1] // scale])
    code_all = []
    for _ in range(4):
        code_all.append(np.zeros([org_img.shape[0] // scale, org_img.shape[1] // scale]))

    for n, inputs in enumerate(ds):
        print(n, '/', len(ds))
        x_i = inputs['offsetx']
        y_i = inputs['offsety']
        x_is = x_i // scale
        y_is = y_i // scale
        x_s = width // scale
        y_s = height // scale

        images = torch.from_numpy(inputs['input'] / 255.).permute(0,3,1,2).to(device=device)
        with torch.no_grad():
            heatmap, features = detector(images)
            heatmap = heatmap.cpu().numpy()
            features = features.cpu().numpy()

        mask = np.zeros([y_s, x_s], dtype=np.float32)
        x_min = int(x_s * 1 / 16) if x_i > 0 else 0
        x_max = int(x_s * 15 / 16) if x_i + width < org_img.shape[1] else x_s
        y_min = int(y_s * 1 / 16) if y_i > 0 else 0
        y_max = int(y_s * 15 / 16) if y_i + height < org_img.shape[0] else y_s
        mask[y_min:y_max, x_min:x_max] = 1

        line_p = sigmoid(heatmap[0,4,:,:])
        seps_p = sigmoid(heatmap[0,5,:,:])
        code_p = []
        for k in range(4):
            code_p.append(sigmoid(heatmap[0,6+k,:,:]))

        lines_all[y_is:y_is+y_s,x_is:x_is+x_s] = np.maximum(line_p * mask, lines_all[y_is:y_is+y_s,x_is:x_is+x_s])
        seps_all[y_is:y_is+y_s,x_is:x_is+x_s] = np.maximum(seps_p * mask, seps_all[y_is:y_is+y_s,x_is:x_is+x_s])
        for k in range(4):
            code_all[k][y_is:y_is+y_s,x_is:x_is+x_s] = np.maximum(code_p[k] * mask, code_all[k][y_is:y_is+y_s,x_is:x_is+x_s])

        peak = sigmoid(heatmap[0,1,:,:]) * mask
        idxy, idxx  = np.unravel_index(np.argsort(-peak.ravel()), peak.shape)

        for y, x in zip(idxy, idxx):
            if peak[y,x] < cut_off:
                break
            w = np.exp(heatmap[0,2,y,x] - 3) * 1024
            h = np.exp(heatmap[0,3,y,x] - 3) * 1024
            if w <= 0 or h <= 0:
                continue
            if w > org_img.shape[1] or h > org_img.shape[0]:
                continue
            ix = x * scale + x_i
            iy = y * scale + y_i

            codes = []
            for k in range(4):
                codes.append(code_p[k][y,x])

            locations.append(np.array([peak[y,x], ix, iy, w, h, *codes]))
            glyphfeatures.append(features[0,:,y,x])

    locations = np.array(locations)
    glyphfeatures = np.array(glyphfeatures)

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
        fill_map = np.zeros([int(w), int(h)], dtype=bool)
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
            if inter_vol.max() > area0_vol * 0.75:
                continue
            dx = done_area[:,0] - cx
            dy = done_area[:,1] - cy
            d = np.sqrt(dx * dx + dy * dy)
            if d.min() < max(8, min(w,h)/2):
                continue
            idx_overlap = np.where(iou > 0)[0]
            for j in idx_overlap:
                cx1 = done_area[j,0]
                cy1 = done_area[j,1]
                w1 = done_area[j,2]
                h1 = done_area[j,3]
                p1x = int(max(cx1 - w1/2, cx - w/2) - (cx - w/2))
                p2x = int(min(cx1 + w1/2, cx + w/2) - (cx - w/2))
                p1y = int(max(cy1 - h1/2, cy - h/2) - (cy - h/2))+1
                p2y = int(min(cy1 + h1/2, cy + h/2) - (cy - h/2))+1
                fill_map[p1x:p2x,p1y:p2y] = True
            if np.mean(fill_map) > 0.5:
                continue

        done_area = np.vstack([done_area, np.array([cx, cy, w, h])])
        selected_idx.append(i)

    idx = selected_idx
    selected_idx = []
    for i in idx:
        cx = locations[i,1]
        cy = locations[i,2]
        x = int(cx / scale)
        y = int(cy / scale)
        if x >= 0 and x < org_img.shape[1] // scale and y >= 0 and y < org_img.shape[0] // scale:
            if seps_all[y,x] > 0.1:
                continue
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
        w = locations[i,3]
        h = locations[i,4]
        x = int(cx / scale)
        y = int(cy / scale)
        x_min = int(cx / scale - 1)
        y_min = int(cy / scale - 1)
        x_max = int(cx / scale + 1) + 1
        y_max = int(cy / scale + 1) + 1
        if x >= 0 and x < org_img.shape[1] // scale and y >= 0 and y < org_img.shape[0] // scale:
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(org_img.shape[1] // scale, x_max)
            y_max = min(org_img.shape[0] // scale, y_max)
            for k in range(4):
                locations[i,5+k] = max(np.max(code_all[k][y_min:y_max,x_min:x_max]), locations[i,5+k])

    return locations, glyphfeatures, lines_all, seps_all

def decode(glyphfeatures):
    print("decode")
    glyphids = []
    glyphprobs = []
    for data in glyphfeatures:
        with torch.no_grad():
            decode_outputs = decoder(torch.from_numpy(data).to(device=device).unsqueeze(0))
        p = []
        id = []
        for k,prob in enumerate(decode_outputs):
            prob = prob[0].cpu().numpy()
            idx = np.where(prob > 0.01)[0]
            if len(idx) == 0:
                idx = [np.argmax(prob)]
            if k == 0:
                for i in idx[:3]:
                    id.append([i])
                    p.append([prob[i]])
            else:
                id = [i1 + [i2] for i1, i2 in itertools.product(id, idx[:3])]
                p = [i1 + [prob[i2]] for i1, i2 in itertools.product(p, idx[:3])]
        p = [np.exp(np.mean([np.log(prob) for prob in probs])) for probs in p]
        i = [calc_predid(*ids) for ids in id]
        g = sorted([(prob, id) for prob,id in zip(p,i)], key=lambda x: x[0] if x[1] <= 0x10FFFF else 0, reverse=True)
        prob,idx = g[0]
        glyphids.append(idx)
        glyphprobs.append(prob)

    glyphids = np.atleast_1d(glyphids)
    glyphprobs = np.atleast_1d(glyphprobs)
    
    return  glyphids, glyphprobs

stepx = width * 1 // 2
stepy = height * 1 // 2

im0 = Image.open(target_file).convert('RGB')
if resize != 1.0:
    im0 = im0.resize((int(im0.width * resize), int(im0.height * resize)), resample=Image.Resampling.BILINEAR)
#im0 = im0.filter(ImageFilter.SHARPEN)
im0 = np.asarray(im0)

padx = max(0, (width - im0.shape[1]) % stepx, width - im0.shape[1])
pady = max(0, (height - im0.shape[0]) % stepy, height - im0.shape[0])
im0 = np.pad(im0, [[0,pady],[0,padx],[0,0]], 'constant', constant_values=((255,255),(255,255),(255,255)))

im = im0.astype(np.float32)

ds0 = []
for y in range(0, im0.shape[0] - height + 1, stepy):
    for x in range(0, im0.shape[1] - width + 1, stepx):
        ds0.append({
            'input': np.expand_dims(im[y:y+height,x:x+width,:], 0),
            'offsetx': x,
            'offsety': y,
        })

locations, glyphfeatures, lines_all, seps_all = eval(ds0, im, cut_off=cutoff)
glyphids, glyphprobs = decode(glyphfeatures)

linesfile = target_file + '.lines.png'
lines_all = (lines_all * 255).astype(np.uint8)
Image.fromarray(lines_all).save(linesfile)

sepsfile = target_file + '.seps.png'
seps_all = (seps_all * 255).astype(np.uint8)
Image.fromarray(seps_all).save(sepsfile)

out_dict = {}
out_dict['textbox'] = []
for i, loc in enumerate(locations):
    p_loc = loc[0]
    cx = loc[1]
    cy = loc[2]
    w = loc[3]
    h = loc[4]
    codes = loc[5:]
    cid = glyphids[i]
    p_chr = glyphprobs[i]
    if cid < 0x10FFFF:
        pred_char = chr(cid)
    else:
        pred_char = None

    out_dict['textbox'].append({
        'cx': float(cx),
        'cy': float(cy),
        'w': float(w),
        'h': float(h),
        'text': pred_char,
        'p_loc': float(p_loc),
        'p_chr': float(p_chr),
        'p_code1': float(codes[0]),
        'p_code2': float(codes[1]),
        'p_code4': float(codes[2]),
        'p_code8': float(codes[3]),
    })

with open(target_file+'.json', 'w', encoding='utf-8') as file:
    json.dump(out_dict, file, indent=2, ensure_ascii=False)
