#!/usr/bin/env python3

import torch

import numpy as np
import sys
import os
from PIL import Image, ImageFilter
import itertools
import subprocess

try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass

from util_func import calc_predid, width, height, scale, feature_dim, sigmoid, decode_ruby
from models.detector import TextDetectorModel, CenterNetDetector, CodeDecoder
from models.transformer import ModelDimensions, Transformer, TransformerPredictor
from const import encoder_add_dim, max_encoderlen
encoder_dim = feature_dim + encoder_add_dim

if len(sys.argv) < 2:
    print(sys.argv[0],'target.png','(resize ratio)')
    exit(1)

step_ratio = 0.66
stepx = int(width * step_ratio)
stepy = int(height * step_ratio)

target_file = sys.argv[1]
model_size = 'xl'
resize = 1.0
if len(sys.argv) > 2:
    for arg in sys.argv[2:]:
        if arg == 's':
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
if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'
device = torch.device(device)
detector.to(device=device)
detector.eval()

if os.path.exists('model3.pt'):
    data = torch.load('model3.pt', map_location="cpu", weights_only=True)
    config = ModelDimensions(**data['config'])
    model = Transformer(**config.__dict__)
    model.load_state_dict(data['model_state_dict'])
else:
    config = ModelDimensions()
    model = Transformer(**config.__dict__)
model2 = TransformerPredictor(model.encoder, model.decoder)
model2.to(device)
model2.eval()

def cluster_dist(hist):
    sum_y = np.sum(hist)
    if sum_y == 0:
        return 0

    i = np.arange(hist.shape[0])
    v = hist * i
    mean_y = np.sum(v) / sum_y
    s1 = np.sum(hist[:int(mean_y+0.5)])
    s2 = np.sum(hist[int(mean_y+0.5):])
    if s1 == 0:
        return 0
    if s2 == 0:
        return 0
    g1 = v[:int(mean_y+0.5)]
    g2 = v[int(mean_y+0.5):]
    k1 = np.sum(g1) / s1
    k2 = np.sum(g2) / s2
    dist1 = 256.0
    dist2 = abs(k1 - k2)
    while dist1 != dist2:
        dist1 = dist2
        s1 = np.sum(hist[np.abs(i - k1) < np.abs(i - k2)])
        s2 = np.sum(hist[np.abs(i - k1) >= np.abs(i - k2)])
        if s1 == 0:
            return 0
        if s2 == 0:
            return 0
        g1 = v[np.abs(i - k1) < np.abs(i - k2)]
        g2 = v[np.abs(i - k1) >= np.abs(i - k2)]
        k1 = np.sum(g1) / s1
        k2 = np.sum(g2) / s2
        dist2 = abs(k1 - k2)
    return dist1

def imageHist(im):
    maxPeakDiff = -1
    maxPeakDiff = max(maxPeakDiff, cluster_dist(np.histogram(im[:,:,0], bins=256, range=(0,256))[0]))
    maxPeakDiff = max(maxPeakDiff, cluster_dist(np.histogram(im[:,:,1], bins=256, range=(0,256))[0]))
    maxPeakDiff = max(maxPeakDiff, cluster_dist(np.histogram(im[:,:,2], bins=256, range=(0,256))[0]))
    return maxPeakDiff

def eval(ds, org_img, cut_off = 0.5, locations0 = None, glyphfeatures0 = None):
    print(org_img.shape)
    print("test")

    locations = [np.zeros(5+4)]
    glyphfeatures = [np.zeros(feature_dim, dtype=np.float32)]
    keymap_all = np.zeros([org_img.shape[0] // scale, org_img.shape[1] // scale], dtype=np.float32)
    lines_all = np.zeros([org_img.shape[0] // scale, org_img.shape[1] // scale], dtype=np.float32)
    seps_all = np.zeros([org_img.shape[0] // scale, org_img.shape[1] // scale], dtype=np.float32)
    code_all = []
    for _ in range(4):
        code_all.append(np.zeros([org_img.shape[0] // scale, org_img.shape[1] // scale], dtype=np.float32))

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

        mask = np.zeros([y_s, x_s], dtype=bool)
        x_min = int(x_s * (1-step_ratio)/2) if x_i > 0 else 0
        x_max = int(x_s * (1-(1-step_ratio)/2)) + 1 if x_i + width < org_img.shape[1] else x_s
        y_min = int(y_s * (1-step_ratio)/2) if y_i > 0 else 0
        y_max = int(y_s * (1-(1-step_ratio)/2)) + 1 if y_i + height < org_img.shape[0] else y_s
        mask[y_min:y_max, x_min:x_max] = True

        keymap_p = sigmoid(heatmap[0,0,:,:])
        line_p = sigmoid(heatmap[0,4,:,:])
        seps_p = sigmoid(heatmap[0,5,:,:])
        code_p = []
        for k in range(4):
            code_p.append(sigmoid(heatmap[0,6+k,:,:]))

        keymap_all[y_is:y_is+y_s,x_is:x_is+x_s] = np.maximum(keymap_p * mask, keymap_all[y_is:y_is+y_s,x_is:x_is+x_s])
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
    if locations0 is not None:
        locations = np.concatenate([locations, locations0])
    glyphfeatures = np.array(glyphfeatures)
    if glyphfeatures0 is not None:
        glyphfeatures = np.concatenate([glyphfeatures, glyphfeatures0])

    hists = []
    for i in range(locations.shape[0]):
        p = locations[i,0]
        if p < cut_off:
            continue
        cx = locations[i,1]
        cy = locations[i,2]
        w = locations[i,3]
        h = locations[i,4]
        x_min = int(cx - w/2) - 1
        x_max = int(cx + w/2) + 2
        y_min = int(cy - h/2) - 1
        y_max = int(cy + h/2) + 2
        hists.append(imageHist(org_img[y_min:y_max,x_min:x_max,:]))
    th_hist = np.median(hists) / 5

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
        x_min = max(0, int(cx - w/2))
        x_max = min(org_img.shape[1] - 1, int(cx + w/2) + 1)
        y_min = max(0, int(cy - h/2))
        y_max = min(org_img.shape[0] - 1, int(cy + h/2) + 1)
        if imageHist(org_img[y_min:y_max,x_min:x_max,:]) < th_hist:
            continue
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
            # dx = done_area[:,0] - cx
            # dy = done_area[:,1] - cy
            # d = np.sqrt(dx * dx + dy * dy)
            # if d.min() < max(8, min(w,h)/2):
            #     continue
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
            if seps_all[y,x] > 0.5:
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

    return locations.astype(np.float32), glyphfeatures, lines_all, seps_all


im0 = Image.open(target_file).convert('RGB')
if resize != 1.0:
    im0 = im0.resize((int(im0.width * resize), int(im0.height * resize)), resample=Image.Resampling.BILINEAR)
im0 = np.asarray(im0)

padx = max(0, (width - im0.shape[1]) % stepx, width - im0.shape[1])
pady = max(0, (height - im0.shape[0]) % stepy, height - im0.shape[0])
im0 = np.pad(im0, [[0,pady],[0,padx],[0,0]], 'constant', constant_values=((255,255),(255,255),(255,255)))

locations0, glyphfeatures0 = None, None

im = im0.astype(np.float32)

ds0 = []
for y in range(0, im0.shape[0] - height + 1, stepy):
    for x in range(0, im0.shape[1] - width + 1, stepx):
        ds0.append({
            'input': np.expand_dims(im[y:y+height,x:x+width,:], 0),
            'offsetx': x,
            'offsety': y,
        })

locations, glyphfeatures, lines, seps = eval(ds0, im, cut_off=0.35,
        locations0=locations0, glyphfeatures0=glyphfeatures0)

h, w = lines.shape
input_binary = int(0).to_bytes(4, 'little')
input_binary += int(w).to_bytes(4, 'little')
input_binary += int(h).to_bytes(4, 'little')
input_binary += lines.tobytes()
input_binary += seps.tobytes()
input_binary += int(locations.shape[0]).to_bytes(4, 'little')
input_binary += locations[:,1:].tobytes()

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
    pageidx = int.from_bytes(result[p:p+4], byteorder='little', signed=True)
    p += 4
    sectionidx = int.from_bytes(result[p:p+4], byteorder='little', signed=True)
    p += 4
    detected_boxes.append((id,block,idx,subidx,subtype,pageidx,sectionidx))

# print(detected_boxes)

features = []
prev_block = 0
prev_idx = 0
vertical = 0
for id, block, idx, subidx, subtype, pageidx, sectionidx in detected_boxes:
    if id < 0:
        continue

    # 1 vertical
    # 2 ruby (base)
    # 3 ruby (text)
    # 4 space
    # 5 emphasis
    # 6 newline

    ruby = 0
    rubybase = 0
    space = 0
    emphasis = 0

    if prev_block != block:
        prev_block = block
        g = np.zeros([encoder_dim], np.float32)
        g[feature_dim+0] = 5 * vertical
        g[-1] = 5
        features.append(g)
        prev_idx = -1
    if prev_idx != idx:
        prev_idx = idx
        g = np.zeros([encoder_dim], np.float32)
        g[feature_dim+0] = 5 * vertical
        g[-1] = 5
        features.append(g)

    if subtype & 2+4 == 2+4:
        ruby = 1
    elif subtype & 2+4 == 2:
        rubybase = 1
    
    if subtype & 8 == 8:
        space = 1

    if subtype & 16 == 16:
        emphasis = 1

    if subtype & 1 == 0:
        vertical = 0
    else:
        vertical = 1
    
    g = np.concatenate([glyphfeatures[id,:], 5*np.array([vertical,rubybase,ruby,space,emphasis,0], np.float32)])
    features.append(g)

features = np.array(features, np.float32)
SP_token = np.zeros([encoder_dim], dtype=np.float32)
SP_token[0:feature_dim:2] = 5
SP_token[1:feature_dim:2] = -5

cur_i = 0
prev_j = 0
result_txt = ''
loop_count = 0
keep_back = 0
while cur_i < features.shape[0]:
    r = 0
    s = 0
    for k in range(cur_i, min(cur_i + max_encoderlen - 3, features.shape[0])):
        # space
        if features[k,-3] > 0:
            r += 1
        # rubybase
        if s == 0 and features[k,-5] > 0:
            r += 3
            s = 1
        # ruby
        elif s == 1 and features[k,-4] > 0:
            s = 2
        elif s == 2 and features[k,-4] == 0:
            s = 0
    cur_j = min(features.shape[0], cur_i + (max_encoderlen - 3 - r))
    # horizontal / vertical change point
    for j in range(cur_i+1, cur_j):
        if features[j,-6] != features[cur_i,-6]:
            cur_j = j
            break
    # double newline
    if cur_j < features.shape[0]-1 and cur_i+1 < cur_j-1:
        for j in range(cur_i+1, cur_j-1):
            if features[j,-1] > 0 and features[j+1,-1] > 0:
                cur_j = j+2
                break
    # ruby/rubybase sepatation check
    if cur_j < features.shape[0]:
        # last char is not newline
        if cur_j > 1 and features[cur_j-1, -1] == 0:
            for j in reversed(range(cur_i+1, cur_j)):
                # ruby, ruby base
                if features[j,-4] == 0 and features[j,-5] == 0:
                    cur_j = j+1
                    break

    if prev_j == cur_j:
        keep_back = 0
        cur_i = cur_j
        continue

    print(cur_i,cur_j,'/',features.shape[0])
    encoder_input = np.zeros(shape=(1,max_encoderlen, encoder_dim), dtype=np.float32)
    encoder_input[0,0,:] = SP_token
    encoder_input[0,1:1+cur_j-cur_i,:] = features[cur_i:cur_j,:]
    encoder_input[0,1+cur_j-cur_i,:] = -SP_token

    encoder_input = torch.tensor(encoder_input, device=device)
    pred = model2(encoder_input).squeeze(0).cpu().numpy()
    predstr = ''
    for p in pred:
        if p == 0:
            break
        if p < 0x3FFFF:
            predstr += chr(p)
        else:
            predstr += '\uFFFD'
    # print(keep_back, predstr)
    result_txt += predstr[keep_back:]

    if cur_j < features.shape[0]:
        k = cur_j - 1
        prev_j = cur_j
        keep_back = 0
        while cur_i < k:
            # horizontal / vertical change point
            if features[k,-6] != features[cur_j,-6]:
                k += 1
                break
            # ruby, ruby base
            if features[k,-5] > 0 or features[k,-4] > 0:
                k += 1
                break
            # newline
            if k < cur_j - 1 and features[k,-1] > 0:
                k += 1
                break
            # space
            if features[k,-3] > 0:
                keep_back += 1
            if k > cur_j - 3:
                k -= 1
            else:
                break
        if cur_i < k:
            cur_i = k
            keep_back += cur_j - k
        else:
            keep_back = 0
            cur_i = cur_j
    else:
        break

print("---------------------")
print(decode_ruby(result_txt))
