#!/usr/bin/env python3

import coremltools as ct

import numpy as np
import sys
from PIL import Image
import itertools
import json
import glob
import subprocess

try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass

from util_func import calc_predid, width, height, scale, feature_dim, modulo_list, sigmoid

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
    else:
        target_files += glob.glob(arg)

target_files = sorted(target_files)

print('load')
mlmodel_detector = ct.models.MLModel('TextDetector.mlpackage')
mlmodel_decoder = ct.models.MLModel('CodeDecoder.mlpackage')

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

def eval(ds, org_img, cut_off = 0.5):
    print(org_img.shape)
    print("test")

    locations = [np.zeros(5+4)]
    glyphfeatures = [np.zeros(feature_dim, dtype=np.float32)]
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

        input_image = Image.fromarray(inputs['input'], mode="RGB")

        output = mlmodel_detector.predict({'image': input_image})
        heatmap = output['heatmap']
        features = output['feature']

        mask = np.zeros([y_s, x_s], dtype=bool)
        x_min = int(x_s * 1 / 8) if x_i > 0 else 0
        x_max = int(x_s * 7 / 8) + 1 if x_i + width < org_img.shape[1] else x_s
        y_min = int(y_s * 1 / 8) if y_i > 0 else 0
        y_max = int(y_s * 7 / 8) + 1 if y_i + height < org_img.shape[0] else y_s
        mask[y_min:y_max, x_min:x_max] = True

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
    th_hist = np.median(hists) / 10

    idx = np.argsort(-locations[:,0])
    donefill_map = np.empty([org_img.shape[0], org_img.shape[1]], dtype=int)
    donefill_map.fill(-1)
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
        positive_count = np.sum(np.abs(org_img[y_min:y_max,x_min:x_max,:] - np.mean(org_img[y_min:y_max,x_min:x_max,:], axis=(0,1), keepdims=True)) > th_hist) / org_img.shape[2]
        if positive_count / area0_vol < 0.1:
            continue
        valid = True
        for di in np.unique(donefill_map[y_min:y_max,x_min:x_max]):
            if di < 0:
                continue
            p_cx = locations[di,1]
            p_cy = locations[di,2]
            p_w = locations[di,3]
            p_h = locations[di,4]

            area1_vol = p_w * p_h
            inter_xmin = max(cx - w / 2, p_cx - p_w / 2)
            inter_ymin = max(cy - h / 2, p_cy - p_h / 2)
            inter_xmax = min(cx + w / 2, p_cx + p_w / 2)
            inter_ymax = min(cy + h / 2, p_cy + p_h / 2)
            inter_w = max(inter_xmax - inter_xmin, 0.)
            inter_h = max(inter_ymax - inter_ymin, 0.)
            inter_vol = inter_w * inter_h
            union_vol = area0_vol + area1_vol - inter_vol
            if union_vol > 0:
                iou = inter_vol / union_vol
            else:
                iou = 0

            if iou > 0.25:
                valid = False
                break
            if inter_vol > area0_vol * 0.95:
                valid = False
                break
            if np.sum(donefill_map[y_min:y_max,x_min:x_max] == di) > area1_vol * 0.95:
                valid = False
                break
        if not valid:
            continue
        donefill_map[y_min:y_max,x_min:x_max] = np.where(donefill_map[y_min:y_max,x_min:x_max] < 0, i, donefill_map[y_min:y_max,x_min:x_max])
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
        x_min = int(cx / scale - 1)
        y_min = int(cy / scale - 1)
        x_max = int(cx / scale + 1) + 1
        y_max = int(cy / scale + 1) + 1
        x = int(cx / scale)
        y = int(cy / scale)
        if x >= 0 and x < org_img.shape[1] // scale and y >= 0 and y < org_img.shape[0] // scale:
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(org_img.shape[1] // scale, x_max)
            y_max = min(org_img.shape[0] // scale, y_max)
            for k in range(4):
                locations[i,5+k] = max(np.max(code_all[k][y_min:y_max,x_min:x_max]), locations[i,5+k])

    return locations.astype(np.float32), glyphfeatures, lines_all, seps_all

def decode(glyphfeatures):
    print("decode")
    glyphids = []
    glyphprobs = []
    for data in glyphfeatures:
        decode_output = mlmodel_decoder.predict({'feature_input': np.expand_dims(data,0)})
        p = []
        id = []
        for k,m in enumerate(modulo_list):
            prob = decode_output['modulo_%d'%m][0]
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

    if len(glyphids) > 0:
        glyphids = np.stack(glyphids)
        glyphprobs = np.stack(glyphprobs)
    
    return  glyphids, glyphprobs

stepx = width * 3 // 4
stepy = height * 3 // 4

for target_file in target_files:
    print(target_file)

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

    locations, glyphfeatures, lines, seps = eval(ds0, im, cut_off=cutoff)
    glyphids, glyphprobs = decode(glyphfeatures)

    linesfile = target_file + '.lines.png'
    lines_all = (lines * 255).astype(np.uint8)
    Image.fromarray(lines_all).save(linesfile)

    sepsfile = target_file + '.seps.png'
    seps_all = (seps * 255).astype(np.uint8)
    Image.fromarray(seps_all).save(sepsfile)

    print('construct data')
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
        detected_boxes.append((id,block,idx,subidx,subtype))

    print(detected_boxes)

    out_dict = {}
    out_dict['textbox'] = []
    for id, block, idx, subidx, subtype in detected_boxes:
        if id < 0:
            continue
        p_loc = locations[id, 0]
        cx = locations[id, 1]
        cy = locations[id, 2]
        w = locations[id, 3]
        h = locations[id, 4]
        codes = locations[id, 5:]
        cid = glyphids[id]
        p_chr = glyphprobs[id]
        g = glyphfeatures[id]
        if cid < 0x10FFFF:
            pred_char = chr(cid)
        else:
            pred_char = None

        if subtype & 2+4 == 2:
            if codes[1] < 0.5:
                codes[1] = 1
            if codes[0] > 0.5:
                codes[0] = 0
        elif subtype & 2+4 == 2+4:
            if codes[0] < 0.5:
                codes[0] = 1
            if codes[1] > 0.5:
                codes[1] = 0
        else:
            if codes[0] > 0.5:
                codes[0] = 0
            if codes[1] > 0.5:
                codes[1] = 0

        if subtype & 8 == 8:
            if codes[3] < 0.5:
                codes[3] = 1
        else:
            if codes[3] > 0.5:
                codes[3] = 0

        if subtype & 16 == 16:
            if codes[2] < 0.5:
                codes[2] = 1
        else:
            if codes[2] > 0.5:
                codes[2] = 0

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
