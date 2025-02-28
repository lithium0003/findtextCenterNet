#!/usr/bin/env python3

import coremltools as ct

import numpy as np
import sys
import time
from PIL import Image, ImageFilter, ImageEnhance
import itertools
import subprocess

try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass

import matplotlib.pyplot as plt

from util_func import calc_predid, width, height, scale, feature_dim, modulo_list, sigmoid, softmax, decode_ruby
from const import encoder_add_dim, max_encoderlen, max_decoderlen, decoder_SOT, decoder_EOT, decoder_MSK
encoder_dim = feature_dim + encoder_add_dim

if len(sys.argv) < 2:
    print(sys.argv[0],'target.png','(resize ratio)')
    exit(1)

cutoff = 0.4

step_ratio = 0.6
stepx = int(width * step_ratio)
stepy = int(height * step_ratio)

target_file = sys.argv[1]
resize = 1.0
if len(sys.argv) > 2:
    resize = float(sys.argv[2])
    print('resize: ', resize)

print('load')
mlmodel_detector = ct.models.MLModel('TextDetector.mlpackage')

mlmodel_transformer_encoder = ct.models.MLModel('TransformerEncoder.mlpackage')
mlmodel_transformer_decoder = ct.models.MLModel('TransformerDecoder.mlpackage')

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

        input_image = Image.fromarray(inputs['input'], mode="RGB")
        # plt.figure()
        # plt.imshow(input_image)
        # plt.show()
        # input_image.save('tmp%08d.png'%n)

        output = mlmodel_detector.predict({'image': input_image})
        heatmap = output['heatmap']
        features = output['feature']

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
            if seps_all[y,x] > 0.25:
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


im0 = Image.open(target_file).convert('RGB')
if resize != 1.0:
    im0 = im0.resize((int(im0.width * resize), int(im0.height * resize)), resample=Image.Resampling.BILINEAR)
im0 = np.asarray(im0)

padx = max(0, (width - im0.shape[1]) % stepx, width - im0.shape[1])
pady = max(0, (height - im0.shape[0]) % stepy, height - im0.shape[0])
im0 = np.pad(im0, [[0,pady],[0,padx],[0,0]], 'constant', constant_values=((255,255),(255,255),(255,255)))

locations0, glyphfeatures0 = None, None

ds0 = []
for y in range(0, im0.shape[0] - height + 1, stepy):
    for x in range(0, im0.shape[1] - width + 1, stepx):
        ds0.append({
            'input': im0[y:y+height,x:x+width,:],
            'offsetx': x,
            'offsety': y,
        })
locations, glyphfeatures, lines, seps = eval(ds0, im0, cut_off=cutoff,
        locations0=locations0, glyphfeatures0=glyphfeatures0)

def output_process2(output, i):
    pred_ids = []
    for m in modulo_list:
        pred_id1 = np.argmax(output['modulo_%d'%m][0,i], axis=-1)
        pred_ids.append(pred_id1)
    id = calc_predid(*pred_ids)
    return id

def output_process1(output, i):
    id = output_process2(output, i)
    if id < 0x10FFFF:
        return id
    p = []
    id = []
    for k,m in enumerate(modulo_list):
        prob = softmax(output['modulo_%d'%m][0,i])
        idx = np.where(prob > 0.1)[0]
        if len(idx) == 0:
            idx = [np.argmax(prob)]
        if k == 0:
            for j in idx:
                id.append([j])
                p.append([prob[j]])
        else:
            id = [i1 + [i2] for i1, i2 in itertools.product(id, idx)]
            p = [i1 + [prob[i2]] for i1, i2 in itertools.product(p, idx)]
    p = [np.exp(np.mean([np.log(prob) for prob in probs])) for probs in p]
    i = [calc_predid(*ids) for ids in id]
    g = sorted([(prob, id) for prob,id in zip(p,i)], key=lambda x: x[0] if x[1] <= 0x10FFFF else 0, reverse=True)
    prob,id = g[0]
    return id

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

# print(detected_boxes)


features = []
prev_block = 0
prev_idx = 0
vertical = 0
for id, block, idx, subidx, subtype in detected_boxes:
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
        g = np.zeros([encoder_dim], np.float16)
        g[feature_dim+0] = 5 * vertical
        g[-1] = 5
        features.append(g)
        prev_idx = -1
    if prev_idx != idx:
        prev_idx = idx
        g = np.zeros([encoder_dim], np.float16)
        g[feature_dim+0] = 5 * vertical
        g[-1] = 5
        features.append(g)

    if subtype & (2+4) == 2+4:
        ruby = 1
    elif subtype & (2+4) == 2:
        rubybase = 1
    
    if subtype & 8 == 8:
        space = 1

    if subtype & 16 == 16:
        emphasis = 1

    if subtype & 1 == 0:
        vertical = 0
    else:
        vertical = 1
    
    g = np.concatenate([glyphfeatures[id,:], 5*np.array([vertical,rubybase,ruby,space,emphasis,0], np.float16)])
    features.append(g)
features = np.array(features, np.float16)
SP_token = np.zeros([encoder_dim], dtype=np.float16)
SP_token[0:feature_dim:2] = 5
SP_token[1:feature_dim:2] = -5

start_t = time.time()
cur_i = 0
result_txt = ''
loop_count = 0
while cur_i < features.shape[0]:
    loop_count += 1
    r = 3
    s = 0
    for k in range(cur_i, min(cur_i + max_encoderlen, features.shape[0])):
        # space
        if features[k,-3] > 0:
            r += 1
        # newline
        if features[k,-1] > 0:
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
    cur_j = min(features.shape[0], cur_i + (max_encoderlen - r))
    for j in range(cur_i+1, cur_j):
        if features[j-1,-6] != features[cur_i,-6]:
            cur_j = j
            break
    k = cur_j
    # newline
    if cur_j < features.shape[0]:
        if cur_j > 2:
            while features[cur_j-1,-1] == 0 or features[cur_j-2,-1] == 0:
                if cur_j - 2 <= cur_i + 1:
                    cur_j = k
                    while features[cur_j-1,-1] == 0:
                        if cur_j - 1 <= cur_i + 1:
                            cur_j = k
                            break
                        cur_j -= 1
                    break
                cur_j -= 1
    print(cur_i,cur_j,'/',features.shape[0])
    encoder_input = np.zeros(shape=(1, max_encoderlen, encoder_dim), dtype=np.float16)
    encoder_input[0,0,:] = SP_token
    encoder_input[0,1:1+cur_j-cur_i,:] = features[cur_i:cur_j,:]
    encoder_input[0,1+cur_j-cur_i,:] = -SP_token
    key_mask = np.repeat(np.where((encoder_input == 0).all(axis=-1)[:,None,None,:], float("-inf"), 0), max_encoderlen, axis=2)
    encoder_output = mlmodel_transformer_encoder.predict({
        'encoder_input': encoder_input, 
        'key_mask': key_mask,
    })['encoder_output']

    decoder_input = np.zeros(shape=(1, max_decoderlen), dtype=np.int32)
    decoder_input[0,0] = decoder_SOT
    decoder_input[0,1:] = decoder_MSK
    rep_count = 16
    for k in range(rep_count):
        output = mlmodel_transformer_decoder.predict({
            'encoder_output': encoder_output,
            'decoder_input': decoder_input,
            'key_mask': key_mask,
        })

        listp = []
        listi = []
        for m in modulo_list:
            pred_p1 = softmax(output['modulo_%d'%m])
            topi = np.argpartition(-pred_p1, 5, axis=-1)[...,:5]
            topp = np.take_along_axis(pred_p1, topi, axis=-1)
            listp.append(np.transpose(topp, (2,0,1)))
            listi.append(np.transpose(topi, (2,0,1)))

        pred_ids = np.stack([np.stack(x) for x in itertools.product(*listi)])
        pred_p = np.stack([np.stack(x) for x in itertools.product(*listp)])
        pred_ids = np.transpose(pred_ids, (1,0,2,3))
        pred_p = np.transpose(pred_p, (1,0,2,3))
        pred_p = np.exp(np.mean(np.log(np.maximum(pred_p, 1e-10)), axis=0))
        decoder_output = calc_predid(*pred_ids)
        pred_p[decoder_output > 0x3FFFF] = 0
        maxi = np.argmax(pred_p, axis=0)
        decoder_output = np.take_along_axis(decoder_output, maxi[None,...], axis=0)[0]
        pred_p = np.take_along_axis(pred_p, maxi[None,...], axis=0)[0]
        if k > 0 and np.all(pred_p[decoder_output > 0] > 0.99):
            print(f'[{k} early stop]')
            break
        if k < rep_count-1:
            decoder_input[:,1:] = np.where(pred_p < 1/rep_count*(k+1), decoder_MSK, decoder_output)[:,:-1]
    pred = decoder_output[0]
    predstr = ''
    # print(pred)
    for p in pred:
        if p == 0 or p == decoder_EOT:
            break
        if p >= 0xD800 and p <= 0xDFFF:
            predstr += '\uFFFD'
        elif p < 0x3FFFF:
            predstr += chr(p)
        else:
            predstr += '\uFFFD'
    result_txt += predstr
    cur_i = cur_j

spent_t = time.time() - start_t
print(spent_t, 'sec', spent_t / loop_count * 1000, 'ms/loop', spent_t / features.shape[0] * 1000, 'ms/char')
print("---------------------")
print(decode_ruby(result_txt))
