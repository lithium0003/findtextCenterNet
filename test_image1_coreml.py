#!/usr/bin/env python3

import coremltools as ct

import numpy as np
import sys
import os
from PIL import Image, ImageFilter, ImageEnhance
import itertools
import subprocess

try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass

from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt

from util_func import calc_predid, width, height, scale, feature_dim, modulo_list, sigmoid

if len(sys.argv) < 2:
    print(sys.argv[0],'target.png','(twopass)')
    exit(1)

fprop = FontProperties(fname='data/jpfont/NotoSerifJP-Regular.otf')
cutoff = 0.4

target_file = sys.argv[1]
twopass = False
verbose = False
resize = 1.0
offsetx = 0
offsety = 0
if len(sys.argv) > 2:
    for arg in sys.argv[2:]:
        if arg == 'twopass':
            twopass = True
            print('twopass')
        elif arg == 'kr':
            fprop = FontProperties(fname='data/krfont/NotoSerifKR-Regular.otf')
            print('kr font')
        elif arg == 'verbose':
            verbose = True
            print('verbose')
        elif arg.startswith('x'):
            resize = float(arg[1:])
            print('resize: ', resize)
        elif arg.startswith('offsetx'):
            offsetx = int(arg[7:])
            print('offsetx: ', offsetx)
        elif arg.startswith('offsety'):
            offsety = int(arg[7:])
            print('offsety: ', offsety)

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
        x_min = int(x_s * 1 / 8) if x_i > 0 else 0
        x_max = int(x_s * 7 / 8) + 1 if x_i + width < org_img.shape[1] else x_s
        y_min = int(y_s * 1 / 8) if y_i > 0 else 0
        y_max = int(y_s * 7 / 8) + 1 if y_i + height < org_img.shape[0] else y_s
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

    if verbose:
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
    
    return glyphids, glyphprobs

# stepx = width * 3 // 4
# stepy = height * 3 // 4
stepx = width * 1 // 2
stepy = height * 1 // 2

im0 = Image.open(target_file).convert('RGB')
if resize != 1.0:
    im0 = im0.resize((int(im0.width * resize), int(im0.height * resize)), resample=Image.Resampling.BILINEAR)
# im0 = im0.resize((im0.width // 2, im0.height // 2), resample=Image.Resampling.BILINEAR)
# im0 = im0.filter(ImageFilter.UnsharpMask(radius=20, percent=500, threshold=30))
# im0 = im0.resize((im0.width * 2, im0.height * 2), resample=Image.Resampling.BILINEAR)
# im0 = im0.filter(ImageFilter.SHARPEN)
# im0 = im0.filter(ImageFilter.UnsharpMask(radius=20, percent=500, threshold=30))
# enhancer = ImageEnhance.Brightness(im0)
# im0 = enhancer.enhance(1.5)
# enhancer = ImageEnhance.Contrast(im0)
# im0 = enhancer.enhance(1.5)
im0 = np.asarray(im0)
im = np.array(im0)
im_ave = np.median(im0, axis=(0,1), keepdims=True)
im0 = np.pad(im0, [[offsety,0],[offsetx,0],[0,0]], 'constant', constant_values=((255,255),(255,255),(255,255)))

padx = max(0, (width - im0.shape[1]) % stepx, width - im0.shape[1])
pady = max(0, (height - im0.shape[0]) % stepy, height - im0.shape[0])
im0 = np.pad(im0, [[0,pady],[0,padx],[0,0]], 'constant', constant_values=((255,255),(255,255),(255,255)))

im1 = np.empty_like(im0)
im1[:,:,:] = im_ave
im1[offsety:-(pady+1),offsetx:-(padx+1),:] = im0[offsety:-(pady+1),offsetx:-(padx+1),:]
im0 = im1

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

    locations0, glyphfeatures0, lines0, seps0 = eval(ds1, im1, cut_off=cutoff)
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
locations, glyphfeatures, lines, seps = eval(ds0, im0, cut_off=cutoff,
        locations0=locations0, glyphfeatures0=glyphfeatures0)
glyphids, glyphprobs = decode(glyphfeatures)

if verbose:
    plt.figure()
    plt.hist(np.reshape(glyphfeatures,[-1]), bins=50)
    plt.title('features')

    fig = plt.figure()
    plt.imshow(im0)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

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
            plt.gca().text(cx, cy, pred_char, fontsize=28, color=c, fontproperties=fprop)
        plt.gca().text(cx - w/2, cy + h/2, '%.2f'%(p*100), color='green')
else:
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
        pageidx = int.from_bytes(result[p:p+4], byteorder='little', signed=True)
        p += 4
        sectionidx = int.from_bytes(result[p:p+4], byteorder='little', signed=True)
        p += 4
        detected_boxes.append((id,block,idx,subidx,subtype,pageidx,sectionidx))

    print(detected_boxes)

    fig = plt.figure()
    plt.imshow(im / 255)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    cmap = plt.get_cmap('rainbow', max_block+1)
    for id, block, idx, subidx, subtype, pageidx, sectionidx in detected_boxes:
        if id < 0:
            continue
        cx = locations[id, 1]
        cy = locations[id, 2]
        w = locations[id, 3]
        h = locations[id, 4]
        cid = glyphids[id]
        p = glyphprobs[id]
        g = glyphfeatures[id]

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
        plt.text(cx - w/2, cy - h/2, t, color='black')

        if cid < 0x10FFFF:
            pred_char = chr(cid)
        else:
            pred_char = None
        if pred_char:
            if subtype & 2+4 == 2+4:
                c = 'green'
            else:
                c = 'blue'
            plt.gca().text(cx, cy, pred_char, fontsize=28, color=c, fontproperties=fprop)
        plt.gca().text(cx - w/2, cy + h/2, '%.2f'%(p*100), color='green')
        #print(pred_char,cx,cy,w,h,p)

plt.show()
