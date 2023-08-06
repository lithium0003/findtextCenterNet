#!/usr/bin/env python3

import coremltools as ct

import numpy as np
from PIL import Image, ImageEnhance
import os, glob
import sys
from termios import tcflush, TCIOFLUSH
import matplotlib.pyplot as plt
import matplotlib.widgets as wg
from matplotlib.font_manager import FontProperties

fprop = FontProperties(fname='./NotoSerifJP-Regular.otf')

from util_funcs import calc_predid, calcHist, width, height, scale, feature_dim

output_dir = 'chardata_hand'

mlmodel_detector = ct.models.MLModel('TextDetector.mlpackage')
mlmodel_decoder = ct.models.MLModel('CodeDecoder.mlpackage')

def eval(ds, org_img, cut_off = 0.5):
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

    return locations, glyphfeatures

def filter_boxes(im0, locations, glyphfeatures):
    valid_locations = []
    for i, (p, x, y, w, h, c1, c2, c4, c8) in enumerate(locations):
        x1 = np.clip(int(x - w/2), 0, im0.shape[1])
        y1 = np.clip(int(y - h/2), 0, im0.shape[0])
        x2 = np.clip(int(x + w/2) + 1, 0, im0.shape[1])
        y2 = np.clip(int(y + h/2) + 1, 0, im0.shape[0])
        if calcHist(im0[y1:y2,x1:x2,:]) < 50:
            continue
        valid_locations.append(i)
    locations = locations[valid_locations,:]
    glyphfeatures = glyphfeatures[valid_locations,:]
    print(locations.shape[0],'boxes')
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
    
    return glyphids, glyphprobs

def process(filename):
    im0 = Image.open(filename).convert('RGB')
    #im0 = im0.filter(ImageFilter.SHARPEN)
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
    locations, glyphfeatures = eval(ds0, im0, cut_off=0.35)
    locations, glyphfeatures = filter_boxes(im0, locations, glyphfeatures)
    glyphids, glyphprobs = decode(glyphfeatures)

    if locations.shape[0] < 1:
        print('no box found.')
        return

    box_points = []
    pred_chars = []
    for i, loc in enumerate(locations):
        cx = loc[1]
        cy = loc[2]
        w = loc[3]
        h = loc[4]
        cid = glyphids[i]

        points = [
            [cx - w / 2, cy - h / 2],
            [cx + w / 2, cy - h / 2],
            [cx + w / 2, cy + h / 2],
            [cx - w / 2, cy + h / 2],
            [cx - w / 2, cy - h / 2],
        ]
        points = np.array(points)

        if cid < 0x10FFFF:
            pred_char = chr(cid)
        else:
            pred_char = None

        pred_chars.append(pred_char)
        box_points.append(points)

    box_points = np.array(box_points)

    fig = plt.figure()
    fig.gca().imshow(im0)

    global targetIdx, waiting
    targetIdx = -1
    waiting = False

    def onclick(event):
        global targetIdx, waiting
        if waiting:
            fig.canvas.draw_idle()
            return
        ix, iy = event.xdata, event.ydata
        if ix is None or iy is None:
            fig.canvas.draw_idle()
            return
        b1 = np.logical_and(box_points[:,0,0] < ix, box_points[:,0,1] < iy)
        b2 = np.logical_and(box_points[:,1,0] > ix, box_points[:,1,1] < iy)
        b3 = np.logical_and(box_points[:,2,0] > ix, box_points[:,2,1] > iy)
        b4 = np.logical_and(box_points[:,3,0] < ix, box_points[:,3,1] > iy)
        idx = np.where(np.logical_and(np.logical_and(b1,b2),np.logical_and(b3,b4)))[0]
        if idx.size == 0:
            fig.canvas.draw_idle()
            return
        else:
            idx = idx[0]
            targetIdx = idx
            if pred_chars[idx]:
                pred_char = pred_chars[idx]
            else:
                pred_char = ''
            waiting = True
            tcflush(sys.stdin, TCIOFLUSH)
            ans = input(f'current:{pred_char}>')
            waiting = False
            for txt in plt.gca().texts:
                p = txt.get_position()
                if p[0] == locations[targetIdx,1] and p[1] == locations[targetIdx,2]:
                    txt.remove()
                    fig.canvas.draw_idle()
                    break
            
            if ans == '' or ans[0].isspace():
                newchar = None
            else:
                newchar = ans[0]
            pred_chars[targetIdx] = newchar

            cx = locations[targetIdx,1]
            cy = locations[targetIdx,2]
            pred_char = pred_chars[targetIdx]
            if pred_char:
                plt.text(cx, cy, pred_char, fontsize=28, color='red', fontproperties=fprop)
            fig.canvas.draw_idle()

    for loc, points, pred_char in zip(locations, box_points, pred_chars):
        cx = loc[1]
        cy = loc[2]
        plt.plot(points[:,0], points[:,1],color='cyan')
        if pred_char:
            plt.text(cx, cy, pred_char, fontsize=28, color='blue', fontproperties=fprop)

    fig.canvas.mpl_connect('button_press_event', onclick)

    plt.show()

    for i, pred_char in enumerate(pred_chars):
        if pred_char:
            feature = glyphfeatures[i,:]
            save_codefeature(ord(pred_char), feature)

def save_codefeature(code, feature):
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir,'%d.npy'%code)
    if os.path.exists(filename):
        prev = np.load(filename)
        feature = np.vstack([prev, feature])
        count = feature.shape[0]
    else:
        count = 0
    print(code, count)
    np.save(filename, feature)

if __name__=="__main__":
    if len(sys.argv) < 2:
        print(sys.argv[0], 'image.png')
        exit()

    target_files = []
    for a in sys.argv[1:]:
        target_files += glob.glob(a)

    if len(target_files) < 1:
        print('no image found')
        exit()

    for i, filename in enumerate(target_files):
        print(i,'/',len(target_files), filename)
        process(filename)
