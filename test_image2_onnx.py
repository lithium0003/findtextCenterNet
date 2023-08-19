#!/usr/bin/env python3

import onnxruntime

import numpy as np
from numpy.lib.stride_tricks import as_strided
from scipy.ndimage import gaussian_filter
import sys
import os
from PIL import Image, ImageFilter
import sys
import os
import subprocess

from const import max_encoderlen, max_decoderlen, decoder_SOT, decoder_EOT
from util_funcs import calcHist, calc_predid, decode_ruby, feature_dim, height, width, scale

if len(sys.argv) < 2:
    print(sys.argv[0],'target.png','(twopass)')
    exit(1)

target_file = sys.argv[1]
twopass = False
if len(sys.argv) > 2:
    if 'twopass' in sys.argv[2:]:
        twopass = True

im0 = Image.open(target_file).convert('RGB')
#im0 = im0.filter(ImageFilter.SHARPEN)
im0 = np.asarray(im0)

quantized_filter = False
if os.path.exists("TextDetector.quant.onnx"):
    print('quantized')
    onnx_detector = onnxruntime.InferenceSession("TextDetector.quant.onnx")
    quantized_filter = True
elif os.path.exists("TextDetector.infer.onnx"):
    print('infer')
    onnx_detector = onnxruntime.InferenceSession("TextDetector.infer.onnx")
else:
    onnx_detector = onnxruntime.InferenceSession("TextDetector.onnx")
onnx_encoder = onnxruntime.InferenceSession("TransformerEncoder.onnx")
onnx_decoder = onnxruntime.InferenceSession("TransformerDecoder.onnx")

def maxpool2d(input_matrix, kernel_size):
    # Padding
    pad_size = kernel_size // 2
    pad = (pad_size, pad_size)
    input_matrix = np.pad(input_matrix, [pad]*len(input_matrix.shape), constant_values=-np.inf)

    # Window view of input_matrix
    output_shape = (input_matrix.shape[0] - kernel_size + 1,
                    input_matrix.shape[1] - kernel_size + 1)
    kernel_size = (kernel_size, kernel_size)
    input_matrix_w = as_strided(input_matrix, shape = output_shape + kernel_size,
                        strides = input_matrix.strides + input_matrix.strides)
    input_matrix_w = input_matrix_w.reshape(-1, *kernel_size)
    return input_matrix_w.max(axis=(1,2)).reshape(output_shape)

def eval(ds, org_img, cut_off = 0.5, locations0 = None, glyphfeatures0 = None):
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

        images = inputs['input']
        maps, feature = onnx_detector.run(['maps','feature'], {'image_input': images})

        mask = np.zeros([y_s, x_s], dtype=bool)
        x_min = int(x_s * 1 / 6) if x_i > 0 else 0
        x_max = int(x_s * 5 / 6) if x_i + width < org_img.shape[1] else x_s
        y_min = int(y_s * 1 / 6) if y_i > 0 else 0
        y_max = int(y_s * 5 / 6) if y_i + height < org_img.shape[0] else y_s
        mask[y_min:y_max, x_min:x_max] = True

        keymap_p = 1/(1 + np.exp(-maps[0,:,:,0]))
        line_p = 1/(1 + np.exp(-maps[0,:,:,5]))
        seps_p = 1/(1 + np.exp(-maps[0,:,:,6]))
        code_p = []
        for k in range(4):
            code_p.append(1/(1 + np.exp(-maps[0,:,:,7+k])))

        keymap_all[y_is:y_is+y_s,x_is:x_is+x_s] = np.maximum(keymap_p * mask, keymap_all[y_is:y_is+y_s,x_is:x_is+x_s])
        lines_all[y_is:y_is+y_s,x_is:x_is+x_s] = np.maximum(line_p * mask, lines_all[y_is:y_is+y_s,x_is:x_is+x_s])
        seps_all[y_is:y_is+y_s,x_is:x_is+x_s] = np.maximum(seps_p * mask, seps_all[y_is:y_is+y_s,x_is:x_is+x_s])
        for k in range(4):
            code_all[k][y_is:y_is+y_s,x_is:x_is+x_s] = np.maximum(code_p[k] * mask, code_all[k][y_is:y_is+y_s,x_is:x_is+x_s])

        keypeak = maps[0,:,:,0]
        if quantized_filter:
            keypeak = gaussian_filter(keypeak, sigma=1)
        peak = np.where(maxpool2d(keypeak, 5) == keypeak, keymap_p * mask, 0.)
        idxy, idxx  = np.unravel_index(np.argsort(-peak.ravel()), peak.shape)

        for y, x in zip(idxy, idxx):
            if peak[y,x] < cut_off:
                break
            w = np.exp(maps[0,y,x,1] - 3) * 1024
            h = np.exp(maps[0,y,x,2] - 3) * 1024
            dx = maps[0,y,x,3] * scale
            dy = maps[0,y,x,4] * scale
            if w * h <= 0:
                continue
            ix = x * scale + dx + x_i
            iy = y * scale + dy + y_i

            codes = []
            for k in range(4):
                codes.append(code_p[k][y,x])

            locations.append(np.array([peak[y,x], ix, iy, w, h, *codes]))
            glyphfeatures.append(feature[0, y, x, :])

    locations = np.array(locations, np.float32)
    if locations0 is not None:
        locations = np.concatenate([locations, locations0])
    glyphfeatures = np.array(glyphfeatures, np.float32)
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

    return locations, glyphfeatures, lines_all, seps_all

stepx = width * 1 // 2
stepy = height * 1 // 2

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
    
    im = im1.astype(np.float32)

    ds1 = []
    ds1.append({
        'input': np.expand_dims(im, 0),
        'offsetx': 0,
        'offsety': 0,
        })

    locations0, glyphfeatures0, lines0, seps0 = eval(ds1, im1, cut_off=0.5)
    locations0[:,1:] = locations0[:,1:] * s
else:
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
locations, glyphfeatures, lines, seps = eval(ds0, im0, cut_off=0.5,
        locations0=locations0, glyphfeatures0=glyphfeatures0)

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
print(count)
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

features = []
prev_block = 0
prev_idx = 0
for id, block, idx, subidx, subtype in detected_boxes:
    if id < 0:
        continue

    ruby = 0
    rubybase = 0
    space = 0

    g = np.concatenate([np.zeros([feature_dim], np.float32), np.array([space,ruby,rubybase,1], np.float32)])
    if prev_block != block:
        prev_block = block
        features.append(g)
        features.append(g)
    if prev_idx != idx:
        prev_idx = idx
        features.append(g)

    if subtype & 2+4 == 2+4:
        ruby = 1
    elif subtype & 2+4 == 2:
        rubybase = 1
    
    if subtype & 8 == 8:
        space = 1
    
    g = np.concatenate([glyphfeatures[id,:], np.array([space,ruby,rubybase,0], np.float32)])
    features.append(g)
features = np.array(features, np.float32)


i = 0
result_txt = ''
while i < features.shape[0]:
    j = i + (max_encoderlen - 10)
    if j < features.shape[0]-1:
        while features[j,-1] == 0:
            j -= 1
            if j <= i:
                j = min(features.shape[0]-1, i + (max_encoderlen - 10))
                break
    else:
        j = features.shape[0]-1
    print(i,j)
    encoder_input = features[i:j+1,:]
    #print(list(encoder_input))
    encoder_input = np.pad(encoder_input, [[0, max_encoderlen - encoder_input.shape[0]],[0,0]])
    encoder_input = np.expand_dims(encoder_input, 0).astype(np.float32)

    print('encoder')
    encoder_output, = onnx_encoder.run(['encoder_output'], { 'encoder_input': encoder_input })

    print('decoder')
    decoder_input = np.zeros([1,max_decoderlen], dtype=np.float32)
    decoder_input[0,0] = decoder_SOT
    count = 0
    while count < max_decoderlen - 1 and decoder_input[0,count] != decoder_EOT:
        mod1091, mod1093, mod1097 = onnx_decoder.run(['mod1091','mod1093','mod1097'], { 'decoder_input': decoder_input, 'encoder_output': encoder_output, 'encoder_input': encoder_input })
        i1091 = np.argmax(mod1091[count,:])
        i1093 = np.argmax(mod1093[count,:])
        i1097 = np.argmax(mod1097[count,:])
        code = calc_predid(i1091,i1093,i1097)
        count += 1
        decoder_input[0,count] = code

    code = decoder_input[0].astype(np.int32)
    print(code)
    str_code = code[1:count]
    str_text = ''.join([chr(c) if c < 0x110000 else '\uFFFD' for c in str_code])
    result_txt += str_text
    i = j+1

print("---------------------")
print(decode_ruby(result_txt))
