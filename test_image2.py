#!/usr/bin/env python3

import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0 and tf.config.experimental.get_device_details(physical_devices[0]).get('device_name') != 'METAL':
    tf.keras.mixed_precision.set_global_policy('mixed_float16')

    physical_devices = tf.config.list_physical_devices('GPU')
    try:
        for gpu in physical_devices:
            tf.config.experimental.set_memory_growth(gpu, True)
    except:
        # Invalid device or cannot modify virtual devices once initialized.
        pass

import numpy as np
from PIL import Image
from PIL.Image import Resampling
import sys
import os
import subprocess

import net

from dataset.data_transformer import max_encoderlen, max_decoderlen, decoder_SOT, decoder_EOT
from util_funcs import calcHist, calc_predid, decode_ruby

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

class TextDetectorModel(tf.keras.models.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.detector = net.CenterNetDetectionBlock(pre_weight=False)
        self.decoder = net.SimpleDecoderBlock()

    def eval(self, ds, org_img, cut_off = 0.5, locations0 = None, glyphfeatures0 = None):
        org_img = org_img.numpy()
        print(org_img.shape)
        print("test")

        locations = [np.zeros(5+4, np.float32)]
        glyphfeatures = [np.zeros(net.feature_dim, np.float32)]
        #allfeatures = np.zeros([0,net.feature_dim])
        keymap_all = np.zeros([org_img.shape[0] // net.scale, org_img.shape[1] // net.scale], np.float32)
        lines_all = np.zeros([org_img.shape[0] // net.scale, org_img.shape[1] // net.scale], np.float32)
        seps_all = np.zeros([org_img.shape[0] // net.scale, org_img.shape[1] // net.scale], np.float32)
        code_all = []
        for _ in range(4):
            code_all.append(np.zeros([org_img.shape[0] // net.scale, org_img.shape[1] // net.scale], np.float32))

        for n, inputs in ds.enumerate():
            print(n.numpy())
            offsetx = inputs['offsetx'].numpy()
            offsety = inputs['offsety'].numpy()

            images = inputs['input'].numpy()
            maps, feature = self.detector(inputs['input'])

            keymap = maps[...,0]
            local_peak = tf.nn.max_pool2d(keymap[...,tf.newaxis],5,1,'SAME')
            keep = local_peak[...,0] == keymap
            keymap = tf.math.sigmoid(keymap)
            detectedkey = keymap * tf.cast(keep, tf.float32)

            textlines = tf.math.sigmoid(maps[...,5])
            separator = tf.math.sigmoid(maps[...,6])
            xsize = maps[...,1]
            ysize = maps[...,2]
            xoffset = maps[...,3] * net.scale
            yoffset = maps[...,4] * net.scale
            code_map = []
            for k in range(4):
                code_map.append(tf.math.sigmoid(maps[...,7+k]))

            #allfeatures = np.concatenate([allfeatures, np.reshape(feature, [-1, net.feature_dim])])

            for img_idx in range(images.shape[0]):
                x_i = offsetx[img_idx]
                y_i = offsety[img_idx]
                x_is = x_i // net.scale
                y_is = y_i // net.scale
                x_s = net.width // net.scale
                y_s = net.height // net.scale

                mask = np.zeros([y_s, x_s], dtype=bool)
                x_min = int(x_s * 1 / 6) if x_i > 0 else 0
                x_max = int(x_s * 5 / 6) if x_i + net.width < org_img.shape[1] else x_s
                y_min = int(y_s * 1 / 6) if y_i > 0 else 0
                y_max = int(y_s * 5 / 6) if y_i + net.height < org_img.shape[0] else y_s
                mask[y_min:y_max, x_min:x_max] = True

                keymap_p = keymap[img_idx,...]
                line_p = textlines[img_idx,...]
                seps_p = separator[img_idx,...]
                code_p = [m[img_idx,...] for m in code_map]

                keymap_all[y_is:y_is+y_s,x_is:x_is+x_s] = np.maximum(keymap_p * mask, keymap_all[y_is:y_is+y_s,x_is:x_is+x_s])
                lines_all[y_is:y_is+y_s,x_is:x_is+x_s] = np.maximum(line_p * mask, lines_all[y_is:y_is+y_s,x_is:x_is+x_s])
                seps_all[y_is:y_is+y_s,x_is:x_is+x_s] = np.maximum(seps_p * mask, seps_all[y_is:y_is+y_s,x_is:x_is+x_s])
                for k in range(4):
                    code_all[k][y_is:y_is+y_s,x_is:x_is+x_s] = np.maximum(code_p[k] * mask, code_all[k][y_is:y_is+y_s,x_is:x_is+x_s])

                peak = (detectedkey[img_idx, ...] * mask).numpy()
                idxy, idxx = np.unravel_index(np.argsort(-peak.ravel()), peak.shape)

                for y, x in zip(idxy, idxx):
                    if peak[y,x] < cut_off:
                        break
                    w = tf.math.exp(xsize[img_idx,y,x] - 3) * 1024
                    h = tf.math.exp(ysize[img_idx,y,x] - 3) * 1024
                    if w * h <= 0:
                        continue

                    dx = xoffset[img_idx,y,x]
                    dy = yoffset[img_idx,y,x]

                    ix = x * net.scale + dx + x_i
                    iy = y * net.scale + dy + y_i

                    codes = []
                    for k in range(4):
                        codes.append(code_p[k][y,x])

                    locations.append(np.array([peak[y,x], ix, iy, w, h, *codes]))
                    glyphfeatures.append(feature[img_idx, y, x, :].numpy())

        locations = np.array(locations)
        if locations0 is not None:
            locations = np.concatenate([locations, locations0])
        glyphfeatures = np.array(glyphfeatures)
        if glyphfeatures0 is not None:
            glyphfeatures = np.concatenate([glyphfeatures, glyphfeatures0])

        idx = np.argsort(-locations[:,0])
        done_area = np.zeros([0,4], np.float32)
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
                if inter_vol.max() > area0_vol * 0.8:
                    continue
            done_area = np.vstack([done_area, np.array([cx, cy, w, h])])
            selected_idx.append(i)
        
        if len(selected_idx) > 0:
            selected_idx = np.array(selected_idx)

            locations = locations[selected_idx,:]
            glyphfeatures = glyphfeatures[selected_idx,:]
        else:
            locations = np.zeros([0,5+4], np.float32)
            glyphfeatures = np.zeros([0,net.feature_dim], np.float32)

        for i in range(locations.shape[0]):
            cx = locations[i,1]
            cy = locations[i,2]
            x = int(cx / net.scale)
            y = int(cy / net.scale)
            if x >= 0 and x < org_img.shape[1] // net.scale and y >= 0 and y < org_img.shape[0] // net.scale:
                for k in range(4):
                    locations[i,5+k] = max(code_all[k][y,x], locations[i,5+k])

        return locations, glyphfeatures, lines_all, seps_all

class TransformerDecoderModel(tf.keras.models.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.transformer = net.TextTransformer()
        embedded = tf.keras.Input(shape=(max_encoderlen,net.encoder_dim))
        decoderinput = tf.keras.Input(shape=(max_decoderlen,))
        self.transformer((embedded, decoderinput))

        self.transformer.summary()

model1 = TextDetectorModel()
last = tf.train.latest_checkpoint('ckpt1')
print(last)
model1.load_weights(last).expect_partial()

model2 = TransformerDecoderModel()
last = tf.train.latest_checkpoint('ckpt2')
print(last)
model2.load_weights(last).expect_partial()

stepx = net.width * 1 // 2
stepy = net.height * 1 // 2

padx = max(0, stepx - (im0.shape[1] - net.width) % stepx, net.width - im0.shape[1])
pady = max(0, stepy - (im0.shape[0] - net.height) % stepy, net.height - im0.shape[0])
im0 = np.pad(im0, [[0,pady],[0,padx],[0,0]], 'constant', constant_values=((255,255),(255,255),(255,255)))

if twopass and (im0.shape[1] / stepx > 2 or im0.shape[0] / stepy > 2):
    print('two-pass')
    s = max(im0.shape[1], im0.shape[0]) / max(net.width, net.height)
    im1 = Image.fromarray(im0).resize((int(im0.shape[1] / s), int(im0.shape[0] / s)), resample=Resampling.BILINEAR)
    im1 = np.asarray(im1)
    padx = max(0, net.width - im1.shape[1])
    pady = max(0, net.height - im1.shape[0])
    im1 = np.pad(im1, [[0,pady],[0,padx],[0,0]], 'constant', constant_values=((255,255),(255,255),(255,255)))

    im = tf.image.convert_image_dtype(im1, dtype=tf.float32)
    im = im * 255.

    ds1 = tf.data.Dataset.range(1)
    ds1 = ds1.map(lambda x: {
        'input': im,
        'offsetx': 0,
        'offsety': 0,
        })
    ds1 = ds1.batch(1)
    ds1 = ds1.prefetch(tf.data.AUTOTUNE)

    locations0, glyphfeatures0, lines0, seps0 = model1.eval(ds1, im, cut_off=0.5)
    locations0[:,1:] = locations0[:,1:] * s
else:
    locations0, glyphfeatures0 = None, None

im = tf.image.convert_image_dtype(im0, dtype=tf.float32)
im = im * 255.

yi = tf.data.Dataset.range(0, im0.shape[0] - net.height + 1, stepy)
xi = tf.data.Dataset.range(0, im0.shape[1] - net.width + 1, stepx)
ds0 = yi.flat_map(lambda y: xi.map(lambda x : (x, y)))
ds0 = ds0.map(lambda x,y: {
    'input': im[y:y+net.height,x:x+net.width,:],
    'offsetx': x,
    'offsety': y,
    })
ds0 = ds0.batch(8)
ds0 = ds0.prefetch(tf.data.AUTOTUNE)

locations, glyphfeatures, lines, seps = model1.eval(ds0, im, cut_off=0.5,
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

    g = np.concatenate([np.zeros([net.feature_dim], np.float32), np.array([space,ruby,rubybase,0], np.float32)])
    if prev_block != block:
        prev_block = block
        features.append(g)
    if prev_idx != idx:
        prev_idx = idx
        features.append(g)

    if subtype & 2+4 == 2+4:
        ruby = 1
    elif subtype & 2+4 == 2:
        rubybase = 1
    
    if subtype & 8 == 8:
        space = 0
    
    g = np.concatenate([glyphfeatures[id,:], np.array([space,ruby,rubybase,0], np.float32)])
    features.append(g)
features = np.array(features, np.float32)


@tf.function
def call_loop(decoder_input, i, encoder_output, encoder_input):
    decoder_output = model2.transformer.decoder([decoder_input, encoder_output, encoder_input])

    out1091, out1093, out1097 = decoder_output
    p1091 = tf.math.softmax(out1091[0,i])
    p1093 = tf.math.softmax(out1093[0,i])
    p1097 = tf.math.softmax(out1097[0,i])
    i1091 = tf.argmax(p1091, axis=-1)
    i1093 = tf.argmax(p1093, axis=-1)
    i1097 = tf.argmax(p1097, axis=-1)
    code = calc_predid(i1091,i1093,i1097)
    return tf.where(tf.range(max_decoderlen) == i+1, code, decoder_input), i+1, encoder_output, encoder_input

i = 0
result_txt = ''
while i < features.shape[0]:
    j = min(features.shape[0] - 1, i + (max_decoderlen - 10))
    while features[j,-1] == 0:
        j -= 1
        if j <= i:
            j = min(features.shape[0], i + (max_decoderlen - 10))
            break
    print(i,j)
    encoder_input = tf.constant(features[i:j+1,:], tf.int32)
    encoder_len = tf.shape(encoder_input)[0]
    encoder_input = tf.pad(encoder_input, [[0, max_encoderlen - encoder_len], [0, 0]])
    encoder_input = tf.expand_dims(encoder_input, 0)
    encoder_output = model2.transformer.encoder(encoder_input)

    decoder_input = tf.constant([decoder_SOT], dtype=tf.int64)
    decoder_input = tf.pad(decoder_input, [[0, max_decoderlen - 1]])
    decoder_input = tf.expand_dims(decoder_input, 0)
    i0 = tf.constant(0)
    c = lambda n, i, eo, ei: tf.logical_and(i < max_decoderlen-1, n[0,i] != decoder_EOT)
    output,count,_,_ = tf.while_loop(
        c, call_loop, loop_vars=[decoder_input, i0, encoder_output, encoder_input])

    count = count.numpy()
    code = output[0].numpy().astype(np.int32)
    print(code)
    str_code = code[1:count]
    str_text = ''.join([chr(c) if c < 0x110000 else '\uFFFD' for c in str_code])
    result_txt += str_text
    i = j+1

print(decode_ruby(result_txt))
