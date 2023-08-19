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
import sys
import os
from PIL import Image, ImageFilter
from PIL.Image import Resampling

from matplotlib import rcParams
rcParams['font.serif'] = ['Noto Serif CJK JP', 'IPAexMincho', 'Hiragino Mincho ProN']

import matplotlib.pyplot as plt

from net.detector import CenterNetDetectionBlock, SimpleDecoderBlock
from util_funcs import calc_predid, width, height, scale, feature_dim

if len(sys.argv) < 2:
    print(sys.argv[0],'target.png','(twopass)')
    exit(1)

target_file = sys.argv[1]
twopass = False
feature_plot = False
if len(sys.argv) > 2:
    if 'twopass' in sys.argv[2:]:
        twopass = True
    if 'feature' in sys.argv[2:]:
        feature_plot = True


if feature_plot:
    out_dir = 'feature-%s'%os.path.basename(target_file)
    if os.path.exists(out_dir):
        count = 1
        while os.path.exists('%s-%d'%(out_dir, count)):
            count += 1
        out_dir = '%s-%d'%(out_dir, count)
    os.mkdir(out_dir)

class TextDetectorModel(tf.keras.models.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.detector = CenterNetDetectionBlock(pre_weight=False)
        self.decoder = SimpleDecoderBlock()

    def eval(self, ds, org_img, cut_off = 0.5, locations0 = None, glyphfeatures0 = None):
        org_img = org_img.numpy()
        print(org_img.shape)
        print("test")

        locations = [np.zeros(5+4)]
        glyphfeatures = [np.zeros(feature_dim)]
        #allfeatures = np.zeros([0,feature_dim])
        keymap_all = np.zeros([org_img.shape[0] // scale, org_img.shape[1] // scale])
        lines_all = np.zeros([org_img.shape[0] // scale, org_img.shape[1] // scale])
        seps_all = np.zeros([org_img.shape[0] // scale, org_img.shape[1] // scale])
        code_all = []
        for _ in range(4):
            code_all.append(np.zeros([org_img.shape[0] // scale, org_img.shape[1] // scale]))


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
            xoffset = maps[...,3] * scale
            yoffset = maps[...,4] * scale
            code_map = []
            for k in range(4):
                code_map.append(tf.math.sigmoid(maps[...,7+k]))

            #allfeatures = np.concatenate([allfeatures, np.reshape(feature, [-1, feature_dim])])

            for img_idx in range(images.shape[0]):
                x_i = offsetx[img_idx]
                y_i = offsety[img_idx]
                x_is = x_i // scale
                y_is = y_i // scale
                x_s = width // scale
                y_s = height // scale

                mask = np.zeros([y_s, x_s], dtype=bool)
                x_min = int(x_s * 1 / 6) if x_i > 0 else 0
                x_max = int(x_s * 5 / 6) if x_i + width < org_img.shape[1] else x_s
                y_min = int(y_s * 1 / 6) if y_i > 0 else 0
                y_max = int(y_s * 5 / 6) if y_i + height < org_img.shape[0] else y_s
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

                    ix = x * scale + dx + x_i
                    iy = y * scale + dy + y_i

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
                if inter_vol.max() > area0_vol * 0.8:
                    continue
            done_area = np.vstack([done_area, np.array([cx, cy, w, h])])
            selected_idx.append(i)
        
        if len(selected_idx) > 0:
            selected_idx = np.array(selected_idx)

            locations = locations[selected_idx,:]
            glyphfeatures = glyphfeatures[selected_idx,:]
        else:
            locations = np.zeros([0,5+4])
            glyphfeatures = np.zeros([0,feature_dim])

        # for i in range(locations.shape[0]):
        #     cx = locations[i,1]
        #     cy = locations[i,2]
        #     x = int(cx / scale)
        #     y = int(cy / scale)
        #     if x >= 0 and x < org_img.shape[1] // scale and y >= 0 and y < org_img.shape[0] // scale:
        #         for k in range(4):
        #             locations[i,5+k] = max(code_all[k][y,x], locations[i,5+k])

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
            'ruby base',
            'emphasis',
            'space',
        ]
        for k in range(4):
            plt.figure()
            plt.imshow(code_all[k],interpolation='none',vmin=0.,vmax=1.)
            plt.title('code%d '%(2**(k)) + title_str[k])


        return locations, glyphfeatures

    def decode(self, glyphfeatures):
        print("decode")
        glyphids = []
        glyphprobs = []
        for data in np.array_split(glyphfeatures, 8):
            pred_decoder = self.decoder(data)
            ids = []
            p_id = 0.
            for decoder_id1 in pred_decoder:
                prob_id1 = tf.nn.softmax(decoder_id1, -1)
                pred_id1 = tf.math.argmax(prob_id1, axis=-1)
                index1 = tf.stack([tf.range(tf.shape(prob_id1)[0], dtype=tf.int64), pred_id1], axis=-1)
                p_id += tf.math.log(tf.math.maximum(tf.gather_nd(prob_id1, index1),1e-7))
                ids.append(pred_id1)

            pred_id = calc_predid(*ids).numpy()
            p_id = tf.exp(p_id / len(pred_decoder)).numpy()

            glyphids.append(pred_id)
            glyphprobs.append(p_id)

        glyphids = np.concatenate(glyphids)
        glyphprobs = np.concatenate(glyphprobs)
        
        return  glyphids, glyphprobs

model = TextDetectorModel()
last = tf.train.latest_checkpoint('ckpt1')
print(last)
model.load_weights(last).expect_partial()

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
    im1 = Image.fromarray(im0).resize((int(im0.shape[1] / s), int(im0.shape[0] / s)), resample=Resampling.BILINEAR)
    im1 = np.asarray(im1)
    padx = max(0, width - im1.shape[1])
    pady = max(0, height - im1.shape[0])
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

    locations0, glyphfeatures0 = model.eval(ds1, im, cut_off=0.5)
    locations0[:,1:] = locations0[:,1:] * s
else:
    locations0, glyphfeatures0 = None, None

im = tf.image.convert_image_dtype(im0, dtype=tf.float32)
im = im * 255.

yi = tf.data.Dataset.range(0, im0.shape[0] - height + 1, stepy)
xi = tf.data.Dataset.range(0, im0.shape[1] - width + 1, stepx)
ds0 = yi.flat_map(lambda y: xi.map(lambda x : (x, y)))
ds0 = ds0.map(lambda x,y: {
    'input': im[y:y+height,x:x+width,:],
    'offsetx': x,
    'offsety': y,
    })
ds0 = ds0.batch(8)
ds0 = ds0.prefetch(tf.data.AUTOTUNE)

locations, glyphfeatures = model.eval(ds0, im, cut_off=0.5,
        locations0=locations0, glyphfeatures0=glyphfeatures0)
glyphids, glyphprobs = model.decode(glyphfeatures)

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
    #print(i,codes,cx,cy,pred_char)

    if feature_plot:
        c = ord(pred_char) if pred_char else 0
        im_file = os.path.join(out_dir, '%x-x%d-y%d.png'%(c,cx,cy))

        g = np.reshape(g, [8, -1])
        plt.figure(figsize=[3,3])
        plt.imshow(g, vmin=-2, vmax=2, interpolation='none')
        plt.axis('off')
        plt.savefig(im_file, bbox_inches='tight', pad_inches=0)
        plt.clf()
        plt.close()

plt.show()
