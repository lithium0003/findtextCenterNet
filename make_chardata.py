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
from PIL import Image, ImageEnhance, ImageFilter
import os, glob

from render_font.generate_random_txt import get_random_char
from net.detector import CenterNetDetectionBlock, SimpleDecoderBlock
from net.const import width, height, scale, feature_dim

output_dir = 'chardata_font'

min_delta = 0.5

data_path = '.'
random_background = tf.io.gfile.glob(tf.io.gfile.join(data_path,'data','background','*'))
print(len(random_background),'background files loaded.')

class TextDetectorModel(tf.keras.models.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.detector = CenterNetDetectionBlock(pre_weight=False)
        self.decoder = SimpleDecoderBlock()


    def eval(self, ds, org_img, cut_off = 0.5):
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

        return locations, glyphfeatures

model = TextDetectorModel()
last = tf.train.latest_checkpoint('ckpt1')
print(last)
model.load_weights(last).expect_partial()

rng = np.random.default_rng()

def load_background_images(im_width, im_height):
    import random
    ind = random.choice(range(len(random_background)))
    img0 = Image.open(random_background[ind]).convert('RGB')
    scale_min = max(float(im_width) / float(img0.width), float(im_height) / float(img0.height))
    scale_max = max(scale_min + 0.5, 1.5)
    s = rng.uniform(scale_min, scale_max)
    img = img0.resize((int(float(img0.width) * s)+1, int(float(img0.height) * s)+1),Image.BILINEAR)
    x1 = max(0, int(rng.uniform(0, img.width - im_width)))
    y1 = max(0, int(rng.uniform(0, img.height - im_height)))
    img = np.asarray(img)[y1:y1+im_height, x1:x1+im_width,:]

    if rng.uniform() < 0.5:
        img = img[::-1,:,:]
    if rng.uniform()  < 0.5:
        img = img[:,::-1,:]
    enhancer = ImageEnhance.Brightness(Image.fromarray(img))
    img = enhancer.enhance(rng.uniform())
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(rng.uniform(0.2,1.8))

    img = np.asarray(img).astype(np.float32) / 255.
    img = np.clip(img, 0., 1.)
    return img

def background_image(im_width, im_height):
    bkimg = load_background_images(im_width, im_height)
    bk_c = np.min(bkimg, axis=(0,1))
    bk_std = np.std(bkimg, axis=(0,1))
    fg_c = np.where(
        bk_c > 0.5,
        rng.uniform(np.clip(bk_c - bk_std * 2 - min_delta, None, -1), bk_c - bk_std * 2 - min_delta,[3]),
        rng.uniform(bk_c + bk_std * 2 + min_delta, np.clip(bk_c + bk_std * 2 + min_delta, 1, None), [3]))
    bk_alpha = np.maximum(np.max(np.abs(fg_c)), 1)
    bkimg /= bk_alpha
    fg_c /= bk_alpha
    fg_c = np.clip(fg_c, 0., 1.)
    fgimg = fg_c[None,None,:]
    return fgimg, bkimg

def preprocess_image(image, pos):
    aspect = rng.uniform(0.75,1.3)
    w = int(image.shape[1]*aspect)
    h = int(image.shape[0]/aspect)
    im = Image.fromarray(image).resize((w,h), Image.Resampling.BILINEAR)
    image = np.asarray(im)
    pos *= np.array([aspect,1/aspect,aspect,1/aspect])

    angle = rng.normal() * 2.0
    py1 = max(0,int(image.shape[1]*np.sin(angle/180*np.pi)))
    py2 = max(0,int(image.shape[1]*np.sin(-angle/180*np.pi)))
    px1 = max(0,int(image.shape[0]*np.sin(-angle/180*np.pi)))
    px2 = max(0,int(image.shape[0]*np.sin(angle/180*np.pi)))
    image = np.pad(image, ((py1,py2),(px1,px2)))
    im = Image.fromarray(image).rotate(angle, Image.Resampling.BILINEAR, center=(px1,py1))

    M = np.array([[np.cos(angle/180*np.pi),-np.sin(angle/180*np.pi)],
                 [np.sin(angle/180*np.pi), np.cos(angle/180*np.pi)],])
    pos[:,:2] = (pos[:,:2] @ M)
    pos[:,2:4] += np.array([pos[:,3] * np.abs(np.sin(angle/180*np.pi)), pos[:,2] * np.abs(np.sin(angle/180*np.pi))]).T
    pos += np.array([px1 - 1,py1 - 1,0,0])
    return np.asarray(im), pos

def random_filter(image):
    img = Image.fromarray(image)
    r = rng.uniform()
    if r > 0:
        img = img.filter(ImageFilter.GaussianBlur(radius=r))

    r = rng.uniform()
    if r > 0:
        img = img.filter(ImageFilter.UnsharpMask(radius=r, percent=150, threshold=3))

    return np.array(img)

def process(rng):
    turn = rng.uniform() < 0.01
    d = get_random_char(rng, turn=turn)
    pos = d['position']
    if pos.size == 0:
        return
    codes = d['code_list']
    image = d['image']
    image, pos = preprocess_image(image, pos)
    image = random_filter(image)
    fgimg, bkimg = background_image(image.shape[1], image.shape[0])

    img = image[...,None]
    img = img / 255.
    image = fgimg * img + bkimg * (1 - img)
    image = np.clip(image, 0., 1.)
    image = image * 255

    stepx = width * 1 // 2
    stepy = height * 1 // 2

    im0 = np.asarray(image)

    padx = max(0, stepx - (im0.shape[1] - width) % stepx, width - im0.shape[1])
    pady = max(0, stepy - (im0.shape[0] - height) % stepy, height - im0.shape[0])
    im0 = np.pad(im0, [[0,pady],[0,padx],[0,0]], 'constant', constant_values=((255,255),(255,255),(255,255)))

    im = tf.image.convert_image_dtype(im0, dtype=tf.float32)

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

    locations, glyphfeatures = model.eval(ds0, im)
    for i in range(locations.shape[0]):
        cx = locations[i,1]
        cy = locations[i,2]
        w = locations[i,3]
        h = locations[i,4]
        area0_vol = w * h

        area1_vol = pos[:,2] * pos[:,3]
        inter_xmin = np.maximum(cx - w / 2, pos[:,0] - pos[:,2] / 2)
        inter_ymin = np.maximum(cy - h / 2, pos[:,1] - pos[:,3] / 2)
        inter_xmax = np.minimum(cx + w / 2, pos[:,0] + pos[:,2] / 2)
        inter_ymax = np.minimum(cy + h / 2, pos[:,1] + pos[:,3] / 2)
        inter_w = np.maximum(inter_xmax - inter_xmin, 0.)
        inter_h = np.maximum(inter_ymax - inter_ymin, 0.)
        inter_vol = inter_w * inter_h
        union_vol = area0_vol + area1_vol - inter_vol
        iou = np.where(union_vol > 0., inter_vol / union_vol, 0.)
        j = np.argmax(iou)
        if iou[j] < 0.3:
            continue

        code = codes[j,0]
        feature = glyphfeatures[i,:]

        save_codefeature(code, feature, turn=turn)
    
def save_codefeature(code, feature, turn=False):
    os.makedirs(output_dir, exist_ok=True)
    if turn:
        filename = os.path.join(output_dir,'%dt.npy'%code)
    else:
        filename = os.path.join(output_dir,'%dn.npy'%code)
    if os.path.exists(filename):
        prev = np.load(filename)
        feature = np.vstack([prev, feature])
        count = feature.shape[0]
    else:
        count = 0
    print(code, turn, count)
    np.save(filename, feature)

if __name__=="__main__":
    rng = np.random.default_rng()
    count = 10000
    for i in range(count):
        print(i,'/',count)
        process(rng)
