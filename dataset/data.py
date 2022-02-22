import tensorflow as tf
import numpy as np
from scipy import signal, ndimage
from skimage.morphology import skeletonize
from PIL import Image, ImageDraw
import random
import tqdm
import glob, os
import csv
from multiprocessing import Pool
import subprocess

width = 512
height = 512
scale = 2

class BaseData:
    def __init__(self):
        self.load_idmap()

    def load_idmap(self):
        self.glyph_id = {}
        self.glyphs = {}
        self.glyph_type = {}

        self.glyph_id[''] = 0
        self.glyphs[0] = ''

        with open('data/codepoints.csv','r') as f:
            reader = csv.reader(f)
            for row in reader:
                codehex = row[1]
                if len(codehex) > 7:
                    code = eval('"' + ''.join(['\\u' + codehex[i*4:i*4+4] for i in range(len(codehex) // 4)]) + '"')
                else:
                    code = chr(int(codehex, 16))
                i = int.from_bytes(code.encode('utf-32le'), 'little')
                self.glyph_id[code] = i
                self.glyphs[i] = code

        with open('data/id_map.csv','r') as f:
            reader = csv.reader(f)
            for row in reader:
                code = bytes.fromhex(row[2]).decode()
                if code in self.glyph_id:
                    k = self.glyph_id[code]
                else:
                    i = int.from_bytes(code.encode('utf-32le'), 'little')
                    self.glyph_id[code] = i
                    self.glyphs[i] = code
                    k = i
                self.glyph_type[k] = int(row[3])

        self.id_count = len(self.glyph_id)

def sub_load(args):
    proc = subprocess.Popen([
        './data/load_font/load_font',
        args[0],
        '128',
        ], stdin=subprocess.PIPE, stdout=subprocess.PIPE)

    ret = {}
    for c in args[1]:
        if len(c) == 1:
            charbuf = c.encode("utf-32-le")
            proc.stdin.write(charbuf[:4])
            proc.stdin.flush()

            result = proc.stdout.read(32)

            code = result[:4]
            rows = int.from_bytes(result[4:8], 'little')
            width = int.from_bytes(result[8:12], 'little')
            boundingWidth = int.from_bytes(result[12:16], 'little', signed=True)
            boundingHeight = int.from_bytes(result[16:20], 'little', signed=True)
            horiBearingX = int.from_bytes(result[20:24], 'little', signed=True)
            horiBearingY = int.from_bytes(result[24:28], 'little', signed=True)
            horiAdvance = int.from_bytes(result[28:32], 'little', signed=True)

            if rows * width == 0:
                continue

            assert(charbuf == code)

            boundingWidth = boundingWidth / 64
            boundingHeight = boundingHeight / 64
            horiBearingX = horiBearingX / 64
            horiBearingY = horiBearingY / 64
            horiAdvance = horiAdvance / 64

            buffer = proc.stdout.read(rows*width)
            img = np.frombuffer(buffer, dtype='ubyte').reshape(rows,width)

            value = {
                'horizontal': {
                    'rows': rows, 
                    'width': width, 
                    'boundingWidth': boundingWidth,
                    'boundingHeight': boundingHeight,
                    'horiBearingX': horiBearingX,
                    'horiBearingY': horiBearingY,
                    'horiAdvance': horiAdvance,
                    'image': img,
                }
            }

            result = proc.stdout.read(28)

            rows = int.from_bytes(result[:4], 'little')
            width = int.from_bytes(result[4:8], 'little')
            boundingWidth = int.from_bytes(result[8:12], 'little', signed=True)
            boundingHeight = int.from_bytes(result[12:16], 'little', signed=True)
            vertBearingX = int.from_bytes(result[16:20], 'little', signed=True)
            vertBearingY = int.from_bytes(result[20:24], 'little', signed=True)
            vertAdvance = int.from_bytes(result[24:28], 'little', signed=True)

            boundingWidth = boundingWidth / 64
            boundingHeight = boundingHeight / 64
            vertBearingX = vertBearingX / 64
            vertBearingY = vertBearingY / 64
            vertAdvance = vertAdvance / 64

            buffer = proc.stdout.read(rows*width)
            img = np.frombuffer(buffer, dtype='ubyte').reshape(rows,width)

            value['vertical'] = {
                'rows': rows, 
                'width': width, 
                'boundingWidth': boundingWidth,
                'boundingHeight': boundingHeight,
                'vertBearingX': vertBearingX,
                'vertBearingY': vertBearingY,
                'vertAdvance': vertAdvance,
                'image': img,
            }

            ret[(args[0],c)] = value
        else:
            pass

    proc.stdin.close()
    return ret

def sub_load_image(path):
    dirnames = glob.glob(os.path.join(path, '*'))
    ret = {}
    for d in dirnames:
        c_code = os.path.basename(d)
        char = str(bytes.fromhex(c_code), 'utf-8')
        count = 0
        for f in glob.glob(os.path.join(d, '*.png')):
            rawim = np.asarray(Image.open(f).convert('L'))
            ylines = np.any(rawim < 255, axis=1)
            content = np.where(ylines)[0]
            rows = content[-1] - content[0] + 1
            horiBearingY = 128 - 16 - content[0]
            vertBearingY = content[0] - 16
            y = content[0]
            xlines = np.any(rawim < 255, axis=0)
            content = np.where(xlines)[0]
            width = content[-1] - content[0] + 1
            horiBearingX = content[0] - 16
            vertBearingX = content[0] - 64
            x = content[0]

            if rows == 0 or width == 0:
                continue

            img = 255 - rawim[y:y+rows,x:x+width]

            ret[('hand%06d'%count,char)] = {
                'horizontal': {
                    'rows': rows, 
                    'width': width, 
                    'boundingWidth': width,
                    'boundingHeight': rows,
                    'horiBearingX': horiBearingX,
                    'horiBearingY': horiBearingY,
                    'horiAdvance': 96.0,
                    'image': img,
                },
                'vertical': {
                    'rows': rows, 
                    'width': width, 
                    'boundingWidth': width,
                    'boundingHeight': rows,
                    'vertBearingX': horiBearingX,
                    'vertBearingY': horiBearingY,
                    'vertAdvance': 96.0,
                    'image': img,
                }
            }
            count += 1

        vert_imgs = glob.glob(os.path.join(d, 'vert', '*.png'))
        if 0 < len(vert_imgs) <= count:
            for i in range(count):
                f = vert_imgs[i % len(vert_imgs)]

                rawim = np.asarray(Image.open(f).convert('L'))
                ylines = np.any(rawim < 255, axis=1)
                content = np.where(ylines)[0]
                rows = content[-1] - content[0] + 1
                horiBearingY = 128 - 16 - content[0]
                vertBearingY = content[0] - 16
                y = content[0]
                xlines = np.any(rawim < 255, axis=0)
                content = np.where(xlines)[0]
                width = content[-1] - content[0] + 1
                horiBearingX = content[0] - 16
                vertBearingX = content[0] - 64
                x = content[0]

                if rows == 0 or width == 0:
                    continue

                img = 255 - rawim[y:y+rows,x:x+width]
                ret[('hand%06d'%i,char)]['vertical'] = {
                    'rows': rows, 
                    'width': width, 
                    'boundingWidth': width,
                    'boundingHeight': rows,
                    'vertBearingX': horiBearingX,
                    'vertBearingY': horiBearingY,
                    'vertAdvance': 96.0,
                    'image': img,
                }
        elif 0 < len(vert_imgs):
            vcount = 0
            for f in vert_imgs:
                rawim = np.asarray(Image.open(f).convert('L'))
                ylines = np.any(rawim < 255, axis=1)
                content = np.where(ylines)[0]
                rows = content[-1] - content[0] + 1
                horiBearingY = 128 - 16 - content[0]
                vertBearingY = content[0] - 16
                y = content[0]
                xlines = np.any(rawim < 255, axis=0)
                content = np.where(xlines)[0]
                width = content[-1] - content[0] + 1
                horiBearingX = content[0] - 16
                vertBearingX = content[0] - 64
                x = content[0]

                if rows == 0 or width == 0:
                    continue

                img = 255 - rawim[y:y+rows,x:x+width]

                ret[('hand%06d'%vcount,char)] = {
                    'horizontal': ret[('hand%06d'%(vcount % count),char)]['horizontal'],
                    'vertical': {
                        'rows': rows, 
                        'width': width, 
                        'boundingWidth': width,
                        'boundingHeight': rows,
                        'vertBearingX': vertBearingY,
                        'vertBearingY': vertBearingX,
                        'vertAdvance': 96.0,
                        'image': img,
                    }
                }
                vcount += 1

    return ret

def gaussian_kernel(kernlen=7, xstd=1., ystd=1.):
    gkern1dx = signal.gaussian(kernlen, std=xstd).reshape(kernlen, 1)
    gkern1dy = signal.gaussian(kernlen, std=ystd).reshape(kernlen, 1)
    gkern2d = np.outer(gkern1dy, gkern1dx)
    return gkern2d

def apply_random_filter(images):
    def _gaussian_kernel(kernel_size, sigma, n_channels, dtype):
        x = tf.range(-kernel_size // 2 + 1, kernel_size // 2 + 1, dtype=dtype)
        g = tf.math.exp(-(tf.pow(x, 2) / (2 * tf.pow(tf.cast(sigma, dtype), 2))))
        g_norm2d = tf.pow(tf.reduce_sum(g), 2)
        g_kernel = tf.tensordot(g, g, axes=0) / g_norm2d
        g_kernel = tf.expand_dims(g_kernel, axis=-1)
        return tf.expand_dims(tf.tile(g_kernel, (1, 1, n_channels)), axis=-1)

    def apply_filter_blur(img):
        blur = _gaussian_kernel(7, tf.random.uniform([],0.,1.75), 1, img.dtype)
        img = tf.nn.depthwise_conv2d(img[tf.newaxis,...,tf.newaxis], blur, [1,1,1,1], 'SAME')
        return img[0,...,0]

    def apply_filter_sharpen(img):
        blur = _gaussian_kernel(7, tf.random.uniform([],0.,6.), 1, img.dtype)
        gauss = tf.nn.depthwise_conv2d(img[tf.newaxis,...,tf.newaxis], blur, [1,1,1,1], 'SAME')
        gauss = gauss[0,...,0]
        gain = tf.random.uniform([],0.,5.)
        img = (1 + gain) * img - gain * gauss
        return img

    p = tf.random.uniform([], 0., 1.)
    images = tf.where(p < 0.25, 
        apply_filter_blur(images),
        tf.where(p < 0.5, apply_filter_sharpen(images), images)
    )
    return images

def random_morphology(image):
    n = min(max(image.shape) // 30, 3)
    if n > 0:
        iter = np.random.randint(-n, n+1)
        if iter > 0:
            skeleton = skeletonize(image > 0)
            for _ in range(iter):
                image = ndimage.grey_erosion(image, size=(3,3))
            image = np.maximum(image, skeleton * 255)
        elif iter < 0:
            iter = -iter
            for _ in range(iter):
                image = ndimage.grey_dilation(image, size=(3,3))
    return image

class FontData(BaseData):
    def __init__(self):
        super().__init__()

        self.img_cache = {}
        print('loading handwrite image')
        self.img_cache.update(sub_load_image('data/handwritten'))

        print('loading enfont')
        enfont_files = sorted(glob.glob('data/enfont/*.ttf') + glob.glob('data/enfont/*.otf'))
        en_glyphs = [self.glyphs[key] for key in self.glyphs.keys() if self.glyph_type.get(key,-1) in [0,1,2,6]]
        items = [(f, en_glyphs) for f in enfont_files]
        total = len(enfont_files)
        with Pool() as pool:
            dicts = tqdm.tqdm(pool.imap_unordered(sub_load, items), total=total)
            for dictitem in dicts:
                self.img_cache.update(dictitem)

        print('loading jpfont')
        jpfont_files = sorted(glob.glob('data/jpfont/*.ttf') + glob.glob('data/jpfont/*.otf'))
        items = [(f, list(self.glyphs.values())) for f in jpfont_files]
        total = len(jpfont_files)
        with Pool() as pool:
            dicts = tqdm.tqdm(pool.imap_unordered(sub_load, items), total=total)
            for dictitem in dicts:
                self.img_cache.update(dictitem)

        type_count = max([self.glyph_type[k] for k in self.glyph_type]) + 1
        gtype_count = [0 for _ in range(type_count)] 
        type_count = [0 for _ in range(type_count)]
        for key in self.img_cache:
            t = self.glyph_type.get(self.glyph_id[key[1]],-1)
            if t < 0:
                continue
            type_count[t] += 1
        for k in self.glyph_type:
            gtype_count[self.glyph_type[k]] += 1

        self.prob_map = [
                2.0 / type_count[0],
                1.1 / type_count[1],
                1.25 / type_count[2],
                1.5 / type_count[3],
                1.0 / type_count[4],
                1.0 / type_count[5],
                0.5 / type_count[6],
                0.5 / type_count[7],
                0.25 / type_count[8],
                0.05 / type_count[9],
                0.05 / type_count[10],
                0.
                ]
        self.prob_map_kanji = [
                0 / type_count[0],
                0 / type_count[1],
                0 / type_count[2],
                0 / type_count[3],
                0 / type_count[4],
                3.0 / type_count[5],
                0 / type_count[6],
                0 / type_count[7],
                2.0 / type_count[8],
                1.0 / type_count[9],
                0.5 / type_count[10],
                0.
                ]
        self.prob_map_num = [
                1.0 / type_count[0],
                0 / type_count[1],
                0 / type_count[2],
                0 / type_count[3],
                0 / type_count[4],
                0 / type_count[5],
                0 / type_count[6],
                0 / type_count[7],
                0 / type_count[8],
                0 / type_count[9],
                0 / type_count[10],
                0.
                ]
        self.prob_map_alpha = [
                0 / type_count[0],
                1.0 / type_count[1],
                1.0 / type_count[2],
                0 / type_count[3],
                0 / type_count[4],
                0 / type_count[5],
                0 / type_count[6],
                0 / type_count[7],
                0 / type_count[8],
                0 / type_count[9],
                0 / type_count[10],
                0.
                ]
        self.prob_map_hira = [
                0 / type_count[0],
                0 / type_count[1],
                0 / type_count[2],
                1.0 / type_count[3],
                0 / type_count[4],
                0 / type_count[5],
                0 / type_count[6],
                0 / type_count[7],
                0 / type_count[8],
                0 / type_count[9],
                0 / type_count[10],
                0.
                ]
        self.prob_map_clustering = [
                gtype_count[0] / type_count[0],
                gtype_count[1] / type_count[1],
                gtype_count[2] / type_count[2],
                gtype_count[3] / type_count[3],
                gtype_count[4] / type_count[4],
                gtype_count[5] / type_count[5],
                gtype_count[6] / type_count[6],
                gtype_count[7] / type_count[7],
                0. / type_count[8],
                0.,
                0.,
                0.
                ]

        self.image_keys = list(self.img_cache.keys())
        self.test_keys = self.get_test_keys()
        self.train_keys = self.get_train_keys()

        self.random_background = glob.glob('data/background/*')

        self.max_std = 8.0
        self.min_ker = 4

    def get_test_keys(self):
        def fontname(fontpath):
            return os.path.splitext(os.path.basename(fontpath))[0]

        keys = self.image_keys
        test_keys = [k for k in keys if fontname(k[0]).startswith('Noto')]
        return test_keys

    def get_train_keys(self):
        def fontname(fontpath):
            return os.path.splitext(os.path.basename(fontpath))[0]

        keys = self.image_keys
        train_keys = [k for k in keys if not fontname(k[0]).startswith('Noto')]
        return train_keys

    def load_background_images(self):
        im_file = random.choice(self.random_background)
        raw_im = tf.io.read_file(im_file)
        im = tf.io.decode_image(raw_im, channels=3, expand_animations=False)
        img = tf.image.convert_image_dtype(im, dtype=tf.float32)
        img = img * 2 - 1
        w = tf.cast(tf.shape(img)[1], dtype=tf.float32)
        h = tf.cast(tf.shape(img)[0], dtype=tf.float32)
        scale_min = tf.maximum(width / w, height / h)
        slale_max = tf.maximum(scale_min + 0.5, 1.5)
        s = tf.random.uniform([], scale_min, slale_max)
        img = tf.image.resize(img, [h * s, w * s])
        img = tf.image.random_crop(img, [height, width, 3])

        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_flip_up_down(img)
        img = tf.image.random_brightness(img, max_delta=1.0)
        img = tf.image.random_contrast(img, lower=0.2, upper=1.8)
        img = tf.clip_by_value(img, -1.0, 1.0)

        return img

    def construct_tateyokotext(self, keys, fonts):
        max_count = 256
        angle_max = 15.0

        @tf.autograph.experimental.do_not_convert
        def tate_images():
            min_pixel = 20
            max_pixel = 100
            text_size = random.randint(min_pixel, max_pixel)

            images = np.zeros([height, width], dtype=np.float32)
            keymap = np.zeros([height // scale, width // scale], dtype=np.float32)
            xsizes = np.zeros([height // scale, width // scale], dtype=np.float32)
            ysizes = np.zeros([height // scale, width // scale], dtype=np.float32)
            offsetx = np.zeros([height // scale, width // scale], dtype=np.float32)
            offsety = np.zeros([height // scale, width // scale], dtype=np.float32)
            lines = np.zeros([height // scale, width // scale], dtype=np.float32)
            lines = Image.fromarray(lines)
            draw = ImageDraw.Draw(lines)
            seps = np.zeros([height // scale, width // scale], dtype=np.float32)
            ids = np.zeros([height // scale, width // scale], dtype=np.int32)

            angle = angle_max * np.random.normal() / 180 * np.pi
            angle = np.clip(angle, -np.pi, np.pi)
            pad_x = np.random.normal() * width / 20
            pad_y = np.random.normal() * height / 20

            select_font = random.choices(fonts[0], k=1, weights=fonts[1])[0]
            probs = [1. if key[0] == select_font else 0. for key in keys]
            selection = [key for key in random.choices(keys, k=max_count, weights=probs)]
            probs2 = [1. if key[0] == select_font and self.glyph_type.get(self.glyph_id[key[1]],-1) == 0 else 0. for key in keys]
            selection2 = [key for key in random.choices(keys, k=max_count*2, weights=probs2)]

            base_line = width - text_size // 2
            line_space = int(text_size * random.uniform(1.05, 2.0))
            line_start = 0
            line_end = 0
            isnum = -1
            i = 0
            for key in selection:
                if isnum < 0 or isnum > 1:
                    if np.random.uniform() < 0.1:
                        isnum = 0
                    else:
                        isnum = -1

                if isnum < 0:
                    item = self.img_cache[key]['vertical']
                    if item['width'] * item['rows'] == 0:
                        continue
                    w = item['width'] / 128 * text_size
                    h = item['rows'] / 128 * text_size
                    vertBearingX = item['vertBearingX'] / 128 * text_size
                    vertBearingY = item['vertBearingY'] / 128 * text_size
                    vertAdvance = item['vertAdvance'] / 128 * text_size
                    horiBearingX = 0
                else:
                    item = self.img_cache[key]['vertical']
                    if item['width'] * item['rows'] == 0:
                        continue

                    key = selection2[i]
                    item = self.img_cache[key]['horizontal']
                    if item['width'] * item['rows'] == 0:
                        continue
                    w = item['width'] / 128 * text_size
                    h = item['rows'] / 128 * text_size
                    horiBearingY = item['horiBearingY'] / 128 * text_size
                    horiBearingX = item['horiBearingX'] / 128 * text_size
                    vertBearingX = -text_size * 0.5
                    vertBearingY = 0
                    vertAdvance = text_size

                if line_end + vertAdvance >= height:
                    draw.line(((base_line // scale, line_start // scale), 
                        (base_line // scale, line_end // scale)), fill=255, width=3)

                    base_line -= line_space
                    if base_line - text_size / 2 < 0:
                        break
                    line_start = 0
                    line_end = 0
                
                if isnum >= 0:
                    t = (line_end + vertBearingY + text_size * 0.75 - horiBearingY) / height
                else:
                    t = (line_end + vertBearingY) / height
                if isnum > 0:
                    l = (base_line + horiBearingX) / width
                else:
                    l = (base_line + vertBearingX + horiBearingX) / width
                w = w / width
                h = h / height
                cx = l + w / 2
                cy = t + h / 2

                kernel_size = max(self.min_ker, int(max(w, h) / (2 * scale) * width))
                std_x = min(self.max_std, max(self.min_ker, w / (2 * scale) * width) / 3)
                std_y = min(self.max_std, max(self.min_ker, h / (2 * scale) * height) / 3)
                center_kernel = gaussian_kernel(kernlen=kernel_size*2+1, xstd=std_x, ystd=std_y)

                center_xmin = int(cx / scale * width) - kernel_size
                center_xmax = int(cx / scale * width) + kernel_size + 1
                center_ymin = int(cy / scale * height) - kernel_size
                center_ymax = int(cy / scale * height) + kernel_size + 1
                padx1 = max(0, 0 - center_xmin)
                padx2 = max(0, center_xmax - width // scale)
                pady1 = max(0, 0 - center_ymin)
                pady2 = max(0, center_ymax - height // scale)
                center_xmin += padx1
                center_xmax -= padx2
                center_ymin += pady1
                center_ymax -= pady2
                ker = kernel_size * 2 + 1
                if center_ymax - center_ymin > 1 and center_xmax - center_xmin > 1:
                    keymap[center_ymin:center_ymax, center_xmin:center_xmax] = np.maximum(keymap[center_ymin:center_ymax, center_xmin:center_xmax], center_kernel[pady1:ker-pady2,padx1:ker-padx2])

                    size_xmin = np.clip(int((cx - w/2) * width / scale), 0, width // scale)
                    size_xmax = np.clip(int((cx + w/2) * width / scale) + 1, 0, width // scale)
                    size_ymin = np.clip(int((cy - h/2) * height / scale), 0, height // scale)
                    size_ymax = np.clip(int((cy + h/2) * height / scale) + 1, 0, height // scale)
                    size_mapx, size_mapy = np.meshgrid(np.arange(size_xmin, size_xmax) - cx * width / scale, np.arange(size_ymin, size_ymax) - cy * height / scale)
                    size_map = size_mapx ** 2 / max(w/2 * width / scale, 1) ** 2 + size_mapy ** 2 / max(h/2 * height / scale, 1) ** 2 < 1

                    center_x = int(cx / scale * width)
                    center_y = int(cy / scale * height)
                    offset_x = (cx * width % scale) / width * np.cos(angle)
                    offset_y = (cy * height % scale) / height * np.sin(angle + np.pi / 2)
                    offset_x += pad_x % scale
                    offset_y += pad_y % scale
                    offset_x = offset_x / scale - (np.arange(size_xmin, size_xmax) - center_x) * np.cos(angle)
                    offset_y = offset_y / scale - (np.arange(size_ymin, size_ymax) - center_y) * np.sin(angle + np.pi / 2)
                    offset_x = offset_x[np.newaxis,...] - np.linspace(-(size_ymax-size_ymin) * np.sin(angle) / 2, (size_ymax-size_ymin) * np.sin(angle) / 2, size_ymax-size_ymin)[...,np.newaxis]
                    offset_y = offset_y[...,np.newaxis] - np.linspace(-(size_xmax-size_xmin) * np.cos(angle + np.pi / 2) / 2, (size_xmax-size_xmin) * np.cos(angle + np.pi / 2) / 2, size_xmax-size_xmin)[np.newaxis,...]
                    offsetx[size_ymin:size_ymax, size_xmin:size_xmax] = np.where(size_map, offset_x, offsetx[size_ymin:size_ymax, size_xmin:size_xmax])
                    offsety[size_ymin:size_ymax, size_xmin:size_xmax] = np.where(size_map, offset_y, offsety[size_ymin:size_ymax, size_xmin:size_xmax])

                    fixw = w * np.abs(np.cos(angle)) + h * np.abs(np.sin(angle))
                    fixh = h * np.abs(np.cos(angle)) + w * np.abs(np.sin(angle))
                    fixw = np.log10(fixw * 10)
                    fixh = np.log10(fixh * 10)
                    xsizes[size_ymin:size_ymax, size_xmin:size_xmax] = np.where(size_map, fixw, xsizes[size_ymin:size_ymax, size_xmin:size_xmax])
                    ysizes[size_ymin:size_ymax, size_xmin:size_xmax] = np.where(size_map, fixh, ysizes[size_ymin:size_ymax, size_xmin:size_xmax])

                    id_char = self.glyph_id[key[1]]
                    ids[size_ymin:size_ymax, size_xmin:size_xmax] = np.where(size_map, id_char, ids[size_ymin:size_ymax, size_xmin:size_xmax])

                    w = max(int(item['width'] / 128 * text_size), 1)
                    h = max(int(item['rows'] / 128 * text_size), 1)
                    if isnum > 0:
                        l = int(np.clip(base_line + horiBearingX, 0, width - w))
                    else:
                        l = int(np.clip(base_line + vertBearingX + horiBearingX, 0, width - w))
                    if isnum >= 0:
                        t = int(np.clip(line_end + vertBearingY + text_size * 0.75 - horiBearingY, 0, height - h))
                    else:
                        t = int(np.clip(line_end + vertBearingY, 0, height - h))
                    im = np.asarray(Image.fromarray(item['image']).resize((w,h)))
                    im = random_morphology(im)
                    images[t:t+h,l:l+w] = np.maximum(
                            images[t:t+h,l:l+w], 
                            im)

                if isnum != 0:
                    line_end += vertAdvance            
                if isnum >= 0:
                    isnum += 1
                i += 1

            im = Image.fromarray(images).rotate(angle / np.pi * 180, resample=Image.BILINEAR, translate=(pad_x, pad_y))
            lines = lines.rotate(angle / np.pi * 180, resample=Image.BILINEAR, translate=(pad_x / scale, pad_y / scale))
            keymapim1 = Image.fromarray(keymap).rotate(angle / np.pi * 180, resample=Image.NEAREST, translate=(pad_x / scale, pad_y / scale))
            keymapim2 = Image.fromarray(keymap).rotate(angle / np.pi * 180, resample=Image.BILINEAR, translate=(pad_x / scale, pad_y / scale))
            keymapim = np.maximum(keymapim1, keymapim2)
            xsizeim = Image.fromarray(xsizes).rotate(angle / np.pi * 180, resample=Image.NEAREST, translate=(pad_x / scale, pad_y / scale))
            ysizeim = Image.fromarray(ysizes).rotate(angle / np.pi * 180, resample=Image.NEAREST, translate=(pad_x / scale, pad_y / scale))
            xoffsetim = Image.fromarray(offsetx).rotate(angle / np.pi * 180, resample=Image.BILINEAR, translate=(pad_x / scale, pad_y / scale))
            yoffsetim = Image.fromarray(offsety).rotate(angle / np.pi * 180, resample=Image.BILINEAR, translate=(pad_x / scale, pad_y / scale))
            sepim = Image.fromarray(seps).rotate(angle / np.pi * 180, resample=Image.BILINEAR, translate=(pad_x / scale, pad_y / scale))
            labels = np.stack([keymapim, xsizeim, ysizeim, xoffsetim, yoffsetim, np.asarray(lines) / 255., sepim], -1)
            idsim = Image.fromarray(ids).rotate(angle / np.pi * 180, resample=Image.NEAREST, translate=(pad_x / scale, pad_y / scale))
            return np.asarray(im) / 255., labels, np.asarray(idsim)

        images, labels, ids = tf.py_function(
            func=tate_images, 
            inp=[], 
            Tout=[tf.float32, tf.float32, tf.int32])
        images = tf.ensure_shape(images, [height, width])
        labels = tf.ensure_shape(labels, [height // scale, width // scale, 7])
        ids = tf.ensure_shape(ids, [height // scale, width // scale])

        images = apply_random_filter(images)

        return self.sub_constructimage(images, labels, ids, False)

    def construct_rendertext(self, keys, fonts):
        max_count = 256
        angle_max = 15.0

        @tf.autograph.experimental.do_not_convert
        def yoko_images():
            min_pixel = 20
            max_pixel = 100
            text_size = random.randint(min_pixel, max_pixel)

            line_space = int(text_size * random.uniform(1.05, 2.0))
            block_count = 2
            line_break = int(random.uniform(0.3,0.7) * width)
            break_space = text_size * random.uniform(0.6, 1.5)

            images = np.zeros([height, width], dtype=np.float32)
            keymap = np.zeros([height // scale, width // scale], dtype=np.float32)
            xsizes = np.zeros([height // scale, width // scale], dtype=np.float32)
            ysizes = np.zeros([height // scale, width // scale], dtype=np.float32)
            offsetx = np.zeros([height // scale, width // scale], dtype=np.float32)
            offsety = np.zeros([height // scale, width // scale], dtype=np.float32)
            lines = np.zeros([height // scale, width // scale], dtype=np.float32)
            lines = Image.fromarray(lines)
            draw = ImageDraw.Draw(lines)
            seps = np.zeros([height // scale, width // scale], dtype=np.float32)
            ids = np.zeros([height // scale, width // scale], dtype=np.int32)

            angle = angle_max * np.random.normal() / 180 * np.pi
            angle = np.clip(angle, -np.pi, np.pi)
            pad_x = np.random.normal() * width / 20
            pad_y = np.random.normal() * height / 20

            select_font = random.choices(fonts[0], k=1, weights=fonts[1])[0]
            probs = [1. if key[0] == select_font else 0. for key in keys]
            selection = [key for key in random.choices(keys, k=max_count, weights=probs)]

            base_line = line_space
            block_no = 0
            line_start = int(max(0, 0 if block_count == 1 or block_no == 0 else line_break + break_space))
            line_end = int(min(width, width if block_count == 1 or block_no == 1 else line_break - break_space))
            temp_lineend = line_start
            linebuf = []
            text_count = [0, 0]
            sep_end = 0
            for key in selection:
                item = self.img_cache[key]['horizontal']
                if item['width'] * item['rows'] == 0:
                    continue

                w = item['width'] / 128 * text_size
                h = item['rows'] / 128 * text_size
                horiBearingX = item['horiBearingX'] / 128 * text_size
                horiBearingY = item['horiBearingY'] / 128 * text_size
                horiAdvance = item['horiAdvance'] / 128 * text_size

                if temp_lineend + horiAdvance < line_end:
                    linebuf.append((key, item))
                    temp_lineend += horiAdvance
                else:
                    remain = line_end - temp_lineend
                    if block_no == 0:
                        line_start += remain

                    if len(linebuf) > 1:
                        draw.line(((line_start // scale, base_line // scale), 
                            (line_end // scale, base_line // scale)), fill=255, width=3)

                    text_count[block_no] += len(linebuf)

                    for key, item in linebuf:
                        w = item['width'] / 128 * text_size
                        h = item['rows'] / 128 * text_size
                        horiBearingX = item['horiBearingX'] / 128 * text_size
                        horiBearingY = item['horiBearingY'] / 128 * text_size
                        horiAdvance = item['horiAdvance'] / 128 * text_size

                        l = (line_start + horiBearingX) / width
                        t = (base_line - horiBearingY) / height
                        w = w / width
                        h = h / height
                        cx = l + w / 2
                        cy = t + h / 2

                        kernel_size = max(self.min_ker, int(max(w, h) / (2 * scale) * width))
                        std_x = min(self.max_std, max(self.min_ker, w / (2 * scale) * width) / 3)
                        std_y = min(self.max_std, max(self.min_ker, h / (2 * scale) * height) / 3)
                        center_kernel = gaussian_kernel(kernlen=kernel_size*2+1, xstd=std_x, ystd=std_y)

                        center_xmin = int(cx / scale * width) - kernel_size
                        center_xmax = int(cx / scale * width) + kernel_size + 1
                        center_ymin = int(cy / scale * height) - kernel_size
                        center_ymax = int(cy / scale * height) + kernel_size + 1
                        padx1 = max(0, 0 - center_xmin)
                        padx2 = max(0, center_xmax - width // scale)
                        pady1 = max(0, 0 - center_ymin)
                        pady2 = max(0, center_ymax - height // scale)
                        center_xmin += padx1
                        center_xmax -= padx2
                        center_ymin += pady1
                        center_ymax -= pady2
                        ker = kernel_size * 2 + 1
                        keymap[center_ymin:center_ymax, center_xmin:center_xmax] = np.maximum(keymap[center_ymin:center_ymax, center_xmin:center_xmax], center_kernel[pady1:ker-pady2,padx1:ker-padx2])

                        size_xmin = np.clip(int((cx - w/2) * width / scale), 0, width // scale)
                        size_xmax = np.clip(int((cx + w/2) * width / scale) + 1, 0, width // scale)
                        size_ymin = np.clip(int((cy - h/2) * height / scale), 0, height // scale)
                        size_ymax = np.clip(int((cy + h/2) * height / scale) + 1, 0, height // scale)
                        size_mapx, size_mapy = np.meshgrid(np.arange(size_xmin, size_xmax) - cx * width / scale, np.arange(size_ymin, size_ymax) - cy * height / scale)
                        size_map = size_mapx ** 2 / max(w/2 * width / scale, 1) ** 2 + size_mapy ** 2 / max(h/2 * height / scale, 1) ** 2 < 1

                        center_x = int(cx / scale * width)
                        center_y = int(cy / scale * height)
                        offset_x = (cx * width % scale) / width * np.cos(angle)
                        offset_y = (cy * height % scale) / height * np.sin(angle + np.pi / 2)
                        offset_x += pad_x % scale
                        offset_y += pad_y % scale
                        offset_x = offset_x / scale - (np.arange(size_xmin, size_xmax) - center_x) * np.cos(angle)
                        offset_y = offset_y / scale - (np.arange(size_ymin, size_ymax) - center_y) * np.sin(angle + np.pi / 2)
                        offset_x = offset_x[np.newaxis,...] - np.linspace(-(size_ymax-size_ymin) * np.sin(angle) / 2, (size_ymax-size_ymin) * np.sin(angle) / 2, size_ymax-size_ymin)[...,np.newaxis]
                        offset_y = offset_y[...,np.newaxis] - np.linspace(-(size_xmax-size_xmin) * np.cos(angle + np.pi / 2) / 2, (size_xmax-size_xmin) * np.cos(angle + np.pi / 2) / 2, size_xmax-size_xmin)[np.newaxis,...]
                        offsetx[size_ymin:size_ymax, size_xmin:size_xmax] = np.where(size_map, offset_x, offsetx[size_ymin:size_ymax, size_xmin:size_xmax])
                        offsety[size_ymin:size_ymax, size_xmin:size_xmax] = np.where(size_map, offset_y, offsety[size_ymin:size_ymax, size_xmin:size_xmax])

                        fixw = w * np.abs(np.cos(angle)) + h * np.abs(np.sin(angle))
                        fixh = h * np.abs(np.cos(angle)) + w * np.abs(np.sin(angle))
                        fixw = np.log10(fixw * 10)
                        fixh = np.log10(fixh * 10)
                        xsizes[size_ymin:size_ymax, size_xmin:size_xmax] = np.where(size_map, fixw, xsizes[size_ymin:size_ymax, size_xmin:size_xmax])
                        ysizes[size_ymin:size_ymax, size_xmin:size_xmax] = np.where(size_map, fixh, ysizes[size_ymin:size_ymax, size_xmin:size_xmax])

                        id_char = self.glyph_id[key[1]]
                        ids[size_ymin:size_ymax, size_xmin:size_xmax] = np.where(size_map, id_char, ids[size_ymin:size_ymax, size_xmin:size_xmax])

                        w = max(int(item['width'] / 128 * text_size), 1)
                        h = max(int(item['rows'] / 128 * text_size), 1)
                        top = int(np.clip(base_line - horiBearingY, 0, height - h))
                        left = int(np.clip(line_start + horiBearingX, 0, width - w))
                        im = np.asarray(Image.fromarray(item['image']).resize((w,h)))
                        im = random_morphology(im)
                        images[top:top+h,left:left+w] = np.maximum(
                                images[top:top+h,left:left+w], 
                                im)

                        line_start += int(horiAdvance)

                    base_line += line_space
                    if base_line + text_size >= height:
                        if block_no == 0:
                            sep_end = base_line - line_space
                        base_line = line_space
                        block_no += 1
                    if block_no >= block_count:
                        break
                    line_start = int(max(0, 0 if block_count == 1 or block_no == 0 else line_break + break_space))
                    line_end = int(min(width, width if block_count == 1 or block_no == 1 else line_break - break_space))
                    temp_lineend = line_start
                    linebuf = []

            if all(t > 1 for t in text_count):
                l = max(1,line_break // scale)
                t = line_space // 2 // scale
                b = sep_end // scale
                seps[t:b, l-1:l+2] = 1

            im = Image.fromarray(images).rotate(angle / np.pi * 180, resample=Image.BILINEAR, translate=(pad_x, pad_y))
            lines = lines.rotate(angle / np.pi * 180, resample=Image.BILINEAR, translate=(pad_x / scale, pad_y / scale))
            keymapim1 = Image.fromarray(keymap).rotate(angle / np.pi * 180, resample=Image.NEAREST, translate=(pad_x / scale, pad_y / scale))
            keymapim2 = Image.fromarray(keymap).rotate(angle / np.pi * 180, resample=Image.BILINEAR, translate=(pad_x / scale, pad_y / scale))
            keymapim = np.maximum(keymapim1, keymapim2)
            xsizeim = Image.fromarray(xsizes).rotate(angle / np.pi * 180, resample=Image.NEAREST, translate=(pad_x / scale, pad_y / scale))
            ysizeim = Image.fromarray(ysizes).rotate(angle / np.pi * 180, resample=Image.NEAREST, translate=(pad_x / scale, pad_y / scale))
            xoffsetim = Image.fromarray(offsetx).rotate(angle / np.pi * 180, resample=Image.BILINEAR, translate=(pad_x / scale, pad_y / scale))
            yoffsetim = Image.fromarray(offsety).rotate(angle / np.pi * 180, resample=Image.BILINEAR, translate=(pad_x / scale, pad_y / scale))
            sepim = Image.fromarray(seps).rotate(angle / np.pi * 180, resample=Image.BILINEAR, translate=(pad_x / scale, pad_y / scale))
            labels = np.stack([keymapim, xsizeim, ysizeim, xoffsetim, yoffsetim, np.asarray(lines) / 255., sepim], -1)
            idsim = Image.fromarray(ids).rotate(angle / np.pi * 180, resample=Image.NEAREST, translate=(pad_x / scale, pad_y / scale))
            return np.asarray(im) / 255., labels, np.asarray(idsim)

        @tf.autograph.experimental.do_not_convert
        def tate_images():
            min_pixel = 20
            max_pixel = 100
            text_size = random.randint(min_pixel, max_pixel)

            line_space = int(text_size * random.uniform(1.05, 2.0))
            block_count = 2
            line_break = int(random.uniform(0.3,0.7) * height)
            break_space = text_size * random.uniform(0.6, 1.0)

            images = np.zeros([height, width], dtype=np.float32)
            keymap = np.zeros([height // scale, width // scale], dtype=np.float32)
            xsizes = np.zeros([height // scale, width // scale], dtype=np.float32)
            ysizes = np.zeros([height // scale, width // scale], dtype=np.float32)
            offsetx = np.zeros([height // scale, width // scale], dtype=np.float32)
            offsety = np.zeros([height // scale, width // scale], dtype=np.float32)
            lines = np.zeros([height // scale, width // scale], dtype=np.float32)
            lines = Image.fromarray(lines)
            draw = ImageDraw.Draw(lines)
            seps = np.zeros([height // scale, width // scale], dtype=np.float32)
            ids = np.zeros([height // scale, width // scale], dtype=np.int32)

            angle = angle_max * np.random.normal() / 180 * np.pi
            angle = np.clip(angle, -np.pi, np.pi)
            pad_x = np.random.normal() * width / 20
            pad_y = np.random.normal() * height / 20

            select_font = random.choices(fonts[0], k=1, weights=fonts[1])[0]
            probs = [1. if key[0] == select_font else 0. for key in keys]
            selection = [key for key in random.choices(keys, k=max_count, weights=probs)]

            base_line = width - line_space + text_size // 2
            block_no = 0
            line_start = int(max(0, 0 if block_count == 1 or block_no == 0 else line_break + break_space))
            line_end = int(min(height, height if block_count == 1 or block_no == 1 else line_break - break_space))
            temp_lineend = line_start
            linebuf = []
            text_count = [0, 0]
            sep_end = 0
            for key in selection:
                item = self.img_cache[key]['vertical']
                if item['width'] * item['rows'] == 0:
                    continue

                w = item['width'] / 128 * text_size
                h = item['rows'] / 128 * text_size
                vertBearingX = item['vertBearingX'] / 128 * text_size
                vertBearingY = item['vertBearingY'] / 128 * text_size
                vertAdvance = item['vertAdvance'] / 128 * text_size

                if temp_lineend + vertAdvance < line_end:
                    linebuf.append((key,item))
                    temp_lineend += vertAdvance
                else:
                    remain = line_end - temp_lineend
                    if block_no == 0:
                        line_start += remain

                    if len(linebuf) > 1:
                        draw.line(((base_line // scale, line_start // scale), 
                            (base_line // scale, line_end // scale)), fill=255, width=3)

                    text_count[block_no] += len(linebuf)

                    for key, item in linebuf:
                        w = item['width'] / 128 * text_size
                        h = item['rows'] / 128 * text_size
                        vertBearingX = item['vertBearingX'] / 128 * text_size
                        vertBearingY = item['vertBearingY'] / 128 * text_size
                        vertAdvance = item['vertAdvance'] / 128 * text_size

                        l = (base_line + vertBearingX) / width
                        t = (line_start + vertBearingY) / height
                        w = w / width
                        h = h / height
                        cx = l + w / 2
                        cy = t + h / 2

                        kernel_size = max(self.min_ker, int(max(w, h) / (2 * scale) * width))
                        std_x = min(self.max_std, max(self.min_ker, w / (2 * scale) * width) / 3)
                        std_y = min(self.max_std, max(self.min_ker, h / (2 * scale) * height) / 3)
                        center_kernel = gaussian_kernel(kernlen=kernel_size*2+1, xstd=std_x, ystd=std_y)

                        center_xmin = int(cx / scale * width) - kernel_size
                        center_xmax = int(cx / scale * width) + kernel_size + 1
                        center_ymin = int(cy / scale * height) - kernel_size
                        center_ymax = int(cy / scale * height) + kernel_size + 1
                        padx1 = max(0, 0 - center_xmin)
                        padx2 = max(0, center_xmax - width // scale)
                        pady1 = max(0, 0 - center_ymin)
                        pady2 = max(0, center_ymax - height // scale)
                        center_xmin += padx1
                        center_xmax -= padx2
                        center_ymin += pady1
                        center_ymax -= pady2
                        ker = kernel_size * 2 + 1
                        if center_ymax - center_ymin > 1 and center_xmax - center_xmin > 1:
                            keymap[center_ymin:center_ymax, center_xmin:center_xmax] = np.maximum(keymap[center_ymin:center_ymax, center_xmin:center_xmax], center_kernel[pady1:ker-pady2,padx1:ker-padx2])

                            size_xmin = np.clip(int((cx - w/2) * width / scale), 0, width // scale)
                            size_xmax = np.clip(int((cx + w/2) * width / scale) + 1, 0, width // scale)
                            size_ymin = np.clip(int((cy - h/2) * height / scale), 0, height // scale)
                            size_ymax = np.clip(int((cy + h/2) * height / scale) + 1, 0, height // scale)
                            size_mapx, size_mapy = np.meshgrid(np.arange(size_xmin, size_xmax) - cx * width / scale, np.arange(size_ymin, size_ymax) - cy * height / scale)
                            size_map = size_mapx ** 2 / max(w/2 * width / scale, 1) ** 2 + size_mapy ** 2 / max(h/2 * height / scale, 1) ** 2 < 1

                            center_x = int(cx / scale * width)
                            center_y = int(cy / scale * height)
                            offset_x = (cx * width % scale) / width * np.cos(angle)
                            offset_y = (cy * height % scale) / height * np.sin(angle + np.pi / 2)
                            offset_x += pad_x % scale
                            offset_y += pad_y % scale
                            offset_x = offset_x / scale - (np.arange(size_xmin, size_xmax) - center_x) * np.cos(angle)
                            offset_y = offset_y / scale - (np.arange(size_ymin, size_ymax) - center_y) * np.sin(angle + np.pi / 2)
                            offset_x = offset_x[np.newaxis,...] - np.linspace(-(size_ymax-size_ymin) * np.sin(angle) / 2, (size_ymax-size_ymin) * np.sin(angle) / 2, size_ymax-size_ymin)[...,np.newaxis]
                            offset_y = offset_y[...,np.newaxis] - np.linspace(-(size_xmax-size_xmin) * np.cos(angle + np.pi / 2) / 2, (size_xmax-size_xmin) * np.cos(angle + np.pi / 2) / 2, size_xmax-size_xmin)[np.newaxis,...]
                            offsetx[size_ymin:size_ymax, size_xmin:size_xmax] = np.where(size_map, offset_x, offsetx[size_ymin:size_ymax, size_xmin:size_xmax])
                            offsety[size_ymin:size_ymax, size_xmin:size_xmax] = np.where(size_map, offset_y, offsety[size_ymin:size_ymax, size_xmin:size_xmax])

                            fixw = w * np.abs(np.cos(angle)) + h * np.abs(np.sin(angle))
                            fixh = h * np.abs(np.cos(angle)) + w * np.abs(np.sin(angle))
                            fixw = np.log10(fixw * 10)
                            fixh = np.log10(fixh * 10)
                            xsizes[size_ymin:size_ymax, size_xmin:size_xmax] = np.where(size_map, fixw, xsizes[size_ymin:size_ymax, size_xmin:size_xmax])
                            ysizes[size_ymin:size_ymax, size_xmin:size_xmax] = np.where(size_map, fixh, ysizes[size_ymin:size_ymax, size_xmin:size_xmax])

                            id_char = self.glyph_id[key[1]]
                            ids[size_ymin:size_ymax, size_xmin:size_xmax] = np.where(size_map, id_char, ids[size_ymin:size_ymax, size_xmin:size_xmax])

                            w = max(int(item['width'] / 128 * text_size), 1)
                            h = max(int(item['rows'] / 128 * text_size), 1)
                            l = int(np.clip(base_line + vertBearingX, 0, width - w))
                            t = int(np.clip(line_start + vertBearingY, 0, height - h))
                            im = np.asarray(Image.fromarray(item['image']).resize((w,h)))
                            im = random_morphology(im)
                            images[t:t+h,l:l+w] = np.maximum(
                                    images[t:t+h,l:l+w], 
                                    im)

                        line_start += int(vertAdvance)

                    base_line -= line_space
                    if base_line - text_size / 2 < 0:
                        if block_no == 0:
                            sep_end = base_line + line_space
                        base_line = width - line_space + text_size // 2
                        block_no += 1
                    if block_no >= block_count:
                        break
                    line_start = int(max(0, 0 if block_count == 1 or block_no == 0 else line_break + break_space))
                    line_end = int(min(width, width if block_count == 1 or block_no == 1 else line_break - break_space))
                    temp_lineend = line_start
                    linebuf = []

            if all(t > 1 for t in text_count):
                l = max(1,line_break // scale)
                right = (width - line_space + text_size // 2) // scale
                left = sep_end // scale
                seps[l-1:l+2, left:right] = 1

            im = Image.fromarray(images).rotate(angle / np.pi * 180, resample=Image.BILINEAR, translate=(pad_x, pad_y))
            lines = lines.rotate(angle / np.pi * 180, resample=Image.BILINEAR, translate=(pad_x / scale, pad_y / scale))
            keymapim1 = Image.fromarray(keymap).rotate(angle / np.pi * 180, resample=Image.NEAREST, translate=(pad_x / scale, pad_y / scale))
            keymapim2 = Image.fromarray(keymap).rotate(angle / np.pi * 180, resample=Image.BILINEAR, translate=(pad_x / scale, pad_y / scale))
            keymapim = np.maximum(keymapim1, keymapim2)
            xsizeim = Image.fromarray(xsizes).rotate(angle / np.pi * 180, resample=Image.NEAREST, translate=(pad_x / scale, pad_y / scale))
            ysizeim = Image.fromarray(ysizes).rotate(angle / np.pi * 180, resample=Image.NEAREST, translate=(pad_x / scale, pad_y / scale))
            xoffsetim = Image.fromarray(offsetx).rotate(angle / np.pi * 180, resample=Image.BILINEAR, translate=(pad_x / scale, pad_y / scale))
            yoffsetim = Image.fromarray(offsety).rotate(angle / np.pi * 180, resample=Image.BILINEAR, translate=(pad_x / scale, pad_y / scale))
            sepim = Image.fromarray(seps).rotate(angle / np.pi * 180, resample=Image.BILINEAR, translate=(pad_x / scale, pad_y / scale))
            labels = np.stack([keymapim, xsizeim, ysizeim, xoffsetim, yoffsetim, np.asarray(lines) / 255., sepim], -1)
            idsim = Image.fromarray(ids).rotate(angle / np.pi * 180, resample=Image.NEAREST, translate=(pad_x / scale, pad_y / scale))
            return np.asarray(im) / 255., labels, np.asarray(idsim)

        if np.random.normal() < 0.5:
            images, labels, ids = tf.py_function(
                func=tate_images, 
                inp=[], 
                Tout=[tf.float32, tf.float32, tf.int32])
        else:
            images, labels, ids = tf.py_function(
                func=yoko_images, 
                inp=[], 
                Tout=[tf.float32, tf.float32, tf.int32])
        images = tf.ensure_shape(images, [height, width])
        labels = tf.ensure_shape(labels, [height // scale, width // scale, 7])
        ids = tf.ensure_shape(ids, [height // scale, width // scale])

        images = apply_random_filter(images)

        return self.sub_constructimage(images, labels, ids, False)

    def construct_alphatext(self, keys, fonts, t):
        max_count = 64
        angle_max = 15.0

        def null_images():
            images = np.zeros([height, width], dtype=np.float32)
            keymap = np.zeros([height // scale, width // scale], dtype=np.float32)
            xsizes = np.zeros([height // scale, width // scale], dtype=np.float32)
            ysizes = np.zeros([height // scale, width // scale], dtype=np.float32)
            offsetx = np.zeros([height // scale, width // scale], dtype=np.float32)
            offsety = np.zeros([height // scale, width // scale], dtype=np.float32)
            lines = np.zeros([height // scale, width // scale], dtype=np.float32)
            seps = np.zeros([height // scale, width // scale], dtype=np.float32)
            labels = np.stack([keymap, xsizes, ysizes, offsetx, offsety, lines, seps], -1)
            ids = np.zeros([height // scale, width // scale], dtype=np.int32)
            return images, labels, ids

        @tf.autograph.experimental.do_not_convert
        def load_random_line():
            images = np.zeros([height, width], dtype=np.float32)
            keymap = np.zeros([height // scale, width // scale], dtype=np.float32)
            xsizes = np.zeros([height // scale, width // scale], dtype=np.float32)
            ysizes = np.zeros([height // scale, width // scale], dtype=np.float32)
            offsetx = np.zeros([height // scale, width // scale], dtype=np.float32)
            offsety = np.zeros([height // scale, width // scale], dtype=np.float32)
            lines = np.zeros([height // scale, width // scale], dtype=np.float32)
            seps = np.zeros([height // scale, width // scale], dtype=np.float32)

            seps = Image.fromarray(seps)
            draw1 = ImageDraw.Draw(seps)
            images = Image.fromarray(images)
            draw2 = ImageDraw.Draw(images)
            
            linew = int(np.clip(np.random.uniform() * 20, scale, 20))
            x1 = np.random.normal() * width / 2 + width / 2
            y1 = np.random.normal() * height / 2 + height / 2
            x2 = np.random.normal() * width / 2 + width / 2
            y2 = np.random.normal() * height / 2 + height / 2

            draw1.line(((x1 // scale, y1 // scale), (x2 // scale, y2 // scale)), fill=255, width=linew//scale)
            draw2.line(((x1, y1), (x2, y2)), fill=255, width=linew)

            labels = np.stack([keymap, xsizes, ysizes, offsetx, offsety, lines, np.asarray(seps) / 255.], -1)
            ids = np.zeros([height // scale, width // scale], dtype=np.int32)
            return np.asarray(images) / 255., labels, ids

        @tf.autograph.experimental.do_not_convert
        def load_images_random():
            images = np.zeros([height, width], dtype=np.float32)
            keymap = np.zeros([height // scale, width // scale], dtype=np.float32)
            xsizes = np.zeros([height // scale, width // scale], dtype=np.float32)
            ysizes = np.zeros([height // scale, width // scale], dtype=np.float32)
            offsetx = np.zeros([height // scale, width // scale], dtype=np.float32)
            offsety = np.zeros([height // scale, width // scale], dtype=np.float32)
            lines = np.zeros([height // scale, width // scale], dtype=np.float32)
            seps = np.zeros([height // scale, width // scale], dtype=np.float32)
            ids = np.zeros([height // scale, width // scale], dtype=np.int32)

            angle = angle_max * np.random.normal() / 180 * np.pi
            angle = np.clip(angle, -np.pi, np.pi)
            pad_x = np.random.normal() * width / 20
            pad_y = np.random.normal() * height / 20

            probs = fonts
            selection = [key for key in random.choices(keys, k=np.random.randint(2,max_count), weights=probs)]
            i = 0
            boxprev = np.zeros([0, 4])
            if random.random() < 0.1:
                margin = 20
                line_c = random.randint(0,3)
                lw = random.randint(2, 10)
                if line_c == 0:
                    x = random.randrange(width // 2, width)
                    y = random.randrange(0, height - lw)
                    px = x // scale
                    py = y // scale
                    seps[py:py+lw//scale, :px] = 1
                    images[y:y+lw, :x] = 255
                    boxprev = np.concatenate([boxprev, [[0, (x + margin)/width, (y - margin)/height, (y+lw + margin)/height]]])
                elif line_c == 1:
                    x = random.randrange(0, width // 2)
                    y = random.randrange(0, height - lw)
                    px = x // scale
                    py = y // scale
                    seps[py:py+lw//scale, px:] = 1
                    images[y:y+lw, x:] = 255
                    boxprev = np.concatenate([boxprev, [[(x - margin)/width, 1, (y - margin)/height, (y+lw + margin)/height]]])
                elif line_c == 2:
                    y = random.randrange(height // 2, height)
                    x = random.randrange(0, width - lw)
                    px = x // scale
                    py = y // scale
                    seps[:py, px:px+lw//scale] = 1
                    images[:y, x:x+lw] = 255
                    boxprev = np.concatenate([boxprev, [[(x - margin)/width, (x+lw + margin)/width, 0, (y + margin)/height]]])
                elif line_c == 3:
                    y = random.randrange(0, height // 2)
                    x = random.randrange(0, width - lw)
                    px = x // scale
                    py = y // scale
                    seps[py:, px:px+lw//scale] = 1
                    images[y:, x:x+lw] = 255
                    boxprev = np.concatenate([boxprev, [[(x - margin)/width, (x+lw + margin)/width, (y - margin)/height, 1]]])

            if random.random() < 0.5:
                min_pixel = 24
                max_pixel = width
            else:
                min_pixel = 24
                max_pixel = width // 3
            for key in selection:
                item = self.img_cache[key]['horizontal']
                if item['width'] * item['rows'] == 0:
                    continue

                if random.random() < 0.5:
                    tile_size = random.randint(min_pixel, max_pixel)
                else:
                    tile_size = int(np.exp(random.uniform(np.log(min_pixel), np.log(max_pixel))))

                w = item['width'] / 128 * tile_size
                h = item['rows'] / 128 * tile_size
                aspects = np.clip(np.random.normal() * 0.1 + 1.0, 0.75, 1.25)
                if random.random() < 0.5:
                    aspects = 1.0 / aspects
                w *= aspects
                h /= aspects

                tile_left = random.randint(0, int(width - tile_size))
                tile_top = random.randint(0, int(height - tile_size))

                if tile_top + h >= height or tile_left + w >= width:
                    continue

                left = tile_left / width
                top = tile_top / height
                w = w / width
                h = h / height
                cx = left + w/2
                cy = top + h/2

                if np.random.uniform() < 0.1:
                    invert = True
                    x1 = cx - w/2 * 1.25
                    x2 = cx + w/2 * 1.25
                    y1 = cy - h/2 * 1.25
                    y2 = cy + h/2 * 1.25
                    inter_xmin = np.maximum(boxprev[:,0], x1)
                    inter_ymin = np.maximum(boxprev[:,2], y1)
                    inter_xmax = np.minimum(boxprev[:,1], x2)
                    inter_ymax = np.minimum(boxprev[:,3], y2)
                else:
                    invert = False
                    inter_xmin = np.maximum(boxprev[:,0], cx - w/2 * 1.1)
                    inter_ymin = np.maximum(boxprev[:,2], cy - h/2 * 1.1)
                    inter_xmax = np.minimum(boxprev[:,1], cx + w/2 * 1.1)
                    inter_ymax = np.minimum(boxprev[:,3], cy + h/2 * 1.1)
                inter_w = np.maximum(inter_xmax - inter_xmin, 0.)
                inter_h = np.maximum(inter_ymax - inter_ymin, 0.)
                inter_vol = inter_w * inter_h
                if np.any(inter_vol > 0):
                    continue

                kernel_size = max(self.min_ker, int(max(w, h) / (2 * scale) * width))
                std_x = min(self.max_std, max(self.min_ker, w / (2 * scale) * width) / 3)
                std_y = min(self.max_std, max(self.min_ker, h / (2 * scale) * height) / 3)
                center_kernel = gaussian_kernel(kernlen=kernel_size*2+1, xstd=std_x, ystd=std_y)

                center_xmin = int(cx / scale * width) - kernel_size
                center_xmax = int(cx / scale * width) + kernel_size + 1
                center_ymin = int(cy / scale * height) - kernel_size
                center_ymax = int(cy / scale * height) + kernel_size + 1
                padx1 = max(0, 0 - center_xmin)
                padx2 = max(0, center_xmax - width // scale)
                pady1 = max(0, 0 - center_ymin)
                pady2 = max(0, center_ymax - height // scale)
                center_xmin += padx1
                center_xmax -= padx2
                center_ymin += pady1
                center_ymax -= pady2
                ker = kernel_size * 2 + 1
                keymap[center_ymin:center_ymax, center_xmin:center_xmax] = np.maximum(keymap[center_ymin:center_ymax, center_xmin:center_xmax], center_kernel[pady1:ker-pady2,padx1:ker-padx2])

                size_xmin = np.clip(int((cx - w/2) * width / scale), 0, width // scale)
                size_xmax = np.clip(int((cx + w/2) * width / scale) + 1, 0, width // scale)
                size_ymin = np.clip(int((cy - h/2) * height / scale), 0, height // scale)
                size_ymax = np.clip(int((cy + h/2) * height / scale) + 1, 0, height // scale)
                size_mapx, size_mapy = np.meshgrid(np.arange(size_xmin, size_xmax) - cx * width / scale, np.arange(size_ymin, size_ymax) - cy * height / scale)
                size_map = size_mapx ** 2 / max(w/2 * width / scale, 1) ** 2 + size_mapy ** 2 / max(h/2 * height / scale, 1) ** 2 < 1

                center_x = int(cx / scale * width)
                center_y = int(cy / scale * height)
                offset_x = (cx * width % scale) / width * np.cos(angle)
                offset_y = (cy * height % scale) / height * np.sin(angle + np.pi / 2)
                offset_x += pad_x % scale
                offset_y += pad_y % scale
                offset_x = offset_x / scale - (np.arange(size_xmin, size_xmax) - center_x) * np.cos(angle)
                offset_y = offset_y / scale - (np.arange(size_ymin, size_ymax) - center_y) * np.sin(angle + np.pi / 2)
                offset_x = offset_x[np.newaxis,...] - np.linspace(-(size_ymax-size_ymin) * np.sin(angle) / 2, (size_ymax-size_ymin) * np.sin(angle) / 2, size_ymax-size_ymin)[...,np.newaxis]
                offset_y = offset_y[...,np.newaxis] - np.linspace(-(size_xmax-size_xmin) * np.cos(angle + np.pi / 2) / 2, (size_xmax-size_xmin) * np.cos(angle + np.pi / 2) / 2, size_xmax-size_xmin)[np.newaxis,...]
                offsetx[size_ymin:size_ymax, size_xmin:size_xmax] = np.where(size_map, offset_x, offsetx[size_ymin:size_ymax, size_xmin:size_xmax])
                offsety[size_ymin:size_ymax, size_xmin:size_xmax] = np.where(size_map, offset_y, offsety[size_ymin:size_ymax, size_xmin:size_xmax])

                fixw = w * np.abs(np.cos(angle)) + h * np.abs(np.sin(angle))
                fixh = h * np.abs(np.cos(angle)) + w * np.abs(np.sin(angle))
                fixw = np.log10(fixw * 10)
                fixh = np.log10(fixh * 10)
                xsizes[size_ymin:size_ymax, size_xmin:size_xmax] = np.where(size_map, fixw, xsizes[size_ymin:size_ymax, size_xmin:size_xmax])
                ysizes[size_ymin:size_ymax, size_xmin:size_xmax] = np.where(size_map, fixh, ysizes[size_ymin:size_ymax, size_xmin:size_xmax])

                id_char = self.glyph_id[key[1]]
                ids[size_ymin:size_ymax, size_xmin:size_xmax] = np.where(size_map, id_char, ids[size_ymin:size_ymax, size_xmin:size_xmax])

                if invert:
                    boxprev = np.concatenate([boxprev, [[x1, x2, y1, y2]]])
                else:
                    boxprev = np.concatenate([boxprev, [[cx - w/2, cx + w/2, cy - h/2, cy + h/2]]])

                w = max(int(item['width'] / 128 * tile_size * aspects), 1)
                h = max(int(item['rows'] / 128 * tile_size / aspects), 1)
                im = np.asarray(Image.fromarray(item['image']).resize((w,h)))
                im = random_morphology(im)
                images[tile_top:tile_top+h,tile_left:tile_left+w] = np.maximum(
                        images[tile_top:tile_top+h,tile_left:tile_left+w], 
                        im)
                if invert:
                    x1 = int(x1 * width)
                    x2 = int(x2 * width)
                    y1 = int(y1 * height)
                    y2 = int(y2 * height)
                    crop = images[y1:y2,x1:x2]
                    images[y1:y2,x1:x2] = 255 - crop
                i += 1

            im = Image.fromarray(images).rotate(angle / np.pi * 180, resample=Image.BILINEAR, translate=(pad_x, pad_y))
            keymapim1 = Image.fromarray(keymap).rotate(angle / np.pi * 180, resample=Image.NEAREST, translate=(pad_x / scale, pad_y / scale))
            keymapim2 = Image.fromarray(keymap).rotate(angle / np.pi * 180, resample=Image.BILINEAR, translate=(pad_x / scale, pad_y / scale))
            keymapim = np.maximum(keymapim1, keymapim2)
            xsizeim = Image.fromarray(xsizes).rotate(angle / np.pi * 180, resample=Image.NEAREST, translate=(pad_x / scale, pad_y / scale))
            ysizeim = Image.fromarray(ysizes).rotate(angle / np.pi * 180, resample=Image.NEAREST, translate=(pad_x / scale, pad_y / scale))
            xoffsetim = Image.fromarray(offsetx).rotate(angle / np.pi * 180, resample=Image.BILINEAR, translate=(pad_x / scale, pad_y / scale))
            yoffsetim = Image.fromarray(offsety).rotate(angle / np.pi * 180, resample=Image.BILINEAR, translate=(pad_x / scale, pad_y / scale))
            sepim = Image.fromarray(seps).rotate(angle / np.pi * 180, resample=Image.BILINEAR, translate=(pad_x / scale, pad_y / scale))
            labels = np.stack([keymapim, xsizeim, ysizeim, xoffsetim, yoffsetim, lines, sepim], -1)
            idsim = Image.fromarray(ids).rotate(angle / np.pi * 180, resample=Image.NEAREST, translate=(pad_x / scale, pad_y / scale))
            return np.asarray(im) / 255., labels, np.asarray(idsim)

        @tf.autograph.experimental.do_not_convert
        def load_images_fill():
            min_pixel = 24
            max_pixel = 200
            if random.random() < 0.5:
                tile_size = random.randint(min_pixel, max_pixel)
            else:
                tile_size = int(np.exp(random.uniform(np.log(min_pixel), np.log(max_pixel))))

            images = np.zeros([height, width], dtype=np.float32)
            keymap = np.zeros([height // scale, width // scale], dtype=np.float32)
            xsizes = np.zeros([height // scale, width // scale], dtype=np.float32)
            ysizes = np.zeros([height // scale, width // scale], dtype=np.float32)
            offsetx = np.zeros([height // scale, width // scale], dtype=np.float32)
            offsety = np.zeros([height // scale, width // scale], dtype=np.float32)
            lines = np.zeros([height // scale, width // scale], dtype=np.float32)
            lines = Image.fromarray(lines)
            draw = ImageDraw.Draw(lines)
            seps = np.zeros([height // scale, width // scale], dtype=np.float32)
            ids = np.zeros([height // scale, width // scale], dtype=np.int32)

            tile_left = 0
            tile_base = 0

            angle = angle_max * np.random.normal() / 180 * np.pi
            if np.random.rand() < 0.5:
                angle -= np.pi / 2
            angle = np.clip(angle, -np.pi, np.pi)
            pad_x = np.random.normal() * width / 20
            pad_y = np.random.normal() * height / 20

            aspects = np.clip(np.random.normal() * 0.1 + 1.0, 0.75, 1.25)
            if random.random() < 0.5:
                aspects = 1.0 / aspects

            select_font = random.choice(fonts)
            probs = [1. if key[0] == select_font else 0. for key in keys]
            selection = [key for key in random.choices(keys, k=np.random.randint(2,max_count), weights=probs)]

            boxprev = np.zeros([0, 4])
            if random.random() < 0.1:
                margin = 20
                line_c = random.randint(0,3)
                lw = random.randint(2, 10)
                if line_c == 0:
                    x = random.randrange(width // 2, width)
                    y = random.randrange(0, height - lw)
                    px = x // scale
                    py = y // scale
                    seps[py:py+lw//scale, :px] = 1
                    images[y:y+lw, :x] = 255
                    boxprev = np.concatenate([boxprev, [[0, (x + margin)/width, (y - margin)/height, (y+lw + margin)/height]]])
                elif line_c == 1:
                    x = random.randrange(0, width // 2)
                    y = random.randrange(0, height - lw)
                    px = x // scale
                    py = y // scale
                    seps[py:py+lw//scale, px:] = 1
                    images[y:y+lw, x:] = 255
                    boxprev = np.concatenate([boxprev, [[(x - margin)/width, 1, (y - margin)/height, (y+lw + margin)/height]]])
                elif line_c == 2:
                    y = random.randrange(height // 2, height)
                    x = random.randrange(0, width - lw)
                    px = x // scale
                    py = y // scale
                    seps[:py, px:px+lw//scale] = 1
                    images[:y, x:x+lw] = 255
                    boxprev = np.concatenate([boxprev, [[(x - margin)/width, (x+lw + margin)/width, 0, (y + margin)/height]]])
                elif line_c == 3:
                    y = random.randrange(0, height // 2)
                    x = random.randrange(0, width - lw)
                    px = x // scale
                    py = y // scale
                    seps[py:, px:px+lw//scale] = 1
                    images[y:, x:x+lw] = 255
                    boxprev = np.concatenate([boxprev, [[(x - margin)/width, (x+lw + margin)/width, (y - margin)/height, 1]]])
    
            cont_n = random.randint(1,21)
            cut_p = 10 ** (np.log10(0.5) / cont_n)

            i = 0
            replace = True
            prev = 0
            prev_center = None
            for key in selection:
                item = self.img_cache[key]['horizontal']
                if item['width'] * item['rows'] == 0:
                    continue

                w = item['width'] / 128 * tile_size
                h = item['rows'] / 128 * tile_size
                w *= aspects
                h /= aspects
                horiBearingX = item['horiBearingX'] / 128 * tile_size * aspects
                horiBearingY = item['horiBearingY'] / 128 * tile_size / aspects
                horiAdvance = item['horiAdvance'] / 128 * tile_size * aspects

                if replace:
                    invert = np.random.uniform() < 0.1
                    l = max(0,-int(horiBearingX))
                    tile_left = random.randint(l, 
                            int(width - tile_size)) if int(width - tile_size) > l else l
                    tile_base = random.randint(tile_size, 
                            int(height - tile_size)) if int(height - tile_size) > tile_size else tile_size

                if tile_base - horiBearingY < 0 or tile_base - horiBearingY + h >= height or tile_left + horiBearingX < 0 or tile_left + horiBearingX + w >= width:
                    replace = True
                    prev = 0
                    prev_center = None
                    continue

                l = (tile_left + horiBearingX) / width
                t = (tile_base - horiBearingY) / height
                w = w / width
                h = h / height
                cx = l + w / 2
                cy = t + h / 2

                if invert:
                    x1 = tile_left / width
                    x2 = (tile_left + horiAdvance) / width
                    y1 = cy - h/2 * 1.25
                    y2 = cy + h/2 * 1.25

                    if prev >= 0:
                        inter_xmin = np.maximum(boxprev[:,0], x1)
                        inter_ymin = np.maximum(boxprev[:,2], y1)
                        inter_xmax = np.minimum(boxprev[:,1], x2)
                        inter_ymax = np.minimum(boxprev[:,3], y2)
                    else:
                        # 連続してレンダリングする場合は、前の文字にめり込むことがあるので判定しない
                        inter_xmin = np.maximum(boxprev[:prev,0], x1)
                        inter_ymin = np.maximum(boxprev[:prev,2], y1)
                        inter_xmax = np.minimum(boxprev[:prev,1], x2)
                        inter_ymax = np.minimum(boxprev[:prev,3], y2)
                else:
                    if prev >= 0:
                        inter_xmin = np.maximum(boxprev[:,0], cx - w/2)
                        inter_ymin = np.maximum(boxprev[:,2], cy - h/2)
                        inter_xmax = np.minimum(boxprev[:,1], cx + w/2)
                        inter_ymax = np.minimum(boxprev[:,3], cy + h/2)
                    else:
                        # 連続してレンダリングする場合は、前の文字にめり込むことがあるので判定しない
                        inter_xmin = np.maximum(boxprev[:prev,0], cx - w/2)
                        inter_ymin = np.maximum(boxprev[:prev,2], cy - h/2)
                        inter_xmax = np.minimum(boxprev[:prev,1], cx + w/2)
                        inter_ymax = np.minimum(boxprev[:prev,3], cy + h/2)
                inter_w = np.maximum(inter_xmax - inter_xmin, 0.)
                inter_h = np.maximum(inter_ymax - inter_ymin, 0.)
                inter_vol = inter_w * inter_h
                if np.any(inter_vol > 0):
                    replace = True
                    prev = 0
                    prev_center = None
                    continue

                if prev < 0:
                    draw.line((prev_center, (cx * width // scale, tile_base // scale)), fill=255, width=3)
                prev_center = (cx * width // scale, tile_base // scale)

                kernel_size = max(self.min_ker, int(max(w, h) / (2 * scale) * width))
                std_x = min(self.max_std, max(self.min_ker, w / (2 * scale) * width) / 3)
                std_y = min(self.max_std, max(self.min_ker, h / (2 * scale) * height) / 3)
                center_kernel = gaussian_kernel(kernlen=kernel_size*2+1, xstd=std_x, ystd=std_y)

                center_xmin = int(cx / scale * width) - kernel_size
                center_xmax = int(cx / scale * width) + kernel_size + 1
                center_ymin = int(cy / scale * height) - kernel_size
                center_ymax = int(cy / scale * height) + kernel_size + 1
                padx1 = max(0, 0 - center_xmin)
                padx2 = max(0, center_xmax - width // scale)
                pady1 = max(0, 0 - center_ymin)
                pady2 = max(0, center_ymax - height // scale)
                center_xmin += padx1
                center_xmax -= padx2
                center_ymin += pady1
                center_ymax -= pady2
                ker = kernel_size * 2 + 1
                keymap[center_ymin:center_ymax, center_xmin:center_xmax] = np.maximum(keymap[center_ymin:center_ymax, center_xmin:center_xmax], center_kernel[pady1:ker-pady2,padx1:ker-padx2])

                size_xmin = np.clip(int((cx - w/2) * width / scale), 0, width // scale)
                size_xmax = np.clip(int((cx + w/2) * width / scale) + 1, 0, width // scale)
                size_ymin = np.clip(int((cy - h/2) * height / scale), 0, height // scale)
                size_ymax = np.clip(int((cy + h/2) * height / scale) + 1, 0, height // scale)
                size_mapx, size_mapy = np.meshgrid(np.arange(size_xmin, size_xmax) - cx * width / scale, np.arange(size_ymin, size_ymax) - cy * height / scale)
                size_map = size_mapx ** 2 / max(w/2 * width / scale, 1) ** 2 + size_mapy ** 2 / max(h/2 * height / scale, 1) ** 2 < 1

                center_x = int(cx / scale * width)
                center_y = int(cy / scale * height)
                if angle < -np.pi / 4:
                    offset_y = (cx * width % scale) / width * np.cos(-(angle + np.pi / 2))
                    offset_x = -(cy * height % scale) / height * np.sin(-(angle + np.pi / 2) + np.pi / 2)
                    offset_x += pad_x % scale
                    offset_y += pad_y % scale
                    offset_x = offset_x / scale + (np.arange(size_ymin, size_ymax) - center_y) * np.sin(-(angle + np.pi / 2) + np.pi / 2)
                    offset_y = offset_y / scale - (np.arange(size_xmin, size_xmax) - center_x) * np.cos(-(angle + np.pi / 2))
                    offset_y = offset_y[np.newaxis,...] + np.linspace(-(size_ymax-size_ymin) * np.sin(-(angle + np.pi / 2)) / 2, (size_ymax-size_ymin) * np.sin(-(angle + np.pi / 2)) / 2, size_ymax-size_ymin)[...,np.newaxis]
                    offset_x = offset_x[...,np.newaxis] - np.linspace(-(size_xmax-size_xmin) * np.cos(-(angle + np.pi/ 2) + np.pi / 2) / 2, (size_xmax-size_xmin) * np.cos(-(angle + np.pi/ 2) + np.pi / 2) / 2, size_xmax-size_xmin)[np.newaxis,...]
                else:
                    offset_x = (cx * width % scale) / width * np.sin(angle)
                    offset_y = (cy * height % scale) / height * np.sin(angle + np.pi / 2)
                    offset_x += pad_x % scale
                    offset_y += pad_y % scale
                    offset_x = offset_x / scale - (np.arange(size_xmin, size_xmax) - center_x) * np.cos(angle)
                    offset_y = offset_y / scale - (np.arange(size_ymin, size_ymax) - center_y) * np.sin(angle + np.pi / 2)
                    offset_x = offset_x[np.newaxis,...] - np.linspace(-(size_ymax-size_ymin) * np.sin(angle) / 2, (size_ymax-size_ymin) * np.sin(angle) / 2, size_ymax-size_ymin)[...,np.newaxis]
                    offset_y = offset_y[...,np.newaxis] - np.linspace(-(size_xmax-size_xmin) * np.cos(angle + np.pi / 2) / 2, (size_xmax-size_xmin) * np.cos(angle + np.pi / 2) / 2, size_xmax-size_xmin)[np.newaxis,...]
                offsetx[size_ymin:size_ymax, size_xmin:size_xmax] = np.where(size_map, offset_x, offsetx[size_ymin:size_ymax, size_xmin:size_xmax])
                offsety[size_ymin:size_ymax, size_xmin:size_xmax] = np.where(size_map, offset_y, offsety[size_ymin:size_ymax, size_xmin:size_xmax])

                fixw = w * np.abs(np.cos(angle)) + h * np.abs(np.sin(angle))
                fixh = h * np.abs(np.cos(angle)) + w * np.abs(np.sin(angle))
                fixw = np.log10(fixw * 10)
                fixh = np.log10(fixh * 10)
                xsizes[size_ymin:size_ymax, size_xmin:size_xmax] = np.where(size_map, fixw, xsizes[size_ymin:size_ymax, size_xmin:size_xmax])
                ysizes[size_ymin:size_ymax, size_xmin:size_xmax] = np.where(size_map, fixh, ysizes[size_ymin:size_ymax, size_xmin:size_xmax])

                id_char = self.glyph_id[key[1]]
                ids[size_ymin:size_ymax, size_xmin:size_xmax] = np.where(size_map, id_char, ids[size_ymin:size_ymax, size_xmin:size_xmax])

                if invert:
                    boxprev = np.concatenate([boxprev, [[x1, x2, y1, y2]]])
                else:
                    boxprev = np.concatenate([boxprev, [[cx - w/2, cx + w/2, cy - h/2, cy + h/2]]])

                w = max(int(item['width'] / 128 * tile_size * aspects), 1)
                h = max(int(item['rows'] / 128 * tile_size / aspects), 1)
                tile_top = int(tile_base - horiBearingY) 
                l = int(tile_left + horiBearingX)
                im = np.asarray(Image.fromarray(item['image']).resize((w,h)))
                im = random_morphology(im)
                if invert:
                    im_inv = 255 - im
                    x1 = int(x1 * width)
                    x2 = int(x2 * width)
                    y1 = int(y1 * height)
                    y2 = int(y2 * height)
                    images[y1:y2,x1:x2] = 255
                    images[tile_top:tile_top+h,l:l+w] = np.minimum(
                            images[tile_top:tile_top+h,l:l+w], 
                            im_inv)
                else:
                    images[tile_top:tile_top+h,l:l+w] = np.maximum(
                            images[tile_top:tile_top+h,l:l+w], 
                            im)

                i += 1
                if random.random() > cut_p:
                    replace = True
                    prev_center = None
                    prev = 0
                else:
                    tile_left += int(horiAdvance)
                    replace = False
                    if tile_left >= width - tile_size:
                        replace = True
                        prev_center = None
                        prev = 0
                    else:
                        prev -= 1

            im = Image.fromarray(images).rotate(angle / np.pi * 180, resample=Image.BILINEAR, translate=(pad_x, pad_y))
            lines = lines.rotate(angle / np.pi * 180, resample=Image.BILINEAR, translate=(pad_x / scale, pad_y / scale))
            keymapim1 = Image.fromarray(keymap).rotate(angle / np.pi * 180, resample=Image.NEAREST, translate=(pad_x / scale, pad_y / scale))
            keymapim2 = Image.fromarray(keymap).rotate(angle / np.pi * 180, resample=Image.BILINEAR, translate=(pad_x / scale, pad_y / scale))
            keymapim = np.maximum(keymapim1, keymapim2)
            xsizeim = Image.fromarray(xsizes).rotate(angle / np.pi * 180, resample=Image.NEAREST, translate=(pad_x / scale, pad_y / scale))
            ysizeim = Image.fromarray(ysizes).rotate(angle / np.pi * 180, resample=Image.NEAREST, translate=(pad_x / scale, pad_y / scale))
            xoffsetim = Image.fromarray(offsetx).rotate(angle / np.pi * 180, resample=Image.BILINEAR, translate=(pad_x / scale, pad_y / scale))
            yoffsetim = Image.fromarray(offsety).rotate(angle / np.pi * 180, resample=Image.BILINEAR, translate=(pad_x / scale, pad_y / scale))
            sepim = Image.fromarray(seps).rotate(angle / np.pi * 180, resample=Image.BILINEAR, translate=(pad_x / scale, pad_y / scale))
            labels = np.stack([keymapim, xsizeim, ysizeim, xoffsetim, yoffsetim, np.asarray(lines) / 255., sepim], -1)
            idsim = Image.fromarray(ids).rotate(angle / np.pi * 180, resample=Image.NEAREST, translate=(pad_x / scale, pad_y / scale))
            return np.asarray(im) / 255., labels, np.asarray(idsim)

        @tf.autograph.experimental.do_not_convert
        def load_images_fill_horizontal():
            min_pixel = 20
            max_pixel = 200
            if random.random() < 0.5:
                tile_size = random.randint(min_pixel, max_pixel)
            else:
                tile_size = int(np.exp(random.uniform(np.log(min_pixel), np.log(max_pixel))))

            images = np.zeros([height, width], dtype=np.float32)
            keymap = np.zeros([height // scale, width // scale], dtype=np.float32)
            xsizes = np.zeros([height // scale, width // scale], dtype=np.float32)
            ysizes = np.zeros([height // scale, width // scale], dtype=np.float32)
            offsetx = np.zeros([height // scale, width // scale], dtype=np.float32)
            offsety = np.zeros([height // scale, width // scale], dtype=np.float32)
            lines = np.zeros([height // scale, width // scale], dtype=np.float32)
            lines = Image.fromarray(lines)
            draw = ImageDraw.Draw(lines)
            seps = np.zeros([height // scale, width // scale], dtype=np.float32)
            ids = np.zeros([height // scale, width // scale], dtype=np.int32)

            tile_left = 0
            tile_base = 0

            angle = angle_max * np.random.normal() / 180 * np.pi
            angle = np.clip(angle, -np.pi, np.pi)
            pad_x = np.random.normal() * width / 20
            pad_y = np.random.normal() * height / 20

            aspects = np.clip(np.random.normal() * 0.1 + 1.0, 0.75, 1.25)
            if random.random() < 0.5:
                aspects = 1.0 / aspects

            select_font = random.choices(fonts[0], k=1, weights=fonts[1])[0]
            probs = [1. if key[0] == select_font else 0. for key in keys]
            selection = [key for key in random.choices(keys, k=np.random.randint(2,max_count), weights=probs)]

            boxprev = np.zeros([0, 4])
            if random.random() < 0.1:
                margin = 20
                line_c = random.randint(0,3)
                lw = random.randint(2, 10)
                if line_c == 0:
                    x = random.randrange(width // 2, width)
                    y = random.randrange(0, height - lw)
                    px = x // scale
                    py = y // scale
                    seps[py:py+lw//scale, :px] = 1
                    images[y:y+lw, :x] = 255
                    boxprev = np.concatenate([boxprev, [[0, (x + margin)/width, (y - margin)/height, (y+lw + margin)/height]]])
                elif line_c == 1:
                    x = random.randrange(0, width // 2)
                    y = random.randrange(0, height - lw)
                    px = x // scale
                    py = y // scale
                    seps[py:py+lw//scale, px:] = 1
                    images[y:y+lw, x:] = 255
                    boxprev = np.concatenate([boxprev, [[(x - margin)/width, 1, (y - margin)/height, (y+lw + margin)/height]]])
                elif line_c == 2:
                    y = random.randrange(height // 2, height)
                    x = random.randrange(0, width - lw)
                    px = x // scale
                    py = y // scale
                    seps[:py, px:px+lw//scale] = 1
                    images[:y, x:x+lw] = 255
                    boxprev = np.concatenate([boxprev, [[(x - margin)/width, (x+lw + margin)/width, 0, (y + margin)/height]]])
                elif line_c == 3:
                    y = random.randrange(0, height // 2)
                    x = random.randrange(0, width - lw)
                    px = x // scale
                    py = y // scale
                    seps[py:, px:px+lw//scale] = 1
                    images[y:, x:x+lw] = 255
                    boxprev = np.concatenate([boxprev, [[(x - margin)/width, (x+lw + margin)/width, (y - margin)/height, 1]]])

            i = 0
            replace = True
            prev = 0
            prev_center = None
            org_tile_size = tile_size
            for key in selection:
                item = self.img_cache[key]['horizontal']
                if item['width'] * item['rows'] == 0:
                    continue

                if replace:
                    if random.random() < 0.25:
                        tile_size = int(max(min_pixel, org_tile_size / 2.2))
                    else:
                        tile_size = org_tile_size

                w = item['width'] / 128 * tile_size
                h = item['rows'] / 128 * tile_size
                w *= aspects
                h /= aspects
                horiBearingX = item['horiBearingX'] / 128 * tile_size * aspects
                horiBearingY = item['horiBearingY'] / 128 * tile_size / aspects
                horiAdvance = item['horiAdvance'] / 128 * tile_size * aspects

                if replace:
                    l = max(0,-int(horiBearingX))
                    tile_left = random.randint(l, 
                            int(width - tile_size)) if int(width - tile_size) > l else l
                    tile_base = random.randint(tile_size, 
                            int(height - tile_size)) if int(height - tile_size) > tile_size else tile_size

                if tile_base - horiBearingY < 0 or tile_base - horiBearingY + h >= height or tile_left + horiBearingX < 0 or tile_left + horiBearingX + w >= width:
                    replace = True
                    prev_center = None
                    prev = 0
                    continue

                l = (tile_left + horiBearingX) / width
                t = (tile_base - horiBearingY) / height
                w = w / width
                h = h / height
                cx = l + w / 2
                cy = t + h / 2

                if prev >= 0:
                    inter_xmin = np.maximum(boxprev[:,0], cx - w/2)
                    inter_ymin = np.maximum(boxprev[:,2], cy - h/2)
                    inter_xmax = np.minimum(boxprev[:,1], cx + w/2)
                    inter_ymax = np.minimum(boxprev[:,3], cy + h/2)
                else:
                    # 連続してレンダリングする場合は、前の文字にめり込むことがあるので判定しない
                    inter_xmin = np.maximum(boxprev[:prev,0], cx - w/2)
                    inter_ymin = np.maximum(boxprev[:prev,2], cy - h/2)
                    inter_xmax = np.minimum(boxprev[:prev,1], cx + w/2)
                    inter_ymax = np.minimum(boxprev[:prev,3], cy + h/2)
                inter_w = np.maximum(inter_xmax - inter_xmin, 0.)
                inter_h = np.maximum(inter_ymax - inter_ymin, 0.)
                inter_vol = inter_w * inter_h
                if np.any(inter_vol > 0):
                    replace = True
                    prev_center = None
                    prev = 0
                    continue

                if prev < 0:
                    draw.line((prev_center, (cx * width // scale, tile_base // scale)), fill=255, width=3)
                prev_center = (cx * width // scale, tile_base // scale)

                kernel_size = max(self.min_ker, int(max(w, h) / (2 * scale) * width))
                std_x = min(self.max_std, max(self.min_ker, w / (2 * scale) * width) / 3)
                std_y = min(self.max_std, max(self.min_ker, h / (2 * scale) * height) / 3)
                center_kernel = gaussian_kernel(kernlen=kernel_size*2+1, xstd=std_x, ystd=std_y)

                center_xmin = int(cx / scale * width) - kernel_size
                center_xmax = int(cx / scale * width) + kernel_size + 1
                center_ymin = int(cy / scale * height) - kernel_size
                center_ymax = int(cy / scale * height) + kernel_size + 1
                padx1 = max(0, 0 - center_xmin)
                padx2 = max(0, center_xmax - width // scale)
                pady1 = max(0, 0 - center_ymin)
                pady2 = max(0, center_ymax - height // scale)
                center_xmin += padx1
                center_xmax -= padx2
                center_ymin += pady1
                center_ymax -= pady2
                ker = kernel_size * 2 + 1
                keymap[center_ymin:center_ymax, center_xmin:center_xmax] = np.maximum(keymap[center_ymin:center_ymax, center_xmin:center_xmax], center_kernel[pady1:ker-pady2,padx1:ker-padx2])

                size_xmin = np.clip(int((cx - w/2) * width / scale), 0, width // scale)
                size_xmax = np.clip(int((cx + w/2) * width / scale) + 1, 0, width // scale)
                size_ymin = np.clip(int((cy - h/2) * height / scale), 0, height // scale)
                size_ymax = np.clip(int((cy + h/2) * height / scale) + 1, 0, height // scale)
                size_mapx, size_mapy = np.meshgrid(np.arange(size_xmin, size_xmax) - cx * width / scale, np.arange(size_ymin, size_ymax) - cy * height / scale)
                size_map = size_mapx ** 2 / max(w/2 * width / scale, 1) ** 2 + size_mapy ** 2 / max(h/2 * height / scale, 1) ** 2 < 1

                center_x = int(cx / scale * width)
                center_y = int(cy / scale * height)
                offset_x = (cx * width % scale) / width * np.cos(angle)
                offset_y = (cy * height % scale) / height * np.sin(angle + np.pi / 2)
                offset_x += pad_x % scale
                offset_y += pad_y % scale
                offset_x = offset_x / scale - (np.arange(size_xmin, size_xmax) - center_x) * np.cos(angle)
                offset_y = offset_y / scale - (np.arange(size_ymin, size_ymax) - center_y) * np.sin(angle + np.pi / 2)
                offset_x = offset_x[np.newaxis,...] - np.linspace(-(size_ymax-size_ymin) * np.sin(angle) / 2, (size_ymax-size_ymin) * np.sin(angle) / 2, size_ymax-size_ymin)[...,np.newaxis]
                offset_y = offset_y[...,np.newaxis] - np.linspace(-(size_xmax-size_xmin) * np.cos(angle + np.pi / 2) / 2, (size_xmax-size_xmin) * np.cos(angle + np.pi / 2) / 2, size_xmax-size_xmin)[np.newaxis,...]
                offsetx[size_ymin:size_ymax, size_xmin:size_xmax] = np.where(size_map, offset_x, offsetx[size_ymin:size_ymax, size_xmin:size_xmax])
                offsety[size_ymin:size_ymax, size_xmin:size_xmax] = np.where(size_map, offset_y, offsety[size_ymin:size_ymax, size_xmin:size_xmax])

                fixw = w * np.abs(np.cos(angle)) + h * np.abs(np.sin(angle))
                fixh = h * np.abs(np.cos(angle)) + w * np.abs(np.sin(angle))
                fixw = np.log10(fixw * 10)
                fixh = np.log10(fixh * 10)
                xsizes[size_ymin:size_ymax, size_xmin:size_xmax] = np.where(size_map, fixw, xsizes[size_ymin:size_ymax, size_xmin:size_xmax])
                ysizes[size_ymin:size_ymax, size_xmin:size_xmax] = np.where(size_map, fixh, ysizes[size_ymin:size_ymax, size_xmin:size_xmax])

                id_char = self.glyph_id[key[1]]
                ids[size_ymin:size_ymax, size_xmin:size_xmax] = np.where(size_map, id_char, ids[size_ymin:size_ymax, size_xmin:size_xmax])

                boxprev = np.concatenate([boxprev, [[cx - w/2, cx + w/2, cy - h/2, cy + h/2]]])

                w = max(int(item['width'] / 128 * tile_size * aspects), 1)
                h = max(int(item['rows'] / 128 * tile_size / aspects), 1)
                tile_top = int(tile_base - horiBearingY) 
                l = int(tile_left + horiBearingX)
                im = np.asarray(Image.fromarray(item['image']).resize((w,h)))
                im = random_morphology(im)
                images[tile_top:tile_top+h,l:l+w] = np.maximum(
                        images[tile_top:tile_top+h,l:l+w], 
                        im)
                i += 1
                tile_left += int(horiAdvance)
                replace = False
                if tile_left >= width - tile_size:
                    replace = True
                    prev_center = None
                    prev = 0
                else:
                    prev -= 1

            im = Image.fromarray(images).rotate(angle / np.pi * 180, resample=Image.BILINEAR, translate=(pad_x, pad_y))
            lines = lines.rotate(angle / np.pi * 180, resample=Image.BILINEAR, translate=(pad_x / scale, pad_y / scale))
            keymapim1 = Image.fromarray(keymap).rotate(angle / np.pi * 180, resample=Image.NEAREST, translate=(pad_x / scale, pad_y / scale))
            keymapim2 = Image.fromarray(keymap).rotate(angle / np.pi * 180, resample=Image.BILINEAR, translate=(pad_x / scale, pad_y / scale))
            keymapim = np.maximum(keymapim1, keymapim2)
            xsizeim = Image.fromarray(xsizes).rotate(angle / np.pi * 180, resample=Image.NEAREST, translate=(pad_x / scale, pad_y / scale))
            ysizeim = Image.fromarray(ysizes).rotate(angle / np.pi * 180, resample=Image.NEAREST, translate=(pad_x / scale, pad_y / scale))
            xoffsetim = Image.fromarray(offsetx).rotate(angle / np.pi * 180, resample=Image.BILINEAR, translate=(pad_x / scale, pad_y / scale))
            yoffsetim = Image.fromarray(offsety).rotate(angle / np.pi * 180, resample=Image.BILINEAR, translate=(pad_x / scale, pad_y / scale))
            sepim = Image.fromarray(seps).rotate(angle / np.pi * 180, resample=Image.BILINEAR, translate=(pad_x / scale, pad_y / scale))
            labels = np.stack([keymapim, xsizeim, ysizeim, xoffsetim, yoffsetim, np.asarray(lines) / 255., sepim], -1)
            idsim = Image.fromarray(ids).rotate(angle / np.pi * 180, resample=Image.NEAREST, translate=(pad_x / scale, pad_y / scale))
            return np.asarray(im) / 255., labels, np.asarray(idsim)

        @tf.autograph.experimental.do_not_convert
        def load_images_fill_vertical(small=0):
            min_pixel = 20
            max_pixel = 200
            if random.random() < 0.5:
                tile_size = random.randint(min_pixel, max_pixel)
            else:
                tile_size = int(np.exp(random.uniform(np.log(min_pixel), np.log(max_pixel))))

            images = np.zeros([height, width], dtype=np.float32)
            keymap = np.zeros([height // scale, width // scale], dtype=np.float32)
            xsizes = np.zeros([height // scale, width // scale], dtype=np.float32)
            ysizes = np.zeros([height // scale, width // scale], dtype=np.float32)
            offsetx = np.zeros([height // scale, width // scale], dtype=np.float32)
            offsety = np.zeros([height // scale, width // scale], dtype=np.float32)
            lines = np.zeros([height // scale, width // scale], dtype=np.float32)
            lines = Image.fromarray(lines)
            draw = ImageDraw.Draw(lines)
            seps = np.zeros([height // scale, width // scale], dtype=np.float32)
            ids = np.zeros([height // scale, width // scale], dtype=np.int32)

            tile_base = 0

            angle = angle_max * np.random.normal() / 180 * np.pi
            angle = np.clip(angle, -np.pi, np.pi)
            pad_x = np.random.normal() * width / 20
            pad_y = np.random.normal() * height / 20

            aspects = np.clip(np.random.normal() * 0.1 + 1.0, 0.75, 1.25)
            if random.random() < 0.5:
                aspects = 1.0 / aspects

            select_font = random.choices(fonts[0], k=1, weights=fonts[1])[0]
            probs = [1. if key[0] == select_font else 0. for key in keys]
            selection = [key for key in random.choices(keys, k=np.random.randint(2,max_count), weights=probs)]

            boxprev = np.zeros([0, 4])
            if random.random() < 0.1:
                margin = 20
                line_c = random.randint(0,3)
                lw = random.randint(2, 10)
                if line_c == 0:
                    x = random.randrange(width // 2, width)
                    y = random.randrange(0, height - lw)
                    px = x // scale
                    py = y // scale
                    seps[py:py+lw//scale, :px] = 1
                    images[y:y+lw, :x] = 255
                    boxprev = np.concatenate([boxprev, [[0, (x + margin)/width, (y - margin)/height, (y+lw + margin)/height]]])
                elif line_c == 1:
                    x = random.randrange(0, width // 2)
                    y = random.randrange(0, height - lw)
                    px = x // scale
                    py = y // scale
                    seps[py:py+lw//scale, px:] = 1
                    images[y:y+lw, x:] = 255
                    boxprev = np.concatenate([boxprev, [[(x - margin)/width, 1, (y - margin)/height, (y+lw + margin)/height]]])
                elif line_c == 2:
                    y = random.randrange(height // 2, height)
                    x = random.randrange(0, width - lw)
                    px = x // scale
                    py = y // scale
                    seps[:py, px:px+lw//scale] = 1
                    images[:y, x:x+lw] = 255
                    boxprev = np.concatenate([boxprev, [[(x - margin)/width, (x+lw + margin)/width, 0, (y + margin)/height]]])
                elif line_c == 3:
                    y = random.randrange(0, height // 2)
                    x = random.randrange(0, width - lw)
                    px = x // scale
                    py = y // scale
                    seps[py:, px:px+lw//scale] = 1
                    images[y:, x:x+lw] = 255
                    boxprev = np.concatenate([boxprev, [[(x - margin)/width, (x+lw + margin)/width, (y - margin)/height, 1]]])

            i = 0
            replace = True
            prev_center = None
            prev = 0
            if small == 0:
                adv_ratio = 1
            else:
                adv_ratio = random.random() * 0.25 + 0.75
            org_tile_size = tile_size
            for key in selection:
                item = self.img_cache[key]['vertical']
                if item['width'] * item['rows'] == 0:
                    continue

                if replace:
                    if random.random() < 0.25:
                        tile_size = int(max(min_pixel, org_tile_size / 2.2))
                    else:
                        tile_size = org_tile_size

                w = item['width'] / 128 * tile_size
                h = item['rows'] / 128 * tile_size
                w *= aspects
                h /= aspects
                vertBearingX = item['vertBearingX'] / 128 * tile_size * aspects
                vertBearingY = item['vertBearingY'] / 128 * tile_size / aspects
                vertAdvance = item['vertAdvance'] / 128 * tile_size / aspects

                if replace:
                    l = max(0,-int(vertBearingX))
                    tile_base = random.randint(l, 
                            int(width - tile_size)) if int(width - tile_size) > l else l
                    tile_top = random.randint(tile_size, 
                            int(height - tile_size)) if int(height - tile_size) > tile_size else 0

                if tile_base + vertBearingX < 0 or tile_base + vertBearingX + w >= width or tile_top + vertBearingY < 0 or tile_top + vertBearingY + h >= height:
                    replace = True
                    prev_center = None
                    prev = 0
                    continue

                l = (tile_base + vertBearingX) / width
                t = (tile_top + vertBearingY) / height
                w = w / width
                h = h / height
                cx = l + w / 2
                cy = t + h / 2

                if prev >= 0:
                    inter_xmin = np.maximum(boxprev[:,0], cx - w/2)
                    inter_ymin = np.maximum(boxprev[:,2], cy - h/2)
                    inter_xmax = np.minimum(boxprev[:,1], cx + w/2)
                    inter_ymax = np.minimum(boxprev[:,3], cy + h/2)
                else:
                    # 連続してレンダリングする場合は、前の文字にめり込むことがあるので判定しない
                    inter_xmin = np.maximum(boxprev[:prev,0], cx - w/2)
                    inter_ymin = np.maximum(boxprev[:prev,2], cy - h/2)
                    inter_xmax = np.minimum(boxprev[:prev,1], cx + w/2)
                    inter_ymax = np.minimum(boxprev[:prev,3], cy + h/2)
                inter_w = np.maximum(inter_xmax - inter_xmin, 0.)
                inter_h = np.maximum(inter_ymax - inter_ymin, 0.)
                inter_vol = inter_w * inter_h
                if np.any(inter_vol > 0):
                    replace = True
                    prev_center = None
                    prev = 0
                    continue

                if prev < 0:
                    draw.line((prev_center, (tile_base // scale, cy * height // scale)), fill=255, width=3)
                prev_center = (tile_base // scale, cy * height // scale)

                kernel_size = max(self.min_ker, int(max(w, h) / (2 * scale) * width))
                std_x = min(self.max_std, max(self.min_ker, w / (2 * scale) * width) / 3)
                std_y = min(self.max_std, max(self.min_ker, h / (2 * scale) * height) / 3)
                center_kernel = gaussian_kernel(kernlen=kernel_size*2+1, xstd=std_x, ystd=std_y)

                center_xmin = int(cx / scale * width) - kernel_size
                center_xmax = int(cx / scale * width) + kernel_size + 1
                center_ymin = int(cy / scale * height) - kernel_size
                center_ymax = int(cy / scale * height) + kernel_size + 1
                padx1 = max(0, 0 - center_xmin)
                padx2 = max(0, center_xmax - width // scale)
                pady1 = max(0, 0 - center_ymin)
                pady2 = max(0, center_ymax - height // scale)
                center_xmin += padx1
                center_xmax -= padx2
                center_ymin += pady1
                center_ymax -= pady2
                ker = kernel_size * 2 + 1
                keymap[center_ymin:center_ymax, center_xmin:center_xmax] = np.maximum(keymap[center_ymin:center_ymax, center_xmin:center_xmax], center_kernel[pady1:ker-pady2,padx1:ker-padx2])

                size_xmin = np.clip(int((cx - w/2) * width / scale), 0, width // scale)
                size_xmax = np.clip(int((cx + w/2) * width / scale) + 1, 0, width // scale)
                size_ymin = np.clip(int((cy - h/2) * height / scale), 0, height // scale)
                size_ymax = np.clip(int((cy + h/2) * height / scale) + 1, 0, height // scale)
                size_mapx, size_mapy = np.meshgrid(np.arange(size_xmin, size_xmax) - cx * width / scale, np.arange(size_ymin, size_ymax) - cy * height / scale)
                size_map = size_mapx ** 2 / max(w/2 * width / scale, 1) ** 2 + size_mapy ** 2 / max(h/2 * height / scale, 1) ** 2 < 1

                center_x = int(cx / scale * width)
                center_y = int(cy / scale * height)
                offset_x = (cx * width % scale) / width * np.cos(angle)
                offset_y = (cy * height % scale) / height * np.sin(angle + np.pi / 2)
                offset_x += pad_x % scale
                offset_y += pad_y % scale
                offset_x = offset_x / scale - (np.arange(size_xmin, size_xmax) - center_x) * np.cos(angle)
                offset_y = offset_y / scale - (np.arange(size_ymin, size_ymax) - center_y) * np.sin(angle + np.pi / 2)
                offset_x = offset_x[np.newaxis,...] - np.linspace(-(size_ymax-size_ymin) * np.sin(angle) / 2, (size_ymax-size_ymin) * np.sin(angle) / 2, size_ymax-size_ymin)[...,np.newaxis]
                offset_y = offset_y[...,np.newaxis] - np.linspace(-(size_xmax-size_xmin) * np.cos(angle + np.pi / 2) / 2, (size_xmax-size_xmin) * np.cos(angle + np.pi / 2) / 2, size_xmax-size_xmin)[np.newaxis,...]
                offsetx[size_ymin:size_ymax, size_xmin:size_xmax] = np.where(size_map, offset_x, offsetx[size_ymin:size_ymax, size_xmin:size_xmax])
                offsety[size_ymin:size_ymax, size_xmin:size_xmax] = np.where(size_map, offset_y, offsety[size_ymin:size_ymax, size_xmin:size_xmax])

                fixw = w * np.abs(np.cos(angle)) + h * np.abs(np.sin(angle))
                fixh = h * np.abs(np.cos(angle)) + w * np.abs(np.sin(angle))
                fixw = np.log10(fixw * 10)
                fixh = np.log10(fixh * 10)
                xsizes[size_ymin:size_ymax, size_xmin:size_xmax] = np.where(size_map, fixw, xsizes[size_ymin:size_ymax, size_xmin:size_xmax])
                ysizes[size_ymin:size_ymax, size_xmin:size_xmax] = np.where(size_map, fixh, ysizes[size_ymin:size_ymax, size_xmin:size_xmax])

                id_char = self.glyph_id[key[1]]
                ids[size_ymin:size_ymax, size_xmin:size_xmax] = np.where(size_map, id_char, ids[size_ymin:size_ymax, size_xmin:size_xmax])

                boxprev = np.concatenate([boxprev, [[cx - w/2, cx + w/2, cy - h/2, cy + h/2]]])

                w = max(int(item['width'] / 128 * tile_size * aspects), 1)
                h = max(int(item['rows'] / 128 * tile_size / aspects), 1)
                l = int(tile_base + vertBearingX)
                t = int(tile_top + vertBearingY)
                im = np.asarray(Image.fromarray(item['image']).resize((w,h)))
                im = random_morphology(im)
                images[t:t+h,l:l+w] = np.maximum(
                        images[t:t+h,l:l+w], 
                        im)
                i += 1
                tile_top += int(vertAdvance * adv_ratio)

                replace = False
                if tile_top >= height - tile_size:
                    replace = True
                    prev_center = None
                    prev = 0
                else:
                    prev -= 1

            im = Image.fromarray(images).rotate(angle / np.pi * 180, resample=Image.BILINEAR, translate=(pad_x, pad_y))
            lines = lines.rotate(angle / np.pi * 180, resample=Image.BILINEAR, translate=(pad_x / scale, pad_y / scale))
            keymapim1 = Image.fromarray(keymap).rotate(angle / np.pi * 180, resample=Image.NEAREST, translate=(pad_x / scale, pad_y / scale))
            keymapim2 = Image.fromarray(keymap).rotate(angle / np.pi * 180, resample=Image.BILINEAR, translate=(pad_x / scale, pad_y / scale))
            keymapim = np.maximum(keymapim1, keymapim2)
            xsizeim = Image.fromarray(xsizes).rotate(angle / np.pi * 180, resample=Image.NEAREST, translate=(pad_x / scale, pad_y / scale))
            ysizeim = Image.fromarray(ysizes).rotate(angle / np.pi * 180, resample=Image.NEAREST, translate=(pad_x / scale, pad_y / scale))
            xoffsetim = Image.fromarray(offsetx).rotate(angle / np.pi * 180, resample=Image.BILINEAR, translate=(pad_x / scale, pad_y / scale))
            yoffsetim = Image.fromarray(offsety).rotate(angle / np.pi * 180, resample=Image.BILINEAR, translate=(pad_x / scale, pad_y / scale))
            sepim = Image.fromarray(seps).rotate(angle / np.pi * 180, resample=Image.BILINEAR, translate=(pad_x / scale, pad_y / scale))
            labels = np.stack([keymapim, xsizeim, ysizeim, xoffsetim, yoffsetim, np.asarray(lines) / 255., sepim], -1)
            idsim = Image.fromarray(ids).rotate(angle / np.pi * 180, resample=Image.NEAREST, translate=(pad_x / scale, pad_y / scale))
            return np.asarray(im) / 255., labels, np.asarray(idsim)

        @tf.autograph.experimental.do_not_convert
        def load_images_randomline():
            min_pixel = 24
            max_pixel = 250

            images = np.zeros([height, width], dtype=np.float32)
            keymap = np.zeros([height // scale, width // scale], dtype=np.float32)
            xsizes = np.zeros([height // scale, width // scale], dtype=np.float32)
            ysizes = np.zeros([height // scale, width // scale], dtype=np.float32)
            offsetx = np.zeros([height // scale, width // scale], dtype=np.float32)
            offsety = np.zeros([height // scale, width // scale], dtype=np.float32)
            lines = np.zeros([height // scale, width // scale], dtype=np.float32)
            lines = Image.fromarray(lines)
            draw = ImageDraw.Draw(lines)
            seps = np.zeros([height // scale, width // scale], dtype=np.float32)
            ids = np.zeros([height // scale, width // scale], dtype=np.int32)

            angle = angle_max * np.random.normal() / 180 * np.pi
            angle = np.clip(angle, -np.pi, np.pi)
            pad_x = np.random.normal() * width / 20
            pad_y = np.random.normal() * height / 20

            probs = fonts
            selection = [key for key in random.choices(keys, k=np.random.randint(2,max_count), weights=probs)]

            boxprev = np.zeros([0, 4])
            if random.random() < 0.1:
                margin = 20
                line_c = random.randint(0,3)
                lw = random.randint(2, 10)
                if line_c == 0:
                    x = random.randrange(width // 2, width)
                    y = random.randrange(0, height - lw)
                    px = x // scale
                    py = y // scale
                    seps[py:py+lw//scale, :px] = 1
                    images[y:y+lw, :x] = 255
                    boxprev = np.concatenate([boxprev, [[0, (x + margin)/width, (y - margin)/height, (y+lw + margin)/height]]])
                elif line_c == 1:
                    x = random.randrange(0, width // 2)
                    y = random.randrange(0, height - lw)
                    px = x // scale
                    py = y // scale
                    seps[py:py+lw//scale, px:] = 1
                    images[y:y+lw, x:] = 255
                    boxprev = np.concatenate([boxprev, [[(x - margin)/width, 1, (y - margin)/height, (y+lw + margin)/height]]])
                elif line_c == 2:
                    y = random.randrange(height // 2, height)
                    x = random.randrange(0, width - lw)
                    px = x // scale
                    py = y // scale
                    seps[:py, px:px+lw//scale] = 1
                    images[:y, x:x+lw] = 255
                    boxprev = np.concatenate([boxprev, [[(x - margin)/width, (x+lw + margin)/width, 0, (y + margin)/height]]])
                elif line_c == 3:
                    y = random.randrange(0, height // 2)
                    x = random.randrange(0, width - lw)
                    px = x // scale
                    py = y // scale
                    seps[py:, px:px+lw//scale] = 1
                    images[y:, x:x+lw] = 255
                    boxprev = np.concatenate([boxprev, [[(x - margin)/width, (x+lw + margin)/width, (y - margin)/height, 1]]])
    
            i = 0
            replace = True
            prev = 0
            prev_center = None
            for key in selection:
                if replace:
                    invert = np.random.uniform() < 0.1
                    if self.glyph_type.get(self.glyph_id[key[1]],11) in [3,4,5,8,9,10,11]:
                        direction = random.randint(0,1) * 2
                    else:
                        p = random.random()
                        if p < 0.45:
                            direction = 0
                        elif p < 0.9:
                            direction = 2
                        else:
                            direction = 1
                    aspects = np.clip(np.random.normal() * 0.1 + 1.0, 0.75, 1.25)
                    if random.random() < 0.5:
                        aspects = 1.0 / aspects

                if direction < 2:
                    item = self.img_cache[key]['horizontal']
                else:
                    item = self.img_cache[key]['vertical']

                if item['width'] * item['rows'] == 0:
                    continue

                if self.glyph_type.get(self.glyph_id[key[1]],11) in [3,4,5,8,9,10,11]:
                    if direction == 1:
                        continue 

                if replace:
                    if random.random() < 0.5:
                        tile_size = random.randint(min_pixel, max_pixel)
                    else:
                        tile_size = int(np.exp(random.uniform(np.log(min_pixel), np.log(max_pixel))))

                if direction < 2:
                    w = item['width'] / 128 * tile_size
                    h = item['rows'] / 128 * tile_size
                    w *= aspects
                    h /= aspects
                    horiBearingX = item['horiBearingX'] / 128 * tile_size * aspects
                    horiBearingY = item['horiBearingY'] / 128 * tile_size / aspects
                    horiAdvance = item['horiAdvance'] / 128 * tile_size * aspects
                else:
                    w = item['width'] / 128 * tile_size
                    h = item['rows'] / 128 * tile_size
                    w *= aspects
                    h /= aspects
                    vertBearingX = item['vertBearingX'] / 128 * tile_size * aspects
                    vertBearingY = item['vertBearingY'] / 128 * tile_size / aspects
                    vertAdvance = item['vertAdvance'] / 128 * tile_size / aspects

                if direction == 0:
                    if replace:
                        l = max(0,-int(horiBearingX))
                        tile_left = random.randint(l, 
                                int(width - tile_size)) if int(width - tile_size) > l else l
                        tile_base = random.randint(tile_size, 
                                int(height - tile_size)) if int(height - tile_size) > tile_size else tile_size

                    if tile_base - horiBearingY < 0 or tile_base - horiBearingY + h >= height or tile_left + horiBearingX < 0 or tile_left + horiBearingX + w >= width:
                        replace = True
                        prev_center = None
                        prev = 0
                        continue

                    l = (tile_left + horiBearingX) / width
                    t = (tile_base - horiBearingY) / height
                    w = w / width
                    h = h / height
                    cx = l + w / 2
                    cy = t + h / 2

                    if invert:
                        x1 = tile_left / width
                        x2 = (tile_left + horiAdvance) / width
                        y1 = cy - h/2 * 1.25
                        y2 = cy + h/2 * 1.25
                        if prev >= 0:
                            inter_xmin = np.maximum(boxprev[:,0], x1)
                            inter_ymin = np.maximum(boxprev[:,2], y1)
                            inter_xmax = np.minimum(boxprev[:,1], x2)
                            inter_ymax = np.minimum(boxprev[:,3], y2)
                        else:
                            # 連続してレンダリングする場合は、前の文字にめり込むことがあるので判定しない
                            inter_xmin = np.maximum(boxprev[:prev,0], x1)
                            inter_ymin = np.maximum(boxprev[:prev,2], y1)
                            inter_xmax = np.minimum(boxprev[:prev,1], x2)
                            inter_ymax = np.minimum(boxprev[:prev,3], y2)
                    else:
                        if prev >= 0:
                            inter_xmin = np.maximum(boxprev[:,0], cx - w/2 * 1.2)
                            inter_ymin = np.maximum(boxprev[:,2], cy - h/2 * 1.2)
                            inter_xmax = np.minimum(boxprev[:,1], cx + w/2 * 1.2)
                            inter_ymax = np.minimum(boxprev[:,3], cy + h/2 * 1.2)
                        else:
                            # 連続してレンダリングする場合は、前の文字にめり込むことがあるので判定しない
                            inter_xmin = np.maximum(boxprev[:prev,0], cx - w/2 * 1.2)
                            inter_ymin = np.maximum(boxprev[:prev,2], cy - h/2 * 1.2)
                            inter_xmax = np.minimum(boxprev[:prev,1], cx + w/2 * 1.2)
                            inter_ymax = np.minimum(boxprev[:prev,3], cy + h/2 * 1.2)
                    inter_w = np.maximum(inter_xmax - inter_xmin, 0.)
                    inter_h = np.maximum(inter_ymax - inter_ymin, 0.)
                    inter_vol = inter_w * inter_h
                    if np.any(inter_vol > 0):
                        replace = True
                        prev_center = None
                        prev = 0
                        continue

                    if prev < 0:
                        draw.line((prev_center, (cx * width // scale, tile_base // scale)), fill=255, width=3)
                    prev_center = (cx * width // scale, tile_base // scale)

                    kernel_size = max(self.min_ker, int(max(w, h) / (2 * scale) * width))
                    std_x = min(self.max_std, max(self.min_ker, w / (2 * scale) * width) / 3)
                    std_y = min(self.max_std, max(self.min_ker, h / (2 * scale) * height) / 3)
                    center_kernel = gaussian_kernel(kernlen=kernel_size*2+1, xstd=std_x, ystd=std_y)

                    center_xmin = int(cx / scale * width) - kernel_size
                    center_xmax = int(cx / scale * width) + kernel_size + 1
                    center_ymin = int(cy / scale * height) - kernel_size
                    center_ymax = int(cy / scale * height) + kernel_size + 1
                    padx1 = max(0, 0 - center_xmin)
                    padx2 = max(0, center_xmax - width // scale)
                    pady1 = max(0, 0 - center_ymin)
                    pady2 = max(0, center_ymax - height // scale)
                    center_xmin += padx1
                    center_xmax -= padx2
                    center_ymin += pady1
                    center_ymax -= pady2
                    ker = kernel_size * 2 + 1
                    keymap[center_ymin:center_ymax, center_xmin:center_xmax] = np.maximum(keymap[center_ymin:center_ymax, center_xmin:center_xmax], center_kernel[pady1:ker-pady2,padx1:ker-padx2])

                    size_xmin = np.clip(int((cx - w/2) * width / scale), 0, width // scale)
                    size_xmax = np.clip(int((cx + w/2) * width / scale) + 1, 0, width // scale)
                    size_ymin = np.clip(int((cy - h/2) * height / scale), 0, height // scale)
                    size_ymax = np.clip(int((cy + h/2) * height / scale) + 1, 0, height // scale)
                    size_mapx, size_mapy = np.meshgrid(np.arange(size_xmin, size_xmax) - cx * width / scale, np.arange(size_ymin, size_ymax) - cy * height / scale)
                    size_map = size_mapx ** 2 / max(w/2 * width / scale, 1) ** 2 + size_mapy ** 2 / max(h/2 * height / scale, 1) ** 2 < 1

                    center_x = int(cx / scale * width)
                    center_y = int(cy / scale * height)
                    offset_x = (cx * width % scale) / width * np.cos(angle)
                    offset_y = (cy * height % scale) / height * np.sin(angle + np.pi / 2)
                    offset_x += pad_x % scale
                    offset_y += pad_y % scale
                    offset_x = offset_x / scale - (np.arange(size_xmin, size_xmax) - center_x) * np.cos(angle)
                    offset_y = offset_y / scale - (np.arange(size_ymin, size_ymax) - center_y) * np.sin(angle + np.pi / 2)
                    offset_x = offset_x[np.newaxis,...] - np.linspace(-(size_ymax-size_ymin) * np.sin(angle) / 2, (size_ymax-size_ymin) * np.sin(angle) / 2, size_ymax-size_ymin)[...,np.newaxis]
                    offset_y = offset_y[...,np.newaxis] - np.linspace(-(size_xmax-size_xmin) * np.cos(angle + np.pi / 2) / 2, (size_xmax-size_xmin) * np.cos(angle + np.pi / 2) / 2, size_xmax-size_xmin)[np.newaxis,...]
                    offsetx[size_ymin:size_ymax, size_xmin:size_xmax] = np.where(size_map, offset_x, offsetx[size_ymin:size_ymax, size_xmin:size_xmax])
                    offsety[size_ymin:size_ymax, size_xmin:size_xmax] = np.where(size_map, offset_y, offsety[size_ymin:size_ymax, size_xmin:size_xmax])

                    fixw = w * np.abs(np.cos(angle)) + h * np.abs(np.sin(angle))
                    fixh = h * np.abs(np.cos(angle)) + w * np.abs(np.sin(angle))
                    fixw = np.log10(fixw * 10)
                    fixh = np.log10(fixh * 10)
                    xsizes[size_ymin:size_ymax, size_xmin:size_xmax] = np.where(size_map, fixw, xsizes[size_ymin:size_ymax, size_xmin:size_xmax])
                    ysizes[size_ymin:size_ymax, size_xmin:size_xmax] = np.where(size_map, fixh, ysizes[size_ymin:size_ymax, size_xmin:size_xmax])

                    id_char = self.glyph_id[key[1]]
                    ids[size_ymin:size_ymax, size_xmin:size_xmax] = np.where(size_map, id_char, ids[size_ymin:size_ymax, size_xmin:size_xmax])

                    if invert:
                        boxprev = np.concatenate([boxprev, [[x1, x2, y1, y2]]])
                    else:
                        boxprev = np.concatenate([boxprev, [[cx - w/2 * 1.2, cx + w/2 * 1.2, cy - h/2 * 1.2, cy + h/2 * 1.2]]])

                    w = max(int(item['width'] / 128 * tile_size * aspects), 1)
                    h = max(int(item['rows'] / 128 * tile_size / aspects), 1)
                    tile_top = int(tile_base - horiBearingY) 
                    l = int(tile_left + horiBearingX)
                    im = np.asarray(Image.fromarray(item['image']).resize((w,h)))
                    im = random_morphology(im)
                    if invert:
                        im_inv = 255 - im
                        x1 = int(x1 * width)
                        x2 = int(x2 * width)
                        y1 = int(y1 * height)
                        y2 = int(y2 * height)
                        images[y1:y2,x1:x2] = 255
                        images[tile_top:tile_top+h,l:l+w] = np.minimum(
                                images[tile_top:tile_top+h,l:l+w], 
                                im_inv)
                    else:
                        images[tile_top:tile_top+h,l:l+w] = np.maximum(
                                images[tile_top:tile_top+h,l:l+w], 
                                im)

                    i += 1
                    tile_left += int(horiAdvance)
                    replace = False
                    if tile_left >= width - tile_size:
                        replace = True
                        prev_center = None
                        prev = 0
                    else:
                        prev -= 1

                elif direction == 1:
                    if replace:
                        t = max(0,-int(horiBearingX))
                        tile_top = random.randint(t, 
                                int(height - tile_size)) if int(height - tile_size) > t else t
                        tile_base = random.randint(tile_size, 
                                int(width - tile_size)) if int(width - tile_size) > tile_size else tile_size

                    if tile_base - horiBearingY < 0 or tile_base - horiBearingY + h >= width or tile_top + horiBearingX < 0 or tile_top + horiBearingX + w >= height:
                        replace = True
                        prev_center = None
                        prev = 0
                        continue

                    t = (tile_top + horiBearingX) / height
                    l = (tile_base - horiBearingY) / width
                    normal_h = h / width
                    normal_w = w / height
                    h = normal_w
                    w = normal_h
                    cx = l + w / 2
                    cy = t + h / 2

                    if invert:
                        y1 = tile_top / height
                        y2 = (tile_top + horiAdvance) / height
                        x1 = cx - w/2 * 1.25
                        x2 = cx + w/2 * 1.25
                        if prev >= 0:
                            inter_xmin = np.maximum(boxprev[:,0], x1)
                            inter_ymin = np.maximum(boxprev[:,2], y1)
                            inter_xmax = np.minimum(boxprev[:,1], x2)
                            inter_ymax = np.minimum(boxprev[:,3], y2)
                        else:
                            # 連続してレンダリングする場合は、前の文字にめり込むことがあるので判定しない
                            inter_xmin = np.maximum(boxprev[:prev,0], x1)
                            inter_ymin = np.maximum(boxprev[:prev,2], y1)
                            inter_xmax = np.minimum(boxprev[:prev,1], x2)
                            inter_ymax = np.minimum(boxprev[:prev,3], y2)
                    else:
                        if prev >= 0:
                            inter_xmin = np.maximum(boxprev[:,0], cx - w/2 * 1.2)
                            inter_ymin = np.maximum(boxprev[:,2], cy - h/2 * 1.2)
                            inter_xmax = np.minimum(boxprev[:,1], cx + w/2 * 1.2)
                            inter_ymax = np.minimum(boxprev[:,3], cy + h/2 * 1.2)
                        else:
                            # 連続してレンダリングする場合は、前の文字にめり込むことがあるので判定しない
                            inter_xmin = np.maximum(boxprev[:prev,0], cx - w/2 * 1.2)
                            inter_ymin = np.maximum(boxprev[:prev,2], cy - h/2 * 1.2)
                            inter_xmax = np.minimum(boxprev[:prev,1], cx + w/2 * 1.2)
                            inter_ymax = np.minimum(boxprev[:prev,3], cy + h/2 * 1.2)
                    inter_w = np.maximum(inter_xmax - inter_xmin, 0.)
                    inter_h = np.maximum(inter_ymax - inter_ymin, 0.)
                    inter_vol = inter_w * inter_h
                    if np.any(inter_vol > 0):
                        replace = True
                        prev_center = None
                        prev = 0
                        continue

                    if prev < 0:
                        draw.line((prev_center, (tile_base // scale, cy * height // scale)), fill=255, width=3)
                    prev_center = (tile_base // scale, cy * height // scale)

                    kernel_size = max(self.min_ker, int(max(w, h) / (2 * scale) * width))
                    std_x = min(self.max_std, max(self.min_ker, w / (2 * scale) * width) / 3)
                    std_y = min(self.max_std, max(self.min_ker, h / (2 * scale) * height) / 3)
                    center_kernel = gaussian_kernel(kernlen=kernel_size*2+1, xstd=std_x, ystd=std_y)

                    center_xmin = int(cx / scale * width) - kernel_size
                    center_xmax = int(cx / scale * width) + kernel_size + 1
                    center_ymin = int(cy / scale * height) - kernel_size
                    center_ymax = int(cy / scale * height) + kernel_size + 1
                    padx1 = max(0, 0 - center_xmin)
                    padx2 = max(0, center_xmax - width // scale)
                    pady1 = max(0, 0 - center_ymin)
                    pady2 = max(0, center_ymax - height // scale)
                    center_xmin += padx1
                    center_xmax -= padx2
                    center_ymin += pady1
                    center_ymax -= pady2
                    ker = kernel_size * 2 + 1
                    keymap[center_ymin:center_ymax, center_xmin:center_xmax] = np.maximum(keymap[center_ymin:center_ymax, center_xmin:center_xmax], center_kernel[pady1:ker-pady2,padx1:ker-padx2])

                    size_xmin = np.clip(int((cx - w/2) * width / scale), 0, width // scale)
                    size_xmax = np.clip(int((cx + w/2) * width / scale) + 1, 0, width // scale)
                    size_ymin = np.clip(int((cy - h/2) * height / scale), 0, height // scale)
                    size_ymax = np.clip(int((cy + h/2) * height / scale) + 1, 0, height // scale)
                    size_mapx, size_mapy = np.meshgrid(np.arange(size_xmin, size_xmax) - cx * width / scale, np.arange(size_ymin, size_ymax) - cy * height / scale)
                    size_map = size_mapx ** 2 / max(w/2 * width / scale, 1) ** 2 + size_mapy ** 2 / max(h/2 * height / scale, 1) ** 2 < 1

                    center_x = int(cx / scale * width)
                    center_y = int(cy / scale * height)
                    offset_x = (cx * width % scale) / width * np.cos(angle)
                    offset_y = (cy * height % scale) / height * np.sin(angle + np.pi / 2)
                    offset_x += pad_x % scale
                    offset_y += pad_y % scale
                    offset_x = offset_x / scale - (np.arange(size_xmin, size_xmax) - center_x) * np.cos(angle)
                    offset_y = offset_y / scale - (np.arange(size_ymin, size_ymax) - center_y) * np.sin(angle + np.pi / 2)
                    offset_x = offset_x[np.newaxis,...] - np.linspace(-(size_ymax-size_ymin) * np.sin(angle) / 2, (size_ymax-size_ymin) * np.sin(angle) / 2, size_ymax-size_ymin)[...,np.newaxis]
                    offset_y = offset_y[...,np.newaxis] - np.linspace(-(size_xmax-size_xmin) * np.cos(angle + np.pi / 2) / 2, (size_xmax-size_xmin) * np.cos(angle + np.pi / 2) / 2, size_xmax-size_xmin)[np.newaxis,...]
                    offsetx[size_ymin:size_ymax, size_xmin:size_xmax] = np.where(size_map, offset_x, offsetx[size_ymin:size_ymax, size_xmin:size_xmax])
                    offsety[size_ymin:size_ymax, size_xmin:size_xmax] = np.where(size_map, offset_y, offsety[size_ymin:size_ymax, size_xmin:size_xmax])

                    fixw = w * np.abs(np.cos(angle)) + h * np.abs(np.sin(angle))
                    fixh = h * np.abs(np.cos(angle)) + w * np.abs(np.sin(angle))
                    fixw = np.log10(fixw * 10)
                    fixh = np.log10(fixh * 10)
                    xsizes[size_ymin:size_ymax, size_xmin:size_xmax] = np.where(size_map, fixw, xsizes[size_ymin:size_ymax, size_xmin:size_xmax])
                    ysizes[size_ymin:size_ymax, size_xmin:size_xmax] = np.where(size_map, fixh, ysizes[size_ymin:size_ymax, size_xmin:size_xmax])

                    id_char = self.glyph_id[key[1]]
                    ids[size_ymin:size_ymax, size_xmin:size_xmax] = np.where(size_map, id_char, ids[size_ymin:size_ymax, size_xmin:size_xmax])

                    if invert:
                        boxprev = np.concatenate([boxprev, [[x1, x2, y1, y2]]])
                    else:
                        boxprev = np.concatenate([boxprev, [[cx - w/2 * 1.2, cx + w/2 * 1.2, cy - h/2 * 1.2, cy + h/2 * 1.2]]])

                    h = max(int(item['width'] / 128 * tile_size * aspects), 1)
                    w = max(int(item['rows'] / 128 * tile_size / aspects), 1)
                    tile_left = int(tile_base - horiBearingY) 
                    t = int(tile_top + horiBearingX)
                    im = item['image'][:,::-1].T
                    im = np.asarray(Image.fromarray(im[::-1,::-1]).resize((w,h)))
                    im = random_morphology(im)
                    if invert:
                        im_inv = 255 - im
                        x1 = int(x1 * width)
                        x2 = int(x2 * width)
                        y1 = int(y1 * height)
                        y2 = int(y2 * height)
                        images[y1:y2,x1:x2] = 255
                        images[t:t+h,tile_left:tile_left+w] = np.minimum(
                                images[t:t+h,tile_left:tile_left+w], 
                                im_inv)
                    else:
                        images[t:t+h,tile_left:tile_left+w] = np.maximum(
                                images[t:t+h,tile_left:tile_left+w], 
                                im)


                    i += 1
                    tile_top += int(horiAdvance)
                    replace = False
                    if tile_top >= height - tile_size:
                        replace = True
                        prev_center = None
                        prev = 0
                    else:
                        prev -= 1

                elif direction == 2:
                    if replace:
                        l = max(0,-int(vertBearingX))
                        tile_base = random.randint(l, 
                                int(width - tile_size)) if int(width - tile_size) > l else l
                        tile_top = random.randint(tile_size, 
                                int(height - tile_size)) if int(height - tile_size) > tile_size else 0

                    if tile_base + vertBearingX < 0 or tile_base + vertBearingX + w >= width or tile_top + vertBearingY < 0 or tile_top + vertBearingY + h >= height:
                        replace = True
                        prev_center = None
                        prev = 0
                        continue

                    l = (tile_base + vertBearingX) / width
                    t = (tile_top + vertBearingY) / height
                    w = w / width
                    h = h / height
                    cx = l + w / 2
                    cy = t + h / 2

                    if invert:
                        x1 = cx - w/2 * 1.25
                        x2 = cx + w/2 * 1.25
                        y1 = tile_top / height
                        y2 = (tile_top + vertAdvance) / height
                        if prev >= 0:
                            inter_xmin = np.maximum(boxprev[:,0], x1)
                            inter_ymin = np.maximum(boxprev[:,2], y1)
                            inter_xmax = np.minimum(boxprev[:,1], x2)
                            inter_ymax = np.minimum(boxprev[:,3], y2)
                        else:
                            # 連続してレンダリングする場合は、前の文字にめり込むことがあるので判定しない
                            inter_xmin = np.maximum(boxprev[:prev,0], x1)
                            inter_ymin = np.maximum(boxprev[:prev,2], y1)
                            inter_xmax = np.minimum(boxprev[:prev,1], x2)
                            inter_ymax = np.minimum(boxprev[:prev,3], y2)
                    else:
                        if prev >= 0:
                            inter_xmin = np.maximum(boxprev[:,0], cx - w/2 * 1.2)
                            inter_ymin = np.maximum(boxprev[:,2], cy - h/2 * 1.2)
                            inter_xmax = np.minimum(boxprev[:,1], cx + w/2 * 1.2)
                            inter_ymax = np.minimum(boxprev[:,3], cy + h/2 * 1.2)
                        else:
                            # 連続してレンダリングする場合は、前の文字にめり込むことがあるので判定しない
                            inter_xmin = np.maximum(boxprev[:prev,0], cx - w/2 * 1.2)
                            inter_ymin = np.maximum(boxprev[:prev,2], cy - h/2 * 1.2)
                            inter_xmax = np.minimum(boxprev[:prev,1], cx + w/2 * 1.2)
                            inter_ymax = np.minimum(boxprev[:prev,3], cy + h/2 * 1.2)
                    inter_w = np.maximum(inter_xmax - inter_xmin, 0.)
                    inter_h = np.maximum(inter_ymax - inter_ymin, 0.)
                    inter_vol = inter_w * inter_h
                    if np.any(inter_vol > 0):
                        replace = True
                        prev_center = None
                        prev = 0
                        continue

                    if prev < 0:
                        draw.line((prev_center, (tile_base // scale, cy * height // scale)), fill=255, width=3)
                    prev_center = (tile_base // scale, cy * height // scale)

                    kernel_size = max(self.min_ker, int(max(w, h) / (2 * scale) * width))
                    std_x = min(self.max_std, max(self.min_ker, w / (2 * scale) * width) / 3)
                    std_y = min(self.max_std, max(self.min_ker, h / (2 * scale) * height) / 3)
                    center_kernel = gaussian_kernel(kernlen=kernel_size*2+1, xstd=std_x, ystd=std_y)

                    center_xmin = int(cx / scale * width) - kernel_size
                    center_xmax = int(cx / scale * width) + kernel_size + 1
                    center_ymin = int(cy / scale * height) - kernel_size
                    center_ymax = int(cy / scale * height) + kernel_size + 1
                    padx1 = max(0, 0 - center_xmin)
                    padx2 = max(0, center_xmax - width // scale)
                    pady1 = max(0, 0 - center_ymin)
                    pady2 = max(0, center_ymax - height // scale)
                    center_xmin += padx1
                    center_xmax -= padx2
                    center_ymin += pady1
                    center_ymax -= pady2
                    ker = kernel_size * 2 + 1
                    keymap[center_ymin:center_ymax, center_xmin:center_xmax] = np.maximum(keymap[center_ymin:center_ymax, center_xmin:center_xmax], center_kernel[pady1:ker-pady2,padx1:ker-padx2])

                    size_xmin = np.clip(int((cx - w/2) * width / scale), 0, width // scale)
                    size_xmax = np.clip(int((cx + w/2) * width / scale) + 1, 0, width // scale)
                    size_ymin = np.clip(int((cy - h/2) * height / scale), 0, height // scale)
                    size_ymax = np.clip(int((cy + h/2) * height / scale) + 1, 0, height // scale)
                    size_mapx, size_mapy = np.meshgrid(np.arange(size_xmin, size_xmax) - cx * width / scale, np.arange(size_ymin, size_ymax) - cy * height / scale)
                    size_map = size_mapx ** 2 / max(w/2 * width / scale, 1) ** 2 + size_mapy ** 2 / max(h/2 * height / scale, 1) ** 2 < 1

                    center_x = int(cx / scale * width)
                    center_y = int(cy / scale * height)
                    offset_x = (cx * width % scale) / width * np.cos(angle)
                    offset_y = (cy * height % scale) / height * np.sin(angle + np.pi / 2)
                    offset_x += pad_x % scale
                    offset_y += pad_y % scale
                    offset_x = offset_x / scale - (np.arange(size_xmin, size_xmax) - center_x) * np.cos(angle)
                    offset_y = offset_y / scale - (np.arange(size_ymin, size_ymax) - center_y) * np.sin(angle + np.pi / 2)
                    offset_x = offset_x[np.newaxis,...] - np.linspace(-(size_ymax-size_ymin) * np.sin(angle) / 2, (size_ymax-size_ymin) * np.sin(angle) / 2, size_ymax-size_ymin)[...,np.newaxis]
                    offset_y = offset_y[...,np.newaxis] - np.linspace(-(size_xmax-size_xmin) * np.cos(angle + np.pi / 2) / 2, (size_xmax-size_xmin) * np.cos(angle + np.pi / 2) / 2, size_xmax-size_xmin)[np.newaxis,...]
                    offsetx[size_ymin:size_ymax, size_xmin:size_xmax] = np.where(size_map, offset_x, offsetx[size_ymin:size_ymax, size_xmin:size_xmax])
                    offsety[size_ymin:size_ymax, size_xmin:size_xmax] = np.where(size_map, offset_y, offsety[size_ymin:size_ymax, size_xmin:size_xmax])

                    fixw = w * np.abs(np.cos(angle)) + h * np.abs(np.sin(angle))
                    fixh = h * np.abs(np.cos(angle)) + w * np.abs(np.sin(angle))
                    fixw = np.log10(fixw * 10)
                    fixh = np.log10(fixh * 10)
                    xsizes[size_ymin:size_ymax, size_xmin:size_xmax] = np.where(size_map, fixw, xsizes[size_ymin:size_ymax, size_xmin:size_xmax])
                    ysizes[size_ymin:size_ymax, size_xmin:size_xmax] = np.where(size_map, fixh, ysizes[size_ymin:size_ymax, size_xmin:size_xmax])

                    id_char = self.glyph_id[key[1]]
                    ids[size_ymin:size_ymax, size_xmin:size_xmax] = np.where(size_map, id_char, ids[size_ymin:size_ymax, size_xmin:size_xmax])

                    if invert:
                        boxprev = np.concatenate([boxprev, [[x1, x2, y1, y2]]])
                    else:
                        boxprev = np.concatenate([boxprev, [[cx - w/2 * 1.2, cx + w/2 * 1.2, cy - h/2 * 1.2, cy + h/2 * 1.2]]])

                    w = max(int(item['width'] / 128 * tile_size * aspects), 1)
                    h = max(int(item['rows'] / 128 * tile_size / aspects), 1)
                    l = int(tile_base + vertBearingX)
                    t = int(tile_top + vertBearingY)
                    im = np.asarray(Image.fromarray(item['image']).resize((w,h)))
                    im = random_morphology(im)
                    if invert:
                        im_inv = 255 - im
                        x1 = int(x1 * width)
                        x2 = int(x2 * width)
                        y1 = int(y1 * height)
                        y2 = int(y2 * height)
                        images[y1:y2,x1:x2] = 255
                        images[t:t+h,l:l+w] = np.minimum(
                                images[t:t+h,l:l+w], 
                                im_inv)
                    else:
                        images[t:t+h,l:l+w] = np.maximum(
                                images[t:t+h,l:l+w], 
                                im)

                    i += 1
                    tile_top += int(vertAdvance)

                    replace = False
                    if tile_top >= height - tile_size:
                        replace = True
                        prev_center = None
                        prev = 0
                    else:
                        prev -= 1


            im = Image.fromarray(images).rotate(angle / np.pi * 180, resample=Image.BILINEAR, translate=(pad_x, pad_y))
            lines = lines.rotate(angle / np.pi * 180, resample=Image.BILINEAR, translate=(pad_x / scale, pad_y / scale))
            keymapim1 = Image.fromarray(keymap).rotate(angle / np.pi * 180, resample=Image.NEAREST, translate=(pad_x / scale, pad_y / scale))
            keymapim2 = Image.fromarray(keymap).rotate(angle / np.pi * 180, resample=Image.BILINEAR, translate=(pad_x / scale, pad_y / scale))
            keymapim = np.maximum(keymapim1, keymapim2)
            xsizeim = Image.fromarray(xsizes).rotate(angle / np.pi * 180, resample=Image.NEAREST, translate=(pad_x / scale, pad_y / scale))
            ysizeim = Image.fromarray(ysizes).rotate(angle / np.pi * 180, resample=Image.NEAREST, translate=(pad_x / scale, pad_y / scale))
            xoffsetim = Image.fromarray(offsetx).rotate(angle / np.pi * 180, resample=Image.BILINEAR, translate=(pad_x / scale, pad_y / scale))
            yoffsetim = Image.fromarray(offsety).rotate(angle / np.pi * 180, resample=Image.BILINEAR, translate=(pad_x / scale, pad_y / scale))
            sepim = Image.fromarray(seps).rotate(angle / np.pi * 180, resample=Image.BILINEAR, translate=(pad_x / scale, pad_y / scale))
            labels = np.stack([keymapim, xsizeim, ysizeim, xoffsetim, yoffsetim, np.asarray(lines) / 255., sepim], -1)
            idsim = Image.fromarray(ids).rotate(angle / np.pi * 180, resample=Image.NEAREST, translate=(pad_x / scale, pad_y / scale))
            return np.asarray(im) / 255., labels, np.asarray(idsim)

        if t == 0:
            images, labels, ids = tf.py_function(
                func=load_images_random, 
                inp=[], 
                Tout=[tf.float32, tf.float32, tf.int32])
        elif t == 1:
            images, labels, ids = tf.py_function(
                func=load_images_fill, 
                inp=[], 
                Tout=[tf.float32, tf.float32, tf.int32])
        elif t == 2:
            images, labels, ids = tf.py_function(
                func=load_images_fill_horizontal, 
                inp=[], 
                Tout=[tf.float32, tf.float32, tf.int32])
        elif t == 3:
            images, labels, ids = tf.py_function(
                func=load_images_fill_vertical, 
                inp=[0], 
                Tout=[tf.float32, tf.float32, tf.int32])
        elif t == 4:
            images, labels, ids = tf.py_function(
                func=load_images_fill_vertical, 
                inp=[1], 
                Tout=[tf.float32, tf.float32, tf.int32])
        elif t == 5:
            images, labels, ids = tf.py_function(
                func=load_images_randomline, 
                inp=[], 
                Tout=[tf.float32, tf.float32, tf.int32])
        elif t == 6:
            images, labels, ids = tf.py_function(
                func=load_random_line, 
                inp=[], 
                Tout=[tf.float32, tf.float32, tf.int32])
        elif t == 7:
            images, labels, ids = tf.py_function(
                func=null_images, 
                inp=[], 
                Tout=[tf.float32, tf.float32, tf.int32])
        images = tf.ensure_shape(images, [height, width])
        labels = tf.ensure_shape(labels, [height // scale, width // scale, 7])
        ids = tf.ensure_shape(ids, [height // scale, width // scale])

        images = apply_random_filter(images)

        return self.sub_constructimage(images, labels, ids, len(self.random_background) > 0)

    @tf.function
    def sub_constructimage(self, images, labels, ids, bk):

        img = images[...,tf.newaxis]

        c = tf.random.uniform([], 0., 1.)
        if c < 0.25:
            fg_c = tf.random.uniform([], minval=-1., maxval=1.)
            bk_c = tf.random.uniform([], minval=-1., maxval=1.)
            if tf.math.abs(fg_c - bk_c) < 0.5:
                d = fg_c - bk_c
                if d < 0:
                    d = -0.5 - d
                else:
                    d = 0.5 - d
                fg_c += d
                bk_c -= d
            fg_c = fg_c[tf.newaxis]
            bk_c = bk_c[tf.newaxis]
            fg_c += tf.random.normal([3]) * 0.1
            bk_c += tf.random.normal([3]) * 0.1
            fgimg = fg_c[tf.newaxis,tf.newaxis,:]
            bkimg = bk_c[tf.newaxis,tf.newaxis,:]
        elif c < 0.5:
            fg1_c = tf.random.uniform([3], minval=-1., maxval=1.)
            fg2_c = tf.random.uniform([3], minval=-1., maxval=1.)
            bk_c = tf.random.uniform([3], minval=-1., maxval=1.)
            if tf.math.reduce_min(tf.math.abs(fg1_c - bk_c)) < 0.5:
                ind = tf.math.argmax(-tf.math.abs(fg1_c - bk_c))
                d = (fg1_c - bk_c)[ind]
                if d < 0:
                    d = -0.5 - d
                else:
                    d = 0.5 - d
                fg1_c += d
                bk_c -= d
            if tf.math.reduce_min(tf.math.abs(fg2_c - bk_c)) < 0.5:
                ind = tf.math.argmax(-tf.math.abs(fg2_c - bk_c))
                d = (fg2_c - bk_c)[ind]
                if d < 0:
                    d = -0.5 - d
                else:
                    d = 0.5 - d
                fg2_c += d
                bk_c -= d
            X, Y = tf.meshgrid(tf.range(tf.shape(images)[-1]), tf.range(tf.shape(images)[-2]))
            X = X[...,tf.newaxis]
            Y = Y[...,tf.newaxis]
            startX = tf.random.uniform([], 0, tf.shape(images)[-1], dtype=tf.int32)
            startY = tf.random.uniform([], 0, tf.shape(images)[-2], dtype=tf.int32)
            fgimg = tf.where(
                tf.logical_and(startX > X, startY > Y),
                fg1_c[tf.newaxis,tf.newaxis,:], fg2_c[tf.newaxis,tf.newaxis,:])
            bkimg = bk_c[tf.newaxis,tf.newaxis,:]
        elif not bk or c < 0.75:
            fg_c = tf.random.uniform([3], minval=-1., maxval=1.)
            bk_c = tf.random.uniform([3], minval=-1., maxval=1.)
            if tf.math.reduce_min(tf.math.abs(fg_c - bk_c)) < 0.5:
                ind = tf.math.argmax(-tf.math.abs(fg_c - bk_c))
                d = (fg_c - bk_c)[ind]
                if d < 0:
                    d = -0.5 - d
                else:
                    d = 0.5 - d
                fg_c += d
                bk_c -= d
            fgimg = fg_c[tf.newaxis,tf.newaxis,:]
            bkimg = bk_c[tf.newaxis,tf.newaxis,:]
        else:
            bkimg = tf.py_function(
                func=self.load_background_images, 
                inp=[], 
                Tout=tf.float32)
            bkimg = tf.clip_by_value(bkimg, -1.0, 1.0)
            bk_c = tf.math.reduce_mean(bkimg, axis=[0,1])
            bk_std = tf.math.reduce_std(bkimg, axis=[0,1])
            fg_c = tf.where(bk_c > 0, 
                tf.where(bk_c - bk_std * 2 - 0.5 > -1., tf.random.uniform([3], -1., bk_c - bk_std * 2 - 0.5), bk_c - bk_std * 2 - 0.5), 
                tf.where(bk_c + bk_std * 2 + 0.5 < 1., tf.random.uniform([3], bk_c + bk_std * 2 + 0.5, 1.), bk_c + bk_std * 2 + 0.5))
            bk_alpha = tf.maximum(tf.abs(fg_c), 1)
            bkimg /= bk_alpha
            fg_c /= bk_alpha
            fg_c = tf.clip_by_value(fg_c, -1.0, 1.0)
            fgimg = fg_c[tf.newaxis,tf.newaxis,:]

        bkimg = tf.clip_by_value(bkimg, -1.0, 1.0)

        image = fgimg * img + bkimg * (1 - img)
        image = tf.clip_by_value(image, -1.0, 1.0)

        noise_v = tf.random.normal([]) * 0.15
        image = tf.where(noise_v <= 0,
            image,
            image + tf.where(tf.random.uniform(tf.shape(image)) > noise_v,
                0.,
                tf.random.normal(tf.shape(image)),
                )
            )
        image = tf.clip_by_value(image, -1.0, 1.0)

        image = image * 127 + 127

        image = tf.reshape(image,[height, width, 3])
        labels = tf.reshape(labels, [height//scale, width//scale, 7])
        ids = tf.reshape(ids, [height//scale, width//scale])

        return image, labels, ids

    def prob_images(self, keys, batch_size):
        def num_func():
            k = [x for x in keys if self.glyph_type.get(self.glyph_id[x[1]],-1) == 0]
            fonts = list(set([key[0] for key in k]))
            return self.construct_alphatext(k, fonts, 1)
        
        def capital_func():
            k = [x for x in keys if self.glyph_type.get(self.glyph_id[x[1]],-1) == 1]
            fonts = list(set([key[0] for key in k]))
            return self.construct_alphatext(k, fonts, 1)
        
        def small_func():
            k = [x for x in keys if self.glyph_type.get(self.glyph_id[x[1]],-1) == 2]
            fonts = list(set([key[0] for key in k]))
            return self.construct_alphatext(k, fonts, 1)
        
        def alpha_func():
            k = [x for x in keys if self.glyph_type.get(self.glyph_id[x[1]],-1) in [0,1,2,6]]
            fonts = list(set([key[0] for key in k]))
            return self.construct_alphatext(k, fonts, 1)
        
        def random_func():
            probs = [self.prob_map[self.glyph_type.get(self.glyph_id[key[1]],11)] for key in keys]
            return self.construct_alphatext(keys, probs, 0)

        def yokogaki_func():
            k = [x for x in keys if self.glyph_type.get(self.glyph_id[x[1]],-1) in [3,4,5,7]]
            fonts = list(set([key[0] for key in k]))
            p_sum = sum([0 if '.' in f else 1 for f in fonts])
            p = [1. if '.' in f else 1/p_sum for f in fonts]
            return self.construct_alphatext(k, (fonts, p), 2)

        def tategaki_func():
            k = [x for x in keys if self.glyph_type.get(self.glyph_id[x[1]],-1) in [3,4,5,7]]
            fonts = list(set([key[0] for key in k]))
            p_sum = sum([0 if '.' in f else 1 for f in fonts])
            p = [1. if '.' in f else 1/p_sum for f in fonts]
            return self.construct_alphatext(k, (fonts, p), 3)

        def tategaki_hira_func():
            k = [x for x in keys if self.glyph_type.get(self.glyph_id[x[1]],-1) in [3,4]]
            fonts = list(set([key[0] for key in k]))
            p_sum = sum([0 if '.' in f else 1 for f in fonts])
            p = [1. if '.' in f else 1/p_sum for f in fonts]
            return self.construct_alphatext(k, (fonts, p), 4)

        def random_kanji_func():
            probs = [self.prob_map_kanji[self.glyph_type.get(self.glyph_id[key[1]],11)] for key in keys]
            return self.construct_alphatext(keys, probs, 0)

        def random_num_func():
            probs = [self.prob_map_num[self.glyph_type.get(self.glyph_id[key[1]],11)] for key in keys]
            return self.construct_alphatext(keys, probs, 0)

        def random_alpha_func():
            probs = [self.prob_map_alpha[self.glyph_type.get(self.glyph_id[key[1]],11)] for key in keys]
            return self.construct_alphatext(keys, probs, 0)

        def random_hira_func():
            probs = [self.prob_map_hira[self.glyph_type.get(self.glyph_id[key[1]],11)] for key in keys]
            return self.construct_alphatext(keys, probs, 0)

        def renderling_func():
            k = [x for x in keys if self.glyph_type.get(self.glyph_id[x[1]],-1) in [3,4,5,7]]
            fonts = list(set([key[0] for key in k]))
            p_sum = sum([0 if '.' in f else 1 for f in fonts])
            p = [1. if '.' in f else 1/p_sum for f in fonts]
            return self.construct_rendertext(k, (fonts, p))

        def renderling2_func():
            k = [x for x in keys if self.glyph_type.get(self.glyph_id[x[1]],-1) in [0,3,4,5,7]]
            fonts = list(set([key[0] for key in k if self.glyph_type.get(self.glyph_id[key[1]],-1) > 0]))
            p = [1. if '.' in f else 0. for f in fonts]
            return self.construct_tateyokotext(k, (fonts, p))

        def randomline_func():
            probs = [self.prob_map[self.glyph_type.get(self.glyph_id[key[1]],11)] for key in keys]
            return self.construct_alphatext(keys, probs, 5)

        def line_func():
            return self.construct_alphatext(None, None, 6)

        def null_func():
            return self.construct_alphatext(None, None, 7)

        @tf.function
        def switch_func(i):
            return tf.switch_case(tf.cast(i, tf.int32), 
                    branch_fns={
                        0: num_func, 1: capital_func, 2: small_func, 3: alpha_func, 
                        4: yokogaki_func, 5: tategaki_func, 6: tategaki_hira_func,
                        7: null_func, 8: random_func, 9: random_kanji_func,
                        10: random_num_func, 11: random_alpha_func, 12: random_hira_func,
                        13: renderling_func,
                        14: randomline_func,
                        15: line_func,
                        16: renderling2_func,
                        }, 
                    default=null_func)
        #                      0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16
        prob = tf.math.log([[3.0,1.0,2.0,3.0,1.5,1.5,2.0,0.1,1.0,2.5,3.0,3.0,3.0,3.0,5.0,1.0,2.0]])
        ds = tf.data.Dataset.range(1)
        ds = ds.repeat()
        ds = ds.map(lambda x: tf.squeeze(tf.random.categorical(prob,1)))
        ds = ds.map(switch_func, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)
        ds = ds.batch(batch_size, drop_remainder=True, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)
        ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds

    def prob_images2(self, keys, batch_size):
        def random_func():
            probs = [self.prob_map_clustering[self.glyph_type.get(self.glyph_id[key[1]],11)] for key in keys]
            return self.construct_alphatext(keys, probs, 0)

        @tf.function
        def switch_func(i):
            return tf.switch_case(tf.cast(i, tf.int32), 
                    branch_fns={
                        0: random_func}, 
                    default=random_func)

        ds = tf.data.Dataset.range(1)
        ds = ds.repeat()
        ds = ds.map(switch_func, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)
        ds = ds.batch(batch_size, drop_remainder=True, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)
        ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds

    def test_data(self, batch_size):
        keys = self.test_keys
        return self.prob_images(keys, batch_size)

    def train_data(self, batch_size):
        keys = self.train_keys
        return self.prob_images(keys, batch_size)

    def cluster_data(self, batch_size):
        keys = self.image_keys
        return self.prob_images2(keys, batch_size)



if __name__ == '__main__':
    import matplotlib.pyplot as plt
    data = FontData()
    ds = data.train_data(16)

    for i,d in enumerate(ds):
        image, labels, ids = d
        for i in range(image.shape[0]):
            img1 = image[i,...].numpy()
            img1 = img1 / 255.
            plt.figure(figsize=(5,5))
            plt.imshow(img1)

            label1 = labels[i,...]

            plt.figure(figsize=(5,5))
            plt.imshow(label1[...,0], vmin=0, vmax=1)

            plt.figure(figsize=(5,5))
            plt.imshow(label1[...,1], vmin=-1, vmax=1)

            plt.figure(figsize=(5,5))
            plt.imshow(label1[...,2], vmin=-1, vmax=1)

            plt.figure(figsize=(5,5))
            plt.imshow(label1[...,3], vmin=-5, vmax=5)

            plt.figure(figsize=(5,5))
            plt.imshow(label1[...,4], vmin=-5, vmax=5)

            plt.figure(figsize=(5,5))
            plt.imshow(label1[...,5], vmin=0, vmax=1)

            plt.figure(figsize=(5,5))
            plt.imshow(label1[...,6], vmin=0, vmax=1)

            plt.show()
            
            
