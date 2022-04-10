import tensorflow as tf
import numpy as np
from scipy import signal
from scipy.ndimage import gaussian_filter
from PIL import Image, ImageDraw
import random
import glob, os
import csv
from multiprocessing import Pool
import subprocess
import time

width = 512
height = 512
scale = 2

np.random.seed(os.getpid() + int(time.time()))
random.seed(os.getpid() + int(time.time()))

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
    p = np.random.uniform(0., 1.)

    if p < 0.25:
        sigma = np.random.uniform(0., 1.75)
        return gaussian_filter(images, sigma=sigma)
    if p < 0.5:
        sigma = np.random.uniform(0., 6.)
        gauss = gaussian_filter(images, sigma=sigma)
        gain = np.random.uniform(0., 5.)
        return (1 + gain) * images - gain * gauss
    return images

def is_Font_match(font, target):
    if target.startswith('hand'):
        return font.startswith('hand')
    else:
        return font == target 

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
            progress = tf.keras.utils.Progbar(total, unit_name='item')
            dicts = pool.imap_unordered(sub_load, items)
            for dictitem in dicts:
                self.img_cache.update(dictitem)
                progress.add(1)

        print('loading jpfont')
        jpfont_files = sorted(glob.glob('data/jpfont/*.ttf') + glob.glob('data/jpfont/*.otf'))
        items = [(f, list(self.glyphs.values())) for f in jpfont_files]
        total = len(jpfont_files)
        with Pool() as pool:
            progress = tf.keras.utils.Progbar(total, unit_name='item')
            dicts = pool.imap_unordered(sub_load, items)
            for dictitem in dicts:
                self.img_cache.update(dictitem)
                progress.add(1)

        type_count_max = max([self.glyph_type[k] for k in self.glyph_type]) + 1
        for key in self.img_cache:
            i = self.glyph_id[key[1]]
            if i not in self.glyph_type:
                self.glyph_type[i] = type_count_max

        type_count_max = max([self.glyph_type[k] for k in self.glyph_type]) + 1
        gtype_count = [0 for _ in range(type_count_max)] 
        type_count = [0 for _ in range(type_count_max)]

        for key in self.img_cache:
            t = self.glyph_type[self.glyph_id[key[1]]]
            type_count[t] += 1
        for k in self.glyph_type:
            gtype_count[self.glyph_type[k]] += 1

        self.image_keys = list(self.img_cache.keys())
        self.test_keys = self.get_test_keys()
        self.train_keys = self.get_train_keys()

        #                  0    1    2    3    4    5    6    7    8    9   10   11
        self.prob_map = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.1, 1.0]
        self.prob_map = [p/t for p,t in zip(self.prob_map, type_count)]
        self.random_probs_train = [self.prob_map[self.glyph_type[self.glyph_id[key[1]]]] for key in self.train_keys]
        self.random_probs_test = [self.prob_map[self.glyph_type[self.glyph_id[key[1]]]] for key in self.test_keys]

        #                      0  1  2  3  4    5  6  7    8    9   10  11
        self.prob_map_kanji = [0, 0, 0, 0, 0, 1.0, 0, 0, 1.0, 1.0, 0.5, 0]
        self.prob_map_kanji = [p/t for p,t in zip(self.prob_map_kanji, type_count)]
        self.kanji_probs_train = [self.prob_map_kanji[self.glyph_type[self.glyph_id[key[1]]]] for key in self.train_keys]
        self.kanji_probs_test = [self.prob_map_kanji[self.glyph_type[self.glyph_id[key[1]]]] for key in self.test_keys]

        #                      0  1  2  3  4  5  6  7  8  9 10 11
        self.prob_map_num = [1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.prob_map_num = [p/t for p,t in zip(self.prob_map_num, type_count)]
        self.num_probs_train = [self.prob_map_num[self.glyph_type[self.glyph_id[key[1]]]] for key in self.train_keys]
        self.num_probs_test = [self.prob_map_num[self.glyph_type[self.glyph_id[key[1]]]] for key in self.test_keys]

        #                      0    1    2  3  4  5  6  7  8  9 10 11
        self.prob_map_alpha = [0, 1.0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.prob_map_alpha = [p/t for p,t in zip(self.prob_map_alpha, type_count)]
        self.alpha_probs_train = [self.prob_map_alpha[self.glyph_type[self.glyph_id[key[1]]]] for key in self.train_keys]
        self.alpha_probs_test = [self.prob_map_alpha[self.glyph_type[self.glyph_id[key[1]]]] for key in self.test_keys]

        #                     0  1  2    3  4  5  6  7  8  9 10 11
        self.prob_map_hira = [0, 0, 0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.prob_map_hira = [p/t for p,t in zip(self.prob_map_hira, type_count)]
        self.hira_probs_train = [self.prob_map_hira[self.glyph_type[self.glyph_id[key[1]]]] for key in self.train_keys]
        self.hira_probs_test = [self.prob_map_hira[self.glyph_type[self.glyph_id[key[1]]]] for key in self.test_keys]


        self.train_keys_num = [x for x in self.train_keys if self.glyph_type[self.glyph_id[x[1]]] == 0]
        self.train_num_fonts = list(set([key[0] for key in self.train_keys_num]))
        self.test_keys_num = [x for x in self.test_keys if self.glyph_type[self.glyph_id[x[1]]] == 0]
        self.test_num_fonts = list(set([key[0] for key in self.test_keys_num]))

        self.train_keys_capital = [x for x in self.train_keys if self.glyph_type[self.glyph_id[x[1]]] == 1]
        self.train_capital_fonts = list(set([key[0] for key in self.train_keys_capital]))
        self.test_keys_capital = [x for x in self.test_keys if self.glyph_type[self.glyph_id[x[1]]] == 1]
        self.test_capital_fonts = list(set([key[0] for key in self.test_keys_capital]))

        self.train_keys_small = [x for x in self.train_keys if self.glyph_type[self.glyph_id[x[1]]] == 2]
        self.train_small_fonts = list(set([key[0] for key in self.train_keys_small]))
        self.test_keys_small = [x for x in self.test_keys if self.glyph_type[self.glyph_id[x[1]]] == 2]
        self.test_small_fonts = list(set([key[0] for key in self.test_keys_small]))

        self.train_keys_alpha = [x for x in self.train_keys if self.glyph_type[self.glyph_id[x[1]]] in [0,1,2,6]]
        self.train_alpha_fonts = list(set([key[0] for key in self.train_keys_alpha]))
        self.test_keys_alpha = [x for x in self.test_keys if self.glyph_type[self.glyph_id[x[1]]] in [0,1,2,6]]
        self.test_alpha_fonts = list(set([key[0] for key in self.test_keys_alpha]))


        self.train_keys_jp = [x for x in self.train_keys if self.glyph_type[self.glyph_id[x[1]]] in [3,4,5,7,8,9]]
        self.test_keys_jp = [x for x in self.test_keys if self.glyph_type[self.glyph_id[x[1]]] in [3,4,5,7,8,9]]
        self.train_jp_fonts = list(set([key[0] for key in self.train_keys_jp]))
        p_sum = sum([0 if '.' in f else 1 for f in self.train_jp_fonts])
        self.train_jp_fonts_p = [1. if '.' in f else 1/p_sum for f in self.train_jp_fonts]
        self.test_jp_fonts = list(set([key[0] for key in self.test_keys_jp]))
        p_sum = sum([0 if '.' in f else 1 for f in self.test_jp_fonts])
        self.test_jp_fonts_p = [1. if '.' in f else 1/p_sum for f in self.test_jp_fonts]

        self.train_keys_hira = [x for x in self.train_keys if self.glyph_type[self.glyph_id[x[1]]] in [3,4]]
        self.test_keys_hira = [x for x in self.test_keys if self.glyph_type[self.glyph_id[x[1]]] in [3,4]]
        self.train_hira_fonts = list(set([key[0] for key in self.train_keys_hira]))
        p_sum = sum([0 if '.' in f else 1 for f in self.train_hira_fonts])
        self.train_hira_fonts_p = [1. if '.' in f else 1/p_sum for f in self.train_hira_fonts]
        self.test_hira_fonts = list(set([key[0] for key in self.test_keys_hira]))
        p_sum = sum([0 if '.' in f else 1 for f in self.test_hira_fonts])
        self.test_hira_fonts_p = [1. if '.' in f else 1/p_sum for f in self.test_hira_fonts]

        self.train_keys_jpnum = [x for x in self.train_keys if (self.glyph_type[self.glyph_id[x[1]]] in [0,3,4,5,7]) and (x[0] in self.train_jp_fonts)]
        self.test_keys_jpnum  = [x for x in self.test_keys if (self.glyph_type[self.glyph_id[x[1]]] in [0,3,4,5,7]) and (x[0] in self.test_jp_fonts)]
        self.train_jpnum_fonts = list(set([key[0] for key in self.train_keys_jpnum]))
        self.train_jpnum_fonts_p = [1. if '.' in f else 0. for f in self.train_jpnum_fonts]
        self.test_jpnum_fonts = list(set([key[0] for key in self.test_keys_jpnum]))
        self.test_jpnum_fonts_p = [1. if '.' in f else 0. for f in self.test_jpnum_fonts]

        self.prob_map_clustering = [
                gtype_count[0] / type_count[0],
                gtype_count[1] / type_count[1],
                gtype_count[2] / type_count[2],
                gtype_count[3] / type_count[3],
                gtype_count[4] / type_count[4],
                gtype_count[5] / type_count[5],
                gtype_count[6] / type_count[6],
                0.,
                0.,
                0.,
                0.,
                0.
                ]

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
        def remove_transparency(im, bg_colour=(255, 255, 255)):
            # Only process if image has transparency (http://stackoverflow.com/a/1963146)
            if im.mode in ('RGBA', 'LA') or (im.mode == 'P' and 'transparency' in im.info):
                # Need to convert to RGBA if LA format due to a bug in PIL (http://stackoverflow.com/a/1963146)
                alpha = im.convert('RGBA').getchannel('A')

                # Create a new background image of our matt color.
                # Must be RGBA because paste requires both images have the same format
                # (http://stackoverflow.com/a/8720632  and  http://stackoverflow.com/a/9459208)
                bg = Image.new("RGBA", im.size, bg_colour + (255,))
                bg.paste(im, mask=alpha)
                return bg
            else:
                return im        
        im_file = random.choice(self.random_background)
        im = Image.open(im_file)
        im = remove_transparency(im).convert('RGB')
        scale_min = max(width / im.width, height / im.height)
        scale_max = max(scale_min + 0.5, 1.5)
        s = np.random.uniform(scale_min, scale_max)
        im = im.resize((int(im.width * s)+1, int(im.height * s)+1))
        x1 = np.random.randint(0, im.width - width)
        y1 = np.random.randint(0, im.height - height)
        im_crop = im.crop((x1, y1, x1 + width, y1 + height))

        img = np.asarray(im_crop).astype(np.float32)
        img = img / 128. - 1.
        if np.random.uniform() < 0.5:
            img = img[::-1,:,:]
        if np.random.uniform() < 0.5:
            img = img[:,::-1,:]
        brightness = np.random.uniform(-1.0, 1.0)
        brightness = np.array([brightness,brightness,brightness])        
        img += brightness[None,None,:]
        contrast = np.random.uniform(0.2, 1.8)
        contrast = np.array([contrast,contrast,contrast])
        img = img * contrast[None,None,:]  

        img = np.clip(img, -1.0, 1.0)

        return img

    def tateyokotext_images(self, keys, fonts, font_p):
        max_count = 256
        angle_max = 15.0

        min_pixel = 16
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

        select_font = random.choices(fonts, k=1, weights=font_p)[0]
        probs = [1. if is_Font_match(key[0], select_font) else 0. for key in keys]
        selection = [key for key in random.choices(keys, k=max_count, weights=probs)]
        probs2 = [1. if is_Font_match(key[0], select_font) and self.glyph_type[self.glyph_id[key[1]]] == 0 else 0. for key in keys]
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

        images = np.asarray(im) / 255.
        ids = np.asarray(idsim)

        images = apply_random_filter(images)

        return self.sub_constructimage(images, labels, ids, False)

    def yoko_images(self, keys, fonts, font_p):
        max_count = 256
        angle_max = 15.0

        min_pixel = 16
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

        select_font = random.choices(fonts, k=1, weights=font_p)[0]
        probs = [1. if is_Font_match(key[0], select_font) else 0. for key in keys]
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
                        top = int(np.clip(base_line - horiBearingY, 0, height - h))
                        left = int(np.clip(line_start + horiBearingX, 0, width - w))
                        im = np.asarray(Image.fromarray(item['image']).resize((w,h)))
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

        images = np.asarray(im) / 255.
        ids = np.asarray(idsim)

        images = apply_random_filter(images)

        return self.sub_constructimage(images, labels, ids, False)

    def tate_images(self, keys, fonts, font_p):
        max_count = 256
        angle_max = 15.0

        min_pixel = 16
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

        select_font = random.choices(fonts, k=1, weights=font_p)[0]
        probs = [1. if is_Font_match(key[0], select_font) else 0. for key in keys]
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

        images = np.asarray(im) / 255.
        ids = np.asarray(idsim)

        images = apply_random_filter(images)

        return self.sub_constructimage(images, labels, ids, False)

    def tatefurigana_images(self, keys, fonts, font_p):
        max_count = 256
        angle_max = 15.0

        min_pixel = 12
        max_pixel = 50
        text_size = random.randint(min_pixel, max_pixel)
        text_size2 = text_size * 2

        line_space = int(text_size2 * random.uniform(1.45, 1.7))
        block_count = 2
        line_break = int(random.uniform(0.3,0.7) * height)
        break_space = text_size2 * random.uniform(0.6, 1.0)

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

        select_font = random.choices(fonts, k=1, weights=font_p)[0]
        probs = [1. if is_Font_match(key[0], select_font) else 0. for key in keys]
        selection = [key for key in random.choices(keys, k=max_count, weights=probs)]
        probs2 = [1. if is_Font_match(key[0], select_font) and self.glyph_type[self.glyph_id[key[1]]] in [3,4] else 0. for key in keys]
        selection2 = iter([key for key in random.choices(keys, k=max_count*2, weights=probs2)])

        base_line = width - line_space + text_size2 // 2
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

            w = item['width'] / 128 * text_size2
            h = item['rows'] / 128 * text_size2
            vertBearingX = item['vertBearingX'] / 128 * text_size2
            vertBearingY = item['vertBearingY'] / 128 * text_size2
            vertAdvance = item['vertAdvance'] / 128 * text_size2

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
                    w = item['width'] / 128 * text_size2
                    h = item['rows'] / 128 * text_size2
                    vertBearingX = item['vertBearingX'] / 128 * text_size2
                    vertBearingY = item['vertBearingY'] / 128 * text_size2
                    vertAdvance = item['vertAdvance'] / 128 * text_size2

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

                        w = max(int(item['width'] / 128 * text_size2), 1)
                        h = max(int(item['rows'] / 128 * text_size2), 1)
                        l = int(np.clip(base_line + vertBearingX, 0, width - w))
                        t = int(np.clip(line_start + vertBearingY, 0, height - h))
                        im = np.asarray(Image.fromarray(item['image']).resize((w,h)))
                        images[t:t+h,l:l+w] = np.maximum(
                                images[t:t+h,l:l+w], 
                                im)

                    line_start += int(vertAdvance)

                # ふりがな処理
                base_line2 = base_line + text_size2 // 2 + text_size // 2
                line_start2 = int(max(0, 0 if block_count == 1 or block_no == 0 else line_break + break_space))
                if block_no == 0:
                    line_start2 += remain
                line_end2 = line_start
                line_start2p = line_start2
                while line_start2 < line_end2:
                    key2 = next(selection2, None)
                    if key2 is None:
                        break

                    item = self.img_cache[key2]['vertical']
                    if item['width'] * item['rows'] == 0:
                        continue

                    w = item['width'] / 128 * text_size
                    h = item['rows'] / 128 * text_size
                    vertBearingX = item['vertBearingX'] / 128 * text_size
                    vertBearingY = item['vertBearingY'] / 128 * text_size
                    vertAdvance = item['vertAdvance'] / 128 * text_size

                    if np.random.uniform() < 0.2:
                        # ここは空ける
                        if line_start2 != line_start2p:
                            draw.line(((base_line2 // scale, line_start2p // scale), 
                                (base_line2 // scale, line_start2 // scale)), fill=255, width=3)
                        
                        line_start2 += int(vertAdvance)
                        line_start2p = line_start2
                        continue

                    # ここはふりがな
                    l = (base_line2 + vertBearingX) / width
                    t = (line_start2 + vertBearingY) / height
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
                        l = int(np.clip(base_line2 + vertBearingX, 0, width - w))
                        t = int(np.clip(line_start2 + vertBearingY, 0, height - h))
                        im = np.asarray(Image.fromarray(item['image']).resize((w,h)))
                        images[t:t+h,l:l+w] = np.maximum(
                                images[t:t+h,l:l+w], 
                                im)

                    line_start2 += int(vertAdvance)

                if line_start2 != line_start2p:
                    draw.line(((base_line2 // scale, line_start2p // scale), 
                        (base_line2 // scale, line_start2 // scale)), fill=255, width=3)

                base_line -= line_space
                if base_line - text_size2 / 2 < 0:
                    if block_no == 0:
                        sep_end = base_line + line_space
                    base_line = width - line_space + text_size2 // 2
                    block_no += 1
                if block_no >= block_count:
                    break
                line_start = int(max(0, 0 if block_count == 1 or block_no == 0 else line_break + break_space))
                line_end = int(min(width, width if block_count == 1 or block_no == 1 else line_break - break_space))
                temp_lineend = line_start
                linebuf = []

        if all(t > 1 for t in text_count):
            l = max(1,line_break // scale)
            right = (width - line_space + text_size2 // 2) // scale
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

        images = np.asarray(im) / 255.
        ids = np.asarray(idsim)

        images = apply_random_filter(images)

        return self.sub_constructimage(images, labels, ids, False)

    def yokofurigana_images(self, keys, fonts, font_p):
        max_count = 256
        angle_max = 15.0

        min_pixel = 12
        max_pixel = 50
        text_size = random.randint(min_pixel, max_pixel)
        text_size2 = text_size * 2

        line_space = int(text_size2 * random.uniform(1.45, 1.7))
        block_count = 2
        line_break = int(random.uniform(0.3,0.7) * width)
        break_space = text_size2 * random.uniform(0.6, 1.5)

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

        select_font = random.choices(fonts, k=1, weights=font_p)[0]
        probs = [1. if is_Font_match(key[0], select_font) else 0. for key in keys]
        selection = [key for key in random.choices(keys, k=max_count, weights=probs)]
        probs2 = [1. if is_Font_match(key[0], select_font) and self.glyph_type[self.glyph_id[key[1]]] in [3,4] else 0. for key in keys]
        selection2 = iter([key for key in random.choices(keys, k=max_count*2, weights=probs2)])

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

            w = item['width'] / 128 * text_size2
            h = item['rows'] / 128 * text_size2
            horiBearingX = item['horiBearingX'] / 128 * text_size2
            horiBearingY = item['horiBearingY'] / 128 * text_size2
            horiAdvance = item['horiAdvance'] / 128 * text_size2

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
                    w = item['width'] / 128 * text_size2
                    h = item['rows'] / 128 * text_size2
                    horiBearingX = item['horiBearingX'] / 128 * text_size2
                    horiBearingY = item['horiBearingY'] / 128 * text_size2
                    horiAdvance = item['horiAdvance'] / 128 * text_size2

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

                        w = max(int(item['width'] / 128 * text_size2), 1)
                        h = max(int(item['rows'] / 128 * text_size2), 1)
                        top = int(np.clip(base_line - horiBearingY, 0, height - h))
                        left = int(np.clip(line_start + horiBearingX, 0, width - w))
                        im = np.asarray(Image.fromarray(item['image']).resize((w,h)))
                        images[top:top+h,left:left+w] = np.maximum(
                                images[top:top+h,left:left+w], 
                                im)

                    line_start += int(horiAdvance)

                # ふりがな処理
                base_line2 = base_line - text_size2
                line_start2 = int(max(0, 0 if block_count == 1 or block_no == 0 else line_break + break_space))
                if block_no == 0:
                    line_start2 += remain
                line_end2 = line_start
                line_start2p = line_start2
                while line_start2 < line_end2:
                    key2 = next(selection2, None)
                    if key2 is None:
                        break

                    item = self.img_cache[key2]['horizontal']
                    if item['width'] * item['rows'] == 0:
                        continue

                    w = item['width'] / 128 * text_size
                    h = item['rows'] / 128 * text_size
                    horiBearingX = item['horiBearingX'] / 128 * text_size
                    horiBearingY = item['horiBearingY'] / 128 * text_size
                    horiAdvance = item['horiAdvance'] / 128 * text_size

                    if np.random.uniform() < 0.2:
                        # ここは空ける
                        if line_start2 != line_start2p:
                            draw.line(((line_start2p // scale, base_line // scale), 
                                (line_start // scale, base_line // scale)), fill=255, width=3)
                        
                        line_start2 += int(horiAdvance)
                        line_start2p = line_start2
                        continue

                    # ここはふりがな
                    l = (line_start2 + horiBearingX) / width
                    t = (base_line2 - horiBearingY) / height
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
                        top = int(np.clip(base_line2 - horiBearingY, 0, height - h))
                        left = int(np.clip(line_start2 + horiBearingX, 0, width - w))
                        im = np.asarray(Image.fromarray(item['image']).resize((w,h)))
                        images[top:top+h,left:left+w] = np.maximum(
                                images[top:top+h,left:left+w], 
                                im)

                    line_start2 += int(horiAdvance)

                if line_start2 != line_start2p:
                    draw.line(((line_start2p // scale, base_line // scale), 
                        (line_start // scale, base_line // scale)), fill=255, width=3)

                base_line += line_space
                if base_line + text_size2 >= height:
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

        images = np.asarray(im) / 255.
        ids = np.asarray(idsim)

        images = apply_random_filter(images)

        return self.sub_constructimage(images, labels, ids, False)

    def null_images(self):
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

        images = apply_random_filter(images)

        return self.sub_constructimage(images, labels, ids, len(self.random_background) > 0)


    def load_random_line(self):
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

        draw1.line(((x1 // scale, y1 // scale), (x2 // scale, y2 // scale)), fill=255, width=linew//scale+1)
        draw2.line(((x1, y1), (x2, y2)), fill=255, width=linew)

        labels = np.stack([keymap, xsizes, ysizes, offsetx, offsety, lines, np.asarray(seps) / 255.], -1)
        ids = np.zeros([height // scale, width // scale], dtype=np.int32)
        images = np.asarray(images) / 255.

        images = apply_random_filter(images)

        return self.sub_constructimage(images, labels, ids, len(self.random_background) > 0)

    def load_images_random(self, keys, probs):
        max_count = 64
        angle_max = 15.0

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
                seps[py:py+lw//scale+1, :px] = 1
                images[y:y+lw, :x] = 255
                boxprev = np.concatenate([boxprev, [[0, (x + margin)/width, (y - margin)/height, (y+lw + margin)/height]]])
            elif line_c == 1:
                x = random.randrange(0, width // 2)
                y = random.randrange(0, height - lw)
                px = x // scale
                py = y // scale
                seps[py:py+lw//scale+1, px:] = 1
                images[y:y+lw, x:] = 255
                boxprev = np.concatenate([boxprev, [[(x - margin)/width, 1, (y - margin)/height, (y+lw + margin)/height]]])
            elif line_c == 2:
                y = random.randrange(height // 2, height)
                x = random.randrange(0, width - lw)
                px = x // scale
                py = y // scale
                seps[:py, px:px+lw//scale+1] = 1
                images[:y, x:x+lw] = 255
                boxprev = np.concatenate([boxprev, [[(x - margin)/width, (x+lw + margin)/width, 0, (y + margin)/height]]])
            elif line_c == 3:
                y = random.randrange(0, height // 2)
                x = random.randrange(0, width - lw)
                px = x // scale
                py = y // scale
                seps[py:, px:px+lw//scale+1] = 1
                images[y:, x:x+lw] = 255
                boxprev = np.concatenate([boxprev, [[(x - margin)/width, (x+lw + margin)/width, (y - margin)/height, 1]]])

        if random.random() < 0.5:
            min_pixel = 20
            max_pixel = width
        else:
            min_pixel = 20
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

        images = np.asarray(im) / 255.
        ids = np.asarray(idsim)

        images = apply_random_filter(images)

        return self.sub_constructimage(images, labels, ids, len(self.random_background) > 0)

    def load_images_fill(self, keys, fonts):
        max_count = 64
        angle_max = 15.0

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
        probs = [1. if is_Font_match(key[0], select_font) else 0. for key in keys]
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
                seps[py:py+lw//scale+1, :px] = 1
                images[y:y+lw, :x] = 255
                boxprev = np.concatenate([boxprev, [[0, (x + margin)/width, (y - margin)/height, (y+lw + margin)/height]]])
            elif line_c == 1:
                x = random.randrange(0, width // 2)
                y = random.randrange(0, height - lw)
                px = x // scale
                py = y // scale
                seps[py:py+lw//scale+1, px:] = 1
                images[y:y+lw, x:] = 255
                boxprev = np.concatenate([boxprev, [[(x - margin)/width, 1, (y - margin)/height, (y+lw + margin)/height]]])
            elif line_c == 2:
                y = random.randrange(height // 2, height)
                x = random.randrange(0, width - lw)
                px = x // scale
                py = y // scale
                seps[:py, px:px+lw//scale+1] = 1
                images[:y, x:x+lw] = 255
                boxprev = np.concatenate([boxprev, [[(x - margin)/width, (x+lw + margin)/width, 0, (y + margin)/height]]])
            elif line_c == 3:
                y = random.randrange(0, height // 2)
                x = random.randrange(0, width - lw)
                px = x // scale
                py = y // scale
                seps[py:, px:px+lw//scale+1] = 1
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

        images = np.asarray(im) / 255.
        ids = np.asarray(idsim)

        images = apply_random_filter(images)

        return self.sub_constructimage(images, labels, ids, len(self.random_background) > 0)


    def load_images_randomline(self, keys, probs):
        max_count = 64
        angle_max = 15.0

        min_pixel = 20
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
                seps[py:py+lw//scale+1, :px] = 1
                images[y:y+lw, :x] = 255
                boxprev = np.concatenate([boxprev, [[0, (x + margin)/width, (y - margin)/height, (y+lw + margin)/height]]])
            elif line_c == 1:
                x = random.randrange(0, width // 2)
                y = random.randrange(0, height - lw)
                px = x // scale
                py = y // scale
                seps[py:py+lw//scale+1, px:] = 1
                images[y:y+lw, x:] = 255
                boxprev = np.concatenate([boxprev, [[(x - margin)/width, 1, (y - margin)/height, (y+lw + margin)/height]]])
            elif line_c == 2:
                y = random.randrange(height // 2, height)
                x = random.randrange(0, width - lw)
                px = x // scale
                py = y // scale
                seps[:py, px:px+lw//scale+1] = 1
                images[:y, x:x+lw] = 255
                boxprev = np.concatenate([boxprev, [[(x - margin)/width, (x+lw + margin)/width, 0, (y + margin)/height]]])
            elif line_c == 3:
                y = random.randrange(0, height // 2)
                x = random.randrange(0, width - lw)
                px = x // scale
                py = y // scale
                seps[py:, px:px+lw//scale+1] = 1
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
                    if p < 0.4:
                        direction = 0
                    elif p < 0.8:
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

        images = np.asarray(im) / 255.
        ids = np.asarray(idsim)

        images = apply_random_filter(images)

        return self.sub_constructimage(images, labels, ids, len(self.random_background) > 0)

    def sub_constructimage(self, images, labels, ids, bk):

        h, w = images.shape
        img = images[...,None]

        c = np.random.uniform(0., 1.)
        if c < 0.25:
            fg_c = np.random.uniform(-1., 1.)
            bk_c = np.random.uniform(-1., 1.)
            if abs(fg_c - bk_c) < 0.25:
                d = fg_c - bk_c
                if d < 0:
                    d = -0.25 - d
                else:
                    d = 0.25 - d
                fg_c += d
                bk_c -= d
            fg_c = np.array([fg_c, fg_c, fg_c])
            bk_c = np.array([bk_c, bk_c, bk_c])
            fg_c += np.random.uniform(-1., 1., [3]) * 0.1
            bk_c += np.random.uniform(-1., 1., [3]) * 0.1
            fgimg = fg_c[None,None,:]
            bkimg = bk_c[None,None,:]
        elif c < 0.5:
            fg1_c = np.random.uniform(-1., 1., [3])
            fg2_c = np.random.uniform(-1., 1., [3])
            bk_c = np.random.uniform(-1., 1., [3])
            m1 = np.min(np.abs(fg1_c - bk_c))
            m2 = np.min(np.abs(fg1_c - bk_c))
            m = min(m1, m2)
            if m < 0.25:
                ind = np.argmin(np.abs(fg1_c - bk_c))
                d1 = (fg1_c - bk_c)[ind]
                ind = np.argmin(np.abs(fg2_c - bk_c))
                d2 = (fg2_c - bk_c)[ind]
                if abs(d1) > abs(d2):
                    d = d2
                else:
                    d = d1
                if d < 0:
                    d = -0.25 - d
                else:
                    d = 0.25 - d
                fg1_c += d
                fg2_c += d
                bk_c -= d
            x1 = np.random.randint(0, w)
            y1 = np.random.randint(0, h)

            fgimg = fg1_c[None,None,:] * np.ones([h, w, 3])
            fgimg[:y1,:x1,:] = fg2_c[None,None,:]
            bkimg = bk_c[None,None,:]
        elif not bk or c < 0.75:
            fg_c = np.random.uniform(-1., 1., [3])
            bk_c = np.random.uniform(-1., 1., [3])
            if np.min(np.abs(fg_c - bk_c)) < 0.25:
                ind = ind = np.argmin(np.abs(fg_c - bk_c))
                d = (fg_c - bk_c)[ind]
                if d < 0:
                    d = -0.25 - d
                else:
                    d = 0.25 - d
                fg_c += d
                bk_c -= d
            fgimg = fg_c[None,None,:]
            bkimg = bk_c[None,None,:]
        else:
            bkimg = self.load_background_images()
            bk_c = np.mean(bkimg, axis=(0,1))
            bk_std = np.std(bkimg, axis=(0,1))
            fg_c = np.where(
                bk_c > 0, 
                np.random.uniform(np.clip(bk_c - bk_std * 2 - 0.25, None, -1.), bk_c - bk_std * 2 - 0.25, [3]),
                np.random.uniform(bk_c + bk_std * 2 + 0.25, np.clip(bk_c + bk_std * 2 + 0.25, 1., None), [3]))
            bk_alpha = max(np.max(np.abs(fg_c)), 1)
            bkimg /= bk_alpha
            fg_c /= bk_alpha
            fg_c = np.clip(fg_c, -1.0, 1.0)
            fgimg = fg_c[tf.newaxis,tf.newaxis,:]

        bkimg = np.clip(bkimg, -1.0, 1.0)

        image = fgimg * img + bkimg * (1 - img)
        image = np.clip(image, -1.0, 1.0)

        noise_v = np.random.normal() * 0.1
        if noise_v > 0:
            noise = np.where(
                np.random.uniform(size=image.shape) > noise_v,
                0.,
                np.random.normal(size=image.shape))
            image += noise
        image = np.clip(image, -1.0, 1.0)

        image = image * 127 + 127

        return image, labels, ids

    def prob_images_train(self):
        def num_func():
            return self.load_images_fill(self.train_keys_num, self.train_num_fonts)
        
        def capital_func():
            return self.load_images_fill(self.train_keys_capital, self.train_capital_fonts)
        
        def small_func():
            return self.load_images_fill(self.train_keys_small, self.train_small_fonts)
        
        def alpha_func():
            return self.load_images_fill(self.train_keys_alpha, self.train_alpha_fonts)
        
        def random_func():
            return self.load_images_random(self.train_keys, self.random_probs_train)

        def random_kanji_func():
            return self.load_images_random(self.train_keys, self.kanji_probs_train)

        def renderling_func():
            if np.random.uniform() < 0.5:
                return self.tate_images(self.train_keys_jp, self.train_jp_fonts, self.train_jp_fonts_p)
            else:
                return self.yoko_images(self.train_keys_jp, self.train_jp_fonts, self.train_jp_fonts_p)

        def renderling_furigana_func():
            if np.random.uniform() < 0.5:
                return self.tatefurigana_images(self.train_keys_jp, self.train_jp_fonts, self.train_jp_fonts_p)
            else:
                return self.yokofurigana_images(self.train_keys_jp, self.train_jp_fonts, self.train_jp_fonts_p)

        def renderling2_func():
            return self.tateyokotext_images(self.train_keys_jpnum, self.train_jpnum_fonts, self.train_jpnum_fonts_p)

        def randomline_func():
            return self.load_images_randomline(self.train_keys, self.random_probs_train)

        def line_func():
            return self.load_random_line()

        def null_func():
            return self.null_images()

        funcs = [
            num_func, capital_func, small_func, alpha_func, randomline_func,
            random_func, random_kanji_func, renderling_func, renderling_furigana_func, renderling2_func,
            line_func, null_func,
        ]
        
        return funcs

    def prob_images_test(self):
        def num_func():
            return self.load_images_fill(self.test_keys_num, self.test_num_fonts)
        
        def capital_func():
            return self.load_images_fill(self.test_keys_capital, self.test_capital_fonts)
        
        def small_func():
            return self.load_images_fill(self.test_keys_small, self.test_small_fonts)
        
        def alpha_func():
            return self.load_images_fill(self.test_keys_alpha, self.test_alpha_fonts)
        
        def random_func():
            return self.load_images_random(self.test_keys, self.random_probs_test)

        def random_kanji_func():
            return self.load_images_random(self.test_keys, self.kanji_probs_test)

        def renderling_func():
            if np.random.uniform() < 0.5:
                return self.tate_images(self.test_keys_jp, self.test_jp_fonts, self.test_jp_fonts_p)
            else:
                return self.yoko_images(self.test_keys_jp, self.test_jp_fonts, self.test_jp_fonts_p)

        def renderling_furigana_func():
            if np.random.uniform() < 0.5:
                return self.tatefurigana_images(self.test_keys_jp, self.test_jp_fonts, self.test_jp_fonts_p)
            else:
                return self.yokofurigana_images(self.test_keys_jp, self.test_jp_fonts, self.test_jp_fonts_p)

        def renderling2_func():
            return self.tateyokotext_images(self.test_keys_jpnum, self.test_jpnum_fonts, self.test_jpnum_fonts_p)

        def randomline_func():
            return self.load_images_randomline(self.test_keys, self.random_probs_test)

        def line_func():
            return self.load_random_line()

        def null_func():
            return self.null_images()

        funcs = [
            num_func, capital_func, small_func, alpha_func, randomline_func,
            random_func, random_kanji_func, renderling_func, renderling_furigana_func, renderling2_func,
            line_func, null_func,
        ]
        
        return funcs

    def cluster_data(self, batch_size):
        keys = self.image_keys
        probs = [self.prob_map_clustering[self.glyph_type[self.glyph_id[key[1]]]] for key in keys]

        def random_func():
            return self.load_images_random(keys, probs)

        @tf.function
        def callfunc(i):
            images, labels, ids = tf.py_function(
                func=random_func, 
                inp=[], 
                Tout=[tf.float32, tf.float32, tf.int32])
            images = tf.ensure_shape(images, [height, width, 3])
            labels = tf.ensure_shape(labels, [height // scale, width // scale, 7])
            ids = tf.ensure_shape(ids, [height // scale, width // scale])
            return images, labels, ids

        ds = tf.data.Dataset.range(1)
        ds = ds.repeat()
        ds = ds.map(callfunc, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)
        ds = ds.batch(batch_size, drop_remainder=True, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)
        ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds

    def test_data(self, batch_size):
        return self.loader(batch_size, False)

    def train_data(self, batch_size):
        return self.loader(batch_size, True)

    def loader(self, batch_size, training):

        if training:
            funcs = self.prob_images_train()
        else:
            funcs = self.prob_images_test()

        @tf.autograph.experimental.do_not_convert
        def getfunc(i):
            return funcs[i]()

        @tf.function
        def callfunc(i):
            images, labels, ids = tf.py_function(
                func=getfunc, 
                inp=[i], 
                Tout=[tf.float32, tf.float32, tf.int32])
            images = tf.ensure_shape(images, [height, width, 3])
            labels = tf.ensure_shape(labels, [height // scale, width // scale, 7])
            ids = tf.ensure_shape(ids, [height // scale, width // scale])
            return images, labels, ids

        # 0 num_func, 1 capital_func, 2 small_func, 3 alpha_func, 4 randomline_func,
        # 5 random_func, 6 random_kanji_func, 7 renderling_func, 8 renderling_furigana_func, 9 renderling2_func,
        # 10 line_func, 11 null_func,

        #                      0   1   2   3   4   5   6   7   8   9   10   11
        prob = tf.math.log([[1.0,1.0,1.0,1.0,3.5,3.5,1.5,1.0,1.0,1.0,0.01,0.01]])

        ds = tf.data.Dataset.range(1)
        ds = ds.repeat()
        ds = ds.map(lambda x: tf.squeeze(tf.random.categorical(prob, 1)))
        ds = ds.map(callfunc, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)
        ds = ds.batch(batch_size, drop_remainder=True, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)
        ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    data = FontData()

    ds = data.test_data(8, 100)
    ds = data.train_data(8, 100)

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
            
            
