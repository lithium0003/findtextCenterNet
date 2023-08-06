import numpy as np
from PIL import Image, ImageDraw
from PIL.Image import Resampling
import glob
import os
import re
import random

from .renderer import UNICODE_WHITESPACE_CHARACTERS 

handwrite_dir = os.path.join('data','handwritten')

def load_handwrite(path):
    dirnames = glob.glob(os.path.join(path, '*'))
    ret = {}
    for d in dirnames:
        c_code = os.path.basename(d)
        char = str(bytes.fromhex(c_code), 'utf-8')
        hori_images = []
        for f in glob.glob(os.path.join(d, '*.png')):
            rawim = np.asarray(Image.open(f).convert('L'))
            img = 255 - rawim

            ylines = np.any(rawim < 255, axis=1)
            content = np.where(ylines)[0]
            top = content[0]
            bottom = content[-1]

            xlines = np.any(rawim < 255, axis=0)
            content = np.where(xlines)[0]
            left = content[0]
            right = content[-1]
            hori_images.append({
                'image': img,
                'left': left,
                'right': right,
                'top': top,
                'bottom': bottom,
            })

        vert_images = []
        for f in glob.glob(os.path.join(d, 'vert', '*.png')):
            rawim = np.asarray(Image.open(f).convert('L'))
            img = 255 - rawim

            ylines = np.any(rawim < 255, axis=1)
            content = np.where(ylines)[0]
            top = content[0]
            bottom = content[-1]

            xlines = np.any(rawim < 255, axis=0)
            content = np.where(xlines)[0]
            left = content[0]
            right = content[-1]
            vert_images.append({
                'image': img,
                'left': left,
                'right': right,
                'top': top,
                'bottom': bottom,
            })
        
        ret[char] = {
            'hori': hori_images,
            'vert': vert_images,
        }
    return ret

hand_images = load_handwrite(handwrite_dir)


class HandwriteCanvas:
    def __init__(self, ratio=1.0, horizontal=True):
        self.ratio = ratio
        self.line_space_ratio = 1.5
        self.fontsize = 128 * ratio

        self.start_pad = self.fontsize * 1.5
        self.end_pad = self.fontsize * 1.5
        self.start_linepad = self.fontsize * 1.5
        self.end_linepad = self.fontsize * 1.5
        self.linewidth = 0
        self.linecount = 0
        self.linecount_max = 0
        self.is_horizontal = horizontal

        self.canvas_image = None
        self.cur_pos = 0
        self.baseline_pos = 0
        self.position = np.zeros([0,4]) # cx, cy, w, h
        self.code_list = np.zeros([0,2], dtype=int) # utf-32 code, char type
        # char type
        # 1          ruby text
        #  2         ruby base
        #   4        emphasis
        #    8       space
        #     16     CR
        self.process_str = ''

    def set_linewidth(self, linewidth):
        self.linewidth = linewidth

    def set_linemax(self, maxline):
        self.linecount_max = maxline

    def _prepare_image(self):
        if self.canvas_image is None:
            if self.linecount_max > 0:
                lines = self.linecount_max
            else:
                lines = 10
            if self.is_horizontal:
                width = self.start_pad + self.linewidth + self.end_pad
                height = self.start_linepad + self.fontsize + self.fontsize * self.line_space_ratio * (lines - 1) + self.end_linepad
            else:
                height = self.start_pad + self.linewidth + self.end_pad
                width = self.start_linepad + self.fontsize + self.fontsize * self.line_space_ratio * (lines - 1) + self.end_linepad
            self.canvas_image = np.zeros([int(height), int(width)], dtype='ubyte')
            self._go_startline()
            self._cur_home()

            self.textline_image = np.zeros_like(self.canvas_image, dtype='ubyte')

    def _go_startline(self):
        if self.is_horizontal:
            self.baseline_pos = self.start_linepad + self.fontsize
        else:
            self.baseline_pos = self.canvas_image.shape[1] - self.start_linepad - self.fontsize / 2

    def _cur_home(self):
        self.cur_pos = self.start_pad

    def _get_linewidth(self):
        return self.start_pad + self.linewidth

    def _process_newline(self):
        if self.linecount_max > 0 and self.linecount >= self.linecount_max - 1:
            self.process_str += '\n'
            self.position = np.concatenate([self.position, np.zeros([1,4])], axis=0)
            self.code_list = np.concatenate([self.code_list, np.array([[ord('\n'), 16]])], axis=0)

            return False

        if self.canvas_image is None:
            self._prepare_image()
        
        if self.linecount_max > 0:
            pass
        else:
            lines = 10

            pad = int(self.fontsize * self.line_space_ratio * lines)
            if self.is_horizontal:
                if self.canvas_image.shape[0] - self.end_linepad - self.baseline_pos < self.fontsize * self.line_space_ratio:
                    self.canvas_image = np.pad(self.canvas_image, [[0, pad],[0, 0]])
                    self.textline_image = np.pad(self.textline_image, [[0, pad],[0, 0]])
            else:
                if self.baseline_pos - self.fontsize / 2 - self.end_linepad < self.fontsize * self.line_space_ratio:
                    self.canvas_image = np.pad(self.canvas_image, [[0, 0],[pad, 0]])
                    self.textline_image = np.pad(self.textline_image, [[0, 0],[pad, 0]])
                    self.baseline_pos += pad
                    self.position[:,0] += pad

        if self.is_horizontal:
            self.baseline_pos += self.fontsize * self.line_space_ratio
        else:
            self.baseline_pos -= self.fontsize * self.line_space_ratio
        self._cur_home()

        self.process_str += '\n'
        self.position = np.concatenate([self.position, np.zeros([1,4])], axis=0)
        self.code_list = np.concatenate([self.code_list, np.array([[ord('\n'), 16]])], axis=0)
        self.linecount += 1
        return True

    def _process_result(self):
        position = self.position
        code_list = self.code_list
        opcode = code_list[:,1]
        prev_opcode = np.concatenate([[16], opcode[:-1]])
        rm_code = np.logical_or(opcode & (8+16) > 0, position[:,2] * position[:,3] <= 0)
        opcode &= ~(8 + 16)
        opcode |= (prev_opcode & (8 + 16))
        code_list[:,1] = opcode

        return position[np.logical_not(rm_code),:], code_list[np.logical_not(rm_code),:], np.zeros_like(self.canvas_image, dtype='ubyte')

    def draw(self, text):
        text = re.sub('\uFFF9(.*?)\uFFFA(.*?)\uFFFB', r'\1', text)
        
        self._prepare_image()

        isLive = True
        for line in text.splitlines():
            if not isLive:
                break;
            if len(line.strip()) == 0:
                if not self._process_newline():
                    break
                continue

            remain_str = line
            count = 0
            while len(remain_str) > 0:
                if self.linewidth > 0:
                    line_length = self._get_linewidth() - self.cur_pos
                else:
                    line_length = 0

                im_buf = self._draw_buffer(remain_str, horizontal=self.is_horizontal, line_length=line_length)
                if im_buf is None:
                    return None

                if len(im_buf['str']) == 0:
                    break
                remain_str = im_buf['remain_str']
                self.process_str += im_buf['str']
                image = im_buf['image']
                pos = im_buf['position']
                code_list = im_buf['code_list']

                if self.is_horizontal:
                    left = self.cur_pos + (im_buf['pad_left'] if count == 0 else 0)
                    top = self.baseline_pos - im_buf['base_line']
                    pos[:,0] += left
                    pos[:,1] += top
                    x1 = left
                    x2 = left + im_buf['next_cur'] - im_buf['pad_left']
                    y1 = self.baseline_pos
                    y2 = self.baseline_pos
                else:
                    left = self.baseline_pos - im_buf['base_line']
                    top = self.cur_pos + (im_buf['pad_top'] if count == 0 else 0)
                    pos[:,0] += left
                    pos[:,1] += top
                    x1 = self.baseline_pos
                    x2 = self.baseline_pos
                    y1 = top
                    y2 = top + im_buf['next_cur'] - im_buf['pad_top']
                
                if image is not None:
                    left = int(left)
                    top = int(top)
                    right = left + image.shape[1]
                    bottom = top + image.shape[0]

                    if self.canvas_image.shape[1] < right:
                        self.canvas_image = np.pad(self.canvas_image, [[0, 0],[0, right-self.canvas_image.shape[1]]])
                        self.textline_image = np.pad(self.textline_image, [[0, 0],[0, right-self.textline_image.shape[1]]])

                    if self.canvas_image.shape[0] < bottom:
                        self.canvas_image = np.pad(self.canvas_image, [[0, bottom-self.canvas_image.shape[0]],[0, 0]])
                        self.textline_image = np.pad(self.textline_image, [[0, bottom-self.textline_image.shape[0]],[0, 0]])

                    self.canvas_image[top:bottom, left:right] = np.maximum(self.canvas_image[top:bottom, left:right], image)

                self.position = np.concatenate([self.position, pos], axis=0)
                self.code_list = np.concatenate([self.code_list, code_list], axis=0)

                if im_buf['str'].strip() != '':
                    textline_image = Image.fromarray(self.textline_image)
                    draw = ImageDraw.Draw(textline_image)
                    draw.line(((x1, y1), (x2, y2)), fill=255, width=7)
                    self.textline_image = np.asarray(textline_image)

                if not self._process_newline():
                    isLive = False
                    break
                if remain_str.startswith(' '):
                    remain_str = remain_str[1:]
                count += 1

        position, code_list, sep_image = self._process_result()
        return {
            'image': self.canvas_image,
            'sep_image': sep_image,
            'textline_image': self.textline_image,
            'position': position,
            'code_list': code_list,
            'str': self.process_str,
        }

    def _get_character(self, c, horizontal=True):
        if c == '\n':
            return None

        if c in UNICODE_WHITESPACE_CHARACTERS:
            return {
                'image': None,
                'code': c,
                'left': 0,
                'right': 0,
                'top': 0,
                'bottom': 0,
            }

        if c in hand_images:
            if horizontal and 'hori' in hand_images[c]:
                return random.choice(hand_images[c]['hori'])
            elif not horizontal:
                if 'vert' in hand_images[c] and len(hand_images[c]['vert']) > 0:
                    return random.choice(hand_images[c]['vert'])
                elif 'hori' in hand_images[c]:
                    return random.choice(hand_images[c]['hori'])

        return None
        
    def _draw_buffer(self, text, horizontal=True, line_length=0):
        position = np.zeros([0,4])
        code_list = np.zeros([0,2], dtype=int)
        process_text = ''
        remain_str = text
        if horizontal:
            pad_left = 0
            image = None
            cur_x = 0
            base_line = 0
            for c in text:
                char_data = self._get_character(c, horizontal=True)
                if char_data is None:
                    remain_str = remain_str[1:]
                    continue
                
                im = char_data.get('image', None)
                if im is None:
                    next_cur = cur_x + self.fontsize
                    cx = (next_cur + cur_x) / 2
                    cy = base_line
                    w = next_cur - cur_x
                    h = 0
                    position = np.concatenate([position, np.array([[cx,cy,w,h]])], axis=0)
                    cur_x = int(next_cur)
                    process_text += c
                    code_list = np.concatenate([code_list, np.array([[ord(c), 8]])], axis=0)
                    remain_str = remain_str[1:]
                    continue

                im = np.array(Image.fromarray(im).resize((int(self.fontsize), int(self.fontsize)), resample=Resampling.BILINEAR))

                if image is None:
                    pad_left = cur_x
                    cur_x = 0
                    base_line = self.fontsize * 0.75
                    position[:,0] -= pad_left
                
                cur_x = int(cur_x)
                if cur_x < 0:
                    cur_x = 0

                left = cur_x + char_data['left'] * self.ratio
                right = cur_x + char_data['right'] * self.ratio
                next_cur = right + self.fontsize * (0.1 + 0.15 * np.random.normal())

                top = base_line - self.fontsize * 0.75 + char_data['top'] * self.ratio
                bottom = base_line - self.fontsize * 0.75 + char_data['bottom'] * self.ratio

                if line_length > 0 and right + pad_left > line_length:
                    break

                if image is None:
                    image = np.zeros(im.shape, dtype='ubyte')

                if image.shape[1] < cur_x + im.shape[1]:
                    image = np.pad(image, [[0, 0],[0, cur_x + im.shape[1]-image.shape[1]]])

                if image.shape[0] < im.shape[0]:
                    image = np.pad(image, [[0, im.shape[0]-image.shape[0]],[0,0]])

                w = right - left
                h = bottom - top
                cx = left + w / 2 - 1
                cy = top + h / 2 - 1

                image[:, cur_x:cur_x+im.shape[1]] = np.maximum(image[:, cur_x:cur_x+im.shape[1]], im)
                position = np.concatenate([position, np.array([[cx,cy,w,h]])], axis=0)
                cur_x = int(next_cur)
                process_text += c
                code_list = np.concatenate([code_list, np.array([[ord(c), 0]])], axis=0)
                remain_str = remain_str[1:]

            return {
                "image": image,
                "str": process_text,
                "remain_str": remain_str,
                "position": position,
                "code_list": code_list,
                "pad_left": pad_left,
                "base_line": base_line,
                "next_cur": int(cur_x + pad_left),
            }

        else:
            pad_top = 0
            image = None
            cur_y = 0
            base_line = 0
            for c in text:
                char_data = self._get_character(c, horizontal=False)
                if char_data is None:
                    remain_str = remain_str[1:]
                    continue

                im = char_data.get('image', None)
                if im is None:
                    next_cur = cur_y + self.fontsize
                    cy = (next_cur + cur_y) / 2
                    cx = base_line
                    h = next_cur - cur_y
                    w = 0
                    position = np.concatenate([position, np.array([[cx,cy,w,h]])], axis=0)
                    code_list = np.concatenate([code_list, np.array([[ord(c), 8]])], axis=0)
                    cur_y = int(next_cur)
                    process_text += c
                    remain_str = remain_str[1:]
                    continue

                im = np.array(Image.fromarray(im).resize((int(self.fontsize), int(self.fontsize)), resample=Resampling.BILINEAR))

                if image is None:
                    pad_top = cur_y
                    cur_y = 0
                    base_line = self.fontsize * 0.5
                    position[:,1] -= pad_top

                left = base_line - self.fontsize * 0.5 + char_data['left'] * self.ratio
                right = base_line - self.fontsize * 0.5 + char_data['right'] * self.ratio

                cur_y = int(cur_y)
                if cur_y < 0:
                    cur_y = 0
                top = cur_y + char_data['top'] * self.ratio
                bottom = cur_y + char_data['bottom'] * self.ratio
                next_cur = bottom + self.fontsize * (0.1 + 0.15 * np.random.normal())

                if line_length > 0 and bottom + pad_top > line_length:
                    break

                if image is None:
                    image = np.zeros(im.shape, dtype='ubyte')

                if image.shape[1] < im.shape[1]:
                    image = np.pad(image, [[0, 0],[0, im.shape[1]-image.shape[1]]])

                if image.shape[0] < cur_y + im.shape[0]:
                    image = np.pad(image, [[0, cur_y + im.shape[0]-image.shape[0]],[0,0]])

                w = right - left
                h = bottom - top
                cx = left + w / 2 - 1
                cy = top + h / 2 - 1

                image[cur_y:cur_y+im.shape[0], :] = np.maximum(image[cur_y:cur_y+im.shape[0], :], im)
                position = np.concatenate([position, np.array([[cx,cy,w,h]])], axis=0)
                code_list = np.concatenate([code_list, np.array([[ord(c), 0]])], axis=0)
                cur_y = int(next_cur)
                process_text += c
                remain_str = remain_str[1:]

            return {
                "image": image,
                "str": process_text,
                "remain_str": remain_str,
                "position": position,
                "code_list": code_list,
                "pad_top": pad_top,
                "base_line": base_line,
                "next_cur": int(cur_y + pad_top),
            }

if __name__=='__main__':
    from matplotlib import rcParams
    rcParams['font.serif'] = ['IPAexMincho', 'IPAPMincho', 'Hiragino Mincho ProN']

    import matplotlib.pyplot as plt


    canvas = HandwriteCanvas(horizontal=True)

    text = 'test'
    text = re.sub(r'<ruby><rb>(.*?)</rb>.*?<rt>(.*?)</rt>.*?</ruby>', '\uFFF9\\1\uFFFA\\2\uFFFB', text)


    canvas = HandwriteCanvas(ratio=0.5, horizontal=False)
    canvas.set_linewidth(1000)
    a = canvas.draw(text)
    print(a)
    plt.imshow(a['image'])
    plt.show()

