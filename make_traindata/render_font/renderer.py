import os
import subprocess
import numpy as np
from PIL import Image, ImageDraw
from PIL.Image import Resampling
import functools
import unicodedata

emphasis_characters = ['•','◦','●','○','◎','◉','▲','△','﹅','﹆']

UNICODE_WHITESPACE_CHARACTERS = [
    "\u0009", # character tabulation
    "\u000a", # line feed
    "\u000b", # line tabulation
    "\u000c", # form feed
    "\u000d", # carriage return
    "\u0020", # space
    "\u0085", # next line
    "\u00a0", # no-break space
    "\u1680", # ogham space mark
    "\u2000", # en quad
    "\u2001", # em quad
    "\u2002", # en space
    "\u2003", # em space
    "\u2004", # three-per-em space
    "\u2005", # four-per-em space
    "\u2006", # six-per-em space
    "\u2007", # figure space
    "\u2008", # punctuation space
    "\u2009", # thin space
    "\u200A", # hair space
    "\u2028", # line separator
    "\u2029", # paragraph separator
    "\u202f", # narrow no-break space
    "\u205f", # medium mathematical space
    "\u3000", # ideographic space
]

def is_ascii(s):
    s = unicodedata.normalize('NFD', s).encode('ascii', 'ignore').decode()
    return s and s in '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'

def allow_rubyover(s):
    if s == '':
        return False
    kigou = (
        '，）］｝,)]｝、〕〉》」』】〙〗〟’”｠»'+
        'ー〜〰…‥'+
        '‐゠–〜～'+
        '?!‼⁇⁈⁉'+
        '・:;/'+
        '。.'+
        '（［｛([｛〔〈《「『【〘〖〝‘“｟«')
    for c in s:
        if ord("ぁ") <= ord(c) <= ord("ゖ"):
            continue
        if ord("ァ") <= ord(c) <= ord("ヺ"):
            continue
        if c in kigou:
            continue
        if c in UNICODE_WHITESPACE_CHARACTERS:
            continue
        return False
    return True

def is_hiragana(s):
    for c in s:
        if ord("ぁ") <= ord(c) <= ord("ゖ"):
            continue
        if ord("ァ") <= ord(c) <= ord("ヺ"):
            continue
        return False
    return True

def is_kanji(s):
    for c in s:
        code = ord(c)
        if 0x2E90 <= code <= 0x2FDF:
            continue
        if c in ["々","〇","〻"]:
            continue
        if 0x3400 <= code <= 0x4DBF:
            continue
        if 0x4E00 <= code <= 0x9FFF:
            continue
        if 0xF900 <= code <= 0xFAFF:
            continue
        if 0x20000 <= code <= 0x3FFFF:
            continue
        return False
    return True

linestart_forbid = [
    '，）］｝,)]｝、〕〉》」』】〙〗〟’”｠»',
    'ゝゞーァィゥェォッャュョヮヵヶぁぃぅぇぉっゃゅょゎゕゖㇰㇱㇲㇳㇴㇵㇶㇷㇸㇹㇷ゚ㇺㇻㇼㇽㇾㇿ々〻',
    '‐゠–〜～',
    '?!‼⁇⁈⁉',
    '・:;/',
    '。.',
]
linestart_forbid = functools.reduce(lambda x, y: x + y, linestart_forbid)

lineend_forbid = '（［｛([｛〔〈《「『【〘〖〝‘“｟«'


class Canvas:
    def __init__(self, fontfile, fontsize=96.0, horizontal=True, bold=False, italic=False, turn=True):
        self.fontfile = fontfile
        self.fontsize = fontsize
        self.bold = bold
        self.italic = italic
        self.ruby_ratio = 0.5
        self.line_space_ratio = 1.5

        self.start_pad = fontsize * 1.5
        self.end_pad = fontsize * 1.5
        self.start_linepad = fontsize * 1.5
        self.end_linepad = fontsize * 1.5
        self.linewidth = 0
        self.linecount = 0
        self.linecount_max = 0
        self.is_horizontal = horizontal
        self.turn = turn

        self.section_count = 0
        self.current_section = 0
        self.section_space = fontsize * 2.0

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

        self.header_str = None
        self.footer_str = None
        self.header_sepline = None
        self.footer_sepline = None

    def set_header(self, content):
        self.header_str = content
        self.start_pad = self.fontsize * (1.5 + 1)

    def set_footer(self, content):
        self.footer_str = content
        self.end_pad = self.fontsize * (1.5 + 2)

    def set_linewidth(self, linewidth):
        self.linewidth = linewidth

    def set_linemax(self, maxline):
        self.linecount_max = maxline

    def set_section(self, section_count, space_ratio):
        self.section_count = section_count
        self.section_space = self.fontsize * space_ratio

    def _prepare_image(self):
        if self.canvas_image is None:
            if self.linecount_max > 0:
                lines = self.linecount_max
            else:
                lines = 10
            if self.is_horizontal:
                if self.section_count > 1:
                    width = self.start_pad + self.linewidth + (self.linewidth + self.section_space) * (self.section_count - 1) + self.end_pad
                else:
                    width = self.start_pad + self.linewidth + self.end_pad
                height = self.start_linepad + self.fontsize + self.fontsize * self.line_space_ratio * (lines - 1) + self.end_linepad
            else:
                if self.section_count > 1:
                    height = self.start_pad + self.linewidth + (self.linewidth + self.section_space) * (self.section_count - 1) + self.end_pad
                else:
                    height = self.start_pad + self.linewidth + self.end_pad
                width = self.start_linepad + self.fontsize + self.fontsize * self.line_space_ratio * (lines - 1) + self.end_linepad
            self.canvas_image = np.zeros([int(height), int(width)], dtype='ubyte')
            self._go_startline()
            self._cur_home()

            if self.section_count > 1 and self.section_space / self.fontsize < 0.5:
                sep_image = Image.fromarray(self.canvas_image)
                draw = ImageDraw.Draw(sep_image)
                for i in range(self.section_count - 1):
                    if self.is_horizontal:
                        x = self.start_pad + (self.linewidth + self.section_space) * i + self.linewidth + self.section_space / 2
                        y1 = self.start_linepad
                        y2 = sep_image.height - self.end_linepad
                        x = int(x)
                        draw.line(((x, y1), (x, y2)), fill=255, width=2)
                    else:
                        x1 = self.start_linepad
                        x2 = sep_image.width - self.end_linepad
                        y = self.start_pad + (self.linewidth + self.section_space) * i + self.linewidth + self.section_space / 2
                        y = int(y)
                        draw.line(((x1, y), (x2, y)), fill=255, width=2)
                self.canvas_image = np.array(sep_image)

            self.textline_image = np.zeros_like(self.canvas_image, dtype='ubyte')

    def _go_startline(self):
        if self.is_horizontal:
            self.baseline_pos = self.start_linepad + self.fontsize
        else:
            self.baseline_pos = self.canvas_image.shape[1] - self.start_linepad - self.fontsize / 2

    def _cur_home(self):
        if self.section_count > 1:
            self.cur_pos = self.start_pad + (self.linewidth + self.section_space) * self.current_section
        else:
            self.cur_pos = self.start_pad

    def _get_linewidth(self):
        if self.section_count > 1:
            return self.start_pad + (self.linewidth + self.section_space) * self.current_section + self.linewidth
        else:
            return self.start_pad + self.linewidth

    def _process_newline(self):
        if self.linecount_max > 0 and self.linecount >= self.linecount_max - 1:
            self.process_str += '\n'
            self.position = np.concatenate([self.position, np.zeros([1,4])], axis=0)
            self.code_list = np.concatenate([self.code_list, np.array([[ord('\n'), 16]])], axis=0)
            self.current_section += 1

            if self.current_section >= self.section_count:
                return False

            self._go_startline()
            self._cur_home()

            self.linecount = 0
            return True

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
        rm_code = np.logical_or(opcode & (8 + 16) > 0, position[:,2] * position[:,3] <= 0)
        opcode &= ~(8 + 16)
        opcode |= (prev_opcode & (8 + 16))
        code_list[:,1] = opcode

        sep_image = np.zeros_like(self.canvas_image, dtype='ubyte')
        sep_image = Image.fromarray(sep_image)
        draw = ImageDraw.Draw(sep_image)
        if self.section_count > 1:
            for i in range(min(self.section_count-1,self.current_section)):
                if self.is_horizontal:
                    x = self.start_pad + (self.linewidth + self.section_space) * i + self.linewidth + self.section_space / 2
                    y1 = self.start_linepad
                    y2 = sep_image.height - self.end_linepad
                    x = int(x)
                    draw.line(((x, y1), (x, y2)), fill=255, width=7)
                else:
                    x1 = self.start_linepad
                    x2 = sep_image.width - self.end_linepad
                    y = self.start_pad + (self.linewidth + self.section_space) * i + self.linewidth + self.section_space / 2
                    y = int(y)
                    draw.line(((x1, y), (x2, y)), fill=255, width=7)

        if self.header_sepline is not None:
            x1, y1, x2, y2 = self.header_sepline
            draw.line(((x1, y1), (x2, y2)), fill=255, width=7)

        if self.footer_sepline is not None:
            x1, y1, x2, y2 = self.footer_sepline
            draw.line(((x1, y1), (x2, y2)), fill=255, width=7)

        return position[np.logical_not(rm_code),:], code_list[np.logical_not(rm_code),:], np.asarray(sep_image)

    def _draw_header(self):
        if self.header_str is None:
            return

        im_buf = self._line_render(self.header_str, horizontal=not self.is_horizontal, line_length=self.canvas_image.shape[0] if self.is_horizontal else self.canvas_image.shape[1])
        if im_buf is None:
            return

        im = Image.fromarray(im_buf['image'])
        im = im.resize((im.width*2//3, im.height*2//3), resample=Image.BILINEAR)
        im_buf['image'] = np.asarray(im)

        im_buf['position'] = im_buf['position'] * (2/3,2/3,2/3,2/3)
        im_buf['base_line'] = im_buf['base_line'] * 2/3

        self.process_str += im_buf['str']
        image = im_buf['image']
        pos = im_buf['position']
        code_list = im_buf['code_list']

        if not self.is_horizontal:
            baseline_pos = (self.start_pad - self.fontsize) / 2 + self.fontsize
            left = self.canvas_image.shape[1] - (image.shape[1] + self.start_linepad)
            top = baseline_pos - im_buf['base_line']

            if left < self.start_linepad:
                pad = int(self.start_linepad - left)
                left = self.start_linepad
                self.canvas_image = np.pad(self.canvas_image, [[0, 0],[pad, 0]])
                self.textline_image = np.pad(self.textline_image, [[0, 0],[pad, 0]])
                self.baseline_pos += pad
                self.position[:,0] += pad

            left = int(left)
            top = int(top)
            right = left + image.shape[1]
            bottom = top + image.shape[0]

            x1 = left
            x2 = left + image.shape[1]
            y1 = baseline_pos
            y2 = baseline_pos

            pos[:,0] += left
            pos[:,1] += top

            y3 = (y1 + self.start_pad) / 2
            self.header_sepline = (x1, y3, x2, y3)

        else:
            baseline_pos = (self.start_pad - self.fontsize) / 2 + self.fontsize / 2
            left = baseline_pos - im_buf['base_line']
            top = self.start_linepad

            left = int(left)
            top = int(top)
            right = left + image.shape[1]
            bottom = top + image.shape[0]

            if bottom > self.canvas_image.shape[0] - self.start_linepad:
                pad = int(bottom - (self.canvas_image.shape[0] - self.start_linepad))
                self.canvas_image = np.pad(self.canvas_image, [[0, pad],[0, 0]])
                self.textline_image = np.pad(self.textline_image, [[0, pad],[0, 0]])

                bottom = top + image.shape[0]

            x1 = baseline_pos
            x2 = baseline_pos
            y1 = top
            y2 = top + image.shape[0]

            pos[:,0] += left
            pos[:,1] += top

            x3 = (x1 + self.fontsize / 2 + self.start_pad) / 2
            self.header_sepline = (x3, y1, x3, y2)

        self.canvas_image[top:bottom, left:right] = np.maximum(self.canvas_image[top:bottom, left:right], image)

        self.position = np.concatenate([self.position, pos], axis=0)
        self.code_list = np.concatenate([self.code_list, code_list], axis=0)

        textline_image = Image.fromarray(self.textline_image)
        draw = ImageDraw.Draw(textline_image)
        draw.line(((x1, y1), (x2, y2)), fill=255, width=7)
        self.textline_image = np.asarray(textline_image)

    def _draw_footer(self):
        if self.footer_str is None:
            return

        im_buf = self._line_render(self.footer_str, horizontal=not self.is_horizontal, line_length=self.canvas_image.shape[0] if self.is_horizontal else self.canvas_image.shape[1])
        if im_buf is None:
            return

        im = Image.fromarray(im_buf['image'])
        im = im.resize((im.width*2//3, im.height*2//3), resample=Image.BILINEAR)
        im_buf['image'] = np.asarray(im)

        im_buf['position'] = im_buf['position'] * (2/3,2/3,2/3,2/3)
        im_buf['base_line'] = im_buf['base_line'] * 2/3

        self.process_str += im_buf['str']
        image = im_buf['image']
        pos = im_buf['position']
        code_list = im_buf['code_list']

        if not self.is_horizontal:
            baseline_pos = self.canvas_image.shape[0] - self.end_pad + (self.end_pad - self.fontsize * 2) / 2 + self.fontsize * 2
            left = self.canvas_image.shape[1] - (image.shape[1] + self.start_linepad)
            top = baseline_pos - im_buf['base_line']

            if left < self.start_linepad:
                pad = int(self.start_linepad - left)
                left = self.start_linepad
                self.canvas_image = np.pad(self.canvas_image, [[0, 0],[pad, 0]])
                self.textline_image = np.pad(self.textline_image, [[0, 0],[pad, 0]])
                self.baseline_pos += pad
                self.position[:,0] += pad

            left = int(left)
            top = int(top)
            right = left + image.shape[1]
            bottom = top + image.shape[0]

            x1 = left
            x2 = left + image.shape[1]
            y1 = baseline_pos
            y2 = baseline_pos

            pos[:,0] += left
            pos[:,1] += top

            y3 = (y1 + self.canvas_image.shape[0] - self.end_pad) / 2
            self.footer_sepline = (x1, y3, x2, y3)

        else:
            baseline_pos = self.canvas_image.shape[1] - self.end_pad + (self.end_pad - self.fontsize * 2) / 2 + self.fontsize * 1.5
            left = baseline_pos - im_buf['base_line']
            top = self.start_linepad

            left = int(left)
            top = int(top)
            right = left + image.shape[1]
            bottom = top + image.shape[0]

            if bottom > self.canvas_image.shape[0] - self.start_linepad:
                pad = int(bottom - (self.canvas_image.shape[0] - self.start_linepad))
                self.canvas_image = np.pad(self.canvas_image, [[0, pad],[0, 0]])
                self.textline_image = np.pad(self.textline_image, [[0, pad],[0, 0]])

                bottom = top + image.shape[0]

            x1 = baseline_pos
            x2 = baseline_pos
            y1 = top
            y2 = top + image.shape[0]

            pos[:,0] += left
            pos[:,1] += top

            x3 = (x1 - self.fontsize / 2 + self.canvas_image.shape[1] - self.end_pad) / 2
            self.footer_sepline = (x3, y1, x3, y2)

        self.canvas_image[top:bottom, left:right] = np.maximum(self.canvas_image[top:bottom, left:right], image)

        self.position = np.concatenate([self.position, pos], axis=0)
        self.code_list = np.concatenate([self.code_list, code_list], axis=0)

        textline_image = Image.fromarray(self.textline_image)
        draw = ImageDraw.Draw(textline_image)
        draw.line(((x1, y1), (x2, y2)), fill=255, width=7)
        self.textline_image = np.asarray(textline_image)

    def draw(self, text):
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
                    if line_length <= 0:
                        if not self._process_newline():
                            isLive = False
                            break
                        line_length = self._get_linewidth() - self.cur_pos
                else:
                    line_length = 0

                im_buf = self._line_render(remain_str, horizontal=self.is_horizontal, line_length=line_length)
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
                    left = self.cur_pos + im_buf['pad_left']
                    top = self.baseline_pos - im_buf['base_line']
                    pos[:,0] += left
                    pos[:,1] += top
                    x1 = left
                    x2 = self.cur_pos + im_buf['next_cur']
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

                    if self.canvas_image.shape[1] <= right:
                        self.canvas_image = np.pad(self.canvas_image, [[0, 0],[0, right-self.canvas_image.shape[1]+1]])
                        self.textline_image = np.pad(self.textline_image, [[0, 0],[0, right-self.textline_image.shape[1]+1]])

                    if self.canvas_image.shape[0] <= bottom:
                        self.canvas_image = np.pad(self.canvas_image, [[0, bottom-self.canvas_image.shape[0]+1],[0, 0]])
                        self.textline_image = np.pad(self.textline_image, [[0, bottom-self.textline_image.shape[0]+1],[0, 0]])

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

        self._draw_header()
        self._draw_footer()

        position, code_list, sep_image = self._process_result()
        return {
            'image': self.canvas_image,
            'sep_image': sep_image,
            'textline_image': self.textline_image,
            'position': position,
            'code_list': code_list,
            'str': self.process_str,
        }

    def random_draw(self, textlist, width, height, rng):
        def resize(buf, ratio):
            image = buf['image']
            image = Image.fromarray(image).resize([int(image.shape[1]*ratio)+1,int(image.shape[0]*ratio)+1], resample=Resampling.BILINEAR)
            buf['image'] = np.asarray(image)
            buf['position'] *= np.array([[ratio, ratio, ratio, ratio]])
            buf['base_line'] *= ratio
            if 'pad_top' in buf:
                buf['pad_top'] *= ratio
            if 'pad_left' in buf:
                buf['pad_left'] *= ratio
            buf['next_cur'] *= ratio
            return buf

        canvas_image = np.zeros([int(height), int(width)], dtype='ubyte')
        textline_image = np.zeros_like(canvas_image, dtype='ubyte')
        sep_image = np.zeros_like(canvas_image, dtype='ubyte')
        space_imagex, space_imagey = np.meshgrid(np.arange(int(height))[::-1]+1,np.arange(int(width))[::-1]+1)
        fill_image = np.zeros_like(canvas_image, dtype='ubyte')
        process_str = ''

        position = np.zeros([0,4]) # cx, cy, w, h
        code_list = np.zeros([0,2], dtype=int) # utf-32 code, char type
        ratio_min = 16.0 / self.fontsize
        ratio_max = 300.0 / self.fontsize

        for text in textlist:
            horizontal = rng.random() < 0.5
            im_buf = self._line_render(text, horizontal=horizontal)

            if im_buf is None:
                return None

            if len(im_buf['str']) == 0:
                continue

            if im_buf['image'] is None:
                continue

            image = im_buf['image']
            y_size, x_size = image.shape
            
            max_ratio = max(np.max(space_imagex) / x_size, np.max(space_imagey) / y_size)
            if max_ratio < ratio_min:
                continue
            max_ratio = min(ratio_max, max_ratio)

            im_buf = resize(im_buf, np.exp(rng.uniform(np.log(ratio_min), np.log(max_ratio))))
            image = im_buf['image']
            y_size, x_size = image.shape

            yp, xp = np.where(np.logical_and(space_imagex > x_size, space_imagey > y_size))
            if xp.shape[0] == 0:
                continue
            elif xp.shape[0] == 1:
                ind = 0
            else:
                ind = rng.integers(0, xp.shape[0])

            left = xp[ind]
            top = yp[ind]
            right = left + image.shape[1]
            bottom = top + image.shape[0]

            pos = im_buf['position']
            code = im_buf['code_list']

            opcode = code[:,1]
            prev_opcode = np.concatenate([[16], opcode[:-1]])
            opcode &= ~(8 + 16)
            opcode |= (prev_opcode & (8 + 16))
            code[:,1] = opcode

            pos[:,0] += left
            pos[:,1] += top

            if horizontal:
                x1 = left
                x2 = left + im_buf['next_cur'] - im_buf['pad_left']
                y1 = top + im_buf['base_line']
                y2 = top + im_buf['base_line']
            else:
                x1 = left + im_buf['base_line']
                x2 = left + im_buf['base_line']
                y1 = top
                y2 = top + im_buf['next_cur'] - im_buf['pad_top']

            if np.sum(fill_image[top:bottom, left:right]) > 0:
                continue

            canvas_image[top:bottom, left:right] = np.maximum(canvas_image[top:bottom, left:right], image)
            fill_image[top:bottom, left:right] = 1
            space_imagex[top:bottom, :left] -= space_imagex[top:bottom, left:left+1]
            space_imagey[:top, left:right] -= space_imagey[top:top+1, left:right]
            space_imagex[top:bottom, left:right] = 0
            space_imagey[top:bottom, left:right] = 0

            if len(im_buf['str']) > 1:
                textline_image = Image.fromarray(textline_image)
                draw = ImageDraw.Draw(textline_image)
                draw.line(((x1, y1), (x2, y2)), fill=255, width=7)
                textline_image = np.asarray(textline_image)

            position = np.concatenate([position, pos], axis=0)
            code_list = np.concatenate([code_list, code], axis=0)
            process_str += im_buf['str'] + '\n'

        return {
            'image': canvas_image,
            'sep_image': sep_image,
            'textline_image': textline_image,
            'position': position,
            'code_list': code_list,
            'str': process_str,
        }

    def random_drawgrid(self, values, rng):
        buffer = []
        max_col = 0
        for row in values:
            line_buffer = []
            for col in row:
                buf = []
                for line in col.splitlines():
                    im_buf = self._line_render(line, horizontal=True)
                    if im_buf is None:
                        return None
                    buf.append(im_buf)
                line_buffer.append(buf)
            max_col = max(max_col, len(line_buffer))
            buffer.append(line_buffer)
        max_row = len(buffer)

        line_height = self.fontsize * 1.1
        width_for_col = np.zeros([max_col], dtype=int)
        height_for_row = np.zeros([max_row], dtype=int)
        for i,row in enumerate(buffer):
            for j,col in enumerate(row):
                for buf in col:
                    image = buf['image']
                    if image is None:
                        continue
                    width_for_col[j] = max(width_for_col[j], image.shape[1])
                height_for_row[i] = max(height_for_row[i], len(col) * line_height)
                
        width_for_col += self.fontsize
        height_for_row += self.fontsize
        line_width = rng.integers(2,10)
        margin = self.fontsize * 3

        width = margin * 2 + np.sum(width_for_col) + line_width * (max_col + 1)
        height = margin * 2 + np.sum(height_for_row) + line_width * (max_row + 1)

        canvas_image = np.zeros([int(height), int(width)], dtype='ubyte')
        textline_image = np.zeros_like(canvas_image, dtype='ubyte')
        sep_image = np.zeros_like(canvas_image, dtype='ubyte')

        process_str = ''
        position = np.zeros([0,4]) # cx, cy, w, h
        code_list = np.zeros([0,2], dtype=int) # utf-32 code, char type

        for i,row in enumerate(buffer):
            for j,col in enumerate(row):
                cur_pos = margin + np.sum(width_for_col[:j]) + line_width * j
                max_right = 0
                for buf in col:
                    max_right = max(max_right, buf['next_cur'])
                cur_pos += (width_for_col[j] - max_right) / 2

                baseline_pos = margin + np.sum(height_for_row[:i]) + line_width * i
                baseline_pos += (height_for_row[i] - len(col) * line_height) / 2
                col_str = ''
                for buf in col:
                    baseline_pos += line_height

                    image = buf['image']
                    if image is None:
                        continue
                    pos = buf['position']
                    code = buf['code_list']

                    left = cur_pos + buf['pad_left']
                    top = baseline_pos - buf['base_line']
                    pos[:,0] += left
                    pos[:,1] += top
                    x1 = left
                    x2 = cur_pos + buf['next_cur']
                    y1 = baseline_pos
                    y2 = baseline_pos

                    left = int(left)
                    top = int(top)
                    right = left + image.shape[1]
                    bottom = top + image.shape[0]

                    canvas_image[top:bottom, left:right] = np.maximum(canvas_image[top:bottom, left:right], image)

                    textline_image = Image.fromarray(textline_image)
                    draw = ImageDraw.Draw(textline_image)
                    draw.line(((x1, y1), (x2, y2)), fill=255, width=7)
                    textline_image = np.asarray(textline_image)

                    col_str += buf['str'] + '\n'
                    pos = np.concatenate([pos, np.zeros([1,4])], axis=0)
                    code = np.concatenate([code, np.array([[ord('\n'), 16]])], axis=0)

                    position = np.concatenate([position, pos], axis=0)
                    code_list = np.concatenate([code_list, code], axis=0)

                if len(col_str) == 0:
                    col_str = '\n'
                    position = np.concatenate([position, np.zeros([1,4])], axis=0)
                    code_list = np.concatenate([code_list, np.array([[ord('\n'), 16]])], axis=0)
                code_list[-1,:] = np.array([ord('\n'), 16])
                process_str += col_str[:-1] + '\n'
            code_list[-1,:] = np.array([ord('\n'), 16])
            process_str = process_str[:-1] + '\n'

        opcode = code_list[:,1]
        prev_opcode = np.concatenate([[16], opcode[:-1]])
        rm_code = np.logical_or(opcode & (8 + 16) > 0, position[:,2] * position[:,3] <= 0)
        opcode &= ~(8 + 16)
        opcode |= (prev_opcode & (8 + 16))
        code_list[:,1] = opcode
        position = position[np.logical_not(rm_code),:]
        code_list = code_list[np.logical_not(rm_code),:]

        sep_image = Image.fromarray(sep_image)
        draw = ImageDraw.Draw(sep_image)
        for i in range(max_row+1):
            y = margin + np.sum(height_for_row[:i]) + line_width * i
            x1 = margin
            x2 = width - margin - line_width
            draw.line(((x1, y), (x2, y)), fill=255, width=7)
        for j in range(max_col+1):
            x = margin + np.sum(width_for_col[:j]) + line_width * j
            y1 = margin
            y2 = height - margin - line_width
            draw.line(((x, y1), (x, y2)), fill=255, width=7)
        sep_image = np.asarray(sep_image)

        canvas_image = Image.fromarray(canvas_image)
        draw = ImageDraw.Draw(canvas_image)
        for i in range(max_row+1):
            y = margin + np.sum(height_for_row[:i]) + line_width * i
            x1 = margin
            x2 = width - margin - line_width
            draw.line(((x1, y), (x2, y)), fill=255, width=line_width)
        for j in range(max_col+1):
            x = margin + np.sum(width_for_col[:j]) + line_width * j
            y1 = margin
            y2 = height - margin - line_width
            draw.line(((x, y1), (x, y2)), fill=255, width=line_width)
        canvas_image = np.asarray(canvas_image)

        return {
            'image': canvas_image,
            'sep_image': sep_image,
            'textline_image': textline_image,
            'position': position,
            'code_list': code_list,
            'str': process_str,
        }

    def draw_wari(self, text):
        def wari_small(buf):
            image = buf['image']
            if image is not None:
                image = Image.fromarray(image).resize([int(image.shape[1]*0.5),int(image.shape[0]*0.5)], resample=Resampling.BILINEAR)
                buf['image'] = np.asarray(image)
            buf['position'] *= np.array([[0.5, 0.5, 0.5, 0.5]])
            buf['base_line'] *= 0.5
            if 'pad_top' in buf:
                buf['pad_top'] *= 0.5
            if 'pad_left' in buf:
                buf['pad_left'] *= 0.5
            buf['next_cur'] *= 0.5
            return buf

        def sub_draw(im_buf, baseline, cur_pos):
            start = False
            remain_str = im_buf['remain_str']
            self.process_str += im_buf['str']
            image = im_buf['image']
            pos = im_buf['position']
            code_list = im_buf['code_list']

            if self.is_horizontal:
                left = cur_pos + im_buf['pad_left']
                top = baseline - im_buf['base_line']
                pos[:,0] += left
                pos[:,1] += top
                x1 = left
                x2 = cur_pos + im_buf['next_cur']
                y1 = baseline
                y2 = baseline
            else:
                left = baseline - im_buf['base_line']
                top = cur_pos + (im_buf['pad_top'] if start else 0)
                pos[:,0] += left
                pos[:,1] += top
                x1 = baseline
                x2 = baseline
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

            return remain_str

        self._prepare_image()

        wari_str = ''
        while len(text) > 0 or len(wari_str) > 0:
            if len(text) > 0:
                if '（' in text:
                    text, wari_str = text.split('（', 1)
                    
                line_length = self._get_linewidth() - self.cur_pos
                if line_length <= 0:
                    if not self._process_newline():
                        break
                    line_length = self._get_linewidth() - self.cur_pos
                im_buf = self._line_render(text, horizontal=self.is_horizontal, line_length=line_length)
                if im_buf is None:
                    break

                if len(im_buf['str']) == 0:
                    break

                text = sub_draw(im_buf, self.baseline_pos, self.cur_pos)

                self.cur_pos += im_buf['next_cur']
                
            if text == '' and wari_str and self._get_linewidth() - self.cur_pos > 0:
                wari_str, text = wari_str.split('）', 1)

                line_length = self._get_linewidth() - self.cur_pos                    
                start_curpos = self.cur_pos
                start_baseline = self.baseline_pos

                while len(wari_str) > 0:
                    for i in range(len(wari_str) // 2):
                        if i > 0:
                            wari_text = wari_str[:-i*2]
                        else:
                            wari_text = wari_str

                        wari_len = len(wari_text)
                        wari_count = (wari_len + 1) // 2
                        wari_text1 = wari_text[:wari_count]
                        wari_text2 = wari_text[wari_count:]

                        im_buf = self._line_render(wari_text1, horizontal=self.is_horizontal, line_length=line_length * 2)
                        if im_buf['remain_str'] == '':
                            if i > 0:
                                wari_str = wari_str[-i*2:]
                            else:
                                wari_str = ''
                            break
                    else:
                        if not self._process_newline():
                            break
                        line_length = self._get_linewidth() - self.cur_pos
                        start_curpos = self.cur_pos
                        start_baseline = self.baseline_pos
                        continue

                    if self.is_horizontal:
                        baseline_pos = start_baseline - self.fontsize * 0.5
                    else:
                        baseline_pos = start_baseline + self.fontsize * 0.25

                    im_buf1 = self._line_render(wari_text1, horizontal=self.is_horizontal, line_length=line_length * 2)
                    im_buf1 = wari_small(im_buf1)
                    sub_draw(im_buf1, baseline_pos, start_curpos)

                    if self.is_horizontal:
                        baseline_pos += self.fontsize * 0.5
                    else:
                        baseline_pos -= self.fontsize * 0.5

                    im_buf2 = self._line_render(wari_text2, horizontal=self.is_horizontal, line_length=line_length * 2)
                    im_buf2 = wari_small(im_buf2)
                    sub_draw(im_buf2, baseline_pos, start_curpos)

                    self.cur_pos += im_buf1['next_cur']
                    if len(wari_str) > 0:
                        if not self._process_newline():
                            break
                        line_length = self._get_linewidth() - self.cur_pos
                        start_curpos = self.cur_pos
                        start_baseline = self.baseline_pos

                continue

            if not self._process_newline():
                break

        self._draw_header()
        self._draw_footer()

        position, code_list, sep_image = self._process_result()
        return {
            'image': self.canvas_image,
            'sep_image': sep_image,
            'textline_image': self.textline_image,
            'position': position,
            'code_list': code_list,
            'str': self.process_str,
        }

    def __enter__(self):
        t1 = 1 if self.italic else 0
        t2 = 2 if self.bold else 0
        t = t1 + t2
        # print(self.fontfile,self.fontsize,t)
        self.proc = subprocess.Popen([
            os.path.join(os.path.dirname(__file__), 'render_font'),
            str(self.fontfile),
            str(self.fontsize),
            str(t),
            ], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        charbuf = '\0'.encode("utf-32-le")
        self.proc.stdin.write(charbuf[:4])
        self.proc.stdin.flush()
        self.proc.stdin.close()
        self.proc.wait()

    def _get_glyph(self, text):
        if self.proc.poll() is not None:
            raise RuntimeError(self.fontfile,self.fontsize,text)
        count = 0
        for c in text:
            charbuf = c.encode("utf-32-le")
            self.proc.stdin.write(charbuf[:4])
            self.proc.stdin.flush()
            count += 1
        charbuf = '\0'.encode("utf-32-le")
        self.proc.stdin.write(charbuf[:4])
        self.proc.stdin.flush()

        retcount = 0
        results = []
        while retcount < count and self.proc.poll() is None:
            result = self.proc.stdout.read(32)

            ligacount = int.from_bytes(result[:4], 'little')
            code = text[retcount:retcount+ligacount]
            retcount += ligacount
            rows1 = int.from_bytes(result[4:8], 'little')
            width1 = int.from_bytes(result[8:12], 'little')
            horiBoundingWidth = int.from_bytes(result[12:16], 'little', signed=True)
            horiBoundingHeight = int.from_bytes(result[16:20], 'little', signed=True)
            horiBearingX = int.from_bytes(result[20:24], 'little', signed=True)
            horiBearingY = int.from_bytes(result[24:28], 'little', signed=True)
            horiAdvance = int.from_bytes(result[28:32], 'little', signed=True)

            horiBoundingWidth = horiBoundingWidth / 64
            horiBoundingHeight = horiBoundingHeight / 64
            horiBearingX = horiBearingX / 64
            horiBearingY = horiBearingY / 64
            horiAdvance = horiAdvance / 64

            if rows1 == 0 or width1 == 0:
                if ligacount == 1 and code not in UNICODE_WHITESPACE_CHARACTERS:
                    results.append(None)
                    continue
                results.append({
                    'code': code,
                    'rows1': rows1,
                    'width1': width1,
                    'horiBoundingWidth': horiBoundingWidth,
                    'horiBoundingHeight': horiBoundingHeight,
                    'horiBearingX': horiBearingX,
                    'horiBearingY': horiBearingY,
                    'horiAdvance': horiAdvance,
                    })
                continue

            buffer = self.proc.stdout.read(rows1*width1)
            img1 = np.frombuffer(buffer, dtype='ubyte').reshape(rows1,width1)

            result = self.proc.stdout.read(28)

            rows2 = int.from_bytes(result[:4], 'little')
            width2 = int.from_bytes(result[4:8], 'little')
            vertBoundingWidth = int.from_bytes(result[8:12], 'little', signed=True)
            vertBoundingHeight = int.from_bytes(result[12:16], 'little', signed=True)
            vertBearingX = int.from_bytes(result[16:20], 'little', signed=True)
            vertBearingY = int.from_bytes(result[20:24], 'little', signed=True)
            vertAdvance = int.from_bytes(result[24:28], 'little', signed=True)

            vertBoundingWidth = vertBoundingWidth / 64
            vertBoundingHeight = vertBoundingHeight / 64
            vertBearingX = vertBearingX / 64
            vertBearingY = vertBearingY / 64
            vertAdvance = vertAdvance / 64

            if rows2 == 0 or width2 == 0:
                results.append({
                    'code': code,
                    'rows1': rows1,
                    'width1': width1,
                    'horiBoundingWidth': horiBoundingWidth,
                    'horiBoundingHeight': horiBoundingHeight,
                    'horiBearingX': horiBearingX,
                    'horiBearingY': horiBearingY,
                    'horiAdvance': horiAdvance,
                    'img1': img1,
                    'rows2': rows2,
                    'width2': width2,
                    'vertBoundingWidth': vertBoundingWidth,
                    'vertBoundingHeight': vertBoundingHeight,
                    'vertBearingX': vertBearingX,
                    'vertBearingY': vertBearingY,
                    'vertAdvance': vertAdvance,
                    })
                continue

            buffer = self.proc.stdout.read(rows2*width2)
            img2 = np.frombuffer(buffer, dtype='ubyte').reshape(rows2,width2)

            results.append({
                'code': code,
                'rows1': rows1,
                'width1': width1,
                'horiBoundingWidth': horiBoundingWidth,
                'horiBoundingHeight': horiBoundingHeight,
                'horiBearingX': horiBearingX,
                'horiBearingY': horiBearingY,
                'horiAdvance': horiAdvance,
                'img1': img1,
                'rows2': rows2,
                'width2': width2,
                'vertBoundingWidth': vertBoundingWidth,
                'vertBoundingHeight': vertBoundingHeight,
                'vertBearingX': vertBearingX,
                'vertBearingY': vertBearingY,
                'vertAdvance': vertAdvance,
                'img2': img2,
                })
        if self.proc.poll() is not None:
            raise RuntimeError(self.fontfile,self.fontsize,text)
        return results

    def _draw_buffer(self, text, horizontal=True, line_length=0, pad_space=0):
        position = np.zeros([0,4])
        code_list = np.zeros([0,2], dtype=int)
        process_text = ''
        remain_str = text
        if horizontal:
            pad_left = 0
            image = None
            cur_x = 0
            base_line = 0
            glyphlist = self._get_glyph(text)
            for char_data in glyphlist:
                if char_data is None:
                    remain_str = remain_str[1:]
                    continue

                im = char_data.get('img1', None)
                if im is None:
                    next_cur = cur_x + char_data['horiAdvance'] + pad_space
                    cx = (next_cur + cur_x) / 2
                    cy = base_line
                    w = next_cur - cur_x
                    h = 0
                    position = np.concatenate([position, np.array([[cx,cy,w,h]])], axis=0)
                    cur_x = next_cur
                    process_text += char_data['code']
                    newcode = np.array([ord(c) for c in char_data['code']]).T
                    newcode = np.stack([newcode, np.ones_like(newcode) * 8], 1)
                    code_list = np.concatenate([code_list, newcode], axis=0)
                    remain_str = remain_str[len(char_data['code']):]
                    continue
    
                if image is None:
                    base_line = int(char_data['horiBearingY'])
                    pad_left = cur_x
                    cur_x = 0
                    
                left = cur_x + char_data['horiBearingX']
                right = left + char_data['horiBoundingWidth']
                next_cur = cur_x + char_data['horiAdvance'] + pad_space

                top = base_line - char_data['horiBearingY']
                bottom = top + char_data['horiBoundingHeight']

                left = int(left)
                top = int(top)
                right = left + im.shape[1]
                bottom = top + im.shape[0]

                if line_length > 0 and right + pad_left > line_length:
                    break

                if image is None:
                    if left < 0:
                        right -= left
                        left = 0
                    image = np.zeros([bottom, right], dtype='ubyte')

                if left < 0:
                    pad_left += -left
                    image = np.pad(image, [[0, 0],[-left, 0]])
                    right += -left
                    next_cur += -left
                    position[:,0] += -left
                    left += -left

                if top < 0:
                    pad_top = -top
                    image = np.pad(image, [[pad_top, 0],[0, 0]])
                    base_line += pad_top
                    top += pad_top
                    bottom += pad_top
                    position[:,1] += pad_top

                if image.shape[1] < right:
                    image = np.pad(image, [[0, 0],[0, right-image.shape[1]]])

                if image.shape[0] < bottom:
                    image = np.pad(image, [[0, bottom-image.shape[0]],[0,0]])

                w = char_data['width1']
                h = char_data['rows1']
                cx = left + w / 2
                cy = top + h / 2

                image[top:bottom, left:right] = np.maximum(image[top:bottom, left:right], im)
                if len(char_data['code']) > 1:
                    w1 = w / len(char_data['code'])
                    for i in range(len(char_data['code'])):
                        cx = left + w1 / 2 + i * w1
                        position = np.concatenate([position, np.array([[cx,cy,w1,h]])], axis=0)
                else:
                    position = np.concatenate([position, np.array([[cx,cy,w,h]])], axis=0)
                cur_x = next_cur
                process_text += char_data['code']
                newcode = np.array([ord(c) for c in char_data['code']]).T
                newcode = np.stack([newcode, np.zeros_like(newcode)], 1)
                code_list = np.concatenate([code_list, newcode], axis=0)
                remain_str = remain_str[len(char_data['code']):]

            return {
                "image": image,
                "str": process_text,
                "remain_str": remain_str,
                "position": position,
                "code_list": code_list,
                "pad_left": pad_left,
                "base_line": base_line,
                "next_cur": int(cur_x + pad_left - pad_space),
            }
        else:
            pad_top = 0
            image = None
            cur_y = 0
            base_line = 0
            glyphlist = self._get_glyph(text)
            for char_data in glyphlist:
                if char_data is None:
                    remain_str = remain_str[1:]
                    continue

                im = char_data.get('img2', None)
                if im is None:
                    adv = char_data['horiAdvance']
                    next_cur = cur_y + char_data.get('vertAdvance', adv) + pad_space
                    cy = (next_cur + cur_y) / 2
                    cx = base_line
                    h = next_cur - cur_y
                    w = 0
                    position = np.concatenate([position, np.array([[cx,cy,w,h]])], axis=0)
                    process_text += char_data['code']
                    newcode = np.array([ord(c) for c in char_data['code']]).T
                    newcode = np.stack([newcode, np.ones_like(newcode) * 8], 1)
                    code_list = np.concatenate([code_list, newcode], axis=0)
                    cur_y = next_cur
                    remain_str = remain_str[len(char_data['code']):]
                    continue
                    
                if image is None:
                    base_line = -int(char_data['vertBearingX'])
                    pad_top = cur_y
                    cur_y = 0

                left = base_line + char_data['vertBearingX']
                right = left + char_data['vertBoundingWidth']
                next_cur = cur_y + char_data['vertAdvance'] + pad_space

                top = cur_y + char_data['vertBearingY']
                bottom = top + char_data['vertBoundingHeight']

                left = int(left)
                top = int(top)
                right = left + im.shape[1]
                bottom = top + im.shape[0]

                if line_length > 0 and bottom + pad_top > line_length:
                    break

                if image is None:
                    if top < 0:
                        bottom -= top
                        top = 0
                    image = np.zeros([bottom, right], dtype='ubyte')

                if top < 0:
                    pad_top += -top
                    image = np.pad(image, [[-top, 0],[0, 0]])
                    next_cur += -top
                    bottom += -top
                    position[:,1] += -top
                    top += -top

                if left < 0:
                    pad_left = -left
                    image = np.pad(image, [[0, 0],[pad_left, 0]])
                    base_line += pad_left
                    left += pad_left
                    right += pad_left
                    position[:,0] += pad_left

                if image.shape[0] < bottom:
                    image = np.pad(image, [[0, bottom-image.shape[0]],[0,0]])

                if image.shape[1] < right:
                    image = np.pad(image, [[0, 0],[0, right-image.shape[1]]])

                w = char_data['width2']
                h = char_data['rows2']
                cx = left + w / 2
                cy = top + h / 2

                image[top:bottom, left:right] = np.maximum(image[top:bottom, left:right], im)
                if len(char_data['code']) > 1:
                    h1 = h / len(char_data['code'])
                    for i in range(len(char_data['code'])):
                        cy = top + h1 / 2 + i * h1
                        position = np.concatenate([position, np.array([[cx,cy,w,h1]])], axis=0)
                else:
                    position = np.concatenate([position, np.array([[cx,cy,w,h]])], axis=0)
                process_text += char_data['code']
                newcode = np.array([ord(c) for c in char_data['code']]).T
                newcode = np.stack([newcode, np.zeros_like(newcode)], 1)
                code_list = np.concatenate([code_list, newcode], axis=0)
                cur_y = next_cur
                remain_str = remain_str[len(char_data['code']):]
                
            return {
                "image": image,
                "str": process_text,
                "remain_str": remain_str,
                "position": position,
                "code_list": code_list,
                "pad_top": pad_top,
                "base_line": base_line,
                "next_cur": int(cur_y + pad_top - pad_space),
            }

    def _word_render(self, texts, space=None, line_length=0, pad_space=0):
        if space is None:
            space = self.fontsize / 2
        need_space = space * 0.4
        cur_x = 0
        pad_left = 0
        base_line = 0
        buffer = []
        remain_texts = []
        image = None
        position = []
        code_list = []
        result_str = []
        for i in range(len(texts)):
            cur_txt = texts[i]
            if cur_txt.strip() == '':
                cx = 0.
                cy = 0.
                w = 0.
                h = 0.
                word = {
                    "image": None,
                    "str": '',
                    "position": np.array([[cx,cy,w,h]]),
                    "code_list": np.array([[ord(' '), 8]]),
                    "pad_left": 0,
                    "base_line": 0,
                    "next_cur": 0,
                }
            else:
                word = self._draw_buffer(cur_txt, horizontal=True)
                if word is None:
                    return None

            if line_length > 0 and cur_x + word['next_cur'] > line_length:
                remain_texts = texts[i:]
                break

            cur_x += word['next_cur'] + need_space
            buffer.append(word)
        
        if line_length > 0 and len(buffer) > 1:
            space_fix = need_space + (line_length - cur_x + need_space) / (len(buffer) - 1) + pad_space
            if space_fix > space + pad_space:
                space_fix = space + pad_space
        else:
            space_fix = space + pad_space

        cur_x = 0
        for word in buffer:
            if image is None:
                image = word['image']
                pos = word['position']
                code_list.append(word['code_list'])
                base_line = word['base_line']
                pad_left += cur_x + word['pad_left']
                cur_x += word['next_cur'] - word['pad_left'] + space_fix
                result_str.append(word['str'])
                position.append(pos)
            else:
                result_str.append(word['str'])
                code_list.append(word['code_list'])
                im = word['image']
                if im is None:
                    pos = word['position']
                    pos[:,0] += cur_x
                    pos[:,1] += base_line
                    cur_x += word['next_cur'] + space_fix
                    position.append(pos)
                else:
                    pos = word['position']

                    left = cur_x + word['pad_left']
                    next_cur = cur_x + word['next_cur'] - word['pad_left'] + space_fix
                    left = int(left)
                    next_cur = int(next_cur)

                    if left < 0:
                        pad_left += -left
                        image = np.pad(image, [[0, 0],[-left, 0]])
                        next_cur += -left
                        position = [p - np.array([[left, 0, 0, 0]]) for p in position]
                        left += -left

                    top = base_line - word['base_line']
                    top = int(top)
                    if top < 0:
                        pad_top = -top
                        image = np.pad(image, [[pad_top, 0],[0, 0]])
                        base_line += pad_top
                        top += pad_top
                        position = [p + np.array([[0, pad_top, 0, 0]]) for p in position]

                    right = left + im.shape[1]
                    right = int(right)
                    if right > image.shape[1]:
                        image = np.pad(image, [[0, 0],[0, right-image.shape[1]]])

                    bottom = top + im.shape[0]
                    bottom = int(bottom)
                    if bottom > image.shape[0]:
                        image = np.pad(image, [[0, bottom-image.shape[0]],[0,0]])

                    pos[:,0] += left
                    pos[:,1] += top

                    image[top:bottom, left:right] = np.maximum(image[top:bottom, left:right], im)
                    position.append(pos)

                    cur_x = next_cur
        
        i = 1
        while i < len(code_list):
            position.insert(i, np.zeros([1,4]))
            code_list.insert(i, np.array([[ord(' '), 8]]))
            i += 2
        position.insert(0, np.zeros([0,4]))
        code_list.insert(0, np.zeros([0,2], dtype=int))

        if len(buffer) == 1:
            cur_x -= space_fix

        return {
            'image': image,
            'str': ' '.join(result_str),
            'remain_str': ' '.join(remain_texts),
            'position': np.concatenate(position, axis=0),
            'code_list': np.concatenate(code_list, axis=0),
            'pad_left': pad_left,
            'base_line': base_line,
            'next_cur': int(cur_x),
        }

    def _horizontal_line_render(self, text, line_length=0, pad_space=0):

        def jp_word_sep(text, line_length, pad_space=pad_space):
            line_buf = self._draw_buffer(text, horizontal=True, line_length=line_length, pad_space=pad_space)
            if line_buf is None:
                return None
            # 文末禁則
            if len(line_buf['str']) > 0 and line_buf['str'][-1] in lineend_forbid:
                line_buf2 = self._draw_buffer(line_buf['str'][:-1], horizontal=True, pad_space=pad_space)
                line_buf2['remain_str'] += line_buf['str'][-1] + line_buf['remain_str']
                return line_buf2
            # 文頭禁則
            if len(line_buf['remain_str']) > 0 and line_buf['remain_str'][0] in linestart_forbid:
                if len(line_buf['str']) > 0 and len(line_buf['remain_str']) > 1 and line_buf['remain_str'][1] in linestart_forbid:
                    line_buf2 = self._draw_buffer(line_buf['str'][:-1], horizontal=True, pad_space=pad_space)
                    line_buf2['remain_str'] += line_buf['str'][-1] + line_buf['remain_str']
                    return line_buf2
                line_buf2 = self._draw_buffer(line_buf['str']+line_buf['remain_str'][0], horizontal=True, pad_space=pad_space)
                line_buf2['remain_str'] += line_buf['remain_str'][1:]
                return line_buf2
            return line_buf

        def iseng(c):
            if 0x20 <= ord(c) < 0x7F:
                return True
            if c in '“”':
                return True
            return False

        ascii_count = 0
        nonascii_count = 0
        for c in text:
            if iseng(c):
                ascii_count += 1
            else:
                nonascii_count += 1
        if ascii_count == 0:
            # jp mode
            return jp_word_sep(text, line_length=line_length, pad_space=pad_space)
        elif nonascii_count == 0:
            # en mode
            return self._word_render(text.split(' '), line_length=line_length, pad_space=pad_space)
        else:
            #mix
            splited_text = []
            cur_entext = ''
            cur_jptext = ''
            for c in text:
                if iseng(c):
                    if len(cur_jptext) > 0:
                        splited_text.append(cur_jptext)
                        cur_jptext = ''
                    cur_entext += c
                else:
                    if len(cur_entext) > 0:
                        splited_text.append(cur_entext)
                        cur_entext = ''
                    cur_jptext += c
            else:
                if len(cur_jptext) > 0:
                    splited_text.append(cur_jptext)
                if len(cur_entext) > 0:
                    splited_text.append(cur_entext)

            cur_x = 0
            pad_left = 0
            base_line = 0
            process_text = ''
            remain_str = ''
            image = None
            position = np.zeros([0,4])
            code_list = np.zeros([0,2], dtype=int)
            pad_space /= len(splited_text)
            for i, segment in enumerate(splited_text):
                if line_length > 0 and cur_x + pad_left >= line_length:
                    for seg in splited_text[i:]:
                        remain_str += seg
                    break

                cur_line_length = line_length - cur_x - pad_left
                if cur_line_length <= 0:
                    cur_line_length = 0
                if iseng(segment[0]):
                    seg_buf = self._word_render(segment.split(' '), line_length=cur_line_length, pad_space=pad_space)
                else:
                    seg_buf = jp_word_sep(segment, line_length=cur_line_length, pad_space=pad_space)

                if seg_buf is None:
                    return None

                if image is None:
                    image = seg_buf['image']
                    pos = seg_buf['position']
                    base_line = seg_buf['base_line']
                    pad_left += cur_x + seg_buf['pad_left']
                    process_text += seg_buf['str']
                    remain_str = seg_buf['remain_str']
                    cur_x += seg_buf['next_cur'] + pad_space
                    position = np.concatenate([position, pos], axis=0)   
                    code_list = np.concatenate([code_list, seg_buf['code_list']], axis=0)               
                else:
                    pos = seg_buf['position']
                    code_list = np.concatenate([code_list, seg_buf['code_list']], axis=0)
                    process_text += seg_buf['str']
                    remain_str = seg_buf['remain_str']
                    im = seg_buf['image']
                    if im is None:
                        cur_x += seg_buf['next_cur']
                        pos[:,0] += cur_x
                        position = np.concatenate([position, pos], axis=0)
                    else:                 
                        left = cur_x + seg_buf['pad_left']
                        next_cur = cur_x + seg_buf['next_cur'] + pad_space
                        left = int(left)
                        next_cur = int(next_cur)
                        if left < 0:
                            pad_left += -left
                            image = np.pad(image, [[0, 0],[-left, 0]])
                            next_cur += -left
                            position[:,0] += -left
                            left += -left

                        top = base_line - seg_buf['base_line']
                        top = int(top)
                        if top < 0:
                            pad_top = -top
                            image = np.pad(image, [[pad_top, 0],[0, 0]])
                            base_line += pad_top
                            top += pad_top
                            position[:,1] += pad_top

                        right = left + im.shape[1]
                        right = int(right)
                        if right > image.shape[1]:
                            image = np.pad(image, [[0, 0],[0, right-image.shape[1]]])

                        bottom = top + im.shape[0]
                        bottom = int(bottom)
                        if bottom > image.shape[0]:
                            image = np.pad(image, [[0, bottom-image.shape[0]],[0,0]])

                        pos[:,0] += left
                        pos[:,1] += top
                        image[top:bottom, left:right] = np.maximum(image[top:bottom, left:right], im)
                        position = np.concatenate([position, pos], axis=0)                    

                        cur_x = next_cur

                if len(seg_buf['remain_str']) > 0:
                    for seg in splited_text[i+1:]:
                        remain_str += seg
                    break

            return {
                "image": image,
                "str": process_text,
                "remain_str": remain_str,
                "position": position,
                "code_list": code_list,
                "pad_left": pad_left,
                "base_line": base_line,
                "next_cur": int(cur_x),
            }

    def _vertical_line_render(self, text, line_length=0, pad_space=0):

        def jp_word_sep(text, line_length, pad_space=pad_space):
            line_buf = self._draw_buffer(text, horizontal=False, line_length=line_length, pad_space=pad_space)
            if line_buf is None:
                return None
            # 文末禁則
            if len(line_buf['str']) > 0 and line_buf['str'][-1] in lineend_forbid:
                line_buf2 = self._draw_buffer(line_buf['str'][:-1], horizontal=False, pad_space=pad_space)
                line_buf2['remain_str'] += line_buf['str'][-1] + line_buf['remain_str']
                return line_buf2
            # 文頭禁則
            if len(line_buf['remain_str']) > 0 and line_buf['remain_str'][0] in linestart_forbid:
                if len(line_buf['str']) > 0 and len(line_buf['remain_str']) > 1 and line_buf['remain_str'][1] in linestart_forbid:
                    line_buf2 = self._draw_buffer(line_buf['str'][:-1], horizontal=False, pad_space=pad_space)
                    line_buf2['remain_str'] += line_buf['str'][-1] + line_buf['remain_str']
                    return line_buf2
                line_buf2 = self._draw_buffer(line_buf['str']+line_buf['remain_str'][0], horizontal=False, pad_space=pad_space)
                line_buf2['remain_str'] += line_buf['remain_str'][1:]
                return line_buf2
            return line_buf

        def iseng(c):
            if 0x20 <= ord(c) < 0x7F:
                return True
            if c in '“”':
                return True
            return False

        ascii_count = 0
        nonascii_count = 0
        for c in text:
            if iseng(c):
                ascii_count += 1
            else:
                nonascii_count += 1
        if ascii_count == 0 or not self.turn:
            # jp mode
            return jp_word_sep(text, line_length=line_length, pad_space=pad_space)
        elif nonascii_count == 0:
            # en mode
            buffer = self._word_render(text.split(' '), line_length=line_length, pad_space=pad_space)

            position = buffer['position']
            position[:,1] = position[:,1] - buffer['base_line']
            position = position[:,[1,0,3,2]]

            if buffer['image'] is None:
                return {
                            "image": None,
                            "str": buffer['str'],
                            "remain_str": buffer['remain_str'],
                            "position": position,
                            "code_list": buffer['code_list'],
                            "pad_top": buffer['pad_left'],
                            "base_line": buffer['base_line'],
                            "next_cur": int(buffer['next_cur']),
                        }

            pad_top = buffer['pad_left']
            fix_base_lise = buffer['image'].shape[0]-buffer['base_line'] + self.fontsize * 0.33
            position[:,0] = buffer['image'].shape[0]-buffer['base_line']-position[:,0]

            return {
                "image": buffer['image'].T[:,::-1],
                "str": buffer['str'],
                "remain_str": buffer['remain_str'],
                "position": position,
                "code_list": buffer['code_list'],
                "pad_top": pad_top,
                "base_line": fix_base_lise,
                "next_cur": int(buffer['next_cur']),
            }
        else:
            #mix
            splited_text = []
            cur_entext = ''
            cur_jptext = ''
            for c in text:
                if iseng(c):
                    if len(cur_jptext) > 0:
                        splited_text.append(cur_jptext)
                        cur_jptext = ''
                    cur_entext += c
                else:
                    if len(cur_entext) > 0:
                        splited_text.append(cur_entext)
                        cur_entext = ''
                    cur_jptext += c
            else:
                if len(cur_jptext) > 0:
                    splited_text.append(cur_jptext)
                if len(cur_entext) > 0:
                    splited_text.append(cur_entext)

            cur_y = 0
            pad_top = 0
            base_line = 0
            process_text = ''
            remain_str = ''
            image = None
            position = np.zeros([0,4])
            code_list = np.zeros([0,2], dtype=int)
            pad_space /= len(splited_text)
            for i, segment in enumerate(splited_text):
                if line_length > 0 and cur_y + pad_top >= line_length:
                    for seg in splited_text[i:]:
                        remain_str += seg
                    break

                cur_line_length = line_length - cur_y - pad_top
                if cur_line_length <= 0:
                    cur_line_length = 0
                if iseng(segment[0]):
                    seg_buf = self._word_render(segment.split(' '), line_length=cur_line_length, pad_space=pad_space)

                    if seg_buf is None:
                        return None

                    if seg_buf['image'] is not None and seg_buf['image'].shape[1] < self.fontsize * 1.1 and seg_buf['remain_str'] == '':
                        # 縦中横
                        pos = seg_buf['position']
                        top = self.fontsize - seg_buf['base_line']
                        base = seg_buf['image'].shape[1] / 2
                        next_cur = self.fontsize
                        seg_buf = {
                            "image": seg_buf['image'],
                            "str": seg_buf['str'],
                            "remain_str": seg_buf['remain_str'],
                            "position": pos,
                            "code_list": seg_buf['code_list'],
                            "pad_top": top,
                            "base_line": base,
                            "next_cur": int(next_cur),
                        }
                    else:
                        pos = seg_buf['position']
                        pos[:,1] = pos[:,1] - seg_buf['base_line']
                        pos = pos[:,[1,0,3,2]]

                        if seg_buf['image'] is None:
                            seg_buf = {
                                "image": None,
                                "str": seg_buf['str'],
                                "remain_str": seg_buf['remain_str'],
                                "position": pos,
                                "code_list": seg_buf['code_list'],
                                "pad_top": seg_buf['pad_left'],
                                "base_line": seg_buf['base_line'],
                                "next_cur": int(seg_buf['next_cur']),
                            }
                        else:
                            fix_base_lise = seg_buf['image'].shape[0]-seg_buf['base_line'] + self.fontsize * 0.33
                            pos[:,0] = seg_buf['image'].shape[0] - seg_buf['base_line'] - pos[:,0] - 1

                            seg_buf = {
                                "image": seg_buf['image'].T[:,::-1],
                                "str": seg_buf['str'],
                                "remain_str": seg_buf['remain_str'],
                                "position": pos,
                                "code_list": seg_buf['code_list'],
                                "pad_top": seg_buf['pad_left'],
                                "base_line": fix_base_lise,
                                "next_cur": int(seg_buf['next_cur']),
                            }
                else:
                    seg_buf = jp_word_sep(segment, line_length=cur_line_length, pad_space=pad_space)

                    if seg_buf is None:
                        return None

                if image is None:
                    image = seg_buf['image']
                    pos = seg_buf['position']
                    base_line = seg_buf['base_line']
                    pad_top += cur_y + seg_buf['pad_top']
                    process_text += seg_buf['str']
                    remain_str = seg_buf['remain_str']
                    cur_y = seg_buf['next_cur'] + pad_space - pad_top
                    position = np.concatenate([position, pos], axis=0)             
                    code_list = np.concatenate([code_list, seg_buf['code_list']], axis=0)               
                else:
                    pos = seg_buf['position']
                    code_list = np.concatenate([code_list, seg_buf['code_list']], axis=0)               
                    process_text += seg_buf['str']
                    remain_str = seg_buf['remain_str']
                    im = seg_buf['image']
                    if im is None:
                        cur_y += seg_buf['next_cur']
                        pos[:,1] += cur_y
                        position = np.concatenate([position, pos], axis=0)
                    else:                 
                        top = cur_y + seg_buf['pad_top']
                        next_cur = cur_y + seg_buf['next_cur'] + pad_space
                        top = int(top)
                        next_cur = int(next_cur)
                        if top < 0:
                            pad_top += -top
                            image = np.pad(image, [[-top, 0],[0, 0]])
                            next_cur += -top
                            position[:,1] += -top
                            top += -top

                        left = base_line - seg_buf['base_line']
                        left = int(left)
                        if left < 0:
                            pad_left = -left
                            image = np.pad(image, [[0, 0],[pad_left, 0]])
                            base_line += pad_left
                            left += pad_left
                            position[:,0] += pad_left

                        right = left + im.shape[1]
                        right = int(right)
                        if right > image.shape[1]:
                            image = np.pad(image, [[0, 0],[0, right-image.shape[1]]])

                        bottom = top + im.shape[0]
                        bottom = int(bottom)
                        if bottom > image.shape[0]:
                            image = np.pad(image, [[0, bottom-image.shape[0]],[0,0]])

                        pos[:,0] += left
                        pos[:,1] += top
                        image[top:bottom, left:right] = np.maximum(image[top:bottom, left:right], im)
                        position = np.concatenate([position, pos], axis=0)                    

                        cur_y = next_cur

                if len(seg_buf['remain_str']) > 0:
                    for seg in splited_text[i+1:]:
                        remain_str += seg
                    break;

            return {
                "image": image,
                "str": process_text,
                "remain_str": remain_str,
                "position": position,
                "code_list": code_list,
                "pad_top": pad_top,
                "base_line": base_line,
                "next_cur": int(cur_y),
            }

    def _ruby_line_render(self, text, ruby, pre_allow=True, post_allow=True, horizontal=True, ruby_dist=1.0):
        def ruby_small(buf):
            image = buf['image']
            if image is not None:
                image = Image.fromarray(image).resize([int(image.shape[1]*self.ruby_ratio),int(image.shape[0]*self.ruby_ratio)], resample=Resampling.BILINEAR)
                buf['image'] = np.asarray(image)
            buf['position'] *= np.array([[self.ruby_ratio, self.ruby_ratio, self.ruby_ratio, self.ruby_ratio]])
            buf['base_line'] *= self.ruby_ratio
            if 'pad_top' in buf:
                buf['pad_top'] *= self.ruby_ratio
            if 'pad_left' in buf:
                buf['pad_left'] *= self.ruby_ratio
            buf['next_cur'] *= self.ruby_ratio
            buf['code_list'] += np.array([[0, 1]])
            return buf

        if horizontal:
            text_buf = self._horizontal_line_render(text)
            ruby_buf = ruby_small(self._horizontal_line_render(ruby))

            if text_buf['image'] is None or ruby_buf['image'] is None:
                return {
                    "image": None,
                    "text" : {
                        "str": '',
                        "remain_str": '',
                    },
                    "ruby" : {
                        "str": '',
                        "remain_str": '',
                    },
                    "position": np.zeros([0,4]),
                    "code_list": np.zeros([0,2], dtype=int),
                    "pad_left": 0,
                    "base_line": 0,
                    "next_cur": 0,
                }

            ruby_len = len(ruby_buf['str'])
            text_len = len(text_buf['str'])
            textadd_len = text_len
            if is_hiragana(ruby) and is_kanji(text):
                if pre_allow:
                    textadd_len += 0.5
                if post_allow:
                    textadd_len += 0.5

                if textadd_len * 2 < ruby_len:
                    # ルビの方が長いのでそちらに合わせる
                    text_pad = ruby_buf['image'].shape[1] - text_buf['image'].shape[1]
                    count = sum([0 if 0x20 <= ord(c) < 0x7F and c != ' ' else 1 for c in text])
                    if ' ' in text:
                        count += 1
                    if count > 1:
                        text_pad /= count - 1
                        text_buf = self._horizontal_line_render(text, pad_space=text_pad)
                elif text_len * 2 > ruby_len:
                    # ルビの方が短いので、本文に合わせる
                    ruby_pad = text_buf['image'].shape[1] - ruby_buf['image'].shape[1]
                    count = sum([0 if 0x20 <= ord(c) < 0x7F and c != ' ' else 1 for c in ruby])
                    if ' ' in ruby:
                        count += 1
                    if count > 1:
                        ruby_pad /= count - 1
                        ruby_buf = ruby_small(self._horizontal_line_render(ruby, pad_space=ruby_pad/self.ruby_ratio))
            else:
                if ruby_buf['image'].shape[1] > text_buf['image'].shape[1]:
                    # ルビの方が長いのでそちらに合わせる
                    text_pad = ruby_buf['image'].shape[1] - text_buf['image'].shape[1]
                    count = sum([0 if 0x20 <= ord(c) < 0x7F and c != ' ' else 1 for c in text])
                    if ' ' in text:
                        count += 1
                    if count > 1:
                        text_pad /= count - 1
                        text_buf = self._horizontal_line_render(text, pad_space=text_pad)
                elif text_buf['image'].shape[1] > ruby_buf['image'].shape[1]:
                    # ルビの方が短いので、本文に合わせる
                    ruby_pad = text_buf['image'].shape[1] - ruby_buf['image'].shape[1]
                    count = sum([0 if 0x20 <= ord(c) < 0x7F and c != ' ' else 1 for c in ruby])
                    if ' ' in ruby:
                        count += 1
                    if count > 1:
                        ruby_pad /= count - 1
                        ruby_buf = ruby_small(self._horizontal_line_render(ruby, pad_space=ruby_pad/self.ruby_ratio))


            image = text_buf['image']
            pad_left = text_buf['pad_left']
            base_line = text_buf['base_line']
            text_position = text_buf['position']
            next_cur = text_buf['next_cur']

            ruby_baseline = base_line - self.fontsize * ruby_dist
            ruby_im = ruby_buf['image']
            ruby_position = ruby_buf['position']

            left = (text_buf['image'].shape[1] - ruby_buf['image'].shape[1]) / 2
            left = int(left)
            if textadd_len * 2 >= ruby_len:
                if left < 0:
                    image = np.pad(image, [[0, 0],[-left, 0]])
                    pad_left += left
                    text_position[:,0] += -left
                    left = 0
            elif left < 0:
                pad_left = 0
                image = np.pad(image, [[0, 0],[-left, 0]])
                next_cur = ruby_buf['image'].shape[1]
                text_position[:,0] += -left
                left = 0

            top = ruby_baseline - ruby_buf['base_line']
            top = int(top)
            if top < 0:
                pad_top = -top
                image = np.pad(image, [[pad_top, 0],[0, 0]])
                ruby_baseline += pad_top
                base_line += pad_top
                top += pad_top
                text_position[:,1] += pad_top

            right = left + ruby_im.shape[1]
            right = int(right)
            if right > image.shape[1]:
                image = np.pad(image, [[0, 0],[0, right-image.shape[1]]])

            bottom = top + ruby_im.shape[0]
            bottom = int(bottom)
            if bottom > image.shape[0]:
                image = np.pad(image, [[0, bottom-image.shape[0]],[0,0]])

            ruby_position[:,0] += left
            ruby_position[:,1] += top
            image[top:bottom, left:right] = np.maximum(image[top:bottom, left:right], ruby_im)

            return {
                "image": image,
                "text" : {
                    "str": text_buf['str'],
                    "remain_str": text_buf['remain_str'],
                },
                "ruby" : {
                    "str": ruby_buf['str'],
                    "remain_str": ruby_buf['remain_str'],
                },
                "position": np.concatenate([text_position, ruby_position], axis=0),
                "code_list": np.concatenate([text_buf['code_list'] + np.array([[0, 2]]), ruby_buf['code_list']], axis=0),
                "pad_left": pad_left,
                "base_line": base_line,
                "next_cur": int(next_cur),
            }
        else:
            text_buf = self._vertical_line_render(text)
            ruby_buf = ruby_small(self._vertical_line_render(ruby))

            if text_buf['image'] is None or ruby_buf['image'] is None:
                return {
                    "image": None,
                    "text" : {
                        "str": '',
                        "remain_str": '',
                    },
                    "ruby" : {
                        "str": '',
                        "remain_str": '',
                    },
                    "position": np.zeros([0,4]),
                    "code_list": np.zeros([0,2], dtype=int),
                    "pad_top": 0,
                    "base_line": 0,
                    "next_cur": 0,
                }

            ruby_len = len(ruby_buf['str'])
            text_len = len(text_buf['str'])
            textadd_len = text_len
            if is_hiragana(ruby) and is_kanji(text):
                if pre_allow:
                    textadd_len += 0.5
                if post_allow:
                    textadd_len += 0.5

                if textadd_len * 2 < ruby_len:
                    # ルビの方が長いのでそちらに合わせる
                    text_pad = ruby_buf['image'].shape[0] - text_buf['image'].shape[0]
                    count = sum([0 if 0x20 <= ord(c) < 0x7F and c != ' ' else 1 for c in text])
                    if ' ' in text:
                        count += 1
                    if count > 1:
                        text_pad /= count - 1
                        text_buf = self._vertical_line_render(text, pad_space=text_pad)
                elif text_len * 2 > ruby_len:
                    # ルビの方が短いので、本文に合わせる
                    ruby_pad = text_buf['image'].shape[0] - ruby_buf['image'].shape[0]
                    count = sum([0 if 0x20 <= ord(c) < 0x7F and c != ' ' else 1 for c in ruby])
                    if ' ' in ruby:
                        count += 1
                    if count > 1:
                        ruby_pad /= count - 1
                        ruby_buf = ruby_small(self._vertical_line_render(ruby, pad_space=ruby_pad/self.ruby_ratio))
            else:
                if ruby_buf['image'].shape[0] > text_buf['image'].shape[0]:
                    # ルビの方が長いのでそちらに合わせる
                    text_pad = ruby_buf['image'].shape[0] - text_buf['image'].shape[0]
                    count = sum([0 if 0x20 <= ord(c) < 0x7F and c != ' ' else 1 for c in text])
                    if ' ' in text:
                        count += 1
                    if count > 1:
                        text_pad /= count - 1
                        text_buf = self._vertical_line_render(text, pad_space=text_pad)
                elif text_buf['image'].shape[0] > ruby_buf['image'].shape[0]:
                    # ルビの方が短いので、本文に合わせる
                    ruby_pad = text_buf['image'].shape[0] - ruby_buf['image'].shape[0]
                    count = sum([0 if 0x20 <= ord(c) < 0x7F and c != ' ' else 1 for c in ruby])
                    if ' ' in ruby:
                        count += 1
                    if count > 1:
                        ruby_pad /= count - 1
                        ruby_buf = ruby_small(self._vertical_line_render(ruby, pad_space=ruby_pad/self.ruby_ratio))

            image = text_buf['image']
            pad_top = text_buf['pad_top']
            base_line = text_buf['base_line']
            text_position = text_buf['position']
            next_cur = text_buf['next_cur']

            ruby_baseline = base_line + self.fontsize * (ruby_dist - 0.5 + 0.5 * self.ruby_ratio)
            ruby_im = ruby_buf['image']
            ruby_position = ruby_buf['position']

            top = (text_buf['image'].shape[0] - ruby_buf['image'].shape[0]) / 2
            top = int(top)
            if textadd_len * 2 >= ruby_len:
                if top < 0:
                    image = np.pad(image, [[-top, 0],[0, 0]])
                    pad_top += top
                    text_position[:,1] += -top
                    top = 0
            elif top < 0:
                pad_top = 0
                image = np.pad(image, [[-top, 0],[0, 0]])
                next_cur = ruby_buf['image'].shape[0]
                text_position[:,1] += -top
                top = 0

            left = ruby_baseline - ruby_buf['base_line']
            left = int(left)
            if left < 0:
                pad_left = -left
                image = np.pad(image, [[0, 0],[pad_left, 0]])
                ruby_baseline += pad_left
                base_line += pad_left
                left += pad_left
                text_position[:,0] += pad_left

            right = left + ruby_im.shape[1]
            right = int(right)
            if right > image.shape[1]:
                image = np.pad(image, [[0, 0],[0, right-image.shape[1]]])

            bottom = top + ruby_im.shape[0]
            bottom = int(bottom)
            if bottom > image.shape[0]:
                image = np.pad(image, [[0, bottom-image.shape[0]],[0,0]])

            ruby_position[:,0] += left
            ruby_position[:,1] += top
            image[top:bottom, left:right] = np.maximum(image[top:bottom, left:right], ruby_im)

            return {
                "image": image,
                "text" : {
                    "str": text_buf['str'],
                    "remain_str": text_buf['remain_str'],
                },
                "ruby" : {
                    "str": ruby_buf['str'],
                    "remain_str": ruby_buf['remain_str'],
                },
                "position": np.concatenate([text_position, ruby_position], axis=0),
                "code_list": np.concatenate([text_buf['code_list'] + np.array([[0, 2]]), ruby_buf['code_list']], axis=0),
                "pad_top": pad_top,
                "base_line": base_line,
                "next_cur": int(next_cur),
            }

    def _line_render(self, text, horizontal=True, line_length=0):        
        def split_ruby(text):
            content = text.split('\uFFF9', maxsplit=1)
            pretext = content[0]
            if len(content) > 1:
                if '\uFFFA' in content[1] and '\uFFFB' in content[1]:
                    content = content[1].split('\uFFFA', maxsplit=1)
                    basetext = content[0]
                    content = content[1].split('\uFFFB', maxsplit=1)
                    rubytext = content[0]
                    posttext = content[1]
                    return pretext, basetext, rubytext, posttext
                else:
                    return pretext, None, None, None
            else:
                return pretext, None, None, None

        segment_buf = []
        remain_length = line_length
        remain_text = text
        process_str = ''
        remain_str = ''
        while True:
            pretext, basetext, rubytext, posttext = split_ruby(remain_text)
            if self.line_space_ratio - 1 < self.ruby_ratio:
                # 行間が狭いので、ルビをつけない
                if basetext is not None:
                    pretext += basetext
                    basetext = None
            if horizontal:
                buf = self._horizontal_line_render(pretext, line_length=remain_length)
            else:
                buf = self._vertical_line_render(pretext, line_length=remain_length)
            if buf is None:
                return None
            segment_buf.append(buf)
            remain_length -= buf['next_cur']
            process_str += buf['str']
            remain_str = buf['remain_str']
            if len(remain_str) > 0 or (line_length > 0 and remain_length <= 0):
                if posttext is not None:
                    if basetext is None:
                        remain_str += posttext
                    else:
                        remain_str += '\uFFF9' + basetext + '\uFFFA' + rubytext + '\uFFFB' + posttext
                break
            if posttext is None:
                break
            elif basetext is None:
                remain_text = posttext
            else:
                if rubytext in emphasis_characters:
                    for i in range(len(basetext)):
                        buf = self._ruby_line_render(basetext[i], rubytext, horizontal=horizontal)
                        if buf is None:
                            return None
                        if line_length > 0 and buf['next_cur'] >= remain_length:
                            remain_str += '\uFFF9' + basetext[i:] + '\uFFFA' + rubytext + '\uFFFB' + posttext
                            break
                        if buf['image'] is None:
                            break
                        buf['code_list'] += np.array([[0, 4]])
                        segment_buf.append(buf)
                        remain_length -= buf['next_cur']
                        rstr = '\uFFF9' + buf['text']['str'] + '\uFFFA' + buf['ruby']['str'] + '\uFFFB'
                        process_str += rstr                
                    if line_length > 0 and buf['next_cur'] >= remain_length:
                        break
                    remain_str = posttext
                    remain_text = posttext
                    continue
                pre_allow = allow_rubyover(pretext[-1:])
                post_allow = allow_rubyover(posttext[:1])
                buf = self._ruby_line_render(basetext, rubytext, pre_allow=pre_allow, post_allow=post_allow, horizontal=horizontal)
                if buf is None:
                    return None
                if line_length > 0 and buf['next_cur'] >= remain_length:
                    if basetext is None:
                        remain_str += posttext
                    else:
                        remain_str += '\uFFF9' + basetext + '\uFFFA' + rubytext + '\uFFFB' + posttext
                    break
                if buf['image'] is None:
                    remain_str = posttext
                    remain_text = posttext
                    continue
                segment_buf.append(buf)
                remain_length -= buf['next_cur']
                rstr = '\uFFF9' + buf['text']['str'] + '\uFFFA' + buf['ruby']['str'] + '\uFFFB'
                process_str += rstr                
                remain_str = posttext
                remain_text = posttext

        if horizontal:
            cur_x = 0
            pad_left = 0
            base_line = 0
            position = np.zeros([0,4])
            code_list = np.zeros([0,2], dtype=int)
            image = None
            for seg_buf in segment_buf:
                if image is None:
                    image = seg_buf['image']
                    pos = seg_buf['position']
                    base_line = seg_buf['base_line']
                    pad_left = cur_x + seg_buf['pad_left']
                    cur_x = seg_buf['next_cur'] - seg_buf['pad_left']
                    position = np.concatenate([position, pos], axis=0)                    
                    code_list = np.concatenate([code_list, seg_buf['code_list']], axis=0)                
                else:
                    pos = seg_buf['position']
                    im = seg_buf['image']
                    if im is None:
                        cur_x += seg_buf['next_cur']
                        pos[:,0] += cur_x
                        position = np.concatenate([position, pos], axis=0)
                        code_list = np.concatenate([code_list, seg_buf['code_list']], axis=0)                
                    else:                 
                        left = cur_x + seg_buf['pad_left']
                        next_cur = cur_x + seg_buf['next_cur']
                        left = int(left)
                        next_cur = int(next_cur)
                        if left < 0:
                            image = np.pad(image, [[0, 0],[-left, 0]])
                            next_cur += -left
                            position[:,0] += -left
                            left += -left

                        top = base_line - seg_buf['base_line']
                        top = int(top)
                        if top < 0:
                            pad_top = -top
                            image = np.pad(image, [[pad_top, 0],[0, 0]])
                            base_line += pad_top
                            top += pad_top
                            position[:,1] += pad_top

                        right = left + im.shape[1]
                        right = int(right)
                        if right > image.shape[1]:
                            image = np.pad(image, [[0, 0],[0, right-image.shape[1]]])

                        bottom = top + im.shape[0]
                        bottom = int(bottom)
                        if bottom > image.shape[0]:
                            image = np.pad(image, [[0, bottom-image.shape[0]],[0,0]])

                        pos[:,0] += left
                        pos[:,1] += top
                        image[top:bottom, left:right] = np.maximum(image[top:bottom, left:right], im)
                        position = np.concatenate([position, pos], axis=0)                    
                        code_list = np.concatenate([code_list, seg_buf['code_list']], axis=0)                

                        cur_x = next_cur

            return {
                "image": image,
                "str": process_str,
                "remain_str": remain_str,
                "position": position,
                "code_list": code_list,
                "pad_left": pad_left,
                "base_line": base_line,
                "next_cur": int(cur_x),
            }
        else:
            cur_y = 0
            pad_top = 0
            base_line = 0
            image = None
            position = np.zeros([0,4])
            code_list = np.zeros([0,2], dtype=int)
            for seg_buf in segment_buf:
                if image is None:
                    image = seg_buf['image']
                    pos = seg_buf['position']
                    base_line = seg_buf['base_line']
                    pad_top = cur_y + seg_buf['pad_top']
                    cur_y = seg_buf['next_cur'] - seg_buf['pad_top']
                    position = np.concatenate([position, pos], axis=0)                    
                    code_list = np.concatenate([code_list, seg_buf['code_list']], axis=0)                
                else:
                    pos = seg_buf['position']
                    im = seg_buf['image']
                    if im is None:
                        cur_y += seg_buf['next_cur']
                        pos[:,1] += cur_y
                        position = np.concatenate([position, pos], axis=0)
                        code_list = np.concatenate([code_list, seg_buf['code_list']], axis=0)                
                    else:                 
                        top = cur_y + seg_buf['pad_top']
                        next_cur = cur_y + seg_buf['next_cur']
                        top = int(top)
                        next_cur = int(next_cur)
                        if top < 0:
                            pad_top += -top
                            image = np.pad(image, [[-top, 0],[0, 0]])
                            next_cur += -top
                            position[:,1] += -top
                            top += -top

                        left = base_line - seg_buf['base_line']
                        left = int(left)
                        if left < 0:
                            pad_left = -left
                            image = np.pad(image, [[0, 0],[pad_left, 0]])
                            base_line += pad_left
                            left += pad_left
                            position[:,0] += pad_left

                        right = left + im.shape[1]
                        right = int(right)
                        if right > image.shape[1]:
                            image = np.pad(image, [[0, 0],[0, right-image.shape[1]]])

                        bottom = top + im.shape[0]
                        bottom = int(bottom)
                        if bottom > image.shape[0]:
                            image = np.pad(image, [[0, bottom-image.shape[0]],[0,0]])

                        pos[:,0] += left
                        pos[:,1] += top
                        image[top:bottom, left:right] = np.maximum(image[top:bottom, left:right], im)
                        position = np.concatenate([position, pos], axis=0)                    
                        code_list = np.concatenate([code_list, seg_buf['code_list']], axis=0)                

                        cur_y = next_cur

            return {
                "image": image,
                "str": process_str,
                "remain_str": remain_str,
                "position": position,
                "code_list": code_list,
                "pad_top": pad_top,
                "base_line": base_line,
                "next_cur": int(cur_y + pad_top),
            }


if __name__=="__main__":
    from matplotlib import rcParams
    rcParams['font.serif'] = ['IPAexMincho', 'IPAPMincho', 'Hiragino Mincho ProN']

    import matplotlib.pyplot as plt
    # import glob
    # import os

    # jp_furi_fontlist = [
    #     'GenShinGothic-Monospace-Normal.ttf',
    #     'GenShinGothic-Normal.ttf',
    #     'GenShinGothic-P-Normal.ttf',
    #     'mgenplus-2m-regular.ttf',
    #     'mgenplus-2p-regular.ttf',
    # ]
    # jp_furi_fontlist = [os.path.join('data','jpfont',f) for f in jp_furi_fontlist]

    # text = '𠮷𠮷𠮷𠮷〰〰〰〰〰〰'
    
    # for fontfile in jp_furi_fontlist:
    #     with Canvas(fontfile, fontsize=48, horizontal=False) as canvas:
    #         #a = canvas.random_draw(words, 1024, 1024, rng)
    #         canvas.set_linewidth(870)
    #         canvas.set_linemax(10)
    #         # canvas.set_header('header test')
    #         # canvas.set_footer('footer test')
    #         a = canvas.draw(text)
    #         #print(a)
    #         plt.imshow(a['image'])
    #         plt.show()
    #     if a['str'] != '':
    #         print(fontfile)
    # exit()

    fontfile = 'data/jpfont/NotoSerifJP-Regular.otf'
    #fontfile = 'data/enfont/Gabriola.ttf'
     
    text = 'てすと（ここを割ると）どうなるか。長い文章だった場合、切るところが（あいうえおかきくけこ。さしすせそたちつてと。分割が大変長い文章になっていても）いけるか'

    text = 'テスト\uFFF9漢字\uFFFAかんじあいう\uFFFBのふりがな\nテスト漢字のふりがな\nてすと\nfilter'
    text = '　\uFFF9漢字\uFFFAかんじあいう\uFFFBのふりがな\n　漢字のふりがな\nてすと\nfilter'
    text = '　テスト\uFFF9漢字\uFFFAかんじあいう\uFFFBのふりがな\n　テスト漢字のふりがな\nてすと\nfilter'
    text = '\uFFF9漢字\uFFFA﹅\uFFFBに圏点'
    #text = 'aphs have appeared on the covers of Life, Sports Illustrated, Newsweek, Fortune, and Forbes, and in Time, The New '
    text = 'テスト\uFFF9漢字\uFFFAかんじあいう\uFFFBのふりがな\nテスト漢字のふりがな\nてすと\nfilter'
    text = 'ぱ於がづ、\uFFF9習び惣妨妨王託簪櫓\uFFFAhmiacwcn sy ziqrrhjuu\uFFFB括へ串らぐ冗ろ。'
    text = 'ぱ於がづ、\uFFF9習び惣妨妨王託簪櫓\uFFFAhmiacwcn a b ziqrrhjuu\uFFFB括へ串らぐ冗ろ。'
    text = 'ぱ於がづ、\uFFF9hm あ uu a\uFFFA習び妨王託簪櫓\uFFFB括へ串らぐ冗ろ。'
    # text = '　\uFFF9埀銓う蒋ウサう獗巫医モ鱇ホ\uFFFAｖＮｖＰ\uFFFB屯怜る。'
    values = [
        ['身長','体重','性別'],
        ['192.8','48.4','女'],
        ['152.0','56.5','男'],
        ['136.2','33.3','女'],
        ['122.4','45.1','女'],
        ['172.1','71.0','男'],
        ['156.8','53.8','女'],
        ['','','ここに\nコメント'],
        ['160.4','46.0','男'],
    ]
    words = [
        'test1',
        'test2',
        'test3',
        'test4',
        'test5',
        'test6',
        'test7',
    ]

    rng = np.random.default_rng()

    with Canvas(fontfile, fontsize=48, horizontal=False) as canvas:
        #a = canvas.random_draw(words, 1024, 1024, rng)
        canvas.set_linewidth(1600)
        canvas.set_linemax(10)
        # canvas.set_header('header test')
        # canvas.set_footer('footer test')
        a = canvas.draw(text)
        #a = canvas.draw_wari(text)

        #a = canvas.random_drawgrid(values)
        # a = canvas.random_draw(text.splitlines(), 512*2, 512*2)
        # canvas.set_linewidth(870)
        # canvas.set_linemax(10)
        # canvas.set_section(2, 0.45)
        # # canvas.line_space_ratio = 1.5
        # # canvas.ruby_ratio = 0.5
        # # #a = canvas._horizontal_line_render('      test test   test test', line_length=0)
        # # #a = canvas._vertical_line_render('test val日本語　のtest', line_length=0)
        # a = canvas.draw(text)
        #a = canvas._draw_buffer('　　二文字下げ')
        print(a)
        plt.imshow(a['image'])

        for (c, t), (cx, cy, w, h) in zip(a['code_list'], a['position']):
            points = [
                [cx - w / 2, cy - h / 2],
                [cx + w / 2, cy - h / 2],
                [cx + w / 2, cy + h / 2],
                [cx - w / 2, cy + h / 2],
                [cx - w / 2, cy - h / 2],
            ]
            points = np.array(points)
            plt.plot(points[:,0], points[:,1], linewidth=0.5)
        #     plt.text(cx, cy, chr(c), fontsize=28, color='blue', family='serif')

        plt.figure()
        plt.imshow(a['sep_image'])

        plt.figure()
        plt.imshow(a['textline_image'])

        plt.show()
    

