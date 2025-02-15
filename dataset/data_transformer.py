import torch
import numpy as np
import os
import glob
import csv
import re
import json

from util_func import feature_dim
from const import encoder_add_dim, max_decoderlen, max_encoderlen, decoder_SOT, decoder_EOT, decoder_MSK
encoder_dim = feature_dim + encoder_add_dim

train_data3 = 'train_data3'
train_data4 = 'train_data4'

rng = np.random.default_rng()

emphasis_characters = ['•','◦','●','○','◎','◉','▲','△','﹅','﹆']

doublew1_list = 'ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺ'
doublew2_list = 'ａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚ'
doublew1_list = list(doublew1_list)
doublew2_list = list(doublew2_list)

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
    return s and s in '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz~!@#$%^&*()_+-={}[]|\\:;"\'<>,.?/‘’“”'

jp_type_list = {}
with open(os.path.join('data','id_map.csv'),'r') as f:
    reader = csv.reader(f)
    for row in reader:
        code = bytes.fromhex(row[2]).decode()
        i = int.from_bytes(code.encode('utf-32le'), 'little')
        jp_type_list.setdefault(int(row[3]), [])
        jp_type_list[int(row[3])].append(code)

def skip_remainruby(txt):
    idx2 = txt.find('\uFFFB')
    if idx2 >= 0:
        idx1 = txt.find('\uFFF9')
        if idx1 < 0 or idx1 > idx2:
            return txt[idx2+1:]
    return txt

def find_splitpoint(txt, start=0, split_count=-1):
    # print('sprit:', split_count)
    if split_count == 0:
        return start
    i = start
    if split_count < 0:
        split_count = len(txt) - i
    idx0 = txt.find('\n', i, i+split_count)
    if idx0 >= 0:
        return idx0+1
    idx1 = txt.find('\uFFF9', i, i+split_count)
    if idx1 < 0:
        return min(i+split_count+1, len(txt))
    idx2 = txt.find('\uFFFA', idx1)
    idx3 = txt.find('\uFFFB', idx1)
    if idx3 >= i+split_count:
        return idx3+1
    return find_splitpoint(txt, start=idx3+1, split_count=i+split_count-idx3)

def get_random_furigana():
    # 0: num
    # 1: A
    # 2: a
    # 3: あ
    # 4: ア
    # 5: 亜
    # 8: 弌

    if rng.uniform() < 0.5:
        out_count = max_decoderlen-2
    else:
        out_count = rng.integers(1,max_decoderlen-2)

    count = 100
    txt = '　'
    for _ in range(count):
        if len(txt) > out_count:
            break
        p = rng.random()
        if p < 0.25:
            # 漢字にひらがな
            if rng.random() < 0.2:
                before = ''.join(rng.choice(jp_type_list[5], rng.integers(1,5)))
            else:
                before = ''.join(rng.choice(jp_type_list[3], 1))
            m_l = rng.integers(1, 10)
            main = ''.join(rng.choice(jp_type_list[5]+jp_type_list[8]+jp_type_list[9]+jp_type_list[10], m_l))
            ruby = ''.join(rng.choice(jp_type_list[3] + ['ー'], rng.integers(1, m_l * 2 + 2)))
            if rng.random() < 0.2:
                after = ''.join(rng.choice(jp_type_list[5], rng.integers(1,5)))
            else:
                after = ''.join(rng.choice(jp_type_list[3], 1))
            if rng.random() < 0.5:
                txt += '\uFFF9'+main+'\uFFFA'+ruby+'\uFFFB'
            else:
                txt += before+'\uFFF9'+main+'\uFFFA'+ruby+'\uFFFB'+after
        elif p < 0.35:
            # 日本語に傍点
            m_l = rng.integers(1, 15)
            main1 = ''.join(rng.choice(jp_type_list[3], 20))
            main2 = ''.join(rng.choice(jp_type_list[4], 10))
            main3 = ''.join(rng.choice(jp_type_list[5]+jp_type_list[8]+jp_type_list[9]+jp_type_list[10], 10))
            main = ''.join(rng.choice(list(main1+main2+main3+'ー'), m_l))
            if rng.random() < 0.95:
                ruby = ''.join(list(rng.choice(['●','﹅'],1)) * m_l)
            else:
                ruby = ''.join(list(rng.choice(['•','◦','●','○','◎','◉','▲','△','﹅','﹆'],1)) * m_l)
            txt += '\uFFF9'+main+'\uFFFA'+ruby+'\uFFFB'
        elif p < 0.55:
            # 漢字にカタカナ
            kanjis = list(rng.choice(jp_type_list[5]+jp_type_list[8]+jp_type_list[9]+jp_type_list[10], 40))
            m_l = rng.integers(1, 15)
            main = ''.join(rng.choice(jp_type_list[3]+jp_type_list[4]+kanjis+['ー'], m_l))
            ruby = ''.join(rng.choice(jp_type_list[4] + ['ー'], rng.integers(3, m_l * 2 + 3)))
            txt += '\uFFF9'+main+'\uFFFA'+ruby+'\uFFFB'
        elif p < 0.7:
            # alphabetに日本語
            if rng.random() < 0.5:
                m_l = rng.integers(3, 20)
                if rng.random() < 0.5:
                    main = ''.join(rng.choice(doublew1_list, m_l))
                else:
                    main = ''.join(rng.choice(doublew1_list + doublew2_list, m_l))
            else:
                word = []
                m_l = 0
                while rng.random() < 0.5 or m_l < 6:
                    m_l1 = rng.integers(2, 10)
                    m_l += m_l1
                    word.append(''.join(rng.choice(jp_type_list[2], m_l1)))
                main = ' '.join(word)
                m_l = 10
            kanjis = list(rng.choice(jp_type_list[5]+jp_type_list[8], 100))
            if rng.random() < 0.5:
                m_l2 = rng.integers(3, m_l + 3)
            else:
                m_l2 = rng.integers(m_l // 5 + 3, m_l // 3 + 4)
            ruby = ''.join(rng.choice(jp_type_list[3]+jp_type_list[4]+kanjis+['ー'], m_l2))
            txt += '\uFFF9'+main+'\uFFFA'+ruby+'\uFFFB'
        elif p < 0.85:
            # 日本語にalphabet
            kanjis = list(rng.choice(jp_type_list[5]+jp_type_list[8], 100))
            m_l = rng.integers(3, 20)
            main = ''.join(rng.choice(jp_type_list[3]+jp_type_list[4]+kanjis+['ー'], m_l))
            if rng.random() < 0.5:
                if rng.random() < 0.5:
                    m_l2 = rng.integers(m_l // 5 + 3, m_l // 3 + 4)
                else:
                    m_l2 = rng.integers(m_l, m_l * 3)
                if rng.random() < 0.5:
                    ruby = ''.join(rng.choice(doublew1_list, m_l2))
                else:
                    ruby = ''.join(rng.choice(doublew1_list + doublew2_list, m_l2))
            else:
                word = []
                m_l2 = 0
                while rng.random() < 0.5 or m_l2 < 6 or m_l * 2 > m_l2:
                    m_l1 = rng.integers(2, 10)
                    m_l2 += m_l1
                    word.append(''.join(rng.choice(jp_type_list[2], m_l1)))
                ruby = ' '.join(word)
            txt += '\uFFF9'+main+'\uFFFA'+ruby+'\uFFFB'
        else:
            #日本語に日本語
            kanjis = list(rng.choice(jp_type_list[5]+jp_type_list[8]+jp_type_list[9]+jp_type_list[10], 400))
            m_l = rng.integers(3, 12)
            main = ''.join(rng.choice(jp_type_list[3]+jp_type_list[4]+kanjis+['ー'], m_l))
            kanjis = list(rng.choice(jp_type_list[5]+jp_type_list[8]+jp_type_list[9]+jp_type_list[10], 400))
            if rng.random() < 0.5:
                m_l2 = rng.integers(3, m_l // 5 + 4)
            else:
                m_l2 = rng.integers(m_l, m_l * 2 + 1)
            ruby = ''.join(rng.choice(jp_type_list[3]+jp_type_list[4]+kanjis+['ー'], m_l2))
            txt += '\uFFF9'+main+'\uFFFA'+ruby+'\uFFFB'

        if rng.random() < 0.2:
            txt += '\n'
        else:
            m_l = rng.integers(1, 10)
            main = ''.join(rng.choice(jp_type_list[3]+list(rng.choice(jp_type_list[5]+jp_type_list[8]+jp_type_list[9]+jp_type_list[10], 100)), m_l))
            txt += main

            if rng.random() < 0.05:
                txt += '\n　'
            elif rng.random() < 0.1:
                txt += '　'
            elif rng.random() < 0.4:
                txt += '、'
            elif rng.random() < 0.4:
                txt += '。'
            elif rng.random() < 0.4:
                txt += '——'
            elif rng.random() < 0.1:
                txt += '！　'
            elif rng.random() < 0.1:
                txt += '？　'
            elif rng.random() < 0.1:
                txt += '‼　'
            elif rng.random() < 0.1:
                txt += '⁉　'
            elif rng.random() < 0.1:
                txt += '⁇　'
            elif rng.random() < 0.1:
                txt += '⁈　'

    txt = skip_remainruby(txt)
    for j in range(out_count):
        if txt[j] in UNICODE_WHITESPACE_CHARACTERS:
            out_count -= 1
        if txt[j] == '\uFFF9':
            out_count -= 3
    split_count = rng.integers(20, 80)
    # print('sprit:', split_count, out_count)
    outtxt = ''
    i = 0
    while i < len(txt):
        j = find_splitpoint(txt, i, split_count)
        if outtxt and j > out_count:
            break
        outtxt += txt[i:j] + ('' if txt[j-1] == '\n' else '\n')
        i = j
        if i > out_count:
            break
    return outtxt

class TransformerDataDataset(torch.utils.data.Dataset):
    @staticmethod
    def prepare():
        charparam = {}
        with np.load(os.path.join(train_data3, 'features.npz')) as data:
            codes = sorted(set([int(s.split('_')[1]) for s in data.files]))
            for i,code in enumerate(codes):
                print(i, '/', len(codes), code)
                charparam[code] = (data.get('hori_%d'%code, None), data.get('vert_%d'%code, None))
        return codes, charparam

    def __init__(self, codes, charparam, train=True):
        super().__init__()
        self.SP_token = np.zeros([encoder_dim], dtype=np.float16)
        self.SP_token[0:feature_dim:2] = 5
        self.SP_token[1:feature_dim:2] = -5
        txtfiles = sorted(glob.glob(os.path.join(train_data3,'*','*.txt')))
        if train:
            idx = [i for i in range(len(txtfiles)) if i % 10 > 0]
        else:
            idx = [i for i in range(len(txtfiles)) if i % 10 == 0]
        self.txtfile = list(np.array(txtfiles)[idx])
        self.codes = codes
        self.hcodes = []
        self.vcodes = []
        for c in codes:
            if charparam[c][0] is not None:
                self.hcodes.append(c)
            if charparam[c][1] is not None:
                self.vcodes.append(c)
        self.charparam = charparam

        self.real_ratio = 100
        self.realdata = []
        if train:
            npyfiles = sorted(glob.glob(os.path.join(train_data4, '*.npy')))
            for i,filename in enumerate(npyfiles):
                basename = os.path.splitext(filename)[0]
                feat = np.load(filename)
                print(i, '/', len(npyfiles), basename, feat.shape[0])
                with open(basename+'.json', 'r', encoding='utf-8') as file:
                    data = json.load(file)

                prev_block = 0
                prev_line = 0
                feature_values = []
                target_text = ""
                feature_idx = []
                vertical = 0
                ruby_state = 0
                for box in data['boxlist']:
                    boxid = box['boxid']
                    blockid = box['blockid']
                    lineid = box['lineid']
                    subidx = box['subidx']
                    subtype = box['subtype']
                    text = box['text']

                    if prev_block != blockid:
                        prev_block = blockid
                        g = np.zeros([encoder_dim], np.float16)
                        g[feature_dim+0] = 5 * vertical
                        g[-1] = 5
                        feature_values.append(g)
                        feature_idx.append(len(target_text))
                        if ruby_state == 2:
                            target_text += '\uFFFB'
                        ruby_state = 0
                        target_text += '\n'
                        prev_line = -1
                    if prev_line != lineid:
                        prev_line = lineid
                        g = np.zeros([encoder_dim], np.float16)
                        g[feature_dim+0] = 5 * vertical
                        g[-1] = 5
                        feature_values.append(g)
                        feature_idx.append(len(target_text))
                        if ruby_state == 2:
                            target_text += '\uFFFB'
                        ruby_state = 0
                        target_text += '\n'

                    cur_idx = len(target_text)
                    if subtype & 8 == 8:
                        space = 1
                        if is_ascii(text):
                            target_text += ' '
                        else:
                            target_text += '　'
                    else:
                        space = 0

                    if subtype & (2+4) == 2+4:
                        if ruby_state == 1:
                            target_text += '\uFFFA'
                        ruby_state = 2
                    elif subtype & (2+4) == 2:
                        if ruby_state == 2:
                            target_text += '\uFFFB'
                        if ruby_state == 0:
                            target_text += '\uFFF9'
                        ruby_state = 1
                    elif subtype & (2+4) == 0:
                        if ruby_state == 2:
                            target_text += '\uFFFB'
                        ruby_state = 0

                    if subtype & 16 == 16:
                        emphasis = 1
                    else:
                        emphasis = 0

                    if subtype & 1 == 0:
                        vertical = 0
                    else:
                        vertical = 1

                    if ruby_state == 0:
                        rubybase = 0
                        ruby = 0
                    elif ruby_state == 1:
                        rubybase = 1
                        ruby = 0
                    elif ruby_state == 2:
                        rubybase = 0
                        ruby = 1

                    g = np.concatenate([feat[boxid], 5*np.array([vertical,rubybase,ruby,space,emphasis,0], np.float16)])
                    feature_values.append(g)
                    feature_idx.append(cur_idx)
                    if text is None:
                        target_text += '\uFFFD'
                    else:
                        target_text += text

                if len(feature_values) == 0:
                    continue
                feature_values.append(np.zeros([encoder_dim], np.float16))
                feature_idx.append(len(target_text))

                self.realdata.append({
                    'feature': np.array(feature_values, dtype=np.float16),
                    'index': np.array(feature_idx),
                    'text': target_text,
                })

        self.text = {}
        for i,filename in enumerate(self.txtfile):
            print(i, '/', len(self.txtfile), filename)
            with open(filename) as rf:
                txt = rf.read()
                txt = re.sub(r'　　+','　',txt)
                txt = re.sub(r'  +',' ',txt)
                txt = re.sub('\n\n\n+','\n\n',txt)
            self.text[filename] = txt

    def __len__(self):
        return (len(self.realdata) * self.real_ratio + len(self.txtfile) + 1) * 2
    
    def __getitem__(self, idx):
        if idx < len(self.realdata) * self.real_ratio:
            return self.load_realdata(idx % len(self.realdata))
        idx -= len(self.realdata) * self.real_ratio
        if idx < len(self.txtfile):
            vert_ok = os.path.basename(os.path.dirname(self.txtfile[idx])) in ['aozora','wikipedia_ja']
            return self.load_textfile(self.txtfile[idx], orientation='both' if vert_ok else 'horizontal')
        return self.random_text()

    def load_realdata(self, idx):
        text = self.realdata[idx]['text']
        index = self.realdata[idx]['index']
        if index.shape[0] > 10:
            start_idx = rng.integers(index.shape[0]-10)
        else:
            start_idx = 0
        if start_idx > 0:
            g = self.realdata[idx]['feature'][start_idx]
            # ruby, rubybase
            if g[-4] > 0 or g[-5] > 0:
                j = start_idx-1
                while j >= 0 and (g[-4] > 0 or g[-5] > 0):
                    g = self.realdata[idx]['feature'][j]
                    start_idx = j
                    j -= 1
        out_count = 0
        ruby_state = 0
        
        if rng.uniform() < 0.5:
            count = min(max_decoderlen-2,index.shape[0]-start_idx)
        else:
            count = rng.integers(1, min(max_decoderlen-2,index.shape[0]-start_idx))

        for j in range(start_idx, start_idx+count):
            end_idx = j
            out_count += 1
            if j >= index.shape[0]:
                break
            g = self.realdata[idx]['feature'][j]
            # newline
            if g[-1] > 0:
                out_count += 1
            # space
            if g[-3] > 0:
                out_count += 1
            # rubybase
            if g[-5] > 0:
                if ruby_state == 0:
                    out_count += 3
                ruby_state = 1
            # ruby
            elif g[-4] > 0:
                ruby_state = 2
            else:
                ruby_state = 0
            if ruby_state > 0 and out_count > max_decoderlen-10:
                break
            if out_count > max_decoderlen-3:
                break
        if end_idx < index.shape[0]:
            g = self.realdata[idx]['feature'][end_idx]
            # ruby, rubybase
            if g[-4] > 0 or g[-5] > 0:
                j = end_idx+1
                while j < index.shape[0] and (g[-4] > 0 or g[-5] > 0):
                    g = self.realdata[idx]['feature'][j]
                    end_idx = j
                    j += 1
        if end_idx+1 < index.shape[0]:
            end_idx += 1
        if end_idx-start_idx > max_encoderlen-2:
            end_idx = start_idx+max_encoderlen-2

        feat = np.zeros(shape=(max_encoderlen,feature_dim+encoder_add_dim), dtype=np.float16)
        feat[0,:] = self.SP_token # SOT
        txt = text[index[start_idx]:index[end_idx]]
        feat[0:end_idx-start_idx,:] += self.add_noize(self.realdata[idx]['feature'][start_idx:end_idx])
        if end_idx-start_idx < max_encoderlen:
            feat[end_idx-start_idx,:] = -self.SP_token # EOT
        return self.pad_output(txt, feat)

    def add_noize(self, value):
        return value * (1 + 1e-2 * rng.normal(loc=0, scale=1, size=value.shape)) + 0.1 * rng.normal(loc=0, scale=1, size=value.shape)

    def generage_feature(self, code, horizontal):
        hori, vert = self.charparam.get(code, (None, None))
        value = hori if horizontal else vert
        if value is None:
            value = rng.normal(loc=0, scale=5, size=(1,feature_dim))
        # print('found:',code,chr(code),mu,sd)
        return rng.choice(value, axis=0, replace=False)

    def gen_feature(self, text, orientation='both'):
        # 1 vertical
        # 2 ruby (base)
        # 3 ruby (text)
        # 4 space
        # 5 emphasis
        # 6 newline
        sp = False
        ruby = 0
        if orientation == 'horizontal':
            horizontal = True
        elif orientation == 'vertical':
            horizontal = False
        else:
            horizontal = rng.uniform() < 0.5
        emphasis_idx = []
        if re.findall('['+''.join(emphasis_characters)+']', text):
            ind = text.find('\uFFF9')
            while ind >= 0:
                if ind >= 0:
                    ind2 = text.find('\uFFFA',ind)
                else:
                    ind2 = -1    
                if ind2 >= 0:
                    ind3 = text.find('\uFFFB',ind2)
                else:
                    ind3 = -1    
                if ind >= 0 and ind2 >= 0 and ind3 >= 0:
                    if text[ind2+1] in emphasis_characters:
                        emphasis_idx.extend(list(range(ind+1,ind3)))
                    ind = text.find('\uFFF9', ind3)
                else:
                    break
        # print(emphasis_idx)
        ret = np.zeros(shape=(max_encoderlen,feature_dim+encoder_add_dim), dtype=np.float16)
        ret[0,:] = self.SP_token # SOT
        idx = 1
        for i,c in enumerate(text):
            if idx >= max_encoderlen:
                break
            if not horizontal:
                ret[idx,feature_dim+0] = 5
            if c == '\n':
                ret[idx,feature_dim+5] = 5
                sp = False
                idx += 1
                continue
            elif c in UNICODE_WHITESPACE_CHARACTERS:
                sp = True
                continue
            elif c == '\uFFF9':
                ruby = 1
                continue
            elif c == '\uFFFA':
                ruby = 2
                continue
            elif c == '\uFFFB':
                ruby = 0
                continue
            
            ret[idx,:feature_dim] = self.add_noize(self.generage_feature(ord(c), horizontal))
            if ruby == 1:
                ret[idx,feature_dim+1] = 5
            elif ruby == 2:
                ret[idx,feature_dim+2] = 5
            
            if sp:
                ret[idx,feature_dim+3] = 5
                sp = False

            if i in emphasis_idx:
                ret[idx,feature_dim+4] = 5
            idx += 1

        if idx < max_encoderlen:
            ret[idx,:] = -self.SP_token # EOT

        return ret

    def load_textfile(self, filename, orientation='both'):
        txt = self.text[filename]
        start_idx = rng.integers(len(txt)-1)
        txt = txt[start_idx:]
        txt = skip_remainruby(txt)
        if rng.uniform() < 0.5:
            out_count = min(max_decoderlen-2,len(txt))
        else:
            out_count = rng.integers(1, min(max_decoderlen-2,len(txt)))
        for j in range(out_count):
            if txt[j] in UNICODE_WHITESPACE_CHARACTERS:
                out_count -= 1
            if txt[j] == '\uFFF9':
                out_count -= 3
        split_count = rng.integers(20, 80)
        # print('sprit:', split_count, out_count)
        outtxt = ''
        i = 0
        while i < len(txt):
            j = find_splitpoint(txt, i, split_count)
            if outtxt and j > out_count:
                break
            outtxt += txt[i:j] + ('' if txt[j-1] == '\n' else '\n')
            i = j
            if i > out_count:
                break
        return self.pad_output(*self.format_output(outtxt, orientation=orientation))

    def random_text(self):
        if rng.uniform() < 0.5:
            return self.pad_output(*self.format_output(get_random_furigana(), orientation='both'))

        if rng.uniform() < 0.5:
            out_count = max_decoderlen-2
        else:
            out_count = rng.integers(1,max_decoderlen-2)
        split_count = rng.integers(20, 80)
        # print('sprit:', split_count)
        i = 0
        outtxt = ''
        horizontal = rng.uniform() < 0.5
        while i < out_count:
            if i > 0 and i+split_count+1 >= out_count:
                break
            if horizontal:
                outtxt += ''.join([chr(c) for c in rng.choice(self.hcodes, size=split_count)])+'\n'
            else:
                outtxt += ''.join([chr(c) for c in rng.choice(self.hcodes, size=split_count)])+'\n'
            i += split_count+1
        return self.pad_output(*self.format_output(outtxt, orientation='horizontal' if horizontal else 'vertical'))

    def format_output(self, text, orientation='both'):
        if text and rng.uniform() < 0.5 and text[-1] == '\n':
            text = text[:-1]
        if text:
            return text, self.gen_feature(text, orientation=orientation)
        else:
            ret = np.zeros(shape=(max_encoderlen,feature_dim+encoder_add_dim), dtype=np.float16)
            ret[0,:] = self.SP_token # SOT
            ret[1,:] = -self.SP_token # EOT
            return '', ret

    def pad_output1(self, text, feature):
        b = text.encode('utf-32-le')
        codes = [decoder_SOT] + [int.from_bytes(b[i:i+4], 'little') for i in range(0,len(b),4)] + [decoder_EOT]
        codes += [0] * max(0,max_decoderlen+1-len(codes))
        codes = np.array(codes, dtype=int)
        return text, feature, codes[:max_decoderlen+1]

    def pad_output(self, text, feature):
        b = text.encode('utf-32-le')
        codes = [decoder_SOT] + [int.from_bytes(b[i:i+4], 'little') for i in range(0,len(b),4)] + [decoder_EOT]
        codes += [0] * max(0,max_decoderlen-len(codes))
        codes = np.array(codes, dtype=int)
        input_codes = codes[:max_decoderlen]
        true_codes = np.array(codes[:max_decoderlen])
        p = rng.uniform()
        if p < 0.1:
            input_codes[1:] = decoder_MSK
        elif p < 0.2:
            p = rng.uniform()
            input_codes[1:] = np.where(rng.uniform(size=(max_decoderlen-1,)) < p, rng.integers(20, 0x10FFFF, size=(max_decoderlen-1,)), input_codes[1:])
            p = rng.uniform()
            input_codes[1:] = np.where(rng.uniform(size=(max_decoderlen-1,)) < p, decoder_MSK, input_codes[1:])
        else:
            p = rng.uniform()
            input_codes[1:] = np.where(rng.uniform(size=(max_decoderlen-1,)) < p, decoder_MSK, input_codes[1:])
        return text, feature, input_codes, true_codes


#######################################################
# encoder
# 
# input
#  100(feature) + 6(add)
#                 [vertical,rubybase,ruby,space,emphasis,newline]
#
# sot = [5,-5, ...]
# eot = [-5,5, ...]
#######################################################
# decoder
#
# input/output
#  utf-32 code
#
# sot = 1
# eot = 2
# pad = 0
#######################################################


if __name__=='__main__':
    from torch.utils.data import DataLoader
    from util_func import decode_ruby

    # import coremltools as ct
    # from util_func import modulo_list, calc_predid
    # import itertools

    # mlmodel_decoder = ct.models.MLModel('CodeDecoder.mlpackage')

    prep = TransformerDataDataset.prepare()
    training_dataset = TransformerDataDataset(*prep)

    # for data in training_dataset:
    #     print(data[0])
    #     print('//////////')
    #     print(decode_ruby(data[0]))
    #     print(data[1])
    #     print(data[2])
    #     print('---------')

    #     glyphids = []
    #     for d in data[1]:
    #         if np.all(d == 0):
    #             break
    #         d = d[:feature_dim]
    #         if np.all(d == 0):
    #             continue
    #         decode_output = mlmodel_decoder.predict({'feature_input': np.expand_dims(d, 0)})
    #         p = []
    #         id = []
    #         for k,m in enumerate(modulo_list):
    #             prob = decode_output['modulo_%d'%m][0]
    #             idx = np.where(prob > 0.01)[0]
    #             if len(idx) == 0:
    #                 idx = [np.argmax(prob)]
    #             if k == 0:
    #                 for i in idx[:3]:
    #                     id.append([i])
    #                     p.append([prob[i]])
    #             else:
    #                 id = [i1 + [i2] for i1, i2 in itertools.product(id, idx[:3])]
    #                 p = [i1 + [prob[i2]] for i1, i2 in itertools.product(p, idx[:3])]
    #         p = [np.exp(np.mean([np.log(prob) for prob in probs])) for probs in p]
    #         i = [calc_predid(*ids) for ids in id]
    #         g = sorted([(prob, id) for prob,id in zip(p,i)], key=lambda x: x[0] if x[1] <= 0x10FFFF else 0, reverse=True)
    #         prob,idx = g[0]
    #         glyphids.append(idx)
    #     predstr = ''
    #     for cid in glyphids:
    #         if cid < 0x10FFFF:
    #             predstr += chr(cid)
    #         else:
    #             predstr += '\uFFFD'
    #     print(predstr)
    #     print('++++++++++')

    # exit()

    # training_loader = DataLoader(training_dataset, batch_size=8, shuffle=True, num_workers=8, drop_last=True)
    training_loader = DataLoader(training_dataset, batch_size=1, shuffle=True, drop_last=True)

    # from time import time
    # t = time()
    for data in training_loader:
        print(data[0][0])
        print('//////////')
        print(decode_ruby(data[0][0]))
        print(data[1][0])
        print(data[2][0])
        print(data[3][0])
        print('---------')
