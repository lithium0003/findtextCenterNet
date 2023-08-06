#!/usr/bin/env python3

import tensorflow as tf
my_devices = tf.config.list_physical_devices(device_type='CPU')
tf.config.set_visible_devices(devices= my_devices, device_type='CPU')

import numpy as np
import os
import re
import glob
import time
from multiprocessing import Pool

from render_font.get_aozora import get_aozora_urls, get_contents, decode_ruby
from render_font.get_wikipedia import get_random_wordid, get_word_content
from render_font.renderer import UNICODE_WHITESPACE_CHARACTERS
from const import lines_per_file

tfdata_path = 'train_data2'

with open(os.path.join('data','wordlist.txt'),'r') as f:
    wordlist = f.read().splitlines()
wordlist = np.array(wordlist)

with open(os.path.join('data','en_wordlist.txt'),'r') as f:
    en_wordlist = f.read().splitlines()
en_wordlist = np.array(en_wordlist)

aozora_urls = get_aozora_urls()

npz_file = np.load('charparam.npz')
codes = []
for varname in npz_file.files:
    if 'mu_' in varname:
        codes.append(int(varname[3:-1]))
all_codes = sorted(set(codes))
glyphs_list = [chr(c) for c in all_codes]

rng = np.random.default_rng()

def get_random_string():
    result = []
    for _ in range(1000):
        jpstr = ''.join(rng.choice(wordlist, rng.integers(low=0,high=30)))
        enstr = ' ' + ' '.join(rng.choice(en_wordlist, rng.integers(low=0,high=3))) + ' '
        if rng.uniform() < 0.1:
            result.append('\n')
        result.append(jpstr)
        if rng.uniform() < 0.1:
            result.append('\n')
        result.append(enstr)
    return ''.join(result)

def get_random_special():
    result = ''
    for _ in range(1000):
        p = rng.uniform()
        result += ''.join(rng.choice(wordlist, rng.integers(low=0,high=30)))
        if p < 0.2:
            result += ''.join(['ー'] * rng.integers(1,10))
        elif p < 0.4:
            result += ''.join(['〰'] * rng.integers(1,10))
        elif p < 0.6:
            result += '〜'

        p = rng.uniform()
        if p < 0.1:
            result += '、'
        elif p < 0.2:
            result += '。'
        elif p < 0.3:
            result += '？'
        elif p < 0.4:
            result += '！'
        elif p < 0.5:
            result += '‼'
        elif p < 0.6:
            result += '⁉'
        elif p < 0.7:
            result += '⁈'

        if rng.uniform() < 0.25:
            result += '\n'
        else:
            if p < 0.2:
                pass
            else:
                result += '　'
    return result

def count_prevfile(train=True):
    if train:
        prev_files = sorted(glob.glob(os.path.join(tfdata_path,'train*.tfrecords')))
        if len(prev_files) > 0:
            k = int(os.path.splitext(os.path.basename(prev_files[-1]))[0][-8:]) + 1
        else:
            k = 0
    else:
        prev_files = sorted(glob.glob(os.path.join(tfdata_path,'test*.tfrecords')))
        if len(prev_files) > 0:
            k = int(os.path.splitext(os.path.basename(prev_files[-1]))[0][-8:]) + 1
        else:
            k = 0
    return k

def get_filepath(k=0, train=True):
    os.makedirs(tfdata_path, exist_ok=True)

    if train:
        filename = os.path.join(tfdata_path,'train%08d.tfrecords'%k)
    else:
        filename = os.path.join(tfdata_path,'test%08d.tfrecords'%k)
    return filename

def process_trainfunc(k):
    return process_func(k, train=True)

def process_testfunc(k):
    return process_func(k, train=False)

def process_func(k, train):
    with tf.io.TFRecordWriter(get_filepath(k=k, train=train)) as file_writer:
        linecount = 0
        while linecount < lines_per_file:
            print(k,linecount)

            p = rng.random()
            try:
                if p < 0.01:
                    content = get_random_special()
                    en = False
                elif p < 0.1:
                    content = get_random_string()
                    en = True
                elif p < 0.4:
                    # aozora
                    url = rng.choice(aozora_urls)
                    content = get_contents(url)
                    en = False
                elif p < 0.7:
                    pageid = get_random_wordid(en=False)
                    content = get_word_content(pageid, en=False)
                    en = False
                elif p < 0.9:
                    pageid = get_random_wordid(en=True)
                    content = get_word_content(pageid, en=True)
                    en = True
                else:
                    max_text = 64*1024
                    content = ''.join(rng.choice(glyphs_list, size=max_text))
                    en = False
            except OSError:
                time.sleep(1)
                continue

            str_lines = content.splitlines()
            str_lines = [s for s in str_lines if s.strip()]
            str_lines = [s.rstrip() for s in str_lines]
            str_lines = [re.sub(' +',' ',s) for s in str_lines]
            str_lines = [re.sub('\u3000+','\u3000',s) for s in str_lines]

            lines = []
            for content in str_lines:
                lines.append(''.join([c for c in content if ord(c) in all_codes or c in UNICODE_WHITESPACE_CHARACTERS or c in ['\uFFF9','\uFFFA','\uFFFB']]))
            str_lines = lines

            if len(str_lines) == 0:
                continue

            lines = []
            for content in str_lines:
                if en:
                    while len(content) > 0:
                        max_count = rng.integers(2,80)
                        if len(content) < max_count:
                            lines.append(content)
                            content = []
                        else:
                            i = content.find(' ', max_count)
                            if i < 0:
                                lines.append(content)
                                content = []
                            else:
                                lines.append(content[:i])
                                content = content[i+1:]
                else:
                    while len(content) > 0:
                        max_count = rng.integers(2,80)
                        if len(content) < max_count:
                            lines.append(content)
                            content = []
                        else:
                            i = max_count
                            st = [i for i, c in enumerate(content) if c == '\uFFF9']
                            ed = [i for i, c in enumerate(content) if c == '\uFFFB']
                            for s,e in zip(st,ed):
                                if i < s:
                                    break
                                if s <= i <= e:
                                    i = e+1
                                    break
                            lines.append(content[:i])
                            content = content[i:]

            for content in lines:
                codes = []
                sp = 0
                ruby = 0
                rubybase = 0
                for c in list(content):
                    t = ord(c)
                    if c in UNICODE_WHITESPACE_CHARACTERS:
                        sp = 1
                        continue
                    elif c == '\uFFF9':
                        ruby = 0
                        rubybase = 1
                        continue
                    elif c == '\uFFFA':
                        ruby = 1
                        rubybase = 0
                        continue
                    elif c == '\uFFFB':
                        ruby = 0
                        rubybase = 0
                        continue
                    codes.append([t,sp,ruby,rubybase,0])
                    sp = 0
                content += '\n'
                codes.append([0,0,0,0,1])
                example_proto = tf.train.Example(features=tf.train.Features(feature={
                    'str': tf.train.Feature(bytes_list=tf.train.BytesList(value=[content.encode()])),
                    'code': tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(tf.constant(codes, tf.int32)).numpy()])),
                    'codelen': tf.train.Feature(int64_list=tf.train.Int64List(value=[len(codes)])),
                    'strlen': tf.train.Feature(int64_list=tf.train.Int64List(value=[len(content)])),
                }))
                #print(decode_ruby(content), end='')
                record_bytes = example_proto.SerializeToString()
                file_writer.write(record_bytes)
                linecount+=1
                if linecount >= lines_per_file:
                    break
    return k

def create_data(train=True, count=1):
    with Pool() as p:
        k = count_prevfile(train=train)
        if k >= count:
            return
        if train:
            for i in p.imap_unordered(process_trainfunc, range(k,count)):
                print(i,'done')
        else:
            for i in p.imap_unordered(process_testfunc, range(k,count)):
                print(i,'done')

if __name__=="__main__":
    import sys

    if len(sys.argv) < 3:
        test_count = 5
        train_count = 100
    else:
        test_count = int(sys.argv[1])
        train_count = int(sys.argv[2])

    create_data(train=False, count=test_count)
    create_data(train=True, count=train_count)
