import os
import glob
import numpy as np

from dataset.data_transformer import UNICODE_WHITESPACE_CHARACTERS

train_data3 = 'train_data3'

def process():
    txtfiles = sorted(glob.glob(os.path.join(train_data3,'*','*.txt')))
    with np.load(os.path.join(train_data3, 'features.npz')) as data:
        codes = set([chr(c) for c in set([int(s.split('_')[1]) for s in data.files])])
    pass_char = set(['\n','\uFFF9','\uFFFA','\uFFFB'] + UNICODE_WHITESPACE_CHARACTERS)

    all_remain = set()
    for filename in txtfiles:
        print(filename)
        with open(filename) as rf:
            lines = [s for s in rf.read().splitlines() if s.strip()]
            txt = '\n'.join(lines)
        remain = set(txt) - codes - pass_char
        for c in remain:
            print(c, hex(ord(c)), ord(c), 'not found in', filename)
        all_remain |= remain
    print('--------not-found--------')
    for c in sorted([ord(c) for c in all_remain]):
        print(chr(c), hex(c), c)

if __name__=='__main__':
    process()