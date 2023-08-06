#!/usr/bin/env python3

import numpy as np
import glob
import os
import sys
from net.const import feature_dim

rng = np.random.default_rng()

features = {}
char_dir = 'chardata_font'
print(char_dir)
for filename in glob.glob(os.path.join(char_dir,'*.npy')):
    code = os.path.splitext(os.path.basename(filename))[0]
    
    feature = np.load(filename)
    if len(feature.shape) == 1:
        feature = np.expand_dims(feature,0)
    features[code] = np.concatenate([features.get(code, np.zeros([0,feature_dim], np.float32)), feature])
    print(code, feature.shape[0])

char_dir = 'chardata_hand'
print(char_dir)
for filename in glob.glob(os.path.join(char_dir,'*.npy')):
    code = os.path.splitext(os.path.basename(filename))[0] + 'n'
    
    feature = np.load(filename)
    if len(feature.shape) == 1:
        feature = np.expand_dims(feature,0)
    count = feature.shape[0]
    print(code, count)

    mu = np.mean(feature, axis=0)
    sd = np.std(feature, axis=0) if count > 3 else 0.5 * np.ones_like(mu, dtype=np.float32)

    font_feature = features.get(code, np.zeros([0,feature_dim], np.float32))
    features[code] = np.concatenate([font_feature, feature])

values = {}
for code in sorted(features.keys()):
    feature = features[code]
    count = feature.shape[0]
    print(code, count)

    values[code] = feature

for key in values:
    if not np.all(np.isfinite(values[key])):
        print(key)
        print(values[key])
        exit()

np.savez('charparam', **values)
