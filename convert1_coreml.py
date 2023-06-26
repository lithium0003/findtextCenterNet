#!/usr/bin/env python3
import tensorflow as tf
import coremltools as ct
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import io
from datetime import datetime

import net

class TextDetectorModel(tf.keras.models.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.detector = net.CenterNetDetectionBlock(pre_weight=False)
        self.decoder = net.SimpleDecoderBlock()

        inputs = tf.keras.Input(shape=(net.height,net.width,3))
        self.detector(inputs)

def convert1(ckpt_dir='ckpt1'):
    model = TextDetectorModel()

    last = tf.train.latest_checkpoint(ckpt_dir)
    print(last)
    model.load_weights(last).expect_partial()

    # import logging
    # logging.basicConfig(filename='debug.log', level=logging.DEBUG)

    #########################################################################
    print('detector')

    inputs = tf.keras.Input(shape=(net.height,net.width,3), name='Image')
    heatmap, feature = model.detector(inputs)

    keymap = heatmap[...,0]
    local_peak = tf.nn.max_pool2d(keymap[...,tf.newaxis],5,1,'SAME')
    keep = local_peak[...,0] == keymap
    keymap = tf.math.sigmoid(keymap)
    detectedkey = keymap * tf.cast(keep, tf.float32)

    textlines = tf.math.sigmoid(heatmap[...,5])
    separator = tf.math.sigmoid(heatmap[...,6])
    xsize = heatmap[...,1]
    ysize = heatmap[...,2]
    w = tf.math.exp(xsize - 3) * 1024
    h = tf.math.exp(ysize - 3) * 1024
    xoffset = heatmap[...,3]
    yoffset = heatmap[...,4]
    dx = xoffset * net.scale
    dy = yoffset * net.scale
    code_map = []
    for k in range(4):
        code_map.append(tf.math.sigmoid(heatmap[...,7+k]))

    outputs = {
        'Output_heatmap': tf.stack([keymap, detectedkey, w, h, dx, dy, textlines, separator, *code_map], axis=-1),
        'Output_feature': feature,
    }
    detector = tf.keras.Model(inputs, outputs, name='CenterNetBlock')

    mlmodel_detector = ct.convert(detector,
            inputs=[
                ct.ImageType(shape=(1, net.height, net.width, 3))
            ],
            convert_to="mlprogram",
            compute_units=ct.ComputeUnit.CPU_AND_NE,
            minimum_deployment_target=ct.target.iOS16)
    mlmodel_detector.version = datetime.now().strftime("%Y%m%d%H%M%S")
    spec = mlmodel_detector.get_spec()

    # get output names
    output_names = [out.name for out in spec.description.output]
    output_names = {k: int((k.split('_')[1:]+['0'])[0]) for k in output_names }
    org_output_names = sorted(outputs.keys())

    for name, idx in output_names.items():
        ct.utils.rename_feature(spec, name, org_output_names[idx])

    mlmodel_detector_fix = ct.models.MLModel(spec, weights_dir=mlmodel_detector.weights_dir)
    mlmodel_detector_fix.save("TextDetector.mlpackage")

    ############################################################################
    print('decoder')

    embedded = tf.keras.Input(shape=(net.feature_dim,), name='Input')
    decoder_outputs = model.decoder(embedded)
    ids = []
    p_id = None
    for decoder_id1 in decoder_outputs:
        prob_id1 = tf.nn.softmax(decoder_id1, -1)
        pred_id1 = tf.math.argmax(prob_id1, axis=-1)
        prob_id1 = tf.math.reduce_sum(tf.one_hot(pred_id1, tf.shape(prob_id1)[-1]) * prob_id1, -1)
        if p_id is None:
            p_id = tf.math.log(tf.math.maximum(prob_id1,1e-7))
        else:
            p_id += tf.math.log(tf.math.maximum(prob_id1,1e-7))
        ids.append(pred_id1)
    ids = tf.stack(ids, axis=-1)
    p_id = tf.exp(p_id / len(decoder_outputs))
    outputs = {
        'Output_id': tf.cast(ids, tf.float32),
        'Output_p': p_id,
    }
    decoder = tf.keras.Model(embedded, outputs, name='SimpleDecoderBlock')

    mlmodel_decoder = ct.convert(decoder,
                                 convert_to="mlprogram",
                                 inputs=[
                                    ct.TensorType(shape=(1, net.feature_dim))
                                 ],
                                 compute_units=ct.ComputeUnit.CPU_AND_NE,
                                 minimum_deployment_target=ct.target.iOS16)
    mlmodel_decoder.version = datetime.now().strftime("%Y%m%d%H%M%S")
    spec = mlmodel_decoder.get_spec()

    # get output names
    output_names = [out.name for out in spec.description.output]
    output_names = {k: int((k.split('_')[1:]+['0'])[0]) for k in output_names }
    org_output_names = sorted(outputs.keys())

    for name, idx in output_names.items():
        ct.utils.rename_feature(spec, name, org_output_names[idx])

    mlmodel_decoder_fix = ct.models.MLModel(spec, weights_dir=mlmodel_decoder.weights_dir)
    mlmodel_decoder_fix.save("CodeDecoder.mlpackage")

    return last

def calc_predid(*args):
    m = net.modulo_list
    b = args
    assert(len(m) == len(b))
    t = []

    for k in range(len(m)):
        u = 0
        for j in range(k):
            w = t[j]
            for i in range(j):
                w *= m[i]
            u += w
        tk = b[k] - u
        for j in range(k):
            tk *= pow(m[j], -1, m[k])
        tk = tk % m[k]
        t.append(tk)
    x = 0
    for k in range(len(t)):
        w = t[k]
        for i in range(k):
            w *= m[i]
        x += w
    mk = 1
    for k in range(len(m)):
        mk *= m[k]
    x = x % mk
    return x

def cos_sim(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def test_model():

    plt.figure()
    plt.text(0.1,0.9,'test', fontsize=32)
    plt.axis('off')
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    im = np.array(Image.open(buf).convert("RGB"))
    buf.close()

    im = im[:net.height,:net.width,:]
    im = np.pad(im, [[0,net.height-im.shape[0]], [0,net.width-im.shape[1]], [0,0]], 'constant', constant_values=((255,255),(255,255),(255,255)))
    
    print('test')
    input_image = Image.fromarray(im, mode="RGB")

    print('load')
    mlmodel_detector = ct.models.MLModel('TextDetector.mlpackage')
    mlmodel_decoder = ct.models.MLModel('CodeDecoder.mlpackage')

    output = mlmodel_detector.predict({'Image': input_image})
    peakmap = output['Output_heatmap'][0,:,:,1]

    idxy, idxx  = np.unravel_index(np.argsort(-peakmap.ravel()), peakmap.shape)
    results_dict = []
    for y, x in zip(idxy, idxx):
        print(x,y,peakmap[y,x])
        if peakmap[y,x] < 0.5:
            break
        decode_output = mlmodel_decoder.predict({'Input': output['Output_feature'][:,y,x,:]})
        p = decode_output['Output_p'][0]
        ids = list(decode_output['Output_id'][0].astype(int))
        i = calc_predid(*ids)
        if i < 0x10FFFF:
            c = chr(i)
        else:
            c = None
        print(p, i, c)
        feature = output['Output_feature'][0,y,x,:]
        print(feature.max(), feature.min())
        results_dict.append((feature, i, c))
        print()

    for i in range(len(results_dict)):
        for j in range(i+1, len(results_dict)):
            s = cos_sim(results_dict[i][0], results_dict[j][0])
            d = np.linalg.norm(results_dict[i][0] - results_dict[j][0])
            print(s,d, i,j,results_dict[i][1:],results_dict[j][1:])

if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1:
        ckpt_dir = int(sys.argv[1])
    else:
        ckpt_dir = 'ckpt1'
    convert1(ckpt_dir)
    test_model()
