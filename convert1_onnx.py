#!/usr/bin/env python3
import tensorflow as tf
import tf2onnx
import onnx
import onnxruntime
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io

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

    inputs = tf.keras.Input(shape=(net.height,net.width,3), name='image_input')
    heatmap, feature = model.detector(inputs)

    keymap = heatmap[...,0]
    keymape = tf.expand_dims(keymap, axis=-1)
    local_peak = tf.nn.max_pool2d(keymape,5,1,'SAME')
    local_peak = local_peak[...,0]

    textlines = heatmap[...,5]
    separator = heatmap[...,6]
    xsize = heatmap[...,1]
    ysize = heatmap[...,2]
    xoffset = heatmap[...,3]
    yoffset = heatmap[...,4]
    code_map = []
    for k in range(4):
        code_map.append(heatmap[...,7+k])

    outputs = [
        tf.keras.layers.Lambda(lambda x: x, name='maps', dtype='float32')(tf.stack([keymap, local_peak, xsize, ysize, xoffset, yoffset, textlines, separator, *code_map], axis=-1)),
        tf.keras.layers.Lambda(lambda x: x, name='feature', dtype='float32')(feature),
    ]
    detector = tf.keras.Model(inputs, outputs, name='TextDetector')

    tf2onnx.convert.from_keras(detector, output_path='TextDetector.onnx')

    onnx.checker.check_model('TextDetector.onnx')

    ############################################################################

    embedded = tf.keras.Input(shape=(net.feature_dim,), name='feature_input')
    decoder_outputs = model.decoder(embedded)
    outputs = []
    for decoder_id, mod_id in zip(decoder_outputs, net.modulo_list):
        outputs.append(tf.keras.layers.Lambda(lambda x: x, name='mod_%d'%mod_id, dtype='float32')(decoder_id))
    decoder = tf.keras.Model(embedded, outputs, name='CodeDecoder')

    tf2onnx.convert.from_keras(decoder, output_path='CodeDecoder.onnx')

    onnx.checker.check_model('CodeDecoder.onnx')

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
            tk *= pow(m[j], m[k]-2, m[k])
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

def softmax(a):
    a_max = max(a)
    x = np.exp(a-a_max)
    u = np.sum(x)
    return x/u

def test_model():
    print('test')
    plt.figure()
    plt.text(0.1,0.9,'test', fontsize=32)
    plt.axis('off')
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    im = np.array(Image.open(buf))
    buf.close()

    im = im[:net.height,:net.width,:3]
    im = np.pad(im, [[0,net.height-im.shape[0]], [0,net.width-im.shape[1]], [0,0]], 'constant', constant_values=((255,255),(255,255),(255,255)))

    image_input = im.astype(np.float32)
    image_input = np.expand_dims(image_input, 0)
    print(image_input.shape)

    print('load')
    onnx_detector = onnxruntime.InferenceSession("TextDetector.onnx")
    onnx_decoder = onnxruntime.InferenceSession("CodeDecoder.onnx")

    # print(' [ detector ] ')
    # print('input:')
    # for session_input in onnx_detector.get_inputs():
    #     print(session_input.name, session_input.shape)
    # print('output:')
    # for session_output in onnx_detector.get_outputs():
    #     print(session_output.name, session_output.shape)

    # print(' [ decoder ] ')
    # print('input:')
    # for session_input in onnx_decoder.get_inputs():
    #     print(session_input.name, session_input.shape)
    # print('output:')
    # for session_output in onnx_decoder.get_outputs():
    #     print(session_output.name, session_output.shape)

    maps, feature = onnx_detector.run(['maps','feature'], {'image_input': image_input})
    peakmap = 1/(1 + np.exp(-maps[0,:,:,0]))
    peakmap = np.where(maps[0,:,:,0] == maps[0,:,:,1], peakmap, 0.)
    idxy, idxx  = np.unravel_index(np.argsort(-peakmap.ravel()), peakmap.shape)
    results_dict = []
    for y, x in zip(idxy, idxx):
        print(x,y,peakmap[y,x])
        if peakmap[y,x] < 0.5:
            break
        outnames = ['mod_%d'%m for m in net.modulo_list]
        ids = onnx_decoder.run(outnames, {'feature_input': feature[:,y,x,:]})
        p_id = None
        id_mod = []
        for id in ids:
            p = softmax(id[0,:])
            id_i = np.argmax(p)
            if p_id is None:
                p_id = np.log(max(p[id_i],1e-7))
            else:
                p_id += np.log(max(p[id_i],1e-7))
            id_mod.append(id_i)
        p_id = np.exp(p_id / len(ids))
        i = calc_predid(*id_mod)
        if i < 0x10FFFF:
            c = chr(i)
        else:
            c = None
        print(p_id, i, c)
        feature1 = feature[0,y,x,:]
        print(feature1.max(), feature1.min())
        results_dict.append((feature1, i, c))
        print()

    for i in range(len(results_dict)):
        for j in range(i+1, len(results_dict)):
            s = cos_sim(results_dict[i][0], results_dict[j][0])
            d = np.linalg.norm(results_dict[i][0] - results_dict[j][0])
            print(s,d, i,j,results_dict[i][1:],results_dict[j][1:])

if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1:
        ckpt_dir = sys.argv[1]
    else:
        ckpt_dir = 'ckpt1'
        
    convert1(ckpt_dir)
    test_model()
