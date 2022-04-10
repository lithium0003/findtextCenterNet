#!/usr/bin/env python3
import tensorflow as tf
import tf2onnx
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType
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

def convert1():
    model = TextDetectorModel()
    last = tf.train.latest_checkpoint('ckpt')
    print(last)
    model.load_weights(last).expect_partial()

    # import logging
    # logging.basicConfig(filename='debug.log', level=logging.DEBUG)

    #########################################################################

    inputs = tf.keras.Input(shape=(net.height,net.width,3), name='image_input')
    heatmap, feature = model.detector(inputs)

    keymap = tf.math.sigmoid(heatmap[...,0])
    keymape = tf.expand_dims(keymap, axis=-1)
    local_peak = tf.nn.max_pool2d(keymape,5,1,'SAME')
    keep = local_peak[...,0] - keymap < 1e-6
    detectedkey = keymap * tf.cast(keep, tf.float32)

    textlines = tf.math.sigmoid(heatmap[...,5])
    separator = tf.math.sigmoid(heatmap[...,6])
    xsize = tf.clip_by_value(heatmap[...,1], -2., 1.)
    ysize = tf.clip_by_value(heatmap[...,2], -2., 1.)
    w = tf.math.exp(xsize * tf.math.log(10.)) / 10 * net.width
    h = tf.math.exp(ysize * tf.math.log(10.)) / 10 * net.height
    xoffset = tf.clip_by_value(heatmap[...,3], -5., 5.)
    yoffset = tf.clip_by_value(heatmap[...,4], -5., 5.)
    dx = xoffset * net.scale
    dy = yoffset * net.scale

    outputs = [
        tf.keras.layers.Lambda(lambda x: x, name='maps', dtype='float32')(tf.stack([keymap, detectedkey, w, h, dx, dy, textlines, separator], axis=-1)),
        tf.keras.layers.Lambda(lambda x: x, name='feature', dtype='float32')(feature),
    ]
    detector = tf.keras.Model(inputs, outputs, name='CenterNetBlock')

    tf2onnx.convert.from_keras(detector, output_path='TextDetector.onnx', opset=11)
    onnx.checker.check_model('TextDetector.onnx')

    quantize_dynamic('TextDetector.onnx','TextDetector.quant.onnx',weight_type=QuantType.QUInt8)
    onnx.checker.check_model('TextDetector.quant.onnx')

    ############################################################################

    embedded = tf.keras.Input(shape=(net.feature_dim,), name='feature_input')
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
    outputs = [
        tf.keras.layers.Lambda(lambda x: x, name='ids', dtype='float32')(tf.cast(ids, tf.float32)),
        tf.keras.layers.Lambda(lambda x: x, name='p_id', dtype='float32')(p_id),
    ]
    decoder = tf.keras.Model(embedded, outputs, name='SimpleDecoderBlock')

    tf2onnx.convert.from_keras(decoder, output_path='CodeDecoder.onnx', opset=11)
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
    peakmap = maps[0,:,:,1]
    idxy, idxx  = np.unravel_index(np.argsort(-peakmap.ravel()), peakmap.shape)
    results_dict = []
    for y, x in zip(idxy, idxx):
        print(x,y,peakmap[y,x])
        if peakmap[y,x] < 0.5:
            break
        ids, p_id = onnx_decoder.run(['ids','p_id'], {'feature_input': feature[:,y,x,:]})
        p = p_id[0]
        ids = list(ids[0,:].astype(int))
        i = calc_predid(*ids)
        if i < 0x10FFFF:
            c = chr(i)
        else:
            c = None
        print(p, i, c)
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
    convert1()
    test_model()
