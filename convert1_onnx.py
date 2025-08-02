#!/usr/bin/env python3
import onnx
import onnxruntime
import torch
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import itertools

from models.detector import TextDetectorModel, CenterNetDetector, CodeDecoder
from util_func import calc_predid, width, height, feature_dim, sigmoid, modulo_list

def convert1():
    model = TextDetectorModel(pre_weights=False)
    data = torch.load('model.pt', map_location="cpu", weights_only=True)
    model.load_state_dict(data['model_state_dict'])
    detector = CenterNetDetector(model.detector)
    decoder = CodeDecoder(model.decoder)
    detector.eval()
    decoder.eval()

    #########################################################################
    print('detector')

    example_input = torch.rand(1, 3, height, width)
    torch.onnx.export(detector,
                      example_input,
                      "TextDetector.onnx",
                      input_names=['image'],
                      output_names=['heatmap','feature'],
                      dynamo=True,
                      external_data=False,
                      optimize=True,
                      verify=True,
                      opset_version=20)
    onnx.checker.check_model('TextDetector.onnx')

    ############################################################################
    print('decoder')

    example_input = torch.rand(1, feature_dim)
    torch.onnx.export(decoder,
                      example_input,
                      "CodeDecoder.onnx",
                      input_names=['feature_input'],
                      output_names=['modulo_1091','modulo_1093','modulo_1097'],
                      dynamo=True,
                      external_data=False,
                      optimize=True,
                      verify=True,
                      opset_version=20)
    onnx.checker.check_model('CodeDecoder.onnx')

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
    im = np.array(Image.open(buf).convert('RGB'))
    buf.close()

    im = im[:height,:width,:]
    im = np.pad(im, [[0,height-im.shape[0]], [0,width-im.shape[1]], [0,0]], 'constant', constant_values=((255,255),(255,255),(255,255)))

    image_input = im.astype(np.float32)
    image_input = np.expand_dims(image_input, 0).transpose(0,3,1,2) / 255
    print(image_input.shape)

    print('load')
    onnx_detector = onnxruntime.InferenceSession("TextDetector.onnx")
    onnx_decoder = onnxruntime.InferenceSession("CodeDecoder.onnx")

    print(' [ detector ] ')
    print('input:')
    for session_input in onnx_detector.get_inputs():
        print(session_input.name, session_input.shape)
    print('output:')
    for session_output in onnx_detector.get_outputs():
        print(session_output.name, session_output.shape)

    print(' [ decoder ] ')
    print('input:')
    for session_input in onnx_decoder.get_inputs():
        print(session_input.name, session_input.shape)
    print('output:')
    for session_output in onnx_decoder.get_outputs():
        print(session_output.name, session_output.shape)

    maps, feature = onnx_detector.run(['heatmap','feature'], {'image': image_input})
    peakmap = maps[0,1,:,:]
    idxy, idxx  = np.unravel_index(np.argsort(-peakmap.ravel()), peakmap.shape)
    results_dict = []
    for y, x in zip(idxy, idxx):
        print(x,y,sigmoid(peakmap[y,x]))
        if sigmoid(peakmap[y,x]) < 0.5:
            break
        outnames = ['modulo_%d'%m for m in modulo_list]
        decode_outputs = onnx_decoder.run(outnames, {'feature_input': feature[:,:,y,x]})

        p = []
        id = []
        for k,prob in enumerate(decode_outputs):
            prob = prob[0]
            idx = np.where(prob > 0.01)[0]
            if len(idx) == 0:
                idx = [np.argmax(prob)]
            if k == 0:
                for i in idx[:3]:
                    id.append([i])
                    p.append([prob[i]])
            else:
                id = [i1 + [i2] for i1, i2 in itertools.product(id, idx[:3])]
                p = [i1 + [prob[i2]] for i1, i2 in itertools.product(p, idx[:3])]
        p = [np.exp(np.mean([np.log(prob) for prob in probs])) for probs in p]
        i = [calc_predid(*ids) for ids in id]
        g = sorted([(prob, id) for prob,id in zip(p,i)], key=lambda x: x[0] if x[1] <= 0x10FFFF else 0, reverse=True)
        prob,idx = g[0]
        if idx <= 0x10FFFF:
            c = chr(idx)
        else:
            c = None
        print(prob, idx, c)
        print(feature[0,:,y,x].max(), feature[0,:,y,x].min())
        results_dict.append((feature[0,:,y,x], idx, c))
        print()


    for i in range(len(results_dict)):
        for j in range(i+1, len(results_dict)):
            s = cos_sim(results_dict[i][0], results_dict[j][0])
            d = np.linalg.norm(results_dict[i][0] - results_dict[j][0])
            print(s,d, i,j,results_dict[i][1:],results_dict[j][1:])

if __name__ == '__main__':
    convert1()
    test_model()
