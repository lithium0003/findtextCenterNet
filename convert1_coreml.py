#!/usr/bin/env python3
import coremltools as ct
import torch
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
from datetime import datetime
import itertools

from models.detector import TextDetectorModel, CenterNetDetector, CodeDecoder
from util_func import calc_predid, width, height, feature_dim, sigmoid, modulo_list

def convert1(model_size='xl'):
    # import logging
    # logging.basicConfig(filename='debug.log', level=logging.DEBUG)

    model = TextDetectorModel(model_size=model_size)
    data = torch.load('model.pt', map_location="cpu", weights_only=True)
    model.load_state_dict(data['model_state_dict'])

    # with torch.no_grad():
    #     model.detector.code2.top_conv[-1].bias.copy_(model.detector.code2.top_conv[-1].bias+4)
    #     model.detector.code8.top_conv[-1].bias.copy_(model.detector.code8.top_conv[-1].bias-2)

    detector = CenterNetDetector(model.detector)
    decoder = CodeDecoder(model.decoder)
    detector.eval()
    decoder.eval()

    #########################################################################
    print('detector')

    example_input = torch.rand(1, 3, height, width)
    traced_model = torch.jit.trace(detector, example_input)

    mlmodel_detector = ct.convert(traced_model,
            inputs=[
                ct.ImageType(name='image', shape=(1, 3, height, width), scale=1/255)
            ],
            outputs=[
                ct.TensorType(name='heatmap'),
                ct.TensorType(name='feature'),
            ],
            convert_to="mlprogram",
            minimum_deployment_target=ct.target.iOS18)
    mlmodel_detector.version = datetime.now().strftime("%Y%m%d%H%M%S")
    mlmodel_detector.save("TextDetector.mlpackage")

    ############################################################################
    print('decoder')

    example_input = torch.rand(1, feature_dim)
    traced_model = torch.jit.trace(decoder, example_input)

    mlmodel_decoder = ct.convert(traced_model,
                                 convert_to="mlprogram",
                                 inputs=[
                                    ct.TensorType(name='feature_input', shape=(1, feature_dim))
                                 ],
                                 outputs=[
                                    ct.TensorType(name='modulo_1091'),
                                    ct.TensorType(name='modulo_1093'),
                                    ct.TensorType(name='modulo_1097'),
                                 ],
                                 minimum_deployment_target=ct.target.iOS18)
    mlmodel_decoder.version = datetime.now().strftime("%Y%m%d%H%M%S")
    mlmodel_decoder.save("CodeDecoder.mlpackage")


def cos_sim(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def test_model():

    plt.figure()
    plt.text(0.1,0.9,'test', fontsize=32)
    plt.axis('off')
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    im = np.array(Image.open(buf).convert("RGB"))
    buf.close()

    im = im[:height,:width,:]
    im = np.pad(im, [[0,height-im.shape[0]], [0,width-im.shape[1]], [0,0]], 'constant', constant_values=((255,255),(255,255),(255,255)))
    
    print('test')
    input_image = Image.fromarray(im, mode="RGB")

    print('load')
    mlmodel_detector = ct.models.MLModel('TextDetector.mlpackage')
    mlmodel_decoder = ct.models.MLModel('CodeDecoder.mlpackage')

    output = mlmodel_detector.predict({'image': input_image})
    peakmap = output['heatmap'][0,1,:,:]

    idxy, idxx  = np.unravel_index(np.argsort(-peakmap.ravel()), peakmap.shape)
    results_dict = []
    for y, x in zip(idxy, idxx):
        p1 = sigmoid(peakmap[y,x])
        print(x,y,p1)
        if p1 < 0.5:
            break
        feature = output['feature'][:,:,y,x]
        decode_output = mlmodel_decoder.predict({'feature_input': feature})
        p = []
        id = []
        for k,m in enumerate(modulo_list):
            prob = decode_output['modulo_%d'%m][0]
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
        print(g)
        prob,idx = g[0]
        if idx <= 0x10FFFF:
            c = chr(idx)
        else:
            c = None
        print(prob, idx, c)
        print(feature.max(), feature.min())
        results_dict.append((feature[0], idx, c))
        print()

    for i in range(len(results_dict)):
        for j in range(i+1, len(results_dict)):
            s = cos_sim(results_dict[i][0], results_dict[j][0])
            d = np.linalg.norm(results_dict[i][0] - results_dict[j][0])
            print(s,d, i,j,results_dict[i][1:],results_dict[j][1:])

if __name__ == '__main__':
    import sys
    model_size = 'xl'
    if len(sys.argv) > 1:
        if sys.argv[1] == 's':
            model_size = 's'
        if sys.argv[1] == 'm':
            model_size = 'm'
        if sys.argv[1] == 'l':
            model_size = 'l'
    convert1(model_size)
    test_model()
