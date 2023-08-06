#!/usr/bin/env python3
from onnxruntime.quantization import quantize, CalibrationDataReader, StaticQuantConfig, CalibrationMethod, QuantType, QuantFormat, DynamicQuantConfig

from PIL import Image
import numpy as np

import glob
import os

class ImageDataReader(CalibrationDataReader):
    def __init__(self):
        self.imfile = sorted(glob.glob(os.path.join('images','img*.png')))
        self.datasize = len(self.imfile)
        self.enum_imfile = iter(self.imfile)

    def get_next(self):
        imfile = next(self.enum_imfile, None)
        print(imfile)
        if imfile:
            im = Image.open(imfile).convert('RGB') 
            image = np.asarray(im).astype('float32')
            return {'image_input': np.expand_dims(image, axis=0)}
        else:
            return None

def optimize1():
    dr = ImageDataReader()

    config = StaticQuantConfig(
        dr,
        calibrate_method=CalibrationMethod.MinMax,
        quant_format=QuantFormat.QOperator,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QUInt8,
        nodes_to_exclude=[
            'TextDetector/CenterNetBlock/keyheatmap/keyheatmap_out_conv/BiasAdd',
            'TextDetector/CenterNetBlock/sizes/sizes_out_conv/BiasAdd',
            'TextDetector/CenterNetBlock/offsets/offsets_out_conv/BiasAdd',
            'TextDetector/CenterNetBlock/textline/textline_out_conv/BiasAdd',
            'TextDetector/CenterNetBlock/sepatator/sepatator_out_conv/BiasAdd',
            'TextDetector/CenterNetBlock/codes/codes_out_conv/BiasAdd',
            'TextDetector/CenterNetBlock/concatenate/concat',
            'TextDetector/CenterNetBlock/feature/feature_out_conv/BiasAdd',
        ],
        extra_options={
            'CalibMovingAverage': True,
        })
    quantize('TextDetector.infer.onnx',
             'TextDetector.quant.onnx',
             config)

def optimize2():
    config = DynamicQuantConfig(
        weight_type=QuantType.QUInt8,
    )

    quantize('TextDetector.infer.onnx',
             'TextDetector.quant.onnx',
             config)

if __name__ == "__main__":
    from onnxruntime.quantization.shape_inference import quant_pre_process

    quant_pre_process(
        'TextDetector.onnx',
        'TextDetector.infer.onnx'
    )

    optimize1()
    # >30GB memory needed

    #optimize2()

