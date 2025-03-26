#!/usr/bin/env python3
import onnx
from onnxconverter_common import float16

def convert_fp16(filename, outfilename):
    model = onnx.load(filename)
    model_fp16 = float16.convert_float_to_float16(model)
    onnx.save(model_fp16, outfilename)

convert_fp16("TextDetector.onnx","TextDetector.fp16.onnx")
convert_fp16("CodeDecoder.onnx","CodeDecoder.fp16.onnx")
convert_fp16("TransformerDecoder.onnx","TransformerDecoder.fp16.onnx")
convert_fp16("TransformerEncoder.onnx","TransformerEncoder.fp16.onnx")
