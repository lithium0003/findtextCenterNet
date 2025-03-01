#!/usr/bin/env python3

import os

models = []
if os.path.exists('TextDetector.mlpackage') and os.path.exists('TransformerEncoder.mlpackage') and os.path.exists('TransformerDecoder.mlpackage'):
    models.append('coreml')
if (os.path.exists("TextDetector.quant.onnx") or os.path.exists("TextDetector.onnx")) and os.path.exists('TransformerEncoder.onnx') and os.path.exists('TransformerDecoder.onnx'):
    models.append('onnx')
if os.path.exists('model.pt') and os.path.exists('model3.pt'):
    models.append('torch')

if models[0] == 'coreml':
    print('coreml')
    from process_ocr_coreml import OCR_coreml_Processer as OCR_Processer
elif models[0] == 'onnx':
    print('onnx')
    from process_ocr_onnx import OCR_onnx_Processer as OCR_Processer
elif models[0] == 'torch':
    print('torch')
    from process_ocr_torch import OCR_torch_Processer as OCR_Processer

processer = OCR_Processer()

if __name__=='__main__':
    import sys
    import glob

    if len(sys.argv) < 2:
        print(sys.argv[0], 'target_image')
    
    target_files = []
    resize = 1.0
    for arg in sys.argv[1:]:
        if arg.startswith('--resize='):
            resize = float(arg.split('=')[1])
        else:
            target_files += glob.glob(arg)
    target_files = sorted(target_files)

    for target_file in target_files:
        print(target_file)
        processer.call_OCR(target_file, resize)
