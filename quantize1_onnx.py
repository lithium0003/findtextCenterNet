#!/usr/bin/env python3
from onnxruntime.quantization import quantize, StaticQuantConfig, CalibrationDataReader, QuantType, QuantFormat
from torch.utils.data import DataLoader
from dataset.data_detector import get_dataset

class QuntizationDataReader(CalibrationDataReader):
    def __init__(self):

        dataset = get_dataset(train=False)
        # dataloader = DataLoader(dataset, batch_size=8, num_workers=8)
        self.torch_dl = DataLoader(dataset, batch_size=1)

        self.enum_data = iter(self.torch_dl)
        self.count = 0

    def to_numpy(self, pt_tensor):
        return pt_tensor.detach().cpu().numpy() if pt_tensor.requires_grad else pt_tensor.cpu().numpy()

    def get_next(self):
        print(self.count)
        self.count += 1
        if self.count > 200:
            return None
        batch = next(self.enum_data, None)
        if batch is not None:
          return {'image': self.to_numpy(batch[0].float())}
        else:
          return None

    def rewind(self):
        self.enum_data = iter(self.torch_dl)


def optimize1(nodes_to_quantize=None):
    qdr = QuntizationDataReader()

    config = StaticQuantConfig(qdr, 
                               quant_format=QuantFormat.QOperator, 
                               activation_type=QuantType.QUInt8, 
                               nodes_to_quantize=nodes_to_quantize,
                               extra_options={
                                   'CalibMovingAverage': True,
                               })
    quantize('TextDetector.infer.onnx',
             'TextDetector.quant.onnx',
             config)

if __name__ == "__main__":
    from onnxruntime.quantization.shape_inference import quant_pre_process
    import onnx
    import os

    if os.path.exists('TextDetector.infer.onnx'):
        os.remove('TextDetector.infer.onnx')
    if os.path.exists('TextDetector.quant.onnx'):
        os.remove('TextDetector.quant.onnx')
    
    quant_pre_process(
        'TextDetector.onnx',
        'TextDetector.infer.onnx',
    )

    model = onnx.load('TextDetector.infer.onnx')
    nodes = model.graph.node
    nodes_to_quantize = []
    for node in nodes:
        if '/detector/backbone/' in node.name:
            nodes_to_quantize.append(node.name)
        if '/detector/' in node.name and 'upsamplers' in node.name:
            nodes_to_quantize.append(node.name)
    print(nodes_to_quantize)

    optimize1(nodes_to_quantize)

