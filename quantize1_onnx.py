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


def optimize1(nodes_to_exclude=None):
    qdr = QuntizationDataReader()

    config = StaticQuantConfig(qdr, 
                               quant_format=QuantFormat.QOperator, 
                               activation_type=QuantType.QUInt8, 
                               nodes_to_exclude=nodes_to_exclude,
                               extra_options={
                                   'CalibMovingAverage': True,
                               })
    quantize('TextDetector.onnx',
             'TextDetector.quant.onnx',
             config)

if __name__ == "__main__":
    from onnx import shape_inference
    import onnx
    import os

    if os.path.exists('TextDetector.quant.onnx'):
        os.remove('TextDetector.quant.onnx')
    
    model = onnx.load('TextDetector.onnx')
    model = shape_inference.infer_shapes(model)
    outputs = [o.name for o in model.graph.output]
    nodes_to_exclude = []
    for node in model.graph.node:
        if 'feature' in node.output:
            nodes_to_exclude.append(node.name)

    outputs = ['heatmap']
    while outputs:
        next_intput = []
        for output in outputs:
            for node in model.graph.node:
                if output in node.output:
                    nodes_to_exclude.append(node.name)
                    if node.op_type != 'Conv':
                        next_intput += node.input
        outputs = list(set(next_intput))

    nodes_to_exclude = list(set(nodes_to_exclude))
    print(nodes_to_exclude)

    optimize1(nodes_to_exclude)

