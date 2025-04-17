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
                               weight_type=QuantType.QInt8,
                               nodes_to_exclude=nodes_to_exclude,
                               extra_options={
                                   'CalibMovingAverage': True,
                               })
    quantize('TextDetector.pre.onnx',
             'TextDetector.quant.onnx',
             config)

def convert2():
    import onnx
    model = onnx.load("TextDetector.quant.onnx")

    model.graph.input[0].type.tensor_type.elem_type = 10
    model.graph.output[0].type.tensor_type.elem_type = 10
    model.graph.output[1].type.tensor_type.elem_type = 10

    cast_node = onnx.helper.make_node(op_type='Cast', name='cast_'+model.graph.input[0].name, inputs=[model.graph.input[0].name], outputs=['cast_'+model.graph.input[0].name], to=1)

    node = [node for node in model.graph.node if model.graph.input[0].name in node.input][0]
    node.input[node.input.index(model.graph.input[0].name)] = 'cast_'+model.graph.input[0].name
    model.graph.node.insert(0, cast_node)

    cast_node = onnx.helper.make_node(op_type='Cast', name='cast_'+model.graph.output[0].name, inputs=['cast_'+model.graph.output[0].name], outputs=[model.graph.output[0].name], to=10)

    node = [node for node in model.graph.node if model.graph.output[0].name in node.output][0]
    node.output[0] = 'cast_'+model.graph.output[0].name
    model.graph.node.insert(model.graph.node.index(node)+1, cast_node)

    cast_node = onnx.helper.make_node(op_type='Cast', name='cast_'+model.graph.output[1].name, inputs=['cast_'+model.graph.output[1].name], outputs=[model.graph.output[1].name], to=10)

    node = [node for node in model.graph.node if model.graph.output[1].name in node.output][0]
    node.output[0] = 'cast_'+model.graph.output[1].name
    model.graph.node.insert(model.graph.node.index(node)+1, cast_node)

    graph = onnx.helper.make_graph(model.graph.node, model.graph.name, model.graph.input, model.graph.output, model.graph.initializer)
    info_model = onnx.helper.make_model(graph, opset_imports=model.opset_import)
    model_fixed = onnx.shape_inference.infer_shapes(info_model)

    onnx.checker.check_model(model_fixed)
    onnx.save(model_fixed, 'TextDetector.quant.fp16.onnx')

if __name__ == "__main__":
    from onnxruntime.quantization.shape_inference import quant_pre_process
    from onnx import shape_inference
    import onnx
    import os

    if os.path.exists('TextDetector.quant.onnx'):
        os.remove('TextDetector.quant.onnx')
    if os.path.exists('TextDetector.pre.onnx'):
        os.remove('TextDetector.pre.onnx')
    
    quant_pre_process('TextDetector.onnx', 'TextDetector.pre.onnx', skip_symbolic_shape=True)

    model = onnx.load('TextDetector.pre.onnx')
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

    convert2()

