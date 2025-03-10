from torchvision.models.efficientnet import EfficientNet, FusedMBConvConfig, MBConvConfig
from torchvision.models import efficientnet_v2_l, efficientnet_v2_m, efficientnet_v2_s
import torch
from torch import nn, Tensor
from torch.nn.modules import Conv2d

from functools import partial
from typing import Any

from util_func import feature_dim, modulo_list, width, height

def efficientnet_v2_xl(**kwargs: Any) -> EfficientNet:
    inverted_residual_setting = [
        FusedMBConvConfig(1, 3, 1, 32, 32, 4),
        FusedMBConvConfig(4, 3, 2, 32, 64, 8),
        FusedMBConvConfig(4, 3, 2, 64, 96, 8),
        MBConvConfig(4, 3, 2, 96, 192, 16),
        MBConvConfig(6, 3, 1, 192, 256, 24),
        MBConvConfig(6, 3, 2, 256, 512, 32),
        MBConvConfig(6, 3, 1, 512, 640, 8),
    ]
    last_channel = 1280
    dropout = 0.5

    model = EfficientNet(inverted_residual_setting, dropout, 
                         last_channel=last_channel,
                         norm_layer=partial(nn.BatchNorm2d, eps=1e-03), **kwargs)
    return model

def load_weight(model: EfficientNet, weight_path: str) -> EfficientNet:
    import numpy as np
    import os

    if not os.path.exists(weight_path):
        print('not found:', weight_path)
        return model

    with np.load(weight_path) as weights:
        print('loading weights')
        def apply_weights(func, base, tag=None):
            if isinstance(func, Conv2d):
                state_dict = func.state_dict()
                for key in state_dict.keys():
                    if key == 'weight':
                        if tag:
                            target = base + tag
                            state_dict[key] = torch.from_numpy(weights[target]).permute(2,3,0,1)
                        else:
                            target = base + 'kernel'
                            state_dict[key] = torch.from_numpy(weights[target]).permute(3,2,0,1)
                func.load_state_dict(state_dict)
            elif isinstance(func, nn.BatchNorm2d):
                state_dict = func.state_dict()
                for key in state_dict.keys():
                    if key == 'weight':
                        target = base + 'gamma'
                        state_dict[key] = torch.from_numpy(weights[target])
                    elif key == 'bias':
                        target = base + 'beta'
                        state_dict[key] = torch.from_numpy(weights[target])
                    elif key == 'running_mean':
                        target = base + 'moving_mean'
                        state_dict[key] = torch.from_numpy(weights[target])
                    elif key == 'running_var':
                        target = base + 'moving_variance'
                        state_dict[key] = torch.from_numpy(weights[target])
                func.load_state_dict(state_dict)
                    
        idx = 0
        for i,section in enumerate(model.features):
            if i == 0:
                for func in section:
                    if isinstance(func, Conv2d):
                        apply_weights(func, 'efficientnetv2-xl/stem/conv2d/')
                    elif isinstance(func, nn.BatchNorm2d):
                        apply_weights(func, 'efficientnetv2-xl/stem/tpu_batch_normalization/')
            elif i < 8:
                for sec in section:
                    if len(sec.block) == 1:
                        for func in sec.block[0]:
                            if isinstance(func, Conv2d):
                                apply_weights(func, 'efficientnetv2-xl/blocks_%d/conv2d/'%idx)
                            elif isinstance(func, nn.BatchNorm2d):
                                apply_weights(func, 'efficientnetv2-xl/blocks_%d/tpu_batch_normalization/'%idx)
                    elif len(sec.block) == 2:
                        for func in sec.block[0]:
                            if isinstance(func, Conv2d):
                                apply_weights(func, 'efficientnetv2-xl/blocks_%d/conv2d/'%idx)
                            elif isinstance(func, nn.BatchNorm2d):
                                apply_weights(func, 'efficientnetv2-xl/blocks_%d/tpu_batch_normalization/'%idx)
                        for func in sec.block[1]:
                            if isinstance(func, Conv2d):
                                apply_weights(func, 'efficientnetv2-xl/blocks_%d/conv2d_1/'%idx)
                            elif isinstance(func, nn.BatchNorm2d):
                                apply_weights(func, 'efficientnetv2-xl/blocks_%d/tpu_batch_normalization_1/'%idx)
                    elif len(sec.block) == 4:
                        for func in sec.block[0]:
                            if isinstance(func, Conv2d):
                                apply_weights(func, 'efficientnetv2-xl/blocks_%d/conv2d/'%idx)
                            elif isinstance(func, nn.BatchNorm2d):
                                apply_weights(func, 'efficientnetv2-xl/blocks_%d/tpu_batch_normalization/'%idx)
                        for func in sec.block[1]:
                            if isinstance(func, Conv2d):
                                apply_weights(func, 'efficientnetv2-xl/blocks_%d/depthwise_conv2d/'%idx, 'depthwise_kernel')
                            elif isinstance(func, nn.BatchNorm2d):
                                apply_weights(func, 'efficientnetv2-xl/blocks_%d/tpu_batch_normalization_1/'%idx)
                        apply_weights(sec.block[2].fc1, 'efficientnetv2-xl/blocks_%d/se/conv2d/'%idx)
                        apply_weights(sec.block[2].fc2, 'efficientnetv2-xl/blocks_%d/se/conv2d_1/'%idx)
                        for func in sec.block[3]:
                            if isinstance(func, Conv2d):
                                apply_weights(func, 'efficientnetv2-xl/blocks_%d/conv2d_1/'%idx)
                            elif isinstance(func, nn.BatchNorm2d):
                                apply_weights(func, 'efficientnetv2-xl/blocks_%d/tpu_batch_normalization_2/'%idx)
                    idx += 1
            else:
                for func in section:
                    if isinstance(func, Conv2d):
                        apply_weights(func, 'efficientnetv2-xl/head/conv2d/')
                    elif isinstance(func, nn.BatchNorm2d):
                        apply_weights(func, 'efficientnetv2-xl/head/tpu_batch_normalization/')
    return model

class BackboneModel(nn.Module):
    def __init__(self, pre_weights=True, model_size='xl', **kwargs):
        super().__init__(**kwargs)
        if model_size == 'xl':
            import os
            model = efficientnet_v2_xl()
            if pre_weights:
                load_weight(model, os.path.join(os.path.dirname(__file__),'efficientnetv2-xl-21k.npz'))
        elif model_size == 'l':
            model = efficientnet_v2_l(weights='DEFAULT' if pre_weights else None)
        elif model_size == 'm':
            model = efficientnet_v2_m(weights='DEFAULT' if pre_weights else None)
        elif model_size == 's':
            model = efficientnet_v2_s(weights='DEFAULT' if pre_weights else None)
        self.features = model.features

    def forward(self, x):
        results = []
        for ii,block in enumerate(self.features):
            x = block(x)
            if ii in [2,3,5]:
                results.append(x)
        results.append(x)
        return results

class Leafmap(nn.Module):
    def __init__(self, out_dim=1, model_size='xl', **kwargs) -> None:
        super().__init__(**kwargs)
        if model_size == 'xl':
            in_dims = [64,96,256,1280]
        elif model_size == 'l':
            in_dims = [64,96,224,1280]
        elif model_size == 'm':
            in_dims = [48,80,176,1280]
        elif model_size == 's':
            in_dims = [48,64,160,1280]
        conv_dim = 192
        upsamplers = []
        self.in_bn = nn.ModuleList(
            [nn.BatchNorm2d(dim) for dim in in_dims]
        )
        for i, in_dim in enumerate(reversed(in_dims)):
            if i == 0:
                layers = nn.Sequential(
                    nn.Conv2d(in_dim, conv_dim, 3, padding=1, bias=False),
                    nn.BatchNorm2d(conv_dim),
                    nn.GELU(),
                    nn.UpsamplingBilinear2d(scale_factor=2),
                )
            elif i > 0 and i < len(in_dims) - 1:
                layers = nn.Sequential(
                    nn.Conv2d(in_dim + conv_dim, conv_dim, 3, padding=1, bias=False),
                    nn.BatchNorm2d(conv_dim),
                    nn.GELU(),
                    nn.UpsamplingBilinear2d(scale_factor=2),
                )
            else:
                layers = nn.Sequential(
                    nn.Conv2d(in_dim + conv_dim, conv_dim, 3, padding=1, bias=False),
                    nn.BatchNorm2d(conv_dim),
                    nn.GELU(),
                )
            upsamplers.append(layers)
        self.upsamplers = nn.ModuleList(upsamplers)

        self.top_conv = nn.Sequential(
            nn.Conv2d(conv_dim, out_dim, 3, padding=1),
        )

    def forward(self, x1, x2, x3, x4) -> Tensor:
        y = None
        for x, bn, up in zip(reversed([x1,x2,x3,x4]), reversed(self.in_bn), self.upsamplers):
            x = bn(x)
            if y is None:
                y = up(x)
            else:
                x = torch.cat([y,x], dim=1)
                y = up(x)
        return self.top_conv(y)

class CenterNetDetection(nn.Module):
    def __init__(self, pre_weights=True, model_size='xl', **kwargs) -> None:
        super().__init__(**kwargs)
        self.backbone = BackboneModel(pre_weights=pre_weights, model_size=model_size)
        self.keyheatmap = Leafmap(out_dim=1, model_size=model_size)
        self.sizes = Leafmap(out_dim=2, model_size=model_size)
        self.textline = Leafmap(out_dim=1, model_size=model_size)
        self.sepatator = Leafmap(out_dim=1, model_size=model_size)
        self.code1 = Leafmap(out_dim=1, model_size=model_size)
        self.code2 = Leafmap(out_dim=1, model_size=model_size)
        self.code4 = Leafmap(out_dim=1, model_size=model_size)
        self.code8 = Leafmap(out_dim=1, model_size=model_size)
        self.feature = Leafmap(out_dim=feature_dim, model_size=model_size)

    def forward(self, x):
        x = x * 2 - 1
        x = self.backbone(x)
        y = [
            self.keyheatmap(*x),
            self.sizes(*x),
            self.textline(*x),
            self.sepatator(*x),
            self.code1(*x),
            self.code2(*x),
            self.code4(*x),
            self.code8(*x),
        ]
        return torch.cat(y, dim=1), self.feature(*x)

class SimpleDecoder(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        blocks = []
        mid_dim = 2048
        for modulo in modulo_list:
            layer = nn.Sequential(
                nn.Linear(feature_dim, mid_dim, bias=False),
                nn.BatchNorm1d(mid_dim),
                nn.GELU(),
                nn.Linear(mid_dim, mid_dim, bias=False),
                nn.BatchNorm1d(mid_dim),
                nn.GELU(),
                nn.Linear(mid_dim, modulo),
            )
            blocks.append(layer)
        self.blocks = nn.ModuleList(blocks)
    
    def forward(self, x):
        y = []
        for block in self.blocks:
            y.append(block(x))
        return y

class TextDetectorModel(nn.Module):
    def __init__(self, pre_weights=True, model_size='xl', **kwargs) -> None:
        super().__init__(**kwargs)
        self.detector = CenterNetDetection(pre_weights=pre_weights, model_size=model_size)
        self.decoder = SimpleDecoder()

    def forward(self, x, fmask):
        heatmap, features = self.detector(x)

        features = torch.permute(features, (0,2,3,1)).flatten(0,-2)
        decoder_outputs = self.decoder(features[fmask])

        return heatmap, decoder_outputs

    def get_fmask(self, heatmap, mask) -> Tensor:
        # heatmap: [-1, 9, 192, 192]
        batch_dim = heatmap.shape[0]
        labelmaps = heatmap[:,0,:,:]
        labelmaps = labelmaps.flatten()

        sort_idx = torch.argsort(labelmaps, descending=True)
        if mask is None or mask.shape != sort_idx.shape:
            mask = torch.zeros_like(sort_idx, dtype=torch.bool, device=sort_idx.device)
        mask.fill_(0)
        mask[sort_idx[:1024*batch_dim]] = True
        return mask

class CenterNetDetector(nn.Module):
    def __init__(self, detector, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.detector = detector
        self.minval = torch.tensor(float("-inf"))
        
    def forward(self, x):
        heatmap, features = self.detector(x)
        keymap = heatmap[:,0:1,:,:]
        # keep local maxima of 3x3, other to -inf
        local_peak = torch.nn.functional.pad(keymap, (1,1,1,1), value=self.minval.to(dtype=keymap.dtype))
        local_peak = torch.nn.functional.max_pool2d(local_peak, kernel_size=3, stride=1)
        detectedkey = torch.where(keymap < local_peak, self.minval.to(dtype=keymap.dtype), keymap)
        return torch.cat([keymap, detectedkey, heatmap[:,1:,:,:]], dim=1), features

class CodeDecoder(nn.Module):
    def __init__(self, decoder, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.decoder = decoder

    def forward(self, x):
        x = self.decoder(x)
        return tuple([torch.nn.functional.softmax(x1, dim=-1) for x1 in x])

if __name__=="__main__":
    import os
    from torchinfo import summary

    # model = BackboneModel(model_size='s')
    # print(model)
    # with torch.no_grad():
    #     outputs = model(torch.zeros(1, 3, height, width))
    # print([t.shape for t in outputs])

    # exit()

    model = CenterNetDetection()
    print(model)
    summary(model, input_size=[[1, 3, height, width]])
    with torch.no_grad():
        outputs = model(torch.zeros(1, 3, height, width))
    print([t.shape for t in outputs])

    model = SimpleDecoder()
    print(model)
    summary(model, input_size=[[1, feature_dim]])
    with torch.no_grad():
        outputs = model(torch.zeros(2, feature_dim))
    print([t.shape for t in outputs])
