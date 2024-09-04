import torch
from torch import nn, Tensor

from .vit import mixViT
from util_func import feature_dim, modulo_list, width, height

class BackboneModel(nn.Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = mixViT(
            image_size = width,
            dim = 1024,
            depth = 16,
            heads = 16,
            mlp_dim = 1024,
            dim_head = 64,
        )

    def forward(self, x):
        return self.model(x)

class Leafmap(nn.Module):
    def __init__(self, out_dim=1, mid_dim=64, **kwargs) -> None:
        super().__init__(**kwargs)
        in_dims = [128,256,512,1024]
        conv_dims = [8,8,16,32]
        upsamplers = []
        for i, (in_dim, o_dim) in enumerate(zip(in_dims, conv_dims)):
            layers = nn.Sequential(
                nn.Conv2d(in_dim, o_dim, 3, padding=1, bias=False),
                nn.BatchNorm2d(o_dim),
                nn.Upsample(scale_factor=2**(i+1), mode='bilinear', align_corners=True),
            )
            upsamplers.append(layers)
        self.upsamplers = nn.ModuleList(upsamplers)

        self.top_conv = nn.Sequential(
            nn.GELU(),
            nn.Conv2d(sum(conv_dims), mid_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_dim),
            nn.GELU(),
            nn.Conv2d(mid_dim, out_dim, 1),
        )

    def forward(self, x1, x2, x3, x4) -> Tensor:
        y = []
        for x, up in zip([x1,x2,x3,x4], self.upsamplers):
            y.append(up(x))
        x = torch.cat(y, dim=1)
        return self.top_conv(x)

class Leafmap2(nn.Module):
    def __init__(self, out_dim=1, mid_dim=256, **kwargs) -> None:
        super().__init__(**kwargs)
        in_dims = [128,256,512,1024]
        conv_dims = [32,32,64,128]
        upsamplers = []
        for i, (in_dim, o_dim) in enumerate(zip(in_dims, conv_dims)):
            layers = nn.Sequential(
                nn.Conv2d(in_dim, o_dim, 3, padding=1, bias=False),
                nn.BatchNorm2d(o_dim),
                nn.Upsample(scale_factor=2**(i+1), mode='bilinear', align_corners=True),
            )
            upsamplers.append(layers)
        self.upsamplers = nn.ModuleList(upsamplers)

        self.top_conv = nn.Sequential(
            nn.GELU(),
            nn.Conv2d(sum(conv_dims), mid_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_dim),
            nn.GELU(),
            nn.Conv2d(mid_dim, out_dim, 1),
        )

    def forward(self, x1, x2, x3, x4) -> Tensor:
        y = []
        for x, up in zip([x1,x2,x3,x4], self.upsamplers):
            y.append(up(x))
        x = torch.cat(y, dim=1)
        return self.top_conv(x)

class CenterNetDetection(nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.backbone = BackboneModel()
        self.keyheatmap = Leafmap(out_dim=1)
        self.sizes = Leafmap(out_dim=2)
        self.textline = Leafmap(out_dim=1)
        self.sepatator = Leafmap(out_dim=1)
        self.code1 = Leafmap(out_dim=1)
        self.code2 = Leafmap(out_dim=1)
        self.code4 = Leafmap(out_dim=1)
        self.code8 = Leafmap(out_dim=1)
        self.feature = Leafmap2(out_dim=feature_dim)

    def forward(self, x):
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
                nn.Linear(feature_dim, mid_dim),
                nn.SiLU(inplace=True),
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
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.detector = CenterNetDetection()
        self.decoder = SimpleDecoder()

    def forward(self, x, fmask):
        heatmap, features = self.detector(x)

        features = torch.permute(features, (0,2,3,1)).flatten(0,-2)
        decoder_outputs = self.decoder(features[fmask])

        return heatmap, decoder_outputs

    def get_fmask(self, heatmap) -> Tensor:
        # heatmap: [-1, 11, 256, 256]
        batch_dim = heatmap.shape[0]
        labelmaps = heatmap[:,0,:,:]
        labelmaps = labelmaps.flatten()

        sort_idx = torch.argsort(labelmaps, descending=True)
        return sort_idx[:1024*batch_dim]

class CenterNetDetector(nn.Module):
    def __init__(self, detector, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.detector = detector
        
    def forward(self, x):
        heatmap, features = self.detector(x)
        keymap = heatmap[:,0:1,:,:]
        local_peak = torch.nn.functional.max_pool2d(keymap, kernel_size=5, stride=1, padding=2)
        detectedkey = torch.where(keymap == local_peak, keymap, torch.finfo(torch.float16).min)
        return torch.cat([keymap, detectedkey, heatmap[:,1:,:,:]], dim=1), features

class CodeDecoder(nn.Module):
    def __init__(self, decoder, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.decoder = decoder

    def forward(self, x):
        x = self.decoder(x)
        return tuple([torch.nn.functional.softmax(x1, dim=-1) for x1 in x])

if __name__=="__main__":
    from torchinfo import summary

    model = BackboneModel()
    print(model)
    with torch.no_grad():
        outputs = model(torch.zeros(1, 3, height, width))
    print([t.shape for t in outputs])

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
        outputs = model(torch.zeros(1, feature_dim))
    print([t.shape for t in outputs])
